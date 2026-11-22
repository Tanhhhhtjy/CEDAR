#!/usr/bin/env python3
"""Cross-backend parity verifier (v2): HF live adapter vs vLLM native `--enable-lora` with 100% 4-option coverage.

Audited v2 fixes (incorporating Astra's pre-flight audit):
  1. 100% 4-option (`A/B/C/D`) coverage via `prompt_logprobs`: appends each option token ID directly onto
     the fixed `prompt_token_ids` prefix and reads the exact last-position `prompt_logprobs` entry.
     NEVER skips missing options (any `n_missing_u > 0` or `coverage < 1.0` strictly FAILS!).
  2. Enforces exact thresholds without silent relaxation:
     - `option_match_rate >= 0.95`
     - `abs(du_signed_mean) <= 0.25`
     - `du_abs_mean <= 0.35` (strictly rejecting `0.36`!)
     - `u_coverage_rate == 1.0` (100% of positions compared!)
  3. Cryptographic SHA-256 binding: records `adapter_state_sha256`, `peft_adapter_model_sha256`, `r_export`,
     and `vllm_max_lora_rank` so every parity artifact is cryptographically tied to exact weights.

Usage (GPU pre-flight on 1 GPU, ~2 min):
  CUDA_VISIBLE_DEVICES=0 python experiments/ACL-exp-c1/scripts/c1_check_vllm_lora_parity.py \
      --base-model /mnt/kcl/devuser/ckpts/Qwen3-8B --model-key qwen3_8b \
      --components-file experiments/ACL-exp-c1/data/c1_components_frozen.json \
      --arm Targeted-Dual --adapter-state <PATH>/adapter_state_step0200.pt \
      --peft-lora-dir <TMP_PEFT_DIR> \
      --dataset c1_dev --limit 40 \
      --out-json experiments/ACL-exp-c1/results/step2b_8b_showdown/std_targeted_dual/vllm_lora_parity.json

Usage (CPU self-test):
  python experiments/ACL-exp-c1/scripts/c1_check_vllm_lora_parity.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
C1 = HERE.parents[0]
REPO = C1.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-a1" / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-b1" / "scripts"))

import b1_lib as B1L  # noqa: E402
import c1_eval as EV  # noqa: E402
import c1_export_peft_lora as EXP  # noqa: E402
import c1_modules as M  # noqa: E402
from render_prompts import build_messages, load_config  # noqa: E402


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compare_backend_records(
    hf_recs: list[dict[str, Any]],
    vllm_recs: list[dict[str, Any]],
    *,
    max_abs_mean_du: float = 0.25,
    max_abs_du_mean: float = 0.35,
    min_option_match: float = 0.95,
) -> dict[str, Any]:
    """Compare per-position HF vs vLLM records requiring 100% 4-option coverage and exact thresholds."""
    assert len(hf_recs) == len(vllm_recs) and len(hf_recs) > 0
    du_signed: list[float] = []
    du_abs: list[float] = []
    n_missing_u = 0
    n_missing_any_opt = 0
    n_opt_match = 0
    n_near_boundary = 0
    n_near_boundary_flip = 0
    flips: list[dict[str, Any]] = []

    hf_by_ba: dict[tuple[str, str], dict[str, Any]] = {}
    vl_by_ba: dict[tuple[str, str], dict[str, Any]] = {}

    for hr, vr in zip(hf_recs, vllm_recs, strict=True):
        assert (hr["base_id"], hr["arm"]) == (vr["base_id"], vr["arm"])
        hf_by_ba[(hr["base_id"], hr["arm"])] = hr
        vl_by_ba[(vr["base_id"], vr["arm"])] = vr
        same = hr["pred"] == vr["pred"]
        n_opt_match += int(same)
        uh, uv = hr.get("u_theta"), vr.get("u_theta")
        near = (uh is not None) and (abs(uh) < 1.0)
        n_near_boundary += int(near)
        if near and not same:
            n_near_boundary_flip += 1
        vl_lps = vr.get("opt_logprobs")
        if vl_lps is not None and any(vl_lps.get(k) is None for k in ("A", "B", "C", "D")):
            n_missing_any_opt += 1
        if uh is not None and uv is not None:
            d = float(uv) - float(uh)
            du_signed.append(d)
            du_abs.append(abs(d))
        else:
            n_missing_u += 1
        if not same:
            flips.append(
                {
                    "base_id": hr["base_id"],
                    "arm": hr["arm"],
                    "hf_pred": hr["pred"],
                    "vllm_pred": vr["pred"],
                    "u_hf": round(uh, 4) if uh is not None else None,
                    "u_vllm": round(uv, 4) if uv is not None else None,
                }
            )

    bases = sorted({r["base_id"] for r in hf_recs})
    dc_hf = 0
    dc_vl = 0
    n_vf_pairs = 0
    for b in bases:
        hv, hf = hf_by_ba.get((b, "V")), hf_by_ba.get((b, "F"))
        vv, vf = vl_by_ba.get((b, "V")), vl_by_ba.get((b, "F"))
        if hv and hf and vv and vf:
            n_vf_pairs += 1
            dc_hf += int(hv["pred"] == hv["target_answer"] and hf["pred"] == hf["initial_answer"])
            dc_vl += int(vv["pred"] == vv["target_answer"] and vf["pred"] == vf["initial_answer"])

    n_total = len(hf_recs)
    opt_match_rate = n_opt_match / n_total
    u_coverage_rate = len(du_signed) / n_total
    mean_signed_du = statistics.mean(du_signed) if du_signed else None
    median_signed_du = statistics.median(du_signed) if du_signed else None
    mean_abs_du = statistics.mean(du_abs) if du_abs else None
    max_abs_du = max(du_abs) if du_abs else None
    pos_frac = (sum(d > 0 for d in du_signed) / len(du_signed)) if du_signed else None

    passed = bool(
        n_missing_u == 0
        and n_missing_any_opt == 0
        and u_coverage_rate == 1.0
        and opt_match_rate >= min_option_match
        and mean_signed_du is not None
        and abs(mean_signed_du) <= max_abs_mean_du
        and mean_abs_du is not None
        and mean_abs_du <= max_abs_du_mean
    )
    return {
        "n_positions": n_total,
        "n_vf_pairs": n_vf_pairs,
        "u_coverage_rate": round(u_coverage_rate, 6),
        "n_u_compared": len(du_signed),
        "n_u_missing": n_missing_u,
        "n_positions_missing_any_option": n_missing_any_opt,
        "option_match_count": n_opt_match,
        "option_match_rate": round(opt_match_rate, 4),
        "double_correct_hf": dc_hf,
        "double_correct_vllm": dc_vl,
        "double_correct_delta": dc_vl - dc_hf,
        "du_signed_mean": round(mean_signed_du, 5) if mean_signed_du is not None else None,
        "du_signed_median": round(median_signed_du, 5) if median_signed_du is not None else None,
        "du_abs_mean": round(mean_abs_du, 5) if mean_abs_du is not None else None,
        "du_abs_max": round(max_abs_du, 5) if max_abs_du is not None else None,
        "du_positive_frac": round(pos_frac, 4) if pos_frac is not None else None,
        "n_near_boundary_lt1": n_near_boundary,
        "n_near_boundary_flips": n_near_boundary_flip,
        "thresholds": {
            "max_abs_mean_du": max_abs_mean_du,
            "max_abs_du_mean": max_abs_du_mean,
            "min_option_match": min_option_match,
            "required_u_coverage": 1.0,
        },
        "flips": flips,
        "passed": passed,
    }


@torch.no_grad()
def _score_hf_fixed_ids(model, items: list[dict[str, Any]], opt_tok_ids: dict[str, int]) -> list[dict[str, Any]]:
    dev = EV._input_device(model)
    recs: list[dict[str, Any]] = []
    for it in items:
        t_ids = torch.tensor(it["token_ids"], dtype=torch.long, device=dev)[None]
        logp = F.log_softmax(model(input_ids=t_ids).logits[0, -1].float(), dim=-1)
        opt_lp = {letter: float(logp[tid]) for letter, tid in opt_tok_ids.items()}
        pred = max(opt_lp, key=opt_lp.get)
        u_theta = opt_lp[it["target_answer"]] - opt_lp[it["initial_answer"]]
        recs.append(
            {
                "base_id": it["base_id"],
                "arm": it["arm"],
                "initial_answer": it["initial_answer"],
                "target_answer": it["target_answer"],
                "pred": pred,
                "u_theta": u_theta,
                "opt_logprobs": opt_lp,
            }
        )
    return recs


def _score_vllm_fixed_ids_prompt_logprobs(
    llm,
    items: list[dict[str, Any]],
    opt_tok_ids: dict[str, int],
    *,
    lora_request=None,
) -> list[dict[str, Any]]:
    """Score ALL 4 options ('A','B','C','D') on EVERY position via `prompt_logprobs=1` -> 100% coverage!"""
    from vllm import SamplingParams

    letters = ("A", "B", "C", "D")
    prompts = []
    for it in items:
        prefix = list(it["token_ids"])
        for letter in letters:
            prompts.append({"prompt_token_ids": prefix + [opt_tok_ids[letter]]})

    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=1)
    if lora_request is not None:
        outputs = llm.generate(prompts, sp, lora_request=lora_request)
    else:
        outputs = llm.generate(prompts, sp)

    assert len(outputs) == len(items) * 4
    recs: list[dict[str, Any]] = []
    for idx, it in enumerate(items):
        opt_lp: dict[str, float | None] = {}
        for j, letter in enumerate(letters):
            out = outputs[idx * 4 + j]
            tid = opt_tok_ids[letter]
            last_lp_map = out.prompt_logprobs[-1] if out.prompt_logprobs else None
            entry = last_lp_map.get(tid) if isinstance(last_lp_map, dict) else None
            opt_lp[letter] = float(entry.logprob) if entry is not None else None
        if all(opt_lp[k] is not None for k in letters):
            pred = max(letters, key=lambda k: float(opt_lp[k]))  # type: ignore[arg-type]
            u_theta = float(opt_lp[it["target_answer"]]) - float(opt_lp[it["initial_answer"]])  # type: ignore[arg-type]
        else:
            pred = None
            u_theta = None
        recs.append(
            {
                "base_id": it["base_id"],
                "arm": it["arm"],
                "initial_answer": it["initial_answer"],
                "target_answer": it["target_answer"],
                "pred": pred,
                "u_theta": u_theta,
                "opt_logprobs": opt_lp,
            }
        )
    return recs


def run_selftest_tiny() -> None:
    """CPU self-test covering passing parity + ALL 3 Astra counterexamples (BF16 drift, missing u, du=0.36)."""
    hf_recs, vl_good, vl_drift, vl_missing, vl_036 = [], [], [], [], []
    for i in range(20):
        bid = f"TEST-{i+1:03d}"
        for arm, u_base, pred in (("V", 2.0, "B"), ("F", -2.0, "A")):
            lps_ok = {"A": -2.0, "B": -0.1, "C": -5.0, "D": -5.0}
            hf_recs.append({"base_id": bid, "arm": arm, "initial_answer": "A", "target_answer": "B", "pred": pred, "u_theta": u_base, "opt_logprobs": lps_ok})
            # 1. Good parity: tiny symmetric noise (+/- 0.02)
            noise = 0.02 if (i % 2 == 0) else -0.02
            vl_good.append({"base_id": bid, "arm": arm, "initial_answer": "A", "target_answer": "B", "pred": pred, "u_theta": u_base + noise, "opt_logprobs": lps_ok})
            # 2. Systematic +4.5 logit shift (BF16 merge style)
            bad_u = u_base + 4.5
            vl_drift.append({"base_id": bid, "arm": arm, "initial_answer": "A", "target_answer": "B", "pred": ("B" if bad_u > 0 else "A"), "u_theta": bad_u, "opt_logprobs": lps_ok})
            # 3. Astra CE: 39/40 missing u_theta -> MUST FAIL!
            keep = (i == 0 and arm == "V")
            vl_missing.append({"base_id": bid, "arm": arm, "initial_answer": "A", "target_answer": "B", "pred": pred, "u_theta": (u_base if keep else None), "opt_logprobs": (lps_ok if keep else {"A": None, "B": -0.1, "C": None, "D": None})})
            # 4. Astra CE: zero mean drift, 100% match, BUT du_abs_mean = 0.36 (> 0.35) -> MUST FAIL!
            err036 = 0.36 if (i % 2 == 0) else -0.36
            vl_036.append({"base_id": bid, "arm": arm, "initial_answer": "A", "target_answer": "B", "pred": pred, "u_theta": u_base + err036, "opt_logprobs": lps_ok})

    rep_ok = compare_backend_records(hf_recs, vl_good)
    assert rep_ok["passed"] is True and rep_ok["u_coverage_rate"] == 1.0, rep_ok
    rep_drift = compare_backend_records(hf_recs, vl_drift)
    assert rep_drift["passed"] is False, rep_drift
    rep_miss = compare_backend_records(hf_recs, vl_missing)
    assert rep_miss["passed"] is False and rep_miss["n_u_missing"] == 39, rep_miss
    rep_036 = compare_backend_records(hf_recs, vl_036)
    assert rep_036["passed"] is False and abs(rep_036["du_abs_mean"] - 0.36) < 1e-4, rep_036
    print(
        "[OK] c1_check_vllm_lora_parity v2 CPU self-test passed "
        "(good=PASS, drift+4.5=FAIL, 39/40-missing=FAIL, du_abs=0.36=FAIL)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", help="Base HF model path (e.g. Qwen3-8B)")
    ap.add_argument("--model-key", choices=["qwen3_8b", "qwen3_14b"])
    ap.add_argument("--components-file", type=Path, default=C1 / "data" / "c1_components_frozen.json")
    ap.add_argument("--arm", help="Arm name in components file")
    ap.add_argument("--adapter-state", type=Path, help="Raw adapter_state*.pt for HF live reference")
    ap.add_argument("--peft-lora-dir", type=Path, help="PEFT LoRA dir exported by c1_export_peft_lora.py")
    ap.add_argument("--dataset", default="c1_dev", choices=list(EV.DEFAULT_DATASETS))
    ap.add_argument("--arms", nargs="*", default=["V", "F"])
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-abs-mean-du", type=float, default=0.25)
    ap.add_argument("--max-abs-du-mean", type=float, default=0.35)
    ap.add_argument("--min-option-match", type=float, default=0.95)
    ap.add_argument("--out-json", type=Path, required=False)
    ap.add_argument("--selftest-tiny", action="store_true")
    args = ap.parse_args()

    if args.selftest_tiny:
        run_selftest_tiny()
        return

    assert args.base_model and args.model_key and args.arm and args.adapter_state and args.peft_lora_dir
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    t0 = time.time()
    a1_cfg = load_config()
    tok = AutoTokenizer.from_pretrained(args.base_model)
    opt_tok_ids = EV.get_cached_option_token_ids(tok)
    rows = [json.loads(l) for l in EV.DEFAULT_DATASETS[args.dataset].read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        rows = rows[: args.limit]

    items: list[dict[str, Any]] = []
    for r in rows:
        for arm in args.arms:
            msgs = build_messages(a1_cfg, r, arm, "no_think")
            ids = B1L.encode_prompt(tok, msgs, "Final:", enable_thinking=False).tolist()
            items.append(
                {
                    "base_id": r["base_id"],
                    "arm": arm,
                    "initial_answer": r["initial_answer"],
                    "target_answer": r["target_answer"],
                    "token_ids": ids,
                }
            )

    adapter_sha = _sha256_file(args.adapter_state)
    peft_sf = args.peft_lora_dir / "adapter_model.safetensors"
    peft_sha = _sha256_file(peft_sf)

    # 1. HF base M0 + HF live adapter (free GPU memory completely before starting vLLM!)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="auto").eval()
    hf_m0 = _score_hf_fixed_ids(model, items, opt_tok_ids)

    comp = json.loads(args.components_file.read_text(encoding="utf-8"))
    arm_spec = comp["models"][args.model_key]["arms"][args.arm]
    scaling = EV.checkpoint_scaling(args.adapter_state, args.model_key, args.arm, arm_spec)
    M.apply_c1_adapters(model, arm_spec, **scaling)
    M.load_c1_adapter_state(model, args.adapter_state)
    model.eval()
    hf_live = _score_hf_fixed_ids(model, items, opt_tok_ids)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. vLLM base M0 + vLLM native LoRA on EXACT SAME fixed token IDs with 100% 4-option prompt_logprobs
    peft_cfg = json.loads((args.peft_lora_dir / "adapter_config.json").read_text(encoding="utf-8"))
    r_export = int(peft_cfg["r"])
    max_lora_rank = EXP.recommended_vllm_max_lora_rank(r_export)
    llm = LLM(
        model=args.base_model,
        dtype="bfloat16",
        tensor_parallel_size=args.tp,
        enable_lora=True,
        max_lora_rank=max_lora_rank,
    )
    vl_m0 = _score_vllm_fixed_ids_prompt_logprobs(llm, items, opt_tok_ids, lora_request=None)
    lora_req = LoRARequest("c1_lora", 1, str(args.peft_lora_dir))
    vl_lora = _score_vllm_fixed_ids_prompt_logprobs(llm, items, opt_tok_ids, lora_request=lora_req)

    m0_cmp = compare_backend_records(
        hf_m0, vl_m0,
        max_abs_mean_du=args.max_abs_mean_du,
        max_abs_du_mean=args.max_abs_du_mean,
        min_option_match=args.min_option_match,
    )
    lora_cmp = compare_backend_records(
        hf_live, vl_lora,
        max_abs_mean_du=args.max_abs_mean_du,
        max_abs_du_mean=args.max_abs_du_mean,
        min_option_match=args.min_option_match,
    )

    report = {
        "schema": "cedar.acl_exp_c1.vllm_lora_parity.v2",
        "base_model": args.base_model,
        "model_key": args.model_key,
        "arm": args.arm,
        "adapter_state": str(args.adapter_state),
        "adapter_state_sha256": adapter_sha,
        "peft_lora_dir": str(args.peft_lora_dir),
        "peft_adapter_model_sha256": peft_sha,
        "r_export": r_export,
        "vllm_max_lora_rank": max_lora_rank,
        "dataset": args.dataset,
        "arms": args.arms,
        "n_bases": len(rows),
        "m0_hf_vs_vllm": m0_cmp,
        "lora_hf_live_vs_vllm_native": lora_cmp,
        "passed": bool(m0_cmp["passed"] and lora_cmp["passed"]),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
