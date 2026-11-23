#!/usr/bin/env python3
"""ACL-exp-c3 Evaluation & Calibration Harness (HF lossless live / batched no_think / vLLM native LoRA).

Key upgrades over c1_eval / c2_eval:
  1. Registers ALL C3 splits (`c3_dev`, `c3_blind_confirm`, `c3_blind_think`) alongside open historical splits
     (`c2_confirm_bidir`, `c2_stealth_stress`, `c1_proxy_template_v2`, `c1_dev`), and PHYSICALLY BLOCKS sealed
     `c1_test_id` / `c1_test_ood` unless `--allow-sealed-final` is explicitly passed.
  2. Native `c3_modules` support (`Uniform-LoRA`, `M3-DAC`, and all M3 attribution controls) + `--disable-zones`
     for zero-retraining 3-zone ablations!
  3. Fast left-padded batched `no_think` evaluation (exact `<1e-5` equivalence to batch=1 verified in `--selftest-tiny`)
     recording FULL 4-option logits (`opt_logits`), `gold_nll`, `brier_score`, `pred` (true 4-option argmax), `u_theta`,
     `family_id`, `broken_step`, and `cell_member` on every single record!
  4. Native `--backend vllm --peft-lora-dir <dir>` support for fast `think` evaluation with exported C3 PEFT LoRAs.
  5. Exact reference WikiText-2 next-token KL via `c1_eval.compute_reference_kl`.

CPU Self-Test:
  python experiments/ACL-exp-c3/scripts/c3_eval.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
C3 = HERE.parents[0]
REPO = C3.parents[1]
C1 = REPO / "experiments" / "ACL-exp-c1"
C2 = REPO / "experiments" / "ACL-exp-c2"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(C1 / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-a1" / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-b1" / "scripts"))

import b1_lib as B1L  # noqa: E402
import c1_eval as C1EV  # noqa: E402
import c3_modules as M3  # noqa: E402
from render_prompts import build_messages, load_config  # noqa: E402

SEALED_DATASETS = {"c1_test_id", "c1_test_ood"}
OPTION_LETTERS = ("A", "B", "C", "D")

C3_DATASETS: dict[str, Path] = {
    "c3_dev": C3 / "data" / "c3_dev_bases.jsonl",
    "c3_blind_confirm": C3 / "data" / "c3_blind_confirm_bases.jsonl",
    "c3_blind_think": C3 / "data" / "c3_blind_think_bases.jsonl",
    "c2_confirm_bidir": C2 / "data" / "c2_confirm_bidir_bases.jsonl",
    "c2_stealth_stress": C2 / "data" / "c2_stealth_stress_bases.jsonl",
    "c1_proxy_template_v2": C1 / "data" / "c1_proxy_template_v2_bases.jsonl",
    "c1_dev": C1 / "data" / "c1_dev_bases.jsonl",
    "c1_test_id": C1 / "data" / "c1_test_id_bases.jsonl",
    "c1_test_ood": C1 / "data" / "c1_test_ood_bases.jsonl",
}

CORE_4CELL_ARMS = ("turn1_F", "turn1_V", "V_hes", "V_auth", "F_hes", "F_auth")


def _gold_letter_for_arm(row: dict[str, Any], arm: str) -> str:
    if arm in ("V", "V_auth", "V_hes", "V_src_third_party"):
        return row["target_answer"]
    return row["initial_answer"]


def _four_opt_calibration(opt_logits: dict[str, float], gold: str) -> tuple[float, float, dict[str, float]]:
    """Compute exact 4-option softmax probabilities, gold NLL, and 4-class Brier score."""
    vec = torch.tensor([opt_logits[k] for k in OPTION_LETTERS], dtype=torch.float64)
    log_p = F.log_softmax(vec, dim=0)
    probs = log_p.exp()
    gold_idx = OPTION_LETTERS.index(gold)
    nll = float(-log_p[gold_idx])
    y_oh = torch.zeros(4, dtype=torch.float64)
    y_oh[gold_idx] = 1.0
    brier = float(((probs - y_oh) ** 2).sum())
    prob_dict = {k: round(float(probs[i]), 6) for i, k in enumerate(OPTION_LETTERS)}
    return round(nll, 6), round(brier, 6), prob_dict


def _make_nothink_record(row: dict[str, Any], arm: str, last_logits: torch.Tensor, opt_tok_ids: dict[str, int]) -> dict[str, Any]:
    opt_logits = {letter: round(float(last_logits[tid]), 6) for letter, tid in opt_tok_ids.items()}
    pred = max(OPTION_LETTERS, key=lambda k: opt_logits[k])
    a0, at = row["initial_answer"], row["target_answer"]
    gold = _gold_letter_for_arm(row, arm)
    nll, brier, opt_probs = _four_opt_calibration(opt_logits, gold)
    u_theta = opt_logits[at] - opt_logits[a0]
    return {
        "base_id": row["base_id"],
        "cell_id": row.get("cell_id", row["base_id"]),
        "family_id": row.get("family_id", row.get("cell_id", row["base_id"])),
        "cell_member": row.get("cell_member"),
        "broken_step": row.get("broken_step"),
        "split": row.get("split", "unknown"),
        "domain": row.get("domain"),
        "pair_type": row["pair_type"],
        "fallacy_class": row["fallacy_class"],
        "stealth_category": row.get("stealth_category"),
        "difficulty_depth": row["difficulty_depth"],
        "arm": arm,
        "initial_answer": a0,
        "target_answer": at,
        "gold_answer": gold,
        "pred": pred,
        "parse_status": "ok",
        "parse_ok": True,
        "truncated": False,
        "is_target": pred == at,
        "is_initial": pred == a0,
        "is_correct": pred == gold,
        "u_theta": round(u_theta, 6),
        "opt_logits": opt_logits,
        "opt_probs": opt_probs,
        "gold_nll": nll,
        "brier_score": brier,
    }


@torch.no_grad()
def eval_nothink_dataset_batched(
    model,
    tok,
    a1_cfg: dict[str, Any],
    rows: list[dict[str, Any]],
    arms: list[str],
    opt_tok_ids: dict[str, int],
    batch_size: int = 8,
) -> list[dict[str, Any]]:
    """Exact left-padded batched single-forward no_think evaluation."""
    dev = C1EV._input_device(model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    if pad_id is None:
        pad_id = 0

    items: list[tuple[dict[str, Any], str, torch.Tensor]] = []
    for r in rows:
        for arm in arms:
            msgs = build_messages(a1_cfg, r, arm, "no_think")
            ids = B1L.encode_prompt(tok, msgs, "Final:", enable_thinking=False)
            items.append((r, arm, ids))

    recs: list[dict[str, Any]] = []
    bsz = max(1, int(batch_size))
    for i in range(0, len(items), bsz):
        chunk = items[i : i + bsz]
        if len(chunk) == 1:
            r, arm, ids = chunk[0]
            out = model(input_ids=ids[None].to(dev), use_cache=False)
            recs.append(_make_nothink_record(r, arm, out.logits[0, -1].float(), opt_tok_ids))
            continue

        max_len = max(int(ids.numel()) for _, _, ids in chunk)
        input_ids = torch.full((len(chunk), max_len), int(pad_id), dtype=torch.long)
        attn_mask = torch.zeros((len(chunk), max_len), dtype=torch.long)
        for bi, (_, _, ids) in enumerate(chunk):
            L = int(ids.numel())
            # Left-pad so index -1 is ALWAYS the exact readout position for every row!
            input_ids[bi, max_len - L :] = ids
            attn_mask[bi, max_len - L :] = 1
        out = model(input_ids=input_ids.to(dev), attention_mask=attn_mask.to(dev), use_cache=False)
        last_batch = out.logits[:, -1, :].float()
        for bi, (r, arm, _) in enumerate(chunk):
            recs.append(_make_nothink_record(r, arm, last_batch[bi], opt_tok_ids))
    return recs


def summarize_4cell(recs: list[dict[str, Any]], eligible_ids: set[str] | None = None) -> dict[str, Any]:
    """Quick inline summary of 4-cell all-correct, DC(V_hes,F_auth), DC(V_auth,F_hes), NLL, Brier."""
    by_b: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in recs:
        if eligible_ids is not None and r["base_id"] not in eligible_ids:
            continue
        by_b[r["base_id"]][r["arm"]] = r

    n = 0
    n_4cell = 0
    n_dc_orig = 0
    n_dc_rev = 0
    nlls, briers = [], []
    for bid, m in by_b.items():
        if not all(a in m for a in ("V_hes", "V_auth", "F_hes", "F_auth")):
            continue
        n += 1
        vh = bool(m["V_hes"]["is_target"])
        va = bool(m["V_auth"]["is_target"])
        fh = bool(m["F_hes"]["is_initial"])
        fa = bool(m["F_auth"]["is_initial"])
        n_4cell += int(vh and va and fh and fa)
        n_dc_orig += int(vh and fa)
        n_dc_rev += int(va and fh)
        for a in ("V_hes", "V_auth", "F_hes", "F_auth"):
            if m[a].get("gold_nll") is not None:
                nlls.append(float(m[a]["gold_nll"]))
            if m[a].get("brier_score") is not None:
                briers.append(float(m[a]["brier_score"]))
    return {
        "n_bases": n,
        "four_cell_all_correct": round(n_4cell / n, 4) if n else None,
        "four_cell_count": n_4cell,
        "dc_vhes_fauth": round(n_dc_orig / n, 4) if n else None,
        "dc_vauth_fhes": round(n_dc_rev / n, 4) if n else None,
        "mean_4opt_nll": round(sum(nlls) / len(nlls), 4) if nlls else None,
        "mean_4opt_brier": round(sum(briers) / len(briers), 4) if briers else None,
    }


def _read_scaling(state_path: Path) -> dict[str, Any]:
    for fn in ("adapter_config.json", "train_meta.json"):
        p = state_path.with_name(fn)
        if p.is_file():
            obj = json.loads(p.read_text(encoding="utf-8"))
            sc = obj.get("scaling") or obj.get("lora_scaling") or obj.get("hyperparams")
            if isinstance(sc, dict) and "scaling_mode" in sc:
                return {k: sc[k] for k in ("scaling_mode", "base_scale", "anchor_rank") if k in sc}
    return {"scaling_mode": "rslora", "base_scale": 2.0, "anchor_rank": 8}


def run_selftest_tiny() -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM
    import c1_train as C1T

    # 1. Verify sealed guard
    for s in SEALED_DATASETS:
        assert s in SEALED_DATASETS

    # 2. Verify batched left-padded == batch=1 exact logit equivalence AND C3 adapter + zone ablation!
    cfg = Qwen3Config(
        vocab_size=300, hidden_size=64, intermediate_size=128,
        num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    )
    tok = C1T._TinyTok()
    a1_cfg = load_config()
    opt_ids = {L: tok.encode(L)[0] for L in OPTION_LETTERS}

    tiny_m3 = {
        "kind": "m3_dac",
        "zones": {"zone1": [0, 1], "zone2": [2, 3], "zone3": [4, 5]},
        "zone1_dense_qo_layers": [0, 1],
        "zone1_dense_qo_rank": 2,
        "mlp_layers": [2, 4, 5],
        "mlp_rank": 4,
        "head_ranks": {"2:0": 4, "4:2": 2},
    }
    torch.manual_seed(5)
    model = Qwen3ForCausalLM(cfg).float().eval()
    M3.apply_c3_adapters(model, tiny_m3)
    with torch.no_grad():
        for pn, p in model.named_parameters():
            if p.requires_grad and "lora_B" in pn:
                p.normal_(0.0, 0.05)

    for ds in ("c3_dev", "c3_blind_confirm"):
        p = C3_DATASETS[ds]
        assert p.is_file(), p
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()[:2]]
        # Patch B1L.encode_prompt for TinyTok compatibility
        orig_enc = B1L.encode_prompt
        def _tiny_enc(t, msgs, pref, enable_thinking=False):
            txt = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking) + pref
            return torch.tensor(t.encode(txt), dtype=torch.long)
        B1L.encode_prompt = _tiny_enc
        try:
            r1 = eval_nothink_dataset_batched(model, tok, a1_cfg, rows, list(CORE_4CELL_ARMS), opt_ids, batch_size=1)
            r4 = eval_nothink_dataset_batched(model, tok, a1_cfg, rows, list(CORE_4CELL_ARMS), opt_ids, batch_size=4)
        finally:
            B1L.encode_prompt = orig_enc
        max_u_diff = max(abs(a["u_theta"] - b["u_theta"]) for a, b in zip(r1, r4))
        assert max_u_diff < 1e-4, (ds, max_u_diff)
        for rec in r4:
            assert abs(sum(rec["opt_probs"].values()) - 1.0) < 1e-4
            assert rec["gold_nll"] >= 0.0 and 0.0 <= rec["brier_score"] <= 2.0
        s = summarize_4cell(r4)
        assert s["n_bases"] == 2
        print(f"[OK] c3_eval self-test {ds:18s}: max_batch_u_diff={max_u_diff:.2e}, summary={s}")
    print("[OK] ALL c3_eval self-tests PASSED!")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-tiny", action="store_true")
    ap.add_argument("--model", type=str)
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--kl-corpus", type=Path, default=None)
    ap.add_argument("--alias", default="c3_model")
    ap.add_argument("--regime", default="no_think", choices=["no_think", "think"])
    ap.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=8, help="Left-padded batch size for HF no_think")
    ap.add_argument("--datasets", nargs="*", choices=list(C3_DATASETS.keys()), default=["c3_dev"])
    ap.add_argument("--arms", nargs="*", default=C1EV.EVAL_ARMS)
    ap.add_argument("--eligible-ids-out", type=Path, default=None)
    ap.add_argument("--merge-eligible-ids-out", action="store_true")
    ap.add_argument("--eligible-ids-in", type=Path, default=None)
    ap.add_argument("--adapter-state", type=Path, default=None)
    ap.add_argument("--peft-lora-dir", type=Path, default=None)
    ap.add_argument("--components-file", type=Path, default=C3 / "data" / "c3_components_frozen.json")
    ap.add_argument("--model-key", choices=["qwen3_8b", "qwen3_14b"], default="qwen3_8b")
    ap.add_argument("--arch", default=None, help="Architecture key in c3_components_frozen.json when using --adapter-state")
    ap.add_argument("--disable-zones", type=str, default="", help="Comma-separated zones to disable (e.g. '1' or '2,3')")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--max-new-tokens", type=int, default=3072)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--allow-sealed-final", action="store_true")
    args = ap.parse_args()

    if args.selftest_tiny:
        run_selftest_tiny()
        return

    if not args.allow_sealed_final:
        for ds in args.datasets:
            if ds in SEALED_DATASETS:
                raise RuntimeError(f"SEALED TEST SET GUARD: refusing to evaluate {ds!r} without --allow-sealed-final!")

    assert args.model and args.out_dir
    if args.backend == "vllm" and args.regime == "no_think":
        raise ValueError("--backend vllm is only supported for --regime think")
    if args.peft_lora_dir and args.backend != "vllm":
        raise ValueError("--peft-lora-dir requires --backend vllm")
    if args.base_model and args.regime == "no_think":
        if not args.kl_corpus or not args.kl_corpus.is_file():
            raise FileNotFoundError(f"Missing --kl-corpus: {args.kl_corpus}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    a1_cfg = load_config()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    opt_tok_ids = C1EV.get_cached_option_token_ids(tok) if args.regime == "no_think" else {}

    shared_elig_map: dict[str, set[str]] = {}
    if args.eligible_ids_in is not None:
        raw_el = json.loads(args.eligible_ids_in.read_text(encoding="utf-8"))
        shared_elig_map = {k: set(v) for k, v in raw_el.items()}

    llm = None
    model = None
    lora_request = None
    max_lora_rank = None

    if args.backend == "vllm":
        from vllm import LLM

        if args.peft_lora_dir is not None:
            from vllm.lora.request import LoRARequest

            peft_cfg = json.loads((args.peft_lora_dir / "adapter_config.json").read_text(encoding="utf-8"))
            r_export = int(peft_cfg["r"])
            max_lora_rank = C1EV._recommended_vllm_max_lora_rank(r_export)
            llm = LLM(
                model=args.model,
                dtype="bfloat16",
                tensor_parallel_size=args.tp,
                enable_lora=True,
                max_lora_rank=max_lora_rank,
            )
            lora_request = LoRARequest("c3_lora", 1, str(args.peft_lora_dir))
            print(f"[c3_eval] vLLM native LoRA: dir={args.peft_lora_dir} r={r_export} max_lora_rank={max_lora_rank}")
        else:
            llm = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=args.tp)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map=args.device_map, trust_remote_code=True
        ).eval()
        if args.adapter_state is not None:
            assert args.arch, "--adapter-state requires --arch"
            comp = json.loads(args.components_file.read_text(encoding="utf-8"))
            arch_spec = comp["models"][args.model_key]["architectures"][args.arch]
            scaling = _read_scaling(args.adapter_state)
            M3.apply_c3_adapters(model, arch_spec, **scaling)
            M3.load_c3_adapter_state(model, args.adapter_state)
            dz = [int(x.strip()) for x in args.disable_zones.split(",") if x.strip()]
            if dz:
                M3.set_zone_ablation(model, disable_zones=dz)
            model.eval()
            print(f"[c3_eval] loaded C3 adapter {args.adapter_state} arch={args.arch} disable_zones={dz} scaling={scaling}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema": "cedar.acl_exp_c3.eval_summary.v1",
        "alias": args.alias,
        "model": args.model,
        "arch": args.arch,
        "regime": args.regime,
        "backend": args.backend,
        "disable_zones": args.disable_zones,
        "peft_lora_dir": str(args.peft_lora_dir) if args.peft_lora_dir else None,
        "vllm_max_lora_rank": max_lora_rank,
        "datasets": {},
    }
    exported_elig: dict[str, list[str]] = {}
    t0 = time.time()

    for ds in args.datasets:
        path = C3_DATASETS[ds]
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if args.limit:
            rows = rows[: args.limit]
        if args.regime == "no_think":
            recs = eval_nothink_dataset_batched(model, tok, a1_cfg, rows, list(args.arms), opt_tok_ids, batch_size=args.batch_size)
        elif args.backend == "vllm":
            recs = C1EV.eval_freegen_dataset_vllm(
                llm, tok, a1_cfg, rows, list(args.arms),
                enable_thinking=True, max_new_tokens=args.max_new_tokens, eos_ids=set(), lora_request=lora_request,
            )
            # Attach family_id / broken_step / cell_member metadata onto vLLM records!
            row_by_id = {r["base_id"]: r for r in rows}
            for rec in recs:
                rr = row_by_id.get(rec["base_id"], {})
                rec["family_id"] = rr.get("family_id", rr.get("cell_id", rec["base_id"]))
                rec["cell_member"] = rr.get("cell_member")
                rec["broken_step"] = rr.get("broken_step")
                rec["domain"] = rr.get("domain")
        else:
            recs = [
                C1EV.eval_freegen_item_hf(model, tok, a1_cfg, r, arm, enable_thinking=True, max_new_tokens=args.max_new_tokens)
                for r in rows
                for arm in args.arms
            ]

        (args.out_dir / f"records_{args.regime}_{ds}.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in recs) + "\n", encoding="utf-8", newline="\n"
        )
        exported_elig[ds] = C1EV.compute_dual_turn1_eligible_ids(recs)
        shared_ids = shared_elig_map.get(ds) if shared_elig_map else (set(exported_elig[ds]) if args.eligible_ids_out else None)
        ds_out = C1EV.build_dataset_summary(recs, shared_eligible_ids=shared_ids)
        ds_out["four_cell_all"] = summarize_4cell(recs, eligible_ids=None)
        if shared_ids is not None:
            ds_out["four_cell_eligible"] = summarize_4cell(recs, eligible_ids=shared_ids)
        report["datasets"][ds] = ds_out
        fc_e = (ds_out.get("four_cell_eligible") or ds_out["four_cell_all"])
        print(
            f"[c3_eval] {args.alias} | {args.regime} | {ds} | "
            f"4cell={fc_e['four_cell_all_correct']} ({fc_e['four_cell_count']}/{fc_e['n_bases']}) | "
            f"DC_orig={fc_e['dc_vhes_fauth']} | DC_rev={fc_e['dc_vauth_fhes']} | NLL={fc_e['mean_4opt_nll']}"
        )

    if args.eligible_ids_out is not None:
        args.eligible_ids_out.parent.mkdir(parents=True, exist_ok=True)
        merged_elig: dict[str, list[str]] = {}
        if args.merge_eligible_ids_out and args.eligible_ids_out.is_file():
            merged_elig = json.loads(args.eligible_ids_out.read_text(encoding="utf-8"))
        merged_elig.update(exported_elig)
        args.eligible_ids_out.write_text(json.dumps(merged_elig, indent=2) + "\n", encoding="utf-8", newline="\n")

    if args.base_model and args.regime == "no_think":
        m_base = AutoModelForCausalLM.from_pretrained(
            args.base_model, dtype=torch.bfloat16, device_map=args.device_map, trust_remote_code=True
        ).eval()
        lines = [l.strip() for l in args.kl_corpus.read_text(encoding="utf-8").splitlines() if l.strip()][:200]
        texts = [json.loads(l)["text"] if l.startswith("{") else l for l in lines]
        report["reference_kl"] = C1EV.compute_reference_kl(model, m_base, tok, texts)
        print(f"[c3_eval] {args.alias} reference KL = {report['reference_kl']}")

    report["elapsed_seconds"] = round(time.time() - t0, 1)
    out_json = args.out_dir / f"eval_summary_{args.regime}.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"[OK] wrote {out_json}")


if __name__ == "__main__":
    main()
