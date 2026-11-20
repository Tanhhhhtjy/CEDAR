#!/usr/bin/env python3
"""ACL-exp-c1 evaluation & KL-divergence harness (HF transformers / vLLM + CPU self-test).

Evaluates any base or merged C1 checkpoint on:
  1. Held-out ID (`data/c1_test_id_bases.jsonl`, 100 bases across 5 seen classes)
  2. Held-out OOD (`data/c1_test_ood_bases.jsonl`, 120 bases: 60 unseen converse_error + 60 unseen depth-4)
  3. Zero-shot Natural Probe (`../ACL-exp-a1/data/natural_pilot_seed_12_v2_2.jsonl`, 12 natural bases)
  4. Reference Next-Token KL Divergence D_KL(p_base || p_tuned) on a reference text file (or built-in fallback)

Arms evaluated (all rendered via experiments/ACL-exp-a1/scripts/render_prompts.py):
  turn1_F, N, AO, S, shell_only, F, F_auth, F_src_third_party, V, V_hes, P0_confirm_a0, I0_confirm_a0

Metrics computed per (dataset, split, pair_type):
  - Behavioral: F_follow, shell_follow, S_follow, AO_follow, F_auth_follow, V_accept, V_hes_accept,
    P0/I0_preserve, Paired Selectivity (Double-Correct on (V, F) twins & delta = V_accept - F_follow)
  - Continuous (no_think): u_theta(V), u_theta(F), u_theta(shell_only), u_theta(AO), margin u(V)-u(F),
    compliance jump u(shell)-u(AO), and P(u(V) > u(F))

Usage (GPU):
  CUDA_VISIBLE_DEVICES=0 python experiments/ACL-exp-c1/scripts/c1_eval.py \
      --model checkpoints/qwen3_8b/Targeted-Dual/merged_hf --base-model /mnt/kcl/devuser/ckpts/Qwen3-8B \
      --alias qwen3_8b__Targeted-Dual --regime no_think --out-dir results/qwen3_8b/Targeted-Dual
Usage (CPU self-test):
  python experiments/ACL-exp-c1/scripts/c1_eval.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
C1 = HERE.parents[0]
REPO = C1.parents[1]
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-a1" / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-b1" / "scripts"))

import a1_parse as P  # noqa: E402
import b1_lib as B1L  # noqa: E402
from render_prompts import build_messages, load_config  # noqa: E402

EVAL_ARMS = [
    "turn1_F",
    "N",
    "AO",
    "S",
    "shell_only",
    "F",
    "F_auth",
    "F_src_third_party",
    "V",
    "V_hes",
    "P0_confirm_a0",
    "I0_confirm_a0",
]

DEFAULT_DATASETS = {
    "c1_test_id": C1 / "data" / "c1_test_id_bases.jsonl",
    "c1_test_ood": C1 / "data" / "c1_test_ood_bases.jsonl",
    "natural_probe": REPO / "experiments" / "ACL-exp-a1" / "data" / "natural_pilot_seed_12_v2_2.jsonl",
}


def _input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


@torch.no_grad()
def eval_nothink_item(model, tok, a1_cfg: dict, row: dict, arm: str) -> dict:
    """Fast exact single-forward no_think evaluation: computes u_theta AND argmax option after 'Final:'."""
    msgs = build_messages(a1_cfg, row, arm, "no_think")
    ids = B1L.encode_prompt(tok, msgs, "Final:", enable_thinking=False)
    out = model(input_ids=ids[None].to(_input_device(model)))
    last = out.logits[0, -1].float()

    # Exact option logits for A/B/C/D
    opt_logits = {}
    for letter in ("A", "B", "C", "D"):
        tok_id = B1L.readout_token_ids(tok, "Final:", letter, "A" if letter != "A" else "B")[0]
        opt_logits[letter] = float(last[tok_id])
    pred = max(opt_logits, key=opt_logits.get)
    a0, at = row["initial_answer"], row["target_answer"]
    u_theta = opt_logits[at] - opt_logits[a0]
    return {
        "base_id": row["base_id"],
        "split": row.get("split", "unknown"),
        "pair_type": row["pair_type"],
        "fallacy_class": row["fallacy_class"],
        "difficulty_depth": row["difficulty_depth"],
        "arm": arm,
        "initial_answer": a0,
        "target_answer": at,
        "pred": pred,
        "is_target": pred == at,
        "is_initial": pred == a0,
        "u_theta": u_theta,
    }


@torch.no_grad()
def eval_think_item(model, tok, a1_cfg: dict, row: dict, arm: str, max_new_tokens: int = 2048) -> dict:
    """Greedy generation evaluation in think regime using A1's reference parser."""
    msgs = build_messages(a1_cfg, row, arm, "think")
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    dev = _input_device(model)
    gen = model.generate(
        input_ids=enc["input_ids"].to(dev),
        attention_mask=enc["attention_mask"].to(dev),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    new_toks = gen[0, enc["input_ids"].shape[1] :]
    raw_out = tok.decode(new_toks, skip_special_tokens=True)
    parsed = P.parse_output(raw_out, contract="final_only")
    pred = parsed.final_option
    a0, at = row["initial_answer"], row["target_answer"]
    return {
        "base_id": row["base_id"],
        "split": row.get("split", "unknown"),
        "pair_type": row["pair_type"],
        "fallacy_class": row["fallacy_class"],
        "difficulty_depth": row["difficulty_depth"],
        "arm": arm,
        "initial_answer": a0,
        "target_answer": at,
        "pred": pred,
        "parse_ok": parsed.status == "ok",
        "is_target": pred == at,
        "is_initial": pred == a0,
        "gen_tokens": int(len(new_toks)),
        "u_theta": None,
    }


def summarize_records(recs: list[dict]) -> dict:
    """Compute arm rates, paired (V, F) double-correct selectivity, and continuous u_theta summary."""
    by_arm = defaultdict(list)
    by_base_arm = {}
    for r in recs:
        by_arm[r["arm"]].append(r)
        by_base_arm[(r["base_id"], r["arm"])] = r

    arm_summary = {}
    for arm, rs in sorted(by_arm.items()):
        us = [r["u_theta"] for r in rs if r["u_theta"] is not None]
        arm_summary[arm] = {
            "n": len(rs),
            "p_target": round(sum(r["is_target"] for r in rs) / len(rs), 4),
            "p_initial": round(sum(r["is_initial"] for r in rs) / len(rs), 4),
            "u_theta_median": round(statistics.median(us), 3) if us else None,
            "u_theta_mean": round(statistics.mean(us), 3) if us else None,
        }

    # Paired (V, F) selectivity
    bases = sorted({r["base_id"] for r in recs})
    dc_vf, vf_u_diffs, shell_ao_diffs = [], [], []
    for b in bases:
        rv, rf = by_base_arm.get((b, "V")), by_base_arm.get((b, "F"))
        if rv and rf:
            dc_vf.append(1.0 if (rv["is_target"] and rf["is_initial"]) else 0.0)
            if rv["u_theta"] is not None and rf["u_theta"] is not None:
                vf_u_diffs.append(rv["u_theta"] - rf["u_theta"])
        rsh, rao = by_base_arm.get((b, "shell_only")), by_base_arm.get((b, "AO"))
        if rsh and rao and rsh["u_theta"] is not None and rao["u_theta"] is not None:
            shell_ao_diffs.append(rsh["u_theta"] - rao["u_theta"])

    v_acc = arm_summary.get("V", {}).get("p_target", 0.0)
    f_fol = arm_summary.get("F", {}).get("p_target", 0.0)
    return {
        "n_bases": len(bases),
        "arms": arm_summary,
        "paired_selectivity": {
            "double_correct_VF": round(statistics.mean(dc_vf), 4) if dc_vf else None,
            "delta_V_minus_F": round(v_acc - f_fol, 4),
            "u_V_minus_F_median": round(statistics.median(vf_u_diffs), 3) if vf_u_diffs else None,
            "p_uV_gt_uF": round(sum(d > 0 for d in vf_u_diffs) / len(vf_u_diffs), 4) if vf_u_diffs else None,
            "u_shell_minus_AO_median": round(statistics.median(shell_ao_diffs), 3) if shell_ao_diffs else None,
        },
    }


@torch.no_grad()
def compute_reference_kl(model_tuned, model_base, tok, texts: list[str], max_len: int = 256) -> dict:
    """Compute exact token-level D_KL(p_base || p_tuned) over a reference text corpus."""
    kls = []
    n_tokens = 0
    dev_t, dev_b = _input_device(model_tuned), _input_device(model_base)
    for txt in texts:
        ids = tok.encode(txt, add_special_tokens=False)[:max_len]
        if len(ids) < 4:
            continue
        t_ids = torch.tensor(ids, dtype=torch.long)[None]
        logp_b = F.log_softmax(model_base(input_ids=t_ids.to(dev_b)).logits[0, :-1].float(), dim=-1).cpu()
        logp_t = F.log_softmax(model_tuned(input_ids=t_ids.to(dev_t)).logits[0, :-1].float(), dim=-1).cpu()
        p_b = logp_b.exp()
        kl_tok = (p_b * (logp_b - logp_t)).sum(dim=-1)
        kls.extend(kl_tok.tolist())
        n_tokens += len(kl_tok)
    return {
        "n_texts": len(texts),
        "n_tokens": n_tokens,
        "kl_mean": round(statistics.mean(kls), 6) if kls else None,
        "kl_median": round(statistics.median(kls), 6) if kls else None,
    }


def run_selftest_tiny() -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM
    import c1_train as T

    cfg = Qwen3Config(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=2048,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    m_base = Qwen3ForCausalLM(cfg).float().eval()
    m_tuned = Qwen3ForCausalLM(cfg).float().eval()
    m_tuned.load_state_dict(m_base.state_dict())
    tok = T._TinyTok()
    a1_cfg = load_config()

    rows = [json.loads(l) for l in (C1 / "data" / "c1_test_ood_bases.jsonl").read_text(encoding="utf-8").splitlines()[:4]]
    recs = [eval_nothink_item(m_tuned, tok, a1_cfg, r, a) for r in rows for a in ("AO", "shell_only", "F", "V")]
    summ = summarize_records(recs)
    assert summ["n_bases"] == 4 and "double_correct_VF" in summ["paired_selectivity"]

    kl = compute_reference_kl(m_tuned, m_base, tok, ["A closed harbor berth controller applies exact rules."] * 3)
    assert kl["kl_mean"] is not None and abs(kl["kl_mean"]) < 1e-6, kl
    print(f"[OK] c1_eval CPU self-test passed (n_bases=4, self-KL={kl['kl_mean']})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--base-model", default=None, help="optional base model path to compute reference KL divergence")
    ap.add_argument("--kl-corpus", type=Path, default=None, help="JSONL/txt reference corpus for next-token KL")
    ap.add_argument("--alias", default="model")
    ap.add_argument("--regime", default="no_think", choices=["no_think", "think"])
    ap.add_argument("--datasets", nargs="*", default=["c1_test_id", "c1_test_ood", "natural_probe"])
    ap.add_argument("--arms", nargs="*", default=EVAL_ARMS)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--selftest-tiny", action="store_true")
    args = ap.parse_args()

    if args.selftest_tiny:
        run_selftest_tiny()
        return

    assert args.model and args.out_dir
    from transformers import AutoModelForCausalLM, AutoTokenizer

    a1_cfg = load_config()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map=args.device_map).eval()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"alias": args.alias, "model": args.model, "regime": args.regime, "datasets": {}}
    t0 = time.time()

    for ds in args.datasets:
        path = DEFAULT_DATASETS[ds]
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if args.limit:
            rows = rows[: args.limit]
        recs = []
        for r in rows:
            for arm in args.arms:
                if args.regime == "no_think":
                    recs.append(eval_nothink_item(model, tok, a1_cfg, r, arm))
                else:
                    recs.append(eval_think_item(model, tok, a1_cfg, r, arm, max_new_tokens=args.max_new_tokens))
        (args.out_dir / f"records_{args.regime}_{ds}.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in recs) + "\n", encoding="utf-8", newline="\n"
        )
        ds_out = {"overall": summarize_records(recs)}
        # Breakdown by split (e.g., ood_converse vs ood_depth4) and pair_type (context_flip vs argument_flip)
        for sp in sorted({x["split"] for x in recs}):
            ds_out.setdefault("by_split", {})[sp] = summarize_records([x for x in recs if x["split"] == sp])
        for pt in sorted({x["pair_type"] for x in recs}):
            ds_out.setdefault("by_pair_type", {})[pt] = summarize_records([x for x in recs if x["pair_type"] == pt])
        report["datasets"][ds] = ds_out
        print(f"[c1_eval] {args.alias} | {args.regime} | {ds}: {ds_out['overall']['paired_selectivity']}")

    if args.base_model and args.regime == "no_think":
        m_base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map=args.device_map
        ).eval()
        if args.kl_corpus and args.kl_corpus.exists():
            lines = [l.strip() for l in args.kl_corpus.read_text(encoding="utf-8").splitlines() if l.strip()][:200]
            texts = [json.loads(l)["text"] if l.startswith("{") else l for l in lines]
        else:
            # Deterministic fallback: neutral Turn-1 contexts from c1_test_id
            texts = [json.loads(l)["context_prompt_f"] for l in DEFAULT_DATASETS["c1_test_id"].read_text(encoding="utf-8").splitlines()[:60]]
        report["reference_kl"] = compute_reference_kl(model, m_base, tok, texts)
        print(f"[c1_eval] {args.alias} reference KL = {report['reference_kl']}")

    report["elapsed_seconds"] = round(time.time() - t0, 1)
    out_json = args.out_dir / f"eval_summary_{args.regime}.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"[OK] wrote {out_json}")


if __name__ == "__main__":
    main()
