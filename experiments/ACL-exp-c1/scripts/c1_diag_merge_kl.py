#!/usr/bin/env python3
"""Live-adapter vs merged-weight KL & C1 decision-position fidelity diagnostic (Astra audit check).

Loads a single base model in bfloat16, attaches & loads adapter_state.pt (live unmerged), evaluates:
  1. First N (default 20) frozen WikiText-2 passages -> caches live per-token log-probabilities
  2. First M (default 20) c1_dev bases x (V, F) no_think at exact 'Final:' readout position -> caches live
     4-option logits & u_theta = logit(at) - logit(a0)
Then merges adapters in-place (`M.merge_c1_adapters(model)`) and re-evaluates the exact same prompts,
saving per-passage `KL(p_live || p_merged)` and exact option-argmax & `u_theta` diffs to JSON.

Usage (GPU):
  CUDA_VISIBLE_DEVICES=3 python experiments/ACL-exp-c1/scripts/c1_diag_merge_kl.py \
      --base-model /mnt/kcl/devuser/ckpts/Qwen3-8B --model-key qwen3_8b --arm Uniform-LoRA \
      --adapter-state <PATH>/adapter_state_step0295.pt \
      --out-json experiments/ACL-exp-c1/results/step2a_diag_kl_pareto/merge_kl_diag_unif_w10_s295.json

Usage (CPU self-test):
  python experiments/ACL-exp-c1/scripts/c1_diag_merge_kl.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

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
import c1_modules as M  # noqa: E402
from render_prompts import build_messages, load_config  # noqa: E402


def run_selftest_tiny() -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    spec = {"kind": "uniform", "uniform_rank": 2, "target_modules": ["q_proj", "o_proj", "gate_proj", "down_proj"]}
    torch.manual_seed(0)
    m = Qwen3ForCausalLM(cfg).to(torch.bfloat16)
    M.apply_c1_adapters(m, spec, scaling_mode="rslora", base_scale=2.0, anchor_rank=8)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.requires_grad and "lora_B" in name:
                p.fill_(0.02)
    m.eval()
    seqs = [torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long), torch.tensor([[6, 7, 8, 9, 10]], dtype=torch.long)]
    live_lp = []
    with torch.no_grad():
        for s in seqs:
            live_lp.append(F.log_softmax(m(input_ids=s).logits[0, :-1].float(), dim=-1))
    M.merge_c1_adapters(m)
    m.eval()
    kls = []
    with torch.no_grad():
        for s, lp_u in zip(seqs, live_lp):
            lp_m = F.log_softmax(m(input_ids=s).logits[0, :-1].float(), dim=-1)
            kl = float(torch.sum(lp_u.exp() * (lp_u - lp_m), dim=-1).mean())
            kls.append(kl)
    assert max(kls) < 1e-3, kls
    print(f"[OK] c1_diag_merge_kl CPU self-test passed (max_passage_kl={max(kls):.2e})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model")
    ap.add_argument("--model-key", choices=["qwen3_14b", "qwen3_8b"])
    ap.add_argument("--arm")
    ap.add_argument("--components-file", type=Path, default=C1 / "data" / "c1_components_frozen.json")
    ap.add_argument("--adapter-state", type=Path)
    ap.add_argument("--expected-adapter-sha256", default=None)
    ap.add_argument("--kl-corpus", type=Path, default=C1 / "data" / "c1_kl_wikitext2_200.jsonl")
    ap.add_argument("--n-passages", type=int, default=20)
    ap.add_argument("--n-c1-bases", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--out-json", type=Path)
    ap.add_argument("--selftest-tiny", action="store_true")
    args = ap.parse_args()

    if args.selftest_tiny:
        run_selftest_tiny()
        return

    assert args.base_model and args.model_key and args.arm and args.adapter_state and args.out_json
    t0 = time.time()
    actual_sha = hashlib.sha256(args.adapter_state.read_bytes()).hexdigest()
    if args.expected_adapter_sha256 and actual_sha != args.expected_adapter_sha256:
        raise ValueError(f"Adapter SHA-256 mismatch: {actual_sha} != {args.expected_adapter_sha256}")

    comp = json.loads(args.components_file.read_text(encoding="utf-8"))
    arm_spec = comp["models"][args.model_key]["arms"][args.arm]
    scaling = EV.checkpoint_scaling(args.adapter_state, args.model_key, args.arm, arm_spec)
    if not scaling:
        scaling = {"scaling_mode": "rslora", "base_scale": 2.0, "anchor_rank": 8}

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="auto").eval()
    M.apply_c1_adapters(model, arm_spec, **scaling)
    M.load_c1_adapter_state(model, args.adapter_state)
    model.eval()
    dev = EV._input_device(model)

    # Prepare 20 WikiText-2 passages
    raw_lines = [l.strip() for l in args.kl_corpus.read_text(encoding="utf-8").splitlines() if l.strip()][: args.n_passages]
    wiki_ids = []
    for line in raw_lines:
        txt = json.loads(line)["text"] if line.startswith("{") else line
        enc = tok(txt, return_tensors="pt", truncation=True, max_length=args.max_len, add_special_tokens=False)["input_ids"]
        if enc.shape[1] >= 4:
            wiki_ids.append(enc)

    # Prepare 20 c1_dev bases x (V, F)
    a1_cfg = load_config()
    opt_tok_ids = EV.get_cached_option_token_ids(tok)
    dev_rows = [json.loads(l) for l in (C1 / "data" / "c1_dev_bases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()][: args.n_c1_bases]
    c1_prompts = []
    for row in dev_rows:
        for arm in ("V", "F"):
            msgs = build_messages(a1_cfg, row, arm, "no_think")
            ids = B1L.encode_prompt(tok, msgs, "Final:", enable_thinking=False)[None]
            c1_prompts.append((row["base_id"], arm, row["initial_answer"], row["target_answer"], ids))

    # Pass 1: LIVE unmerged adapter
    live_wiki_lp = []
    with torch.no_grad():
        for enc in wiki_ids:
            logits = model(input_ids=enc.to(dev)).logits[0, :-1].float().cpu()
            live_wiki_lp.append(F.log_softmax(logits, dim=-1))

    live_c1 = []
    with torch.no_grad():
        for bid, arm, a0, at, ids in c1_prompts:
            last = model(input_ids=ids.to(dev)).logits[0, -1].float().cpu()
            opt_l = {k: float(last[tid]) for k, tid in opt_tok_ids.items()}
            pred = max(opt_l, key=opt_l.get)
            live_c1.append((bid, arm, pred, opt_l[at] - opt_l[a0], F.log_softmax(last, dim=-1)))

    # Merge in-place
    M.merge_c1_adapters(model)
    model.eval()

    # Pass 2: MERGED weights
    passage_kls = []
    passage_top1 = []
    with torch.no_grad():
        for idx, (enc, lp_u) in enumerate(zip(wiki_ids, live_wiki_lp)):
            logits_m = model(input_ids=enc.to(dev)).logits[0, :-1].float().cpu()
            lp_m = F.log_softmax(logits_m, dim=-1)
            kl_vec = torch.sum(lp_u.exp() * (lp_u - lp_m), dim=-1)
            t1 = float((lp_u.argmax(dim=-1) == lp_m.argmax(dim=-1)).float().mean())
            passage_kls.append({"idx": idx, "n_tok": int(kl_vec.numel()), "kl_mean": float(kl_vec.mean()), "kl_max": float(kl_vec.max()), "top1_match": t1})
            passage_top1.append(t1)

    c1_diffs = []
    with torch.no_grad():
        for (bid, arm, a0, at, ids), (_, _, pred_u, u_u, lp_u) in zip(c1_prompts, live_c1):
            last_m = model(input_ids=ids.to(dev)).logits[0, -1].float().cpu()
            opt_m = {k: float(last_m[tid]) for k, tid in opt_tok_ids.items()}
            pred_m = max(opt_m, key=opt_m.get)
            u_m = opt_m[at] - opt_m[a0]
            lp_m = F.log_softmax(last_m, dim=-1)
            kl_dec = float(torch.sum(lp_u.exp() * (lp_u - lp_m)))
            c1_diffs.append({
                "base_id": bid,
                "arm": arm,
                "pred_match": pred_u == pred_m,
                "pred_live": pred_u,
                "pred_merged": pred_m,
                "u_live": round(u_u, 4),
                "u_merged": round(u_m, 4),
                "abs_u_diff": round(abs(u_u - u_m), 4),
                "decision_kl": kl_dec,
            })

    kl_means = [x["kl_mean"] for x in passage_kls]
    u_diffs = [x["abs_u_diff"] for x in c1_diffs]
    dec_kls = [x["decision_kl"] for x in c1_diffs]
    out = {
        "schema": "cedar.acl_exp_c1.merge_kl_diag.v1",
        "model_key": args.model_key,
        "arm": args.arm,
        "adapter_state": str(args.adapter_state),
        "adapter_sha256": actual_sha,
        "wiki_summary": {
            "n_passages": len(passage_kls),
            "kl_mean_avg": statistics.mean(kl_means),
            "kl_mean_median": statistics.median(kl_means),
            "kl_mean_max": max(kl_means),
            "top1_match_avg": statistics.mean(passage_top1),
        },
        "c1_decision_summary": {
            "n_items": len(c1_diffs),
            "pred_match_rate": sum(x["pred_match"] for x in c1_diffs) / max(1, len(c1_diffs)),
            "abs_u_diff_mean": statistics.mean(u_diffs),
            "abs_u_diff_max": max(u_diffs),
            "decision_kl_mean": statistics.mean(dec_kls),
            "decision_kl_max": max(dec_kls),
        },
        "wiki_passages": passage_kls,
        "c1_decisions": c1_diffs,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(
        f"[OK] merge KL diag ({args.arm}): wiki_kl_avg={out['wiki_summary']['kl_mean_avg']:.2e}, "
        f"c1_pred_match={out['c1_decision_summary']['pred_match_rate']:.3f}, "
        f"c1_u_diff_max={out['c1_decision_summary']['abs_u_diff_max']:.4f}"
    )


if __name__ == "__main__":
    main()
