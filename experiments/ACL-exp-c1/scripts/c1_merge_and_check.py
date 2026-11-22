#!/usr/bin/env python3
"""Standardized C1 adapter merge + precision-aware logit consistency checker (v2 gate).

Loads clean base model + adapter_state.pt, verifies optional SHA-256 and arm_spec, computes
unmerged vs merged logits on a deterministic prompt (C1-DEV-001 turn1_F think), checks all 4
bfloat16 consistency gates + generation EOS preservation, saves merged_hf + merge_check.json.

Usage (GPU):
  CUDA_VISIBLE_DEVICES=0 python experiments/ACL-exp-c1/scripts/c1_merge_and_check.py \
      --base-model /mnt/kcl/devuser/ckpts/Qwen3-8B \
      --model-key qwen3_8b --arm Targeted-Dual \
      --components-file experiments/ACL-exp-c1/data/c1_components_frozen.json \
      --adapter-state <PATH>/adapter_state_step0295.pt \
      --out-merged-dir <TMP_MERGED_DIR> \
      --out-check-json experiments/ACL-exp-c1/results/step2_8b_showdown/<TAG>/merge_check.json \
      --tag <TAG>

Usage (CPU self-test):
  python experiments/ACL-exp-c1/scripts/c1_merge_and_check.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
C1 = HERE.parents[0]
REPO = C1.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-a1" / "scripts"))

import c1_eval as EV  # noqa: E402
import c1_modules as M  # noqa: E402
from render_prompts import build_messages, load_config  # noqa: E402

V2_THRESHOLDS = {
    "mean_abs_logit_error_lt": 0.25,
    "last_position_kl_lt": 0.01,
    "top1_argmax_match_rate_gte": 0.95,
    "max_abs_logit_error_lt": 2.0,
}


def compute_logit_consistency(logits_unmerged: torch.Tensor, logits_merged: torch.Tensor) -> dict:
    """Compute precision-aware consistency stats in float32 from [1, T, V] logits."""
    lu = logits_unmerged[0].float()
    lm = logits_merged[0].float()
    diff = (lu - lm).abs()
    mean_abs = float(diff.mean())
    max_abs = float(diff.max())
    last_max_abs = float(diff[-1].max())
    top1_match = int((lu.argmax(dim=-1) == lm.argmax(dim=-1)).sum())
    n_pos = int(lu.shape[0])
    top1_rate = top1_match / max(1, n_pos)

    log_pu = F.log_softmax(lu[-1], dim=-1)
    log_pm = F.log_softmax(lm[-1], dim=-1)
    pu = log_pu.exp()
    last_kl = float(torch.sum(pu * (log_pu - log_pm)))

    passed = (
        mean_abs < V2_THRESHOLDS["mean_abs_logit_error_lt"]
        and last_kl < V2_THRESHOLDS["last_position_kl_lt"]
        and top1_rate >= V2_THRESHOLDS["top1_argmax_match_rate_gte"]
        and max_abs < V2_THRESHOLDS["max_abs_logit_error_lt"]
    )
    return {
        "logit_shape": list(logits_unmerged.shape),
        "thresholds": V2_THRESHOLDS,
        "mean_abs_logit_error": mean_abs,
        "last_position_kl": last_kl,
        "top1_argmax_match_count": top1_match,
        "num_positions": n_pos,
        "top1_argmax_match_rate": top1_rate,
        "max_abs_logit_error": max_abs,
        "last_position_max_abs_logit_error": last_max_abs,
        "passed": passed,
    }


def _sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run_selftest_tiny() -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    spec = {"kind": "circuit", "heads": [[0, 1], [1, 2]], "head_rank": 3, "mlp_layers": [1], "mlp_rank": 3}
    torch.manual_seed(0)
    m = Qwen3ForCausalLM(cfg).to(torch.bfloat16)
    M.apply_c1_adapters(m, spec, scaling_mode="rslora", base_scale=2.0, anchor_rank=8)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.requires_grad and "lora_B" in name:
                p.fill_(0.02)
    m.eval()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    with torch.no_grad():
        lu = m(input_ids=ids).logits
    M.merge_c1_adapters(m)
    m.eval()
    with torch.no_grad():
        lm = m(input_ids=ids).logits
    st = compute_logit_consistency(lu, lm)
    assert st["passed"] is True, st
    # Also verify a broken merge fails the gate
    st_bad = compute_logit_consistency(lu, lm + 3.0 * torch.randn_like(lm))
    assert st_bad["passed"] is False, st_bad
    print(
        f"[OK] c1_merge_and_check CPU self-test passed "
        f"(bf16 mean_abs={st['mean_abs_logit_error']:.5f}, kl={st['last_position_kl']:.2e}, top1={st['top1_argmax_match_rate']:.3f})"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model")
    ap.add_argument("--model-key", choices=["qwen3_14b", "qwen3_8b"])
    ap.add_argument("--arm")
    ap.add_argument("--components-file", type=Path, default=C1 / "data" / "c1_components_frozen.json")
    ap.add_argument("--adapter-state", type=Path)
    ap.add_argument("--expected-adapter-sha256", default=None)
    ap.add_argument("--out-merged-dir", type=Path)
    ap.add_argument("--out-check-json", type=Path)
    ap.add_argument("--tag", default="model")
    ap.add_argument("--expected-eos", nargs="*", type=int, default=[151645, 151643])
    ap.add_argument("--selftest-tiny", action="store_true")
    args = ap.parse_args()

    if args.selftest_tiny:
        run_selftest_tiny()
        return

    assert args.base_model and args.model_key and args.arm and args.adapter_state and args.out_merged_dir and args.out_check_json
    t0 = time.time()
    actual_sha = _sha256_file(args.adapter_state)
    if args.expected_adapter_sha256 and actual_sha != args.expected_adapter_sha256:
        raise ValueError(f"Adapter SHA-256 mismatch: got {actual_sha}, expected {args.expected_adapter_sha256}")

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

    # Deterministic check prompt: C1-DEV-001 turn1_F think
    dev_first = json.loads((C1 / "data" / "c1_dev_bases.jsonl").read_text(encoding="utf-8").splitlines()[0])
    a1_cfg = load_config()
    msgs = build_messages(a1_cfg, dev_first, "turn1_F", "think")
    prompt_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False)
    dev = EV._input_device(model)
    ids = enc["input_ids"].to(dev)
    prompt_sha = hashlib.sha256(json.dumps(enc["input_ids"][0].tolist()).encode()).hexdigest()

    with torch.no_grad():
        lu = model(input_ids=ids).logits.detach().cpu()

    M.merge_c1_adapters(model)
    model.eval()
    with torch.no_grad():
        lm = model(input_ids=ids).logits.detach().cpu()

    stats = compute_logit_consistency(lu, lm)
    eos_raw = model.generation_config.eos_token_id
    eos_list = [eos_raw] if isinstance(eos_raw, int) else list(eos_raw or [])
    eos_ok = (eos_list == args.expected_eos) if args.expected_eos else True
    stats["generation_eos_token_id"] = eos_list
    stats["eos_ok"] = eos_ok
    stats["passed"] = bool(stats["passed"] and eos_ok)

    record = {
        "tag": args.tag,
        "model_key": args.model_key,
        "arm": args.arm,
        "components_file": str(args.components_file),
        "adapter_state": str(args.adapter_state),
        "adapter_sha256": actual_sha,
        "base_model": args.base_model,
        "precision": "bfloat16",
        "metrics_dtype": "float32",
        "prompt_base_id": dev_first["base_id"],
        "prompt_arm": "turn1_F",
        "prompt_regime": "think",
        "prompt_token_sha256": prompt_sha,
        "elapsed_seconds": round(time.time() - t0, 2),
        **stats,
    }
    args.out_check_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_check_json.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

    if not record["passed"]:
        raise RuntimeError(f"Merge consistency check FAILED for {args.tag}: {json.dumps(record, indent=2)}")

    args.out_merged_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_merged_dir, safe_serialization=True)
    tok.save_pretrained(args.out_merged_dir)
    print(
        f"[OK] merged & verified {args.tag} -> {args.out_merged_dir} "
        f"(mean_abs={record['mean_abs_logit_error']:.4f}, kl={record['last_position_kl']:.2e}, "
        f"top1={record['top1_argmax_match_rate']:.4f}, eos={eos_list})"
    )


if __name__ == "__main__":
    main()
