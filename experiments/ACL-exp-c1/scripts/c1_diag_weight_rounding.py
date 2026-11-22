#!/usr/bin/env python3
"""Exact real-weight BF16/FP16 merge rounding audit + unified 200-passage KL triad (M0->live, M0->merged, live->merged).

Directly measures on actual base W0 and saved LoRA (A, B) weights:
  1. Per module group (`q_proj`, `o_proj`, `gate_proj`, `down_proj`, plus `attn_all`, `mlp_all`, `overall`):
     - exact active element count & per-element RMS of true FP32 delta `dW`
     - exact zeroed fraction & cosine similarity vs true `dW` under:
         (a) `bf16_single_round`: `(W0.float() + dW).to(bfloat16) - W0`
         (b) `bf16_double_round`: `(W0 + dW.to(bfloat16)) - W0` (legacy)
         (c) `fp16_single_round`: `(W0.float() + dW).to(float16) - W0.float().to(float16)`
  2. Unified 200-passage WikiText-2 KL triad on the EXACT same tokens (max_len=256, token-weighted,
     matching c1_eval.compute_reference_kl):
     - `kl_m0_to_live`
     - `kl_m0_to_merged_bf16` (using updated single-rounding merge_c1_adapters)
     - `kl_live_to_merged_bf16`

Usage (GPU):
  CUDA_VISIBLE_DEVICES=0 python experiments/ACL-exp-c1/scripts/c1_diag_weight_rounding.py \
      --base-model /mnt/kcl/devuser/ckpts/Qwen3-8B --model-key qwen3_8b --arm Uniform-LoRA \
      --adapter-state <PATH>/adapter_state_step0295.pt \
      --out-json experiments/ACL-exp-c1/results/step2a_live_complete/weight_rounding_unif_w10_s295.json

Usage (CPU self-test):
  python experiments/ACL-exp-c1/scripts/c1_diag_weight_rounding.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
C1 = HERE.parents[0]
sys.path.insert(0, str(HERE))

import c1_eval as EV  # noqa: E402
import c1_modules as M  # noqa: E402


def _accum_stats(acc: dict[str, dict[str, float]], key: str, W_bf16: torch.Tensor, dw_f32: torch.Tensor) -> None:
    w_f32 = W_bf16.float()
    dw_norm_sq = float(dw_f32.pow(2).sum())
    n_elem = int(dw_f32.numel())

    # (a) bf16 single-rounding
    d_single = (w_f32 + dw_f32).to(torch.bfloat16).float() - w_f32
    # (b) bf16 double-rounding (legacy)
    d_double = (W_bf16 + dw_f32.to(torch.bfloat16)).float() - w_f32
    # (c) fp16 single-rounding
    w_f16 = w_f32.to(torch.float16)
    d_fp16 = (w_f32 + dw_f32).to(torch.float16).float() - w_f16.float()

    s = acc[key]
    s["n_elem"] += n_elem
    s["dw_norm_sq"] += dw_norm_sq
    for tag, d_act in (("bf16_single", d_single), ("bf16_double", d_double), ("fp16_single", d_fp16)):
        s[f"{tag}_zero"] += int((d_act == 0).sum())
        s[f"{tag}_dot"] += float((d_act * dw_f32).sum())
        s[f"{tag}_norm_sq"] += float(d_act.pow(2).sum())


def _finalize_group(s: dict[str, float]) -> dict[str, float]:
    n = max(1, int(s["n_elem"]))
    dw_norm = math.sqrt(max(0.0, s["dw_norm_sq"]))
    out = {
        "n_active_elements": int(s["n_elem"]),
        "dw_frobenius_norm": round(dw_norm, 6),
        "dw_rms": dw_norm / math.sqrt(n),
    }
    for tag in ("bf16_single", "bf16_double", "fp16_single"):
        act_norm = math.sqrt(max(0.0, s[f"{tag}_norm_sq"]))
        denom = max(1e-12, dw_norm * act_norm)
        out[tag] = {
            "exact_zero_fraction": round(s[f"{tag}_zero"] / n, 6),
            "cos_sim": round(s[f"{tag}_dot"] / denom, 6),
            "norm_ratio": round(act_norm / max(1e-12, dw_norm), 6),
        }
    return out


@torch.no_grad()
def measure_real_weight_rounding(model) -> dict:
    acc: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for layer in model.model.layers:
        for name in ("q_proj", "o_proj"):
            m = getattr(layer.self_attn, name)
            if isinstance(m, M.HeadSliceLoRALinear):
                W = m.base.weight.data
                for i, h in enumerate(m.head_indices):
                    if h in m.disabled_heads:
                        continue
                    dw = ((m.lora_B[i].float() @ m.lora_A[i].float()) * m.scale).to(W.device)
                    sl = slice(h * m.head_dim, (h + 1) * m.head_dim)
                    w_sl = W[sl, :] if m.axis == "out" else W[:, sl]
                    for gk in (name, "attn_all", "overall"):
                        _accum_stats(acc, gk, w_sl, dw)
            elif isinstance(m, M.StandardLoRALinear) and not m.disabled:
                W = m.base.weight.data
                dw = ((m.lora_B.float() @ m.lora_A.float()) * m.scale).to(W.device)
                for gk in (name, "attn_all", "overall"):
                    _accum_stats(acc, gk, W, dw)
        for name in ("gate_proj", "down_proj"):
            m = getattr(layer.mlp, name)
            if isinstance(m, M.StandardLoRALinear) and not m.disabled:
                W = m.base.weight.data
                dw = ((m.lora_B.float() @ m.lora_A.float()) * m.scale).to(W.device)
                for gk in (name, "mlp_all", "overall"):
                    _accum_stats(acc, gk, W, dw)
    return {k: _finalize_group(v) for k, v in sorted(acc.items())}


@torch.no_grad()
def compute_unified_kl_triad(model_base, model_live, tok, texts: list[str], max_len: int = 256) -> dict:
    """Compute M0->live, M0->merged_bf16, live->merged_bf16 on the EXACT same 200 passages (token-weighted)."""
    dev_b = EV._input_device(model_base)
    dev_l = EV._input_device(model_live)
    seqs = []
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=max_len, add_special_tokens=False)["input_ids"]
        if enc.shape[1] >= 4:
            seqs.append(enc)

    # Cache base and live log_softmax on CPU per passage (200 x ~200 tokens is fast & tiny if reduced online?
    # Wait: storing 200 x 256 x 151936 floats would be 30GB! Instead compute M0->live online, save per-passage top/KL,
    # OR run base + live together, then merge live in-place and run base + merged together, and for live->merged
    # we only need a single forward pass of live vs merged if we cache 200 passages? NO — keep a second copy?
    # Even cleaner: load base ONCE on GPU, keep adapter_state on CPU, compute M0->live online; then clone/merge or
    # compute M0->live & live->merged per-passage by toggling adapter_scale? Wait: toggling disabled_heads/disabled
    # lets ONE model compute M0, LIVE, and (after merge) MERGED — Wait: we can compare M0 vs live per passage,
    # and for live vs merged we can keep M0 on GPU (or CPU) and run live+merged? Let's check VRAM:
    # 2 x 8B bf16 = 32 GB fits on 40GB A800! So we pass model_live (16GB) + model_merged (16GB) -> 32GB total!)
    # Actually: computing per-passage with 2 models in memory:
    # Pass A (model_base + model_live): computes kl_m0_to_live AND stores nothing large!
    # Pass B (merge model_live -> model_merged; model_base + model_merged): computes kl_m0_to_merged!
    # For kl_live_to_merged on ALL 200 passages without storing 30GB logits: temporarily reload clean base + live
    # alongside model_merged (2 models = 32GB on A800), compute kl_live_to_merged online!
    raise NotImplementedError("Use compute_triad_two_models")


@torch.no_grad()
def _pair_kl_token_weighted(m_p, m_q, seqs: list[torch.Tensor]) -> dict:
    """Exact token-weighted KL(P || Q) matching c1_eval.compute_reference_kl."""
    dev_p = EV._input_device(m_p)
    dev_q = EV._input_device(m_q)
    tot_kl = 0.0
    tot_tok = 0
    per_p = []
    for idx, ids in enumerate(seqs):
        lp_p = F.log_softmax(m_p(input_ids=ids.to(dev_p)).logits[0, :-1].float(), dim=-1)
        lp_q = F.log_softmax(m_q(input_ids=ids.to(dev_q)).logits[0, :-1].float(), dim=-1).to(lp_p.device)
        kl_vec = torch.sum(lp_p.exp() * (lp_p - lp_q), dim=-1)
        s_kl = float(kl_vec.sum())
        n_t = int(kl_vec.numel())
        tot_kl += s_kl
        tot_tok += n_t
        per_p.append({"idx": idx, "n_tok": n_t, "kl_mean": s_kl / max(1, n_t)})
    return {
        "n_passages": len(seqs),
        "n_tokens": tot_tok,
        "kl_mean_token_weighted": tot_kl / max(1, tot_tok),
        "kl_mean_passage_avg": sum(x["kl_mean"] for x in per_p) / max(1, len(per_p)),
        "kl_mean_passage_max": max(x["kl_mean"] for x in per_p) if per_p else 0.0,
    }


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
    spec = {"kind": "circuit", "heads": [[0, 1], [1, 2]], "head_rank": 3, "mlp_layers": [1], "mlp_rank": 2}
    torch.manual_seed(0)
    m_base = Qwen3ForCausalLM(cfg).to(torch.bfloat16).eval()
    torch.manual_seed(0)
    m_live = Qwen3ForCausalLM(cfg).to(torch.bfloat16).eval()
    M.apply_c1_adapters(m_live, spec, scaling_mode="rslora", base_scale=2.0, anchor_rank=8)
    with torch.no_grad():
        for name, p in m_live.named_parameters():
            if p.requires_grad and "lora_B" in name:
                p.fill_(0.001)
    m_live.eval()
    rw = measure_real_weight_rounding(m_live)
    assert "overall" in rw and "attn_all" in rw and "mlp_all" in rw
    assert rw["overall"]["bf16_single"]["exact_zero_fraction"] <= rw["overall"]["bf16_double"]["exact_zero_fraction"] + 1e-6
    seqs = [torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long), torch.tensor([[6, 7, 8, 9, 10]], dtype=torch.long)]
    kl_bl = _pair_kl_token_weighted(m_base, m_live, seqs)
    assert kl_bl["n_tokens"] == 8 and kl_bl["kl_mean_token_weighted"] >= 0.0
    print(
        f"[OK] c1_diag_weight_rounding CPU self-test passed "
        f"(bf16_single_zero={rw['overall']['bf16_single']['exact_zero_fraction']:.3f} <= "
        f"bf16_double_zero={rw['overall']['bf16_double']['exact_zero_fraction']:.3f})"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model")
    ap.add_argument("--model-key", choices=["qwen3_14b", "qwen3_8b"])
    ap.add_argument("--arm")
    ap.add_argument("--components-file", type=Path, default=C1 / "data" / "c1_components_frozen.json")
    ap.add_argument("--adapter-state", type=Path)
    ap.add_argument("--expected-adapter-sha256", default=None)
    ap.add_argument("--kl-corpus", type=Path, default=C1 / "data" / "c1_kl_wikitext2_200.jsonl")
    ap.add_argument("--max-len", type=int, default=256)
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
    lines = [l.strip() for l in args.kl_corpus.read_text(encoding="utf-8").splitlines() if l.strip()][:200]
    texts = [json.loads(l)["text"] if l.startswith("{") else l for l in lines]
    seqs = []
    for t in texts:
        enc = tok(t, return_tensors="pt", truncation=True, max_length=args.max_len, add_special_tokens=False)["input_ids"]
        if enc.shape[1] >= 4:
            seqs.append(enc)

    # Model A: clean base M0; Model B: live adapter (2 x 8B bf16 = ~32 GB, fits cleanly on 1x A800 40GB)
    m_base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="auto").eval()
    m_live = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, device_map="auto").eval()
    M.apply_c1_adapters(m_live, arm_spec, **scaling)
    M.load_c1_adapter_state(m_live, args.adapter_state)
    m_live.eval()

    # 1. Exact real-weight rounding audit before merging
    weight_rounding = measure_real_weight_rounding(m_live)

    # 2. Unified 200-passage KL triad (all on EXACT same `seqs`, token-weighted!)
    kl_m0_to_live = _pair_kl_token_weighted(m_base, m_live, seqs)

    # Merge m_live in-place (single-rounding FP32->BF16) -> becomes m_merged
    M.merge_c1_adapters(m_live)
    m_merged = m_live.eval()
    kl_m0_to_merged = _pair_kl_token_weighted(m_base, m_merged, seqs)

    # Attach live adapters onto m_base in-place so m_base becomes m_live2 (zero extra VRAM!) -> compute live->merged
    M.apply_c1_adapters(m_base, arm_spec, **scaling)
    M.load_c1_adapter_state(m_base, args.adapter_state)
    m_live2 = m_base.eval()
    kl_live_to_merged = _pair_kl_token_weighted(m_live2, m_merged, seqs)

    out = {
        "schema": "cedar.acl_exp_c1.weight_rounding_and_kl_triad.v1",
        "model_key": args.model_key,
        "arm": args.arm,
        "adapter_state": str(args.adapter_state),
        "adapter_sha256": actual_sha,
        "weight_rounding": weight_rounding,
        "unified_200_passage_kl_triad": {
            "max_len": args.max_len,
            "kl_m0_to_live": kl_m0_to_live,
            "kl_m0_to_merged_bf16_single": kl_m0_to_merged,
            "kl_live_to_merged_bf16_single": kl_live_to_merged,
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    ov = weight_rounding["overall"]
    print(
        f"[OK] {args.arm}: overall_bf16_single zero={ov['bf16_single']['exact_zero_fraction']:.3f} cos={ov['bf16_single']['cos_sim']:.4f} | "
        f"KL(M0->live)={kl_m0_to_live['kl_mean_token_weighted']:.6f} "
        f"KL(M0->merged)={kl_m0_to_merged['kl_mean_token_weighted']:.6f} "
        f"KL(live->merged)={kl_live_to_merged['kl_mean_token_weighted']:.6f}"
    )


if __name__ == "__main__":
    main()
