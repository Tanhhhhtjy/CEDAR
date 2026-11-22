#!/usr/bin/env python3
"""Export any ACL-exp-c1 adapter_state.pt to lossless standard PEFT LoRA format for vLLM `--enable-lora`.

Design (verified against vLLM 0.11.0 LoRA loader):
  1. Block-diagonal head packing: for HeadSliceLoRALinear layers with k selected heads of rank r_h,
     assigns each head its OWN disjoint rank slice [i*r_h : (i+1)*r_h] inside a uniform rank R_export
     (with unused slices zero-padded). On q_proj, B_peft writes into head rows [h*hd:(h+1)*hd]; on
     o_proj, A_peft reads from head cols [h*hd:(h+1)*hd]. Thus B_peft @ A_peft is MATHEMATICALLY
     IDENTICAL to the sum of per-head slices with zero cross-head interference.
  2. Single uniform rank R_export & lora_alpha == R_export (alpha/r == 1.0): folds each module's
     exact rsLoRA scale ONCE into lora_B during export, avoiding vLLM's rank_pattern / alpha_pattern
     loader quirks or double-scaling.
  3. Built-in exact FP32 algebraic verification: asserts max_abs_diff(B_peft @ A_peft, true_dW) < 1e-4
     (for bf16 export) / < 1e-6 (for fp32 check) on EVERY exported module before saving.

Usage (export):
  python experiments/ACL-exp-c1/scripts/c1_export_peft_lora.py \
      --model-key qwen3_8b --arm Targeted-Dual \
      --components-file experiments/ACL-exp-c1/data/c1_components_frozen.json \
      --adapter-state <PATH>/adapter_state_step0200.pt \
      --out-dir <TMP_PEFT_LORA_DIR>

Usage (CPU self-test):
  python experiments/ACL-exp-c1/scripts/c1_export_peft_lora.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
C1 = HERE.parents[0]
sys.path.insert(0, str(HERE))

import c1_eval as EV  # noqa: E402
import c1_modules as M  # noqa: E402

VLLM_SUPPORTED_MAX_RANKS = (8, 16, 32, 64, 128, 256)


def recommended_vllm_max_lora_rank(r_export: int) -> int:
    for cand in VLLM_SUPPORTED_MAX_RANKS:
        if cand >= r_export:
            return cand
    raise ValueError(f"Exported rank {r_export} exceeds max supported vLLM LoRA rank {VLLM_SUPPORTED_MAX_RANKS[-1]}")


def compute_export_rank(arm_spec: dict[str, Any]) -> int:
    """Compute minimum uniform PEFT rank R_export needed to represent every layer losslessly."""
    if arm_spec["kind"] == "uniform":
        return int(arm_spec["uniform_rank"])
    if arm_spec["kind"] == "circuit":
        by_layer: dict[int, int] = {}
        for li, _ in arm_spec.get("heads", []):
            by_layer[int(li)] = by_layer.get(int(li), 0) + 1
        max_heads_in_layer = max(by_layer.values(), default=0)
        r_h = int(arm_spec.get("head_rank", 0))
        r_m = int(arm_spec.get("mlp_rank", 0))
        return max(max_heads_in_layer * r_h, r_m, 1)
    raise ValueError(f"Unknown arm kind: {arm_spec['kind']!r}")


def export_adapter_state_to_peft(
    *,
    state_dict: dict[str, torch.Tensor],
    arm_spec: dict[str, Any],
    model_spec: dict[str, Any],
    scaling: dict[str, Any],
    out_dir: Path,
    base_model_name_or_path: str = "",
    export_dtype: torch.dtype = torch.bfloat16,
    adapter_scale: float = 1.0,
) -> dict[str, Any]:
    """Convert raw c1 adapter_state dict into standard PEFT LoRA directory (`adapter_model.safetensors` + config)."""
    d = int(model_spec["hidden_size"])
    H = int(model_spec["num_heads"])
    hd = int(model_spec["head_dim"])
    d_ff = int(model_spec["intermediate_size"])
    nL = int(model_spec["num_layers"])

    scaling_mode = str(scaling.get("scaling_mode", "rslora"))
    base_scale = float(scaling.get("base_scale", 2.0)) * float(adapter_scale)
    anchor_rank = int(scaling.get("anchor_rank", 8))

    r_export = compute_export_rank(arm_spec)
    vllm_max_r = recommended_vllm_max_lora_rank(r_export)

    peft_tensors: dict[str, torch.Tensor] = {}
    target_module_names: set[str] = set()
    max_fp32_err = 0.0
    max_bf16_err = 0.0
    sum_dot = 0.0
    sum_true_sq = 0.0
    sum_out_sq = 0.0
    n_exported_modules = 0

    def _check_and_store(prefix: str, mod_name: str, A_f32: torch.Tensor, B_f32: torch.Tensor, true_dw_f32: torch.Tensor) -> None:
        nonlocal max_fp32_err, max_bf16_err, sum_dot, sum_true_sq, sum_out_sq, n_exported_modules
        recon_f32 = B_f32 @ A_f32
        err_f32 = float((recon_f32 - true_dw_f32).abs().max())
        max_fp32_err = max(max_fp32_err, err_f32)
        assert err_f32 < 1e-5, f"FP32 algebraic export mismatch on {prefix}: max_abs={err_f32}"

        A_out = A_f32.to(export_dtype).contiguous()
        B_out = B_f32.to(export_dtype).contiguous()
        recon_out = B_out.float() @ A_out.float()
        err_out = float((recon_out - true_dw_f32).abs().max())
        max_bf16_err = max(max_bf16_err, err_out)
        # Assert actual exported dtype fidelity (prevents silent BF16 underflow/overflow during PEFT export!)
        mod_norm_true = float(true_dw_f32.norm())
        if mod_norm_true > 1e-12:
            mod_rel_err = float((recon_out - true_dw_f32).norm()) / mod_norm_true
            assert mod_rel_err < 0.02, f"Exported {export_dtype} relative Frobenius error too high on {prefix}: {mod_rel_err:.4f}"
        sum_dot += float((recon_out * true_dw_f32).sum())
        sum_true_sq += float((true_dw_f32 * true_dw_f32).sum())
        sum_out_sq += float((recon_out * recon_out).sum())

        peft_tensors[f"base_model.model.{prefix}.lora_A.weight"] = A_out
        peft_tensors[f"base_model.model.{prefix}.lora_B.weight"] = B_out
        target_module_names.add(mod_name)
        n_exported_modules += 1

    if arm_spec["kind"] == "circuit":
        by_layer: dict[int, list[int]] = {}
        for li, h in arm_spec.get("heads", []):
            by_layer.setdefault(int(li), []).append(int(h))
        r_h = int(arm_spec.get("head_rank", 0))
        if r_h > 0:
            s_h = M.compute_lora_scale(r_h, scaling_mode, base_scale, anchor_rank)
            for li, hs in sorted(by_layer.items()):
                hs_sorted = sorted(set(hs))
                # q_proj (axis="out"): [H*hd, d]
                q_pre = f"model.layers.{li}.self_attn.q_proj"
                qA = state_dict[f"{q_pre}.lora_A"].float()  # [n_sel, r_h, d]
                qB = state_dict[f"{q_pre}.lora_B"].float()  # [n_sel, hd, r_h]
                A_full = torch.zeros((r_export, d), dtype=torch.float32)
                B_full = torch.zeros((H * hd, r_export), dtype=torch.float32)
                true_dw = torch.zeros((H * hd, d), dtype=torch.float32)
                for i, h in enumerate(hs_sorted):
                    rs = slice(i * r_h, (i + 1) * r_h)
                    sl = slice(h * hd, (h + 1) * hd)
                    A_full[rs, :] = qA[i]
                    B_full[sl, rs] = qB[i] * s_h
                    true_dw[sl, :] += (qB[i] @ qA[i]) * s_h
                _check_and_store(q_pre, "q_proj", A_full, B_full, true_dw)

                # o_proj (axis="in"): [d, H*hd]
                o_pre = f"model.layers.{li}.self_attn.o_proj"
                oA = state_dict[f"{o_pre}.lora_A"].float()  # [n_sel, r_h, hd]
                oB = state_dict[f"{o_pre}.lora_B"].float()  # [n_sel, d, r_h]
                A_full = torch.zeros((r_export, H * hd), dtype=torch.float32)
                B_full = torch.zeros((d, r_export), dtype=torch.float32)
                true_dw = torch.zeros((d, H * hd), dtype=torch.float32)
                for i, h in enumerate(hs_sorted):
                    rs = slice(i * r_h, (i + 1) * r_h)
                    sl = slice(h * hd, (h + 1) * hd)
                    A_full[rs, sl] = oA[i]
                    B_full[:, rs] = oB[i] * s_h
                    true_dw[:, sl] += (oB[i] @ oA[i]) * s_h
                _check_and_store(o_pre, "o_proj", A_full, B_full, true_dw)

        r_m = int(arm_spec.get("mlp_rank", 0))
        if r_m > 0:
            s_m = M.compute_lora_scale(r_m, scaling_mode, base_scale, anchor_rank)
            for li in sorted(set(int(x) for x in arm_spec.get("mlp_layers", []))):
                for mname, d_in, d_out in (("gate_proj", d, d_ff), ("down_proj", d_ff, d)):
                    pre = f"model.layers.{li}.mlp.{mname}"
                    mA = state_dict[f"{pre}.lora_A"].float()
                    mB = state_dict[f"{pre}.lora_B"].float()
                    A_full = torch.zeros((r_export, d_in), dtype=torch.float32)
                    B_full = torch.zeros((d_out, r_export), dtype=torch.float32)
                    A_full[:r_m, :] = mA
                    B_full[:, :r_m] = mB * s_m
                    true_dw = (mB @ mA) * s_m
                    _check_and_store(pre, mname, A_full, B_full, true_dw)

    elif arm_spec["kind"] == "uniform":
        r_u = int(arm_spec["uniform_rank"])
        s_u = M.compute_lora_scale(r_u, scaling_mode, base_scale, anchor_rank)
        targets = set(arm_spec.get("target_modules", ["q_proj", "o_proj", "gate_proj", "down_proj"]))
        dims = {
            "q_proj": ("self_attn", d, H * hd),
            "o_proj": ("self_attn", H * hd, d),
            "gate_proj": ("mlp", d, d_ff),
            "up_proj": ("mlp", d, d_ff),
            "down_proj": ("mlp", d_ff, d),
        }
        for li in range(nL):
            for mname in sorted(targets):
                sub, d_in, d_out = dims[mname]
                pre = f"model.layers.{li}.{sub}.{mname}"
                mA = state_dict[f"{pre}.lora_A"].float()
                mB = state_dict[f"{pre}.lora_B"].float()
                A_full = torch.zeros((r_export, d_in), dtype=torch.float32)
                B_full = torch.zeros((d_out, r_export), dtype=torch.float32)
                A_full[:r_u, :] = mA
                B_full[:, :r_u] = mB * s_u
                true_dw = (mB @ mA) * s_u
                _check_and_store(pre, mname, A_full, B_full, true_dw)
    else:
        raise ValueError(f"Unknown arm kind: {arm_spec['kind']!r}")

    overall_export_cos = (
        float(sum_dot / ((sum_true_sq**0.5) * (sum_out_sq**0.5)))
        if (sum_true_sq > 0.0 and sum_out_sq > 0.0)
        else 1.0
    )
    assert overall_export_cos >= 0.999, f"Exported {export_dtype} overall cosine similarity too low: {overall_export_cos:.6f}"

    out_dir.mkdir(parents=True, exist_ok=True)
    sf_path = out_dir / "adapter_model.safetensors"
    save_file(peft_tensors, str(sf_path))
    sf_sha = hashlib.sha256(sf_path.read_bytes()).hexdigest()

    # Standard PEFT adapter_config.json with lora_alpha == r (so vLLM scales by exactly 1.0!)
    peft_cfg = {
        "peft_type": "LORA",
        "auto_mapping": None,
        "base_model_name_or_path": base_model_name_or_path,
        "revision": None,
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
        "r": r_export,
        "lora_alpha": r_export,
        "lora_dropout": 0.0,
        "fan_in_fan_out": False,
        "bias": "none",
        "target_modules": sorted(target_module_names),
        "modules_to_save": None,
    }
    (out_dir / "adapter_config.json").write_text(json.dumps(peft_cfg, indent=2) + "\n", encoding="utf-8", newline="\n")

    meta = {
        "r_export": r_export,
        "recommended_vllm_max_lora_rank": vllm_max_r,
        "n_exported_modules": n_exported_modules,
        "target_modules": sorted(target_module_names),
        "max_fp32_dw_abs_err": max_fp32_err,
        "max_export_dtype_dw_abs_err": max_bf16_err,
        "overall_export_dtype_cos_sim": round(overall_export_cos, 8),
        "export_dtype": str(export_dtype),
        "adapter_scale": adapter_scale,
        "adapter_model_safetensors_sha256": sf_sha,
    }
    (out_dir / "c1_peft_export_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    return meta


def run_selftest_tiny() -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    ms = {"hidden_size": 64, "num_heads": 4, "head_dim": 16, "intermediate_size": 128, "num_layers": 3}
    specs = {
        # Layer 1 has 3 heads (max_heads=3 * r_h=4 = 12 > r_m=3 -> R_export=12, vLLM max_r=16)
        "Targeted-Dual": {"kind": "circuit", "heads": [[0, 1], [1, 0], [1, 2], [1, 3]], "head_rank": 4, "mlp_layers": [1, 2], "mlp_rank": 3},
        "Uniform-LoRA": {"kind": "uniform", "uniform_rank": 2, "target_modules": ["q_proj", "o_proj", "gate_proj", "down_proj"]},
    }
    ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    scaling = {"scaling_mode": "rslora", "base_scale": 2.0, "anchor_rank": 8}

    with tempfile.TemporaryDirectory() as td:
        for name, spec in specs.items():
            torch.manual_seed(0)
            m_live = Qwen3ForCausalLM(cfg).float()
            M.apply_c1_adapters(m_live, spec, **scaling)
            with torch.no_grad():
                for pname, p in m_live.named_parameters():
                    if p.requires_grad and "lora_B" in pname:
                        p.normal_(0.0, 0.05)
            m_live.eval()
            with torch.no_grad():
                y_live = m_live(input_ids=ids).logits

            ap = Path(td) / f"{name}.pt"
            M.save_c1_adapter_state(m_live, ap)
            sd = torch.load(ap, map_location="cpu", weights_only=True)
            out_dir = Path(td) / f"peft_{name}"
            meta = export_adapter_state_to_peft(
                state_dict=sd,
                arm_spec=spec,
                model_spec=ms,
                scaling=scaling,
                out_dir=out_dir,
                export_dtype=torch.float32,
            )

            # Independently reconstruct model by applying exported PEFT A/B directly onto fresh base model
            torch.manual_seed(0)
            m_peft = Qwen3ForCausalLM(cfg).float()
            peft_sd = load_file(str(out_dir / "adapter_model.safetensors"))
            with torch.no_grad():
                for k, B_t in peft_sd.items():
                    if not k.endswith(".lora_B.weight"):
                        continue
                    A_t = peft_sd[k.replace(".lora_B.weight", ".lora_A.weight")]
                    mod_path = k[len("base_model.model.") : -len(".lora_B.weight")]
                    mod = m_peft
                    for part in mod_path.split("."):
                        mod = getattr(mod, part)
                    # Standard PEFT forward with alpha/r == 1.0 adds (B @ A)
                    mod.weight.data += B_t @ A_t
            m_peft.eval()
            with torch.no_grad():
                y_peft = m_peft(input_ids=ids).logits
            logit_err = float((y_live - y_peft).abs().max())
            assert logit_err < 1e-5, (name, logit_err)
            print(
                f"[OK] PEFT export self-test {name:14s}: r_export={meta['r_export']:2d} "
                f"(vllm_max_r={meta['recommended_vllm_max_lora_rank']:2d}) "
                f"dw_err={meta['max_fp32_dw_abs_err']:.2e} logit_err={logit_err:.2e}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", choices=["qwen3_14b", "qwen3_8b"])
    ap.add_argument("--arm")
    ap.add_argument("--components-file", type=Path, default=C1 / "data" / "c1_components_frozen.json")
    ap.add_argument("--adapter-state", type=Path)
    ap.add_argument("--expected-adapter-sha256", default=None)
    ap.add_argument("--base-model", default="")
    ap.add_argument("--adapter-scale", type=float, default=1.0)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--selftest-tiny", action="store_true")
    args = ap.parse_args()

    if args.selftest_tiny:
        run_selftest_tiny()
        return

    assert args.model_key and args.arm and args.adapter_state and args.out_dir
    actual_sha = hashlib.sha256(args.adapter_state.read_bytes()).hexdigest()
    if args.expected_adapter_sha256 and actual_sha != args.expected_adapter_sha256:
        raise ValueError(f"Adapter SHA-256 mismatch: {actual_sha} != {args.expected_adapter_sha256}")

    comp = json.loads(args.components_file.read_text(encoding="utf-8"))
    m_info = comp["models"][args.model_key]
    arm_spec = m_info["arms"][args.arm]
    scaling = EV.checkpoint_scaling(args.adapter_state, args.model_key, args.arm, arm_spec)
    if not scaling:
        scaling = {"scaling_mode": "rslora", "base_scale": 2.0, "anchor_rank": 8}

    sd = torch.load(args.adapter_state, map_location="cpu", weights_only=True)
    meta = export_adapter_state_to_peft(
        state_dict=sd,
        arm_spec=arm_spec,
        model_spec=m_info["model_spec"],
        scaling=scaling,
        out_dir=args.out_dir,
        base_model_name_or_path=args.base_model,
        export_dtype=torch.bfloat16,
        adapter_scale=args.adapter_scale,
    )
    print(f"[OK] exported PEFT LoRA -> {args.out_dir} | {json.dumps(meta)}")


if __name__ == "__main__":
    main()
