#!/usr/bin/env python3
"""Lossless PEFT LoRA exporter for ACL-exp-c3 (supports per-head tiered ranks, 3-zone M3-DAC, and Uniform-LoRA).

Design Guarantees (verified by `--selftest-tiny` on both `M3-DAC` and `Uniform-LoRA`):
  1. Variable-rank block-diagonal packing: for any layer with heads {(h_i, r_i)}, assigns each head its OWN
     disjoint rank block [off : off + r_i] inside uniform rank R_export = max_layer(sum_i r_i, r_z1, r_mlp) <= 128,
     folding each head's own rsLoRA scale s(r_i) = base_scale * sqrt(anchor_rank / r_i) ONCE into lora_B.
  2. Sets `lora_alpha == r == R_export` (ratio = 1.0) so vLLM applies ZERO secondary scaling and needs ZERO
     `rank_pattern` / `alpha_pattern` loader quirks.
  3. Built-in FP32 algebraic verification (`max_abs_err < 1e-5` on EVERY module) and exported dtype cosine check
     (`overall_cos >= 0.999`) before writing `adapter_model.safetensors`.

CPU Self-Test:
  python experiments/ACL-exp-c3/scripts/c3_export_peft_lora.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
C3 = HERE.parents[0]
sys.path.insert(0, str(HERE))

import c3_modules as M  # noqa: E402

VLLM_SUPPORTED_MAX_RANKS = (8, 16, 32, 64, 128, 256)


def recommended_vllm_max_lora_rank(r_export: int) -> int:
    for cand in VLLM_SUPPORTED_MAX_RANKS:
        if cand >= r_export:
            return cand
    raise ValueError(f"Exported rank {r_export} exceeds max supported vLLM LoRA rank {VLLM_SUPPORTED_MAX_RANKS[-1]}")


def compute_export_rank(arch_spec: dict[str, Any]) -> int:
    if arch_spec["kind"] == "uniform":
        return int(arch_spec["uniform_rank"])
    if arch_spec["kind"] == "m3_dac":
        by_layer: dict[int, int] = defaultdict(int)
        z1_r = int(arch_spec.get("zone1_dense_qo_rank", 0))
        for li in arch_spec.get("zone1_dense_qo_layers", []):
            by_layer[int(li)] += z1_r
        for k, r in arch_spec.get("head_ranks", {}).items():
            li = int(k.split(":")[0])
            by_layer[li] += int(r)
        max_qo = max(by_layer.values(), default=0)
        max_mlp = int(arch_spec.get("mlp_rank", 0)) if arch_spec.get("mlp_layers") else 0
        return max(max_qo, max_mlp, 1)
    raise ValueError(f"Unknown arch kind: {arch_spec['kind']!r}")


def _read_scaling_from_meta(adapter_state_path: Path) -> dict[str, Any]:
    for cand in (
        adapter_state_path.parent / "adapter_config.json",
        adapter_state_path.parent / "train_meta.json",
        adapter_state_path.parent.parent / "adapter_config.json",
        adapter_state_path.parent.parent / "train_meta.json",
    ):
        if cand.exists():
            meta = json.loads(cand.read_text(encoding="utf-8"))
            sc = meta.get("scaling") or meta.get("lora_scaling") or meta.get("hyperparams")
            if isinstance(sc, dict) and "scaling_mode" in sc:
                return {k: sc[k] for k in ("scaling_mode", "base_scale", "anchor_rank") if k in sc}
    return {"scaling_mode": "rslora", "base_scale": 2.0, "anchor_rank": 8}


def export_c3_adapter_to_peft(
    *,
    state_dict: dict[str, torch.Tensor],
    arch_spec: dict[str, Any],
    model_spec: dict[str, Any],
    scaling: dict[str, Any],
    out_dir: Path,
    base_model_name_or_path: str = "",
    export_dtype: torch.dtype = torch.bfloat16,
    adapter_scale: float = 1.0,
) -> dict[str, Any]:
    d = int(model_spec["hidden_size"])
    H = int(model_spec["num_heads"])
    hd = int(model_spec["head_dim"])
    d_ff = int(model_spec["intermediate_size"])
    nL = int(model_spec["num_layers"])

    scaling_mode = str(scaling.get("scaling_mode", "rslora"))
    base_scale = float(scaling.get("base_scale", 2.0)) * float(adapter_scale)
    anchor_rank = int(scaling.get("anchor_rank", 8))

    r_export = compute_export_rank(arch_spec)
    vllm_max_r = recommended_vllm_max_lora_rank(r_export)
    assert vllm_max_r <= 128, f"vLLM max_lora_rank {vllm_max_r} > 128 (r_export={r_export})"

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
        mod_norm_true = float(true_dw_f32.norm())
        if mod_norm_true > 1e-12:
            mod_rel_err = float((recon_out - true_dw_f32).norm()) / mod_norm_true
            assert mod_rel_err < 0.02, f"Exported {export_dtype} rel Frobenius error on {prefix}: {mod_rel_err:.4f}"
        sum_dot += float((recon_out * true_dw_f32).sum())
        sum_true_sq += float((true_dw_f32 * true_dw_f32).sum())
        sum_out_sq += float((recon_out * recon_out).sum())

        peft_tensors[f"base_model.model.{prefix}.lora_A.weight"] = A_out
        peft_tensors[f"base_model.model.{prefix}.lora_B.weight"] = B_out
        target_module_names.add(mod_name)
        n_exported_modules += 1

    if arch_spec["kind"] == "uniform":
        r_u = int(arch_spec["uniform_rank"])
        s_u = M.compute_lora_scale(r_u, scaling_mode, base_scale, anchor_rank)
        targets = set(arch_spec.get("target_modules", ["q_proj", "o_proj", "gate_proj", "down_proj"]))
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

    elif arch_spec["kind"] == "m3_dac":
        # 1. Zone 1 dense Q/O layers
        z1_r = int(arch_spec.get("zone1_dense_qo_rank", 0))
        if z1_r > 0:
            s_z1 = M.compute_lora_scale(z1_r, scaling_mode, base_scale, anchor_rank)
            for li in sorted(int(x) for x in arch_spec.get("zone1_dense_qo_layers", [])):
                for mname, d_in, d_out in (("q_proj", d, H * hd), ("o_proj", H * hd, d)):
                    pre = f"model.layers.{li}.self_attn.{mname}"
                    mA = state_dict[f"{pre}.lora_A"].float()
                    mB = state_dict[f"{pre}.lora_B"].float()
                    A_full = torch.zeros((r_export, d_in), dtype=torch.float32)
                    B_full = torch.zeros((d_out, r_export), dtype=torch.float32)
                    A_full[:z1_r, :] = mA
                    B_full[:, :z1_r] = mB * s_z1
                    true_dw = (mB @ mA) * s_z1
                    _check_and_store(pre, mname, A_full, B_full, true_dw)

        # 2. Zone 2+3 per-head tiered-rank Q/O layers (each head gets its OWN rank block & OWN rsLoRA scale!)
        by_layer_hr: dict[int, dict[int, int]] = defaultdict(dict)
        for key, r in arch_spec.get("head_ranks", {}).items():
            li_s, h_s = key.split(":")
            by_layer_hr[int(li_s)][int(h_s)] = int(r)
        for li, hr_map in sorted(by_layer_hr.items()):
            items = sorted(hr_map.items())
            # q_proj
            q_pre = f"model.layers.{li}.self_attn.q_proj"
            A_full = torch.zeros((r_export, d), dtype=torch.float32)
            B_full = torch.zeros((H * hd, r_export), dtype=torch.float32)
            true_dw = torch.zeros((H * hd, d), dtype=torch.float32)
            off = 0
            for h, r_h in items:
                s_h = M.compute_lora_scale(r_h, scaling_mode, base_scale, anchor_rank)
                qA = state_dict[f"{q_pre}.lora_A.{h}"].float()
                qB = state_dict[f"{q_pre}.lora_B.{h}"].float()
                rs = slice(off, off + r_h)
                sl = slice(h * hd, (h + 1) * hd)
                A_full[rs, :] = qA
                B_full[sl, rs] = qB * s_h
                true_dw[sl, :] += (qB @ qA) * s_h
                off += r_h
            assert off <= r_export
            _check_and_store(q_pre, "q_proj", A_full, B_full, true_dw)

            # o_proj
            o_pre = f"model.layers.{li}.self_attn.o_proj"
            A_full = torch.zeros((r_export, H * hd), dtype=torch.float32)
            B_full = torch.zeros((d, r_export), dtype=torch.float32)
            true_dw = torch.zeros((d, H * hd), dtype=torch.float32)
            off = 0
            for h, r_h in items:
                s_h = M.compute_lora_scale(r_h, scaling_mode, base_scale, anchor_rank)
                oA = state_dict[f"{o_pre}.lora_A.{h}"].float()
                oB = state_dict[f"{o_pre}.lora_B.{h}"].float()
                rs = slice(off, off + r_h)
                sl = slice(h * hd, (h + 1) * hd)
                A_full[rs, sl] = oA
                B_full[:, rs] = oB * s_h
                true_dw[:, sl] += (oB @ oA) * s_h
                off += r_h
            assert off <= r_export
            _check_and_store(o_pre, "o_proj", A_full, B_full, true_dw)

        # 3. Zone 2+3 MLPs (gate_proj + down_proj)
        r_m = int(arch_spec.get("mlp_rank", 0))
        if r_m > 0:
            s_m = M.compute_lora_scale(r_m, scaling_mode, base_scale, anchor_rank)
            for li in sorted(set(int(x) for x in arch_spec.get("mlp_layers", []))):
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
    else:
        raise ValueError(f"Unknown arch kind: {arch_spec['kind']!r}")

    overall_export_cos = (
        float(sum_dot / ((sum_true_sq**0.5) * (sum_out_sq**0.5)))
        if (sum_true_sq > 0.0 and sum_out_sq > 0.0)
        else 1.0
    )
    assert overall_export_cos >= 0.999, f"Overall export cosine too low: {overall_export_cos:.6f}"

    out_dir.mkdir(parents=True, exist_ok=True)
    sf_path = out_dir / "adapter_model.safetensors"
    save_file(peft_tensors, str(sf_path))
    sf_sha = hashlib.sha256(sf_path.read_bytes()).hexdigest()

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
    (out_dir / "c3_peft_export_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8", newline="\n")
    return meta


def run_selftest_tiny() -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
    )
    ms = {"hidden_size": 64, "num_heads": 4, "head_dim": 16, "intermediate_size": 128, "num_layers": 6}
    specs = {
        "M3-DAC": {
            "kind": "m3_dac",
            "zones": {"zone1": [0, 1], "zone2": [2, 3], "zone3": [4, 5]},
            "zone1_dense_qo_layers": [0, 1],
            "zone1_dense_qo_rank": 2,
            "mlp_layers": [2, 4, 5],
            "mlp_rank": 4,
            # Layer 2 has 3 heads with 3 DIFFERENT ranks (8 + 4 + 2 = 14 -> r_export=14, vllm_max_r=16)!
            "head_ranks": {"2:0": 8, "2:1": 4, "2:3": 2, "4:2": 6, "5:1": 2},
        },
        "Uniform-LoRA": {
            "kind": "uniform",
            "uniform_rank": 4,
            "target_modules": ["q_proj", "o_proj", "gate_proj", "down_proj"],
            "zones": {"zone1": [0, 1], "zone2": [2, 3], "zone3": [4, 5]},
        },
    }
    ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    scaling = {"scaling_mode": "rslora", "base_scale": 2.0, "anchor_rank": 8}

    with tempfile.TemporaryDirectory() as td:
        for name, spec in specs.items():
            torch.manual_seed(42)
            m_live = Qwen3ForCausalLM(cfg).float()
            M.apply_c3_adapters(m_live, spec, **scaling)
            with torch.no_grad():
                for pname, p in m_live.named_parameters():
                    if p.requires_grad and "lora_B" in pname:
                        p.normal_(0.0, 0.05)
            m_live.eval()
            with torch.no_grad():
                y_live = m_live(input_ids=ids).logits

            ap = Path(td) / f"{name}.pt"
            M.save_c3_adapter_state(m_live, ap)
            sd = torch.load(ap, map_location="cpu", weights_only=True)
            out_dir = Path(td) / f"peft_{name}"
            meta = export_c3_adapter_to_peft(
                state_dict=sd,
                arch_spec=spec,
                model_spec=ms,
                scaling=scaling,
                out_dir=out_dir,
                export_dtype=torch.float32,
            )

            torch.manual_seed(42)
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
                    mod.weight.data += B_t @ A_t
            m_peft.eval()
            with torch.no_grad():
                y_peft = m_peft(input_ids=ids).logits
            logit_err = float((y_live - y_peft).abs().max())
            assert logit_err < 1e-5, (name, logit_err)
            print(
                f"[OK] c3 PEFT export self-test {name:12s}: r_export={meta['r_export']:2d} "
                f"(vllm_max_r={meta['recommended_vllm_max_lora_rank']:2d}) "
                f"dw_err={meta['max_fp32_dw_abs_err']:.2e} logit_err={logit_err:.2e}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", choices=["qwen3_14b", "qwen3_8b"])
    ap.add_argument("--arch", help="Architecture name in c3_components_frozen.json (e.g. Uniform-LoRA, M3-DAC)")
    ap.add_argument("--components-file", type=Path, default=C3 / "data" / "c3_components_frozen.json")
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

    assert args.model_key and args.arch and args.adapter_state and args.out_dir
    actual_sha = hashlib.sha256(args.adapter_state.read_bytes()).hexdigest()
    if args.expected_adapter_sha256 and actual_sha != args.expected_adapter_sha256:
        raise ValueError(f"Adapter SHA-256 mismatch: {actual_sha} != {args.expected_adapter_sha256}")

    comp = json.loads(args.components_file.read_text(encoding="utf-8"))
    m_info = comp["models"][args.model_key]
    arch_spec = m_info["architectures"][args.arch]
    scaling = _read_scaling_from_meta(args.adapter_state)

    sd = torch.load(args.adapter_state, map_location="cpu", weights_only=True)
    meta = export_c3_adapter_to_peft(
        state_dict=sd,
        arch_spec=arch_spec,
        model_spec=m_info["model_spec"],
        scaling=scaling,
        out_dir=args.out_dir,
        base_model_name_or_path=args.base_model,
        export_dtype=torch.bfloat16,
        adapter_scale=args.adapter_scale,
    )
    print(f"[OK] exported C3 PEFT LoRA -> {args.out_dir} | {json.dumps(meta)}")


if __name__ == "__main__":
    main()
