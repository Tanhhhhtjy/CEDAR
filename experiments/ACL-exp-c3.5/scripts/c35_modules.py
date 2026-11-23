#!/usr/bin/env python3
"""PyTorch adapter modules for ACL-exp-c3.5: extends C3's 3-Zone M3-DAC to support variable per-layer MLP ranks (`M3-v2`).

100% backward compatible with frozen C3 architectures (`Uniform-LoRA`, `M3-DAC` / `M3-v1`) while adding:
  - Per-layer MLP ranks `mlp_ranks: dict[str, int]` (enabling `M3-v2`: Late-Zone-1 bridge MLPs `L8-13 @ r=4`,
    Full Zone-2 MLPs `L14-27` with B1 top-5 N_CF MLPs `@ r=8` and remaining Zone-2 MLPs `@ r=4`, and zero Zone-3 MLPs!).
  - Unified `apply_c3_adapters` / `apply_c35_adapters` API so all C3/C3.5 eval/parity/probe scripts work seamlessly.
  - Exact O(r^2 d) differentiable Zone-1 proximal penalty covering ALL Zone-1 modules (both Late-Z1 Q/O and Late-Z1 MLP!).

CPU Self-Test:
  python experiments/ACL-exp-c3.5/scripts/c35_modules.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
C35 = HERE.parents[0]
REPO = C35.parents[1]
C3 = REPO / "experiments" / "ACL-exp-c3"

sys.path.insert(0, str(C3 / "scripts"))
import c3_modules as M3  # noqa: E402

# Re-export core primitives untouched from frozen C3
compute_lora_scale = M3.compute_lora_scale
PerHeadRankLoRALinear = M3.PerHeadRankLoRALinear
StandardLoRALinear = M3.StandardLoRALinear
iter_c3_adapters = M3.iter_c3_adapters
count_trainable_params = M3.count_trainable_params
count_zone_trainable_params = M3.count_zone_trainable_params
build_zone_optimizer_groups = M3.build_zone_optimizer_groups
zone1_proximal_penalty = M3.zone1_proximal_penalty
set_zone_ablation = M3.set_zone_ablation
zone_delta_norms = M3.zone_delta_norms
zone_grad_norms = M3.zone_grad_norms
save_c3_adapter_state = M3.save_c3_adapter_state
load_c3_adapter_state = M3.load_c3_adapter_state
merge_c3_adapters = M3.merge_c3_adapters


def resolve_mlp_rank_map(arch_spec: dict[str, Any]) -> dict[int, int]:
    """Return `{layer_idx: rank}` supporting both C3.5 `mlp_ranks` dict and legacy C3 `mlp_layers` + `mlp_rank`."""
    if "mlp_ranks" in arch_spec and arch_spec["mlp_ranks"]:
        return {int(k): int(v) for k, v in sorted(arch_spec["mlp_ranks"].items(), key=lambda kv: int(kv[0])) if int(v) > 0}
    mlp_r = int(arch_spec.get("mlp_rank", 0))
    if mlp_r > 0:
        return {int(li): mlp_r for li in sorted(set(int(x) for x in arch_spec.get("mlp_layers", [])))}
    return {}


def apply_c35_adapters(
    model: nn.Module,
    arch_spec: dict[str, Any],
    scaling_mode: str = "rslora",
    base_scale: float = 2.0,
    anchor_rank: int = 8,
    dropout: float = 0.0,
) -> dict[str, Any]:
    """Freeze all base weights and attach exact C3/C3.5 adapters (`uniform`, `m3_dac` / `M3-v1`, or `M3-v2`)."""
    if arch_spec["kind"] == "uniform":
        return M3.apply_c3_adapters(
            model, arch_spec,
            scaling_mode=scaling_mode, base_scale=base_scale, anchor_rank=anchor_rank, dropout=dropout,
        )
    if arch_spec["kind"] != "m3_dac":
        raise ValueError(f"Unknown arch kind: {arch_spec['kind']!r}")

    for p in model.parameters():
        p.requires_grad_(False)

    layers = model.model.layers
    hd = M3._head_dim_of(model)
    zones = arch_spec["zones"]
    attached: dict[str, Any] = {
        "kind": arch_spec["kind"],
        "zone1_modules": 0,
        "head_modules": 0,
        "mlp_modules": 0,
        "uniform_modules": 0,
        "scaling_mode": scaling_mode,
        "base_scale": base_scale,
        "anchor_rank": anchor_rank,
    }

    # 1. Zone 1 (or Late-Zone-1) dense Q/O bridge
    z1_layers = [int(li) for li in arch_spec.get("zone1_dense_qo_layers", [])]
    z1_r = int(arch_spec.get("zone1_dense_qo_rank", 0))
    if z1_layers and z1_r > 0:
        s_z1 = compute_lora_scale(z1_r, scaling_mode, base_scale, anchor_rank)
        for li in z1_layers:
            attn = layers[li].self_attn
            z = M3._layer_zone(li, zones)
            for name in ("q_proj", "o_proj"):
                w = StandardLoRALinear(getattr(attn, name), z1_r, scale=s_z1, dropout=dropout)
                w.c3_zone = z
                w.c3_kind = f"z{z}_qo"
                setattr(attn, name, w)
                attached["zone1_modules"] += 1

    # 2. Zone 2+3 per-head tiered-rank Q/O slices
    by_layer_hr: dict[int, dict[int, int]] = defaultdict(dict)
    for key, r in arch_spec.get("head_ranks", {}).items():
        li_s, h_s = key.split(":")
        li, h = int(li_s), int(h_s)
        assert li not in set(z1_layers), f"Layer {li} cannot be in both Zone 1 dense Q/O and head slices!"
        by_layer_hr[li][h] = int(r)
    for li, hr_map in sorted(by_layer_hr.items()):
        z = M3._layer_zone(li, zones)
        attn = layers[li].self_attn
        wq = PerHeadRankLoRALinear(
            attn.q_proj, hr_map, hd, axis="out",
            scaling_mode=scaling_mode, base_scale=base_scale, anchor_rank=anchor_rank, dropout=dropout,
        )
        wq.c3_zone = z
        wq.c3_kind = f"z{z}_head"
        wo = PerHeadRankLoRALinear(
            attn.o_proj, hr_map, hd, axis="in",
            scaling_mode=scaling_mode, base_scale=base_scale, anchor_rank=anchor_rank, dropout=dropout,
        )
        wo.c3_zone = z
        wo.c3_kind = f"z{z}_head"
        attn.q_proj = wq
        attn.o_proj = wo
        attached["head_modules"] += 2

    # 3. Per-layer variable-rank MLPs (gate_proj + down_proj), supporting Zone-1/2/3 cleanly!
    mlp_rank_map = resolve_mlp_rank_map(arch_spec)
    for li, r_m in sorted(mlp_rank_map.items()):
        z = M3._layer_zone(li, zones)
        s_m = compute_lora_scale(r_m, scaling_mode, base_scale, anchor_rank)
        mlp = layers[li].mlp
        for name in ("gate_proj", "down_proj"):
            w = StandardLoRALinear(getattr(mlp, name), r_m, scale=s_m, dropout=dropout)
            w.c3_zone = z
            w.c3_kind = f"z{z}_mlp"
            setattr(mlp, name, w)
            attached["mlp_modules"] += 1

    attached["trainable_params"] = count_trainable_params(model)
    attached["zone_trainable_params"] = count_zone_trainable_params(model)
    return attached


# Alias for drop-in monkeypatch compatibility with frozen C3 eval/parity utilities
apply_c3_adapters = apply_c35_adapters


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
    # Tiny M3-v2: L0 completely frozen, L1 late-Z1 Q/O r=2 + MLP r=4, L2-3 Zone-2 MLP r=8/4 + heads, L4-5 Zone-3 minimal heads, zero Z3 MLP
    arch_m3v2 = {
        "kind": "m3_dac",
        "zones": {"zone1": [0, 1], "zone2": [2, 3], "zone3": [4, 5]},
        "zone1_dense_qo_layers": [1],
        "zone1_dense_qo_rank": 2,
        "mlp_ranks": {"1": 4, "2": 8, "3": 4},
        "head_ranks": {"2:0": 8, "2:2": 4, "3:1": 6, "4:3": 2, "5:0": 2},
    }
    ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    torch.manual_seed(17)
    m = Qwen3ForCausalLM(cfg).float()
    l0_q_before = m.model.layers[0].self_attn.q_proj.weight.data.clone()
    l0_mlp_before = m.model.layers[0].mlp.gate_proj.weight.data.clone()
    att = apply_c35_adapters(m, arch_m3v2)

    exp_z1_qo = 1 * 2 * (64 + 64) * 2
    exp_mlp = sum(2 * r * (64 + 128) for r in arch_m3v2["mlp_ranks"].values())
    exp_heads = sum(2 * r * (64 + 16) for r in arch_m3v2["head_ranks"].values())
    assert att["trainable_params"] == exp_z1_qo + exp_mlp + exp_heads, (att["trainable_params"], exp_z1_qo + exp_mlp + exp_heads)
    # Zone 1 must contain BOTH L1 Q/O and L1 MLP!
    exp_z1_total = exp_z1_qo + 2 * 4 * (64 + 128)
    assert att["zone_trainable_params"]["zone1"] == exp_z1_total, (att["zone_trainable_params"], exp_z1_total)

    with torch.no_grad():
        y0 = m(input_ids=ids).logits.clone()
        for pn, p in m.named_parameters():
            if p.requires_grad and "lora_B" in pn:
                p.normal_(0.0, 0.05)
        y1 = m(input_ids=ids).logits.clone()
    assert float((y1 - y0).abs().max()) > 1e-4

    # Fast O(r^2 d) Zone-1 prox check covering BOTH Late-Z1 Q/O AND Late-Z1 MLP
    fast_z1 = float(zone1_proximal_penalty(m).detach())
    exact_z1 = 0.0
    for _, mod in iter_c3_adapters(m):
        if mod.c3_zone == 1:
            dw = (mod.lora_B.float() @ mod.lora_A.float()) * mod.scale
            exact_z1 += float(dw.detach().pow(2).sum())
    assert abs(fast_z1 - exact_z1) < 1e-5, (fast_z1, exact_z1)

    # Merge equivalence + untouched early L0 check!
    merge_c3_adapters(m)
    with torch.no_grad():
        y_merged = m(input_ids=ids).logits
    assert float((y_merged - y1).abs().max()) < 1e-5
    assert float((m.model.layers[0].self_attn.q_proj.weight.data - l0_q_before).abs().max()) == 0.0
    assert float((m.model.layers[0].mlp.gate_proj.weight.data - l0_mlp_before).abs().max()) == 0.0

    print(
        f"[OK] c35_modules self-test PASSED: params={att['trainable_params']:,}, "
        f"zones={att['zone_trainable_params']}, prox_err={abs(fast_z1 - exact_z1):.2e}, "
        f"merge_err={float((y_merged - y1).abs().max()):.2e}, early_L0_diff=0.0"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-tiny", action="store_true")
    args = ap.parse_args()
    if args.selftest_tiny:
        run_selftest_tiny()
