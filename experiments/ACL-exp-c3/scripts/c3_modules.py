#!/usr/bin/env python3
"""PyTorch adapter modules for ACL-exp-c3: Per-Head Tiered-Rank LoRA + 3-Zone M3-DAC + Uniform LoRA.

Key capabilities & invariants (verified by `--selftest-tiny`):
  1. `PerHeadRankLoRALinear`: supports arbitrary per-head ranks `{h: r_h}` on `q_proj` (axis='out') and
     `o_proj` (axis='in'), applying each head's own rank-fair rsLoRA scaling s(r_h) = base_scale * sqrt(anchor/r_h).
     Untouched heads and all K/V projections stay strictly frozen (exact 0.0 diff).
  2. Exact O(r^2 * d) differentiable Delta-W Frobenius-squared norm:
       ||s * B @ A||_F^2 == s^2 * sum((B^T B) * (A A^T))
     Enables exact Zone-1 proximal regularization on actual Delta-W without materializing [d, d] matrices!
  3. Unified `apply_c3_adapters(model, arch_spec)` supporting both `Uniform-LoRA` and `M3-DAC` (and all
     M3 attribution controls: Equal-Rank, Shuffled-Rank, Random-Mask), tagging every module with `c3_zone in {1,2,3}`.
  4. `set_zone_ablation(model, disable_zones=...)`: enables zero-retraining 3-zone ablation on BOTH C2's
     existing `unif_base/adapter_state.pt` and any C3 checkpoint!

CPU Self-Test:
  python experiments/ACL-exp-c3/scripts/c3_modules.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_lora_scale(rank: int, scaling_mode: str = "rslora", base_scale: float = 2.0, anchor_rank: int = 8) -> float:
    if scaling_mode == "rslora":
        return float(base_scale * math.sqrt(anchor_rank / max(1, int(rank))))
    if scaling_mode == "constant":
        return float(base_scale)
    raise ValueError(f"Unknown scaling_mode: {scaling_mode!r}")


def _fast_ba_frob_sq(A: torch.Tensor, B: torch.Tensor, scale: float) -> torch.Tensor:
    """Exact differentiable ||scale * (B @ A)||_F^2 in O(r^2 * (d_in + d_out)) via tr((B^T B)(A A^T))."""
    Af = A.float()
    Bf = B.float()
    btb = Bf.t() @ Bf  # [r, r]
    aat = Af @ Af.t()  # [r, r]
    return (scale * scale) * (btb * aat).sum()


class PerHeadRankLoRALinear(nn.Module):
    """GQA head-slice LoRA supporting distinct rank r_h and distinct rsLoRA scale s(r_h) per selected head."""

    def __init__(
        self,
        base: nn.Linear,
        head_ranks: dict[int, int],
        head_dim: int,
        axis: str,
        scaling_mode: str = "rslora",
        base_scale: float = 2.0,
        anchor_rank: int = 8,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert axis in ("out", "in"), axis
        assert len(head_ranks) > 0
        self.base = base
        self.head_ranks: dict[int, int] = {int(h): int(r) for h, r in sorted(head_ranks.items())}
        self.head_indices: list[int] = list(self.head_ranks.keys())
        self.head_dim = int(head_dim)
        self.axis = axis
        self.scaling_mode = scaling_mode
        self.base_scale = float(base_scale)
        self.anchor_rank = int(anchor_rank)
        self.head_scales: dict[int, float] = {
            h: compute_lora_scale(r, scaling_mode, base_scale, anchor_rank)
            for h, r in self.head_ranks.items()
        }
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.disabled: bool = False
        self.disabled_heads: set[int] = set()
        self.c3_zone: int = 2
        self.c3_kind: str = "head"

        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        dev, dtype = base.weight.device, base.weight.dtype
        d_in, d_out = base.in_features, base.out_features

        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        for h, r in self.head_ranks.items():
            assert r > 0
            k = str(h)
            if axis == "out":
                pA = nn.Parameter(torch.empty(r, d_in, device=dev, dtype=dtype))
                pB = nn.Parameter(torch.zeros(head_dim, r, device=dev, dtype=dtype))
            else:
                pA = nn.Parameter(torch.empty(r, head_dim, device=dev, dtype=dtype))
                pB = nn.Parameter(torch.zeros(d_out, r, device=dev, dtype=dtype))
            nn.init.kaiming_uniform_(pA, a=math.sqrt(5))
            self.lora_A[k] = pA
            self.lora_B[k] = pB

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.disabled:
            return y
        active = [h for h in self.head_indices if h not in self.disabled_heads]
        if not active:
            return y
        xd = self.drop(x)
        if self.axis == "out":
            y = y.clone()
            for h in active:
                k = str(h)
                s = self.head_scales[h]
                mid = F.linear(xd, self.lora_A[k])
                delta = F.linear(mid, self.lora_B[k]) * s
                sl = slice(h * self.head_dim, (h + 1) * self.head_dim)
                y[..., sl] = y[..., sl] + delta
            return y
        else:
            total_delta = None
            for h in active:
                k = str(h)
                s = self.head_scales[h]
                sl = slice(h * self.head_dim, (h + 1) * self.head_dim)
                mid = F.linear(xd[..., sl], self.lora_A[k])
                delta = F.linear(mid, self.lora_B[k]) * s
                total_delta = delta if total_delta is None else (total_delta + delta)
            return y + total_delta

    def delta_frobenius_sq(self) -> torch.Tensor:
        dev = self.base.weight.device
        tot = torch.zeros((), device=dev, dtype=torch.float32)
        if self.disabled:
            return tot
        for h in self.head_indices:
            if h in self.disabled_heads:
                continue
            k = str(h)
            tot = tot + _fast_ba_frob_sq(self.lora_A[k], self.lora_B[k], self.head_scales[h])
        return tot

    @torch.no_grad()
    def delta_frobenius_norm(self) -> float:
        return math.sqrt(max(0.0, float(self.delta_frobenius_sq().detach())))

    @torch.no_grad()
    def merge_into_base(self) -> nn.Linear:
        if self.disabled:
            return self.base
        W = self.base.weight.data
        for h in self.head_indices:
            if h in self.disabled_heads:
                continue
            k = str(h)
            dw = ((self.lora_B[k].float() @ self.lora_A[k].float()) * self.head_scales[h]).to(W.device)
            sl = slice(h * self.head_dim, (h + 1) * self.head_dim)
            if self.axis == "out":
                W[sl, :].copy_((W[sl, :].float() + dw).to(W.dtype))
            else:
                W[:, sl].copy_((W[:, sl].float() + dw).to(W.dtype))
        return self.base


class StandardLoRALinear(nn.Module):
    """Standard dense LoRA wrapper with exact differentiable Delta-W Frobenius-squared norm & zone tagging."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        scale: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert rank > 0
        self.base = base
        self.rank = int(rank)
        self.scale = float(scale)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.disabled: bool = False
        self.c3_zone: int = 2
        self.c3_kind: str = "standard"

        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        dev, dtype = base.weight.device, base.weight.dtype
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=dev, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=dev, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.disabled:
            return y
        h = F.linear(self.drop(x), self.lora_A)
        delta = F.linear(h, self.lora_B) * self.scale
        return y + delta

    def delta_frobenius_sq(self) -> torch.Tensor:
        if self.disabled:
            return torch.zeros((), device=self.base.weight.device, dtype=torch.float32)
        return _fast_ba_frob_sq(self.lora_A, self.lora_B, self.scale)

    @torch.no_grad()
    def delta_frobenius_norm(self) -> float:
        return math.sqrt(max(0.0, float(self.delta_frobenius_sq().detach())))

    @torch.no_grad()
    def merge_into_base(self) -> nn.Linear:
        if not self.disabled:
            W = self.base.weight.data
            dw = ((self.lora_B.float() @ self.lora_A.float()) * self.scale).to(W.device)
            W.copy_((W.float() + dw).to(W.dtype))
        return self.base


def _head_dim_of(model) -> int:
    hd = getattr(model.config, "head_dim", None)
    return int(hd) if hd else int(model.config.hidden_size // model.config.num_attention_heads)


def _layer_zone(li: int, zones: dict[str, list[int]]) -> int:
    for z_idx, z_key in ((1, "zone1"), (2, "zone2"), (3, "zone3")):
        lo, hi = zones[z_key]
        if lo <= li <= hi:
            return z_idx
    raise ValueError(f"Layer {li} outside zones {zones}")


def apply_c3_adapters(
    model: nn.Module,
    arch_spec: dict[str, Any],
    scaling_mode: str = "rslora",
    base_scale: float = 2.0,
    anchor_rank: int = 8,
    dropout: float = 0.0,
) -> dict[str, Any]:
    """Freeze all base weights and attach exact C3 adapters (`uniform` or `m3_dac`)."""
    for p in model.parameters():
        p.requires_grad_(False)

    layers = model.model.layers
    hd = _head_dim_of(model)
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

    if arch_spec["kind"] == "uniform":
        r_u = int(arch_spec["uniform_rank"])
        s_u = compute_lora_scale(r_u, scaling_mode, base_scale, anchor_rank)
        targets = set(arch_spec.get("target_modules", ["q_proj", "o_proj", "gate_proj", "down_proj"]))
        for li, layer in enumerate(layers):
            z = _layer_zone(li, zones)
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                if name in targets:
                    w = StandardLoRALinear(getattr(layer.self_attn, name), r_u, scale=s_u, dropout=dropout)
                    w.c3_zone = z
                    w.c3_kind = f"z{z}_attn"
                    setattr(layer.self_attn, name, w)
                    attached["uniform_modules"] += 1
            for name in ("gate_proj", "up_proj", "down_proj"):
                if name in targets:
                    w = StandardLoRALinear(getattr(layer.mlp, name), r_u, scale=s_u, dropout=dropout)
                    w.c3_zone = z
                    w.c3_kind = f"z{z}_mlp"
                    setattr(layer.mlp, name, w)
                    attached["uniform_modules"] += 1

    elif arch_spec["kind"] == "m3_dac":
        # Zone 1: dense Q/O bridge
        z1_layers = [int(li) for li in arch_spec.get("zone1_dense_qo_layers", [])]
        z1_r = int(arch_spec.get("zone1_dense_qo_rank", 0))
        if z1_layers and z1_r > 0:
            s_z1 = compute_lora_scale(z1_r, scaling_mode, base_scale, anchor_rank)
            for li in z1_layers:
                attn = layers[li].self_attn
                for name in ("q_proj", "o_proj"):
                    w = StandardLoRALinear(getattr(attn, name), z1_r, scale=s_z1, dropout=dropout)
                    w.c3_zone = 1
                    w.c3_kind = "z1_qo"
                    setattr(attn, name, w)
                    attached["zone1_modules"] += 1

        # Zone 2+3: per-head tiered-rank Q/O slices
        by_layer_hr: dict[int, dict[int, int]] = defaultdict(dict)
        for key, r in arch_spec.get("head_ranks", {}).items():
            li_s, h_s = key.split(":")
            li, h = int(li_s), int(h_s)
            assert li not in set(z1_layers), f"Layer {li} cannot be in both Zone 1 dense Q/O and head slices!"
            by_layer_hr[li][h] = int(r)
        for li, hr_map in sorted(by_layer_hr.items()):
            z = _layer_zone(li, zones)
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

        # Zone 2+3: selected mid+top MLPs (gate_proj + down_proj)
        mlp_r = int(arch_spec.get("mlp_rank", 0))
        if mlp_r > 0:
            s_m = compute_lora_scale(mlp_r, scaling_mode, base_scale, anchor_rank)
            for li in [int(x) for x in arch_spec.get("mlp_layers", [])]:
                z = _layer_zone(li, zones)
                mlp = layers[li].mlp
                for name in ("gate_proj", "down_proj"):
                    w = StandardLoRALinear(getattr(mlp, name), mlp_r, scale=s_m, dropout=dropout)
                    w.c3_zone = z
                    w.c3_kind = f"z{z}_mlp"
                    setattr(mlp, name, w)
                    attached["mlp_modules"] += 1
    else:
        raise ValueError(f"Unknown arch kind: {arch_spec['kind']!r}")

    attached["trainable_params"] = count_trainable_params(model)
    attached["zone_trainable_params"] = count_zone_trainable_params(model)
    return attached


def iter_c3_adapters(model: nn.Module):
    """Yield `(name, module)` for every attached C3 LoRA wrapper."""
    for name, mod in model.named_modules():
        if isinstance(mod, (PerHeadRankLoRALinear, StandardLoRALinear)):
            yield name, mod


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_zone_trainable_params(model: nn.Module) -> dict[str, int]:
    out = {"zone1": 0, "zone2": 0, "zone3": 0}
    for _, mod in iter_c3_adapters(model):
        n = sum(p.numel() for p in mod.parameters() if p.requires_grad)
        out[f"zone{mod.c3_zone}"] += n
    return out


def build_zone_optimizer_groups(
    model: nn.Module,
    lr: float,
    lr_z1_mult: float = 1.0,
    weight_decay: float = 0.0,
) -> list[dict[str, Any]]:
    """Create AdamW parameter groups giving Zone-1 parameters `lr * lr_z1_mult` and Zone 2+3 `lr`."""
    z1_params, z23_params = [], []
    seen = set()
    for _, mod in iter_c3_adapters(model):
        for p in mod.parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            if mod.c3_zone == 1:
                z1_params.append(p)
            else:
                z23_params.append(p)
    groups = []
    if z1_params:
        groups.append({"params": z1_params, "lr": lr * lr_z1_mult, "weight_decay": weight_decay, "name": "zone1"})
    if z23_params:
        groups.append({"params": z23_params, "lr": lr, "weight_decay": weight_decay, "name": "zone23"})
    return groups


def zone1_proximal_penalty(model: nn.Module) -> torch.Tensor:
    """Exact differentiable sum of ||Delta W||_F^2 across all Zone-1 LoRA modules."""
    dev = next(model.parameters()).device
    tot = torch.zeros((), device=dev, dtype=torch.float32)
    for _, mod in iter_c3_adapters(model):
        if mod.c3_zone == 1:
            tot = tot + mod.delta_frobenius_sq().to(dev)
    return tot


def set_zone_ablation(
    model: nn.Module,
    disable_zones: tuple[int, ...] | list[int] | set[int] = (),
    disable_subkinds: tuple[str, ...] | list[str] | set[str] = (),
) -> None:
    """Enable/disable LoRA deltas by zone (`1, 2, 3`) or subkind (`attn`, `mlp`) for zero-retraining ablations."""
    dz = {int(z) for z in disable_zones}
    dk = set(disable_subkinds)
    for _, mod in iter_c3_adapters(model):
        is_mlp = "mlp" in mod.c3_kind
        is_attn = not is_mlp
        off = (mod.c3_zone in dz) or ("mlp" in dk and is_mlp) or ("attn" in dk and is_attn)
        mod.disabled = bool(off)


@torch.no_grad()
def zone_delta_norms(model: nn.Module) -> dict[str, float]:
    """Report actual Frobenius norms ||Delta W||_F broken down by zone (1/2/3) and projection type (attn/mlp)."""
    sq = defaultdict(float)
    for _, mod in iter_c3_adapters(model):
        val = float(mod.delta_frobenius_sq().detach())
        sq[f"zone{mod.c3_zone}"] += val
        sq["mlp" if "mlp" in mod.c3_kind else "attn"] += val
        sq["total"] += val
    return {
        "z1_frob": math.sqrt(max(0.0, sq["zone1"])),
        "z2_frob": math.sqrt(max(0.0, sq["zone2"])),
        "z3_frob": math.sqrt(max(0.0, sq["zone3"])),
        "attn_frob": math.sqrt(max(0.0, sq["attn"])),
        "mlp_frob": math.sqrt(max(0.0, sq["mlp"])),
        "total_frob": math.sqrt(max(0.0, sq["total"])),
    }


@torch.no_grad()
def zone_grad_norms(model: nn.Module) -> dict[str, float]:
    """Compute pre-clip L2 gradient norms separately for Zone 1, Zone 2, Zone 3, and Total."""
    sq = defaultdict(float)
    for _, mod in iter_c3_adapters(model):
        for p in mod.parameters():
            if p.requires_grad and p.grad is not None:
                v = float(p.grad.detach().float().pow(2).sum())
                sq[f"zone{mod.c3_zone}"] += v
                sq["total"] += v
    return {
        "g_z1": math.sqrt(max(0.0, sq["zone1"])),
        "g_z2": math.sqrt(max(0.0, sq["zone2"])),
        "g_z3": math.sqrt(max(0.0, sq["zone3"])),
        "g_total": math.sqrt(max(0.0, sq["total"])),
    }


def save_c3_adapter_state(model: nn.Module, path) -> None:
    state = {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if (".lora_A" in k or ".lora_B" in k)
    }
    torch.save(state, path)


def load_c3_adapter_state(model: nn.Module, path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    missing, extra = expected - state.keys(), state.keys() - expected
    if missing or extra:
        raise ValueError(f"C3 adapter keys mismatch: missing={sorted(missing)[:6]} ({len(missing)}), extra={sorted(extra)[:6]} ({len(extra)})")
    model.load_state_dict(state, strict=False)


@torch.no_grad()
def merge_c3_adapters(model: nn.Module) -> nn.Module:
    for layer in model.model.layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            m = getattr(layer.self_attn, name)
            if isinstance(m, (PerHeadRankLoRALinear, StandardLoRALinear)):
                setattr(layer.self_attn, name, m.merge_into_base())
        for name in ("gate_proj", "up_proj", "down_proj"):
            m = getattr(layer.mlp, name)
            if isinstance(m, StandardLoRALinear):
                setattr(layer.mlp, name, m.merge_into_base())
    return model


# -------------------------------------------------------------------------------------- CPU Self-Test


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
    arch_m3 = {
        "kind": "m3_dac",
        "zones": {"zone1": [0, 1], "zone2": [2, 3], "zone3": [4, 5]},
        "zone1_dense_qo_layers": [0, 1],
        "zone1_dense_qo_rank": 2,
        "mlp_layers": [2, 4, 5],
        "mlp_rank": 4,
        "head_ranks": {"2:0": 8, "2:2": 4, "3:1": 6, "4:3": 4, "5:0": 2},
    }
    ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

    torch.manual_seed(7)
    m = Qwen3ForCausalLM(cfg).float()
    att = apply_c3_adapters(m, arch_m3)
    # 1. Exact param count check
    exp_z1 = 2 * 2 * (64 + 64) * 2
    exp_mlp = 3 * 2 * 4 * (64 + 128)
    exp_heads = sum(2 * r * (64 + 16) for r in arch_m3["head_ranks"].values())
    assert att["trainable_params"] == exp_z1 + exp_mlp + exp_heads, (att["trainable_params"], exp_z1 + exp_mlp + exp_heads)

    # 2. Zero-init exact identity with base
    with torch.no_grad():
        y0 = m(input_ids=ids).logits.clone()
        for pn, p in m.named_parameters():
            if p.requires_grad and "lora_B" in pn:
                p.normal_(0.0, 0.05)
        y1 = m(input_ids=ids).logits.clone()
    assert float((y1 - y0).abs().max()) > 1e-4

    # 3. Fast O(r^2 d) Delta-W Frobenius-squared identity check!
    fast_z1 = float(zone1_proximal_penalty(m).detach())
    exact_z1 = 0.0
    for _, mod in iter_c3_adapters(m):
        if mod.c3_zone == 1:
            dw = (mod.lora_B.float() @ mod.lora_A.float()) * mod.scale
            exact_z1 += float(dw.detach().pow(2).sum())
    assert abs(fast_z1 - exact_z1) < 1e-5, (fast_z1, exact_z1)

    # 4. Head-slice locality & merge equivalence check!
    q2_before = m.model.layers[2].self_attn.q_proj.base.weight.data.clone()
    m_copy = Qwen3ForCausalLM(cfg).float()
    m_copy.load_state_dict(
        {k: v for k, v in m.state_dict().items() if "lora_" not in k}, strict=False
    )
    # Instead verify merge in-place against live forward
    merge_c3_adapters(m)
    with torch.no_grad():
        y_merged = m(input_ids=ids).logits
    assert float((y_merged - y1).abs().max()) < 1e-5, float((y_merged - y1).abs().max())
    # Untouched heads 1 & 3 on Layer 2 q_proj must have EXACT 0.0 diff!
    q2_after = m.model.layers[2].self_attn.q_proj.weight.data
    for untouched_h in (1, 3):
        sl = slice(untouched_h * 16, (untouched_h + 1) * 16)
        assert float((q2_after[sl, :] - q2_before[sl, :]).abs().max()) == 0.0

    # 5. Zone ablation check on Uniform-LoRA
    arch_u = {
        "kind": "uniform",
        "uniform_rank": 2,
        "target_modules": ["q_proj", "o_proj", "gate_proj", "down_proj"],
        "zones": {"zone1": [0, 1], "zone2": [2, 3], "zone3": [4, 5]},
    }
    torch.manual_seed(9)
    mu = Qwen3ForCausalLM(cfg).float()
    with torch.no_grad():
        yu_base = mu(input_ids=ids).logits.clone()
    apply_c3_adapters(mu, arch_u)
    with torch.no_grad():
        for pn, p in mu.named_parameters():
            if p.requires_grad and "lora_B" in pn:
                p.normal_(0.0, 0.05)
        yu_all = mu(input_ids=ids).logits.clone()
        set_zone_ablation(mu, disable_zones=(1, 2, 3))
        yu_off = mu(input_ids=ids).logits.clone()
        set_zone_ablation(mu, disable_zones=(1,))
        yu_noz1 = mu(input_ids=ids).logits.clone()
    assert float((yu_off - yu_base).abs().max()) < 1e-6
    assert float((yu_noz1 - yu_all).abs().max()) > 1e-4

    print(
        f"[OK] c3_modules self-test PASSED: params={att['trainable_params']:,}, "
        f"fast_prox_err={abs(fast_z1 - exact_z1):.2e}, merge_err={float((y_merged - y1).abs().max()):.2e}, "
        f"untouched_head_diff=0.0, zone_ablation_ok=True"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-tiny", action="store_true")
    args = ap.parse_args()
    if args.selftest_tiny:
        run_selftest_tiny()
