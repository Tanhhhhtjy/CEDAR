#!/usr/bin/env python3
"""PyTorch adapter modules for ACL-exp-c1: exact GQA head-slice LoRA + selective MLP LoRA + Uniform LoRA.

Key guarantees (verified in c1_selftest.py):
1. HeadSliceLoRALinear modifies ONLY the exact [h*hd:(h+1)*hd] output rows (for q_proj) or input columns
   (for o_proj) of selected heads. Untouched heads and all K/V projections stay strictly frozen (exact 0 diff).
2. StandardLoRALinear applies standard low-rank adaptation on selected MLP (gate_proj/down_proj) or uniform layers.
3. merge_c1_adapters(model) merges all deltas cleanly back into standard nn.Linear weights and unwraps, yielding a
   vanilla HF Qwen3ForCausalLM checkpoint directly compatible with vLLM and ACL-exp-b1 post-tuning CT re-scan.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_lora_scale(rank: int, scaling_mode: str = "rslora", base_scale: float = 2.0, anchor_rank: int = 8) -> float:
    """Rank-fair LoRA scaling:
    - 'rslora' (default, Kalajdzievski 2023): s(r) = base_scale * sqrt(anchor_rank / r)
      Anchored at r=8 => s(8)=2.0, s(4)=2.83, s(18)=1.33, s(50)=0.80, equalizing initial r*s^2 variance across ranks.
    - 'constant': fixed s(r) = base_scale (classic alpha = base_scale * r).
    """
    if scaling_mode == "rslora":
        return float(base_scale * math.sqrt(anchor_rank / max(1, rank)))
    if scaling_mode == "constant":
        return float(base_scale)
    raise ValueError(f"Unknown scaling_mode: {scaling_mode!r}")


class HeadSliceLoRALinear(nn.Module):
    """Low-rank adapter restricted strictly to selected attention head slices of q_proj (axis='out') or o_proj (axis='in').

    - For q_proj (weight [H*hd, d_in], axis='out'):
        each selected head h has slice rows [h*hd:(h+1)*hd], A_h: [r, d_in], B_h: [hd, r]
        delta_h(x) = scale * (x @ A_h^T) @ B_h^T  added ONLY to output channels [h*hd:(h+1)*hd].
    - For o_proj (weight [d_out, H*hd], axis='in'):
        each selected head h reads ONLY input channels x[..., h*hd:(h+1)*hd], A_h: [r, hd], B_h: [d_out, r]
        delta_h(x) = scale * (x_h @ A_h^T) @ B_h^T  added to full output [d_out].
    """

    def __init__(
        self,
        base: nn.Linear,
        head_indices: list[int],
        head_dim: int,
        rank: int,
        axis: str,
        scale: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert axis in ("out", "in"), axis
        assert len(head_indices) > 0 and rank > 0
        self.base = base
        self.head_indices = sorted(set(int(h) for h in head_indices))
        self.head_dim = int(head_dim)
        self.rank = int(rank)
        self.axis = axis
        self.scale = float(scale)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.disabled_heads: set[int] = set()  # for zero-training post-hoc pathway ablations

        # Freeze base weights
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        dev = base.weight.device
        dtype = base.weight.dtype
        n_sel = len(self.head_indices)
        d_in, d_out = base.in_features, base.out_features

        # Batched parameters [n_sel, ...] for fast vectorized forward + merge
        if axis == "out":
            self.lora_A = nn.Parameter(torch.empty(n_sel, rank, d_in, device=dev, dtype=dtype))
            self.lora_B = nn.Parameter(torch.zeros(n_sel, head_dim, rank, device=dev, dtype=dtype))
        else:
            self.lora_A = nn.Parameter(torch.empty(n_sel, rank, head_dim, device=dev, dtype=dtype))
            self.lora_B = nn.Parameter(torch.zeros(n_sel, d_out, rank, device=dev, dtype=dtype))

        for i in range(n_sel):
            nn.init.kaiming_uniform_(self.lora_A[i], a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        active = [(i, h) for i, h in enumerate(self.head_indices) if h not in self.disabled_heads]
        if not active:
            return y
        xd = self.drop(x)
        if self.axis == "out":
            h_mid = torch.einsum("...d,srd->...sr", xd, self.lora_A)
            delta = torch.einsum("...sr,skr->...sk", h_mid, self.lora_B) * self.scale
            y = y.clone()
            for i, h in active:
                sl = slice(h * self.head_dim, (h + 1) * self.head_dim)
                y[..., sl] = y[..., sl] + delta[..., i, :]
            return y
        else:
            idxs = [i for i, _ in active]
            x_heads = torch.stack(
                [xd[..., h * self.head_dim : (h + 1) * self.head_dim] for _, h in active], dim=-2
            )
            h_mid = torch.einsum("...sk,srk->...sr", x_heads, self.lora_A[idxs])
            delta = torch.einsum("...sr,sor->...o", h_mid, self.lora_B[idxs]) * self.scale
            return y + delta

    @torch.no_grad()
    def delta_frobenius_norm(self) -> float:
        tot = 0.0
        for i, h in enumerate(self.head_indices):
            if h in self.disabled_heads:
                continue
            dw = (self.lora_B[i].float() @ self.lora_A[i].float()) * self.scale
            tot += float(dw.pow(2).sum())
        return math.sqrt(tot)

    @torch.no_grad()
    def merge_into_base(self) -> nn.Linear:
        W = self.base.weight.data
        for i, h in enumerate(self.head_indices):
            if h in self.disabled_heads:
                continue
            dw = ((self.lora_B[i].float() @ self.lora_A[i].float()) * self.scale).to(W.device)
            sl = slice(h * self.head_dim, (h + 1) * self.head_dim)
            if self.axis == "out":
                W[sl, :].copy_((W[sl, :].float() + dw).to(W.dtype))
            else:
                W[:, sl].copy_((W[:, sl].float() + dw).to(W.dtype))
        return self.base


class StandardLoRALinear(nn.Module):
    """Standard LoRA wrapper for MLP projections (gate_proj/down_proj) or uniform attention layers."""

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
        self.disabled: bool = False  # for zero-training post-hoc pathway ablations

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

    @torch.no_grad()
    def delta_frobenius_norm(self) -> float:
        if self.disabled:
            return 0.0
        dw = (self.lora_B.float() @ self.lora_A.float()) * self.scale
        return float(dw.norm())

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


def apply_c1_adapters(
    model: nn.Module,
    arm_spec: dict[str, Any],
    scaling_mode: str = "rslora",
    base_scale: float = 2.0,
    anchor_rank: int = 8,
    dropout: float = 0.0,
) -> dict[str, Any]:
    """Freeze all base parameters and attach exact C1 adapters specified by `arm_spec`."""
    for p in model.parameters():
        p.requires_grad_(False)

    layers = model.model.layers
    hd = _head_dim_of(model)
    attached: dict[str, Any] = {
        "head_modules": 0,
        "mlp_modules": 0,
        "uniform_modules": 0,
        "scaling_mode": scaling_mode,
        "scales": {},
    }

    if arm_spec["kind"] == "circuit":
        by_layer: dict[int, list[int]] = {}
        for li, h in arm_spec.get("heads", []):
            by_layer.setdefault(int(li), []).append(int(h))
        r_h = int(arm_spec.get("head_rank", 0))
        if r_h > 0:
            s_h = compute_lora_scale(r_h, scaling_mode, base_scale, anchor_rank)
            attached["scales"]["head"] = s_h
            for li, hs in sorted(by_layer.items()):
                attn = layers[li].self_attn
                attn.q_proj = HeadSliceLoRALinear(attn.q_proj, hs, hd, r_h, axis="out", scale=s_h, dropout=dropout)
                attn.o_proj = HeadSliceLoRALinear(attn.o_proj, hs, hd, r_h, axis="in", scale=s_h, dropout=dropout)
                attached["head_modules"] += 2
        r_m = int(arm_spec.get("mlp_rank", 0))
        if r_m > 0:
            s_m = compute_lora_scale(r_m, scaling_mode, base_scale, anchor_rank)
            attached["scales"]["mlp"] = s_m
            for li in arm_spec.get("mlp_layers", []):
                mlp = layers[int(li)].mlp
                mlp.gate_proj = StandardLoRALinear(mlp.gate_proj, r_m, scale=s_m, dropout=dropout)
                mlp.down_proj = StandardLoRALinear(mlp.down_proj, r_m, scale=s_m, dropout=dropout)
                attached["mlp_modules"] += 2
    elif arm_spec["kind"] == "uniform":
        r_u = int(arm_spec["uniform_rank"])
        s_u = compute_lora_scale(r_u, scaling_mode, base_scale, anchor_rank)
        attached["scales"]["uniform"] = s_u
        targets = set(arm_spec.get("target_modules", ["q_proj", "o_proj", "gate_proj", "down_proj"]))
        for layer in layers:
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                if name in targets:
                    mod = getattr(layer.self_attn, name)
                    setattr(layer.self_attn, name, StandardLoRALinear(mod, r_u, scale=s_u, dropout=dropout))
                    attached["uniform_modules"] += 1
            for name in ("gate_proj", "up_proj", "down_proj"):
                if name in targets:
                    mod = getattr(layer.mlp, name)
                    setattr(layer.mlp, name, StandardLoRALinear(mod, r_u, scale=s_u, dropout=dropout))
                    attached["uniform_modules"] += 1
    else:
        raise ValueError(f"Unknown arm kind: {arm_spec['kind']!r}")

    attached["trainable_params"] = count_trainable_params(model)
    return attached


def save_c1_adapter_state(model: nn.Module, path) -> None:
    """Save ONLY the tiny unmerged LoRA parameters (enables zero-training post-hoc pathway ablations)."""
    state = {k: v.detach().cpu() for k, v in model.state_dict().items() if ".lora_A" in k or ".lora_B" in k}
    torch.save(state, path)


def load_c1_adapter_state(model: nn.Module, path) -> None:
    state = torch.load(path, map_location="cpu", weights_only=True)
    expected = {name for name, _ in model.named_parameters() if name.endswith((".lora_A", ".lora_B"))}
    missing, extra = expected - state.keys(), state.keys() - expected
    if missing or extra:
        raise ValueError(f"Adapter keys do not match: missing={sorted(missing)}, extra={sorted(extra)}")
    model.load_state_dict(state, strict=False)


def set_pathway_ablation(
    model: nn.Module,
    disable_mlp: bool = False,
    disable_heads: list[tuple[int, int]] | None = None,
) -> None:
    """Toggle specific MLP or head adapters off/on for zero-training ablation on the exact same checkpoint."""
    dis_by_l: dict[int, set[int]] = {}
    for li, h in (disable_heads or []):
        dis_by_l.setdefault(int(li), set()).add(int(h))
    for li, layer in enumerate(model.model.layers):
        for name in ("q_proj", "o_proj"):
            m = getattr(layer.self_attn, name)
            if isinstance(m, HeadSliceLoRALinear):
                m.disabled_heads = set(dis_by_l.get(li, set()))
        for name in ("gate_proj", "up_proj", "down_proj"):
            m = getattr(layer.mlp, name)
            if isinstance(m, StandardLoRALinear):
                m.disabled = bool(disable_mlp)


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def module_delta_norms(model: nn.Module) -> dict[str, float]:
    """Report separate Frobenius norms of weight updates for attention heads vs MLPs (diagnostic)."""
    head_sq, mlp_sq = 0.0, 0.0
    for layer in model.model.layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            m = getattr(layer.self_attn, name)
            if isinstance(m, (HeadSliceLoRALinear, StandardLoRALinear)):
                head_sq += m.delta_frobenius_norm() ** 2
        for name in ("gate_proj", "up_proj", "down_proj"):
            m = getattr(layer.mlp, name)
            if isinstance(m, StandardLoRALinear):
                mlp_sq += m.delta_frobenius_norm() ** 2
    return {"attn_delta_frob": math.sqrt(head_sq), "mlp_delta_frob": math.sqrt(mlp_sq)}


@torch.no_grad()
def merge_c1_adapters(model: nn.Module) -> nn.Module:
    """Merge all C1 adapters in-place into base nn.Linear modules and unwrap wrappers."""
    for layer in model.model.layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            m = getattr(layer.self_attn, name)
            if isinstance(m, (HeadSliceLoRALinear, StandardLoRALinear)):
                setattr(layer.self_attn, name, m.merge_into_base())
        for name in ("gate_proj", "up_proj", "down_proj"):
            m = getattr(layer.mlp, name)
            if isinstance(m, StandardLoRALinear):
                setattr(layer.mlp, name, m.merge_into_base())
    return model
