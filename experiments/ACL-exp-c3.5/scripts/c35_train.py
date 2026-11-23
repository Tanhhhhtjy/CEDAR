#!/usr/bin/env python3
"""ACL-exp-c3.5 Unified Trainer: Probe-1-Guided `M3-v2` + Soft-Gated IIT + Exact Step-120 Fork + Same-Batch Grad Telemetry.

Directly resolves every mechanistic & optimization finding from ACL-exp-c3 (96f2859), Stage 1 Probe 1 (a472922),
Stage 1 Probe 2 (920493c), and Astra's C3.5 review:
  1. PROBE-1-GUIDED `M3-v2` SUPPORT (`c35_modules.py` + `c35_components_frozen.json`):
       - Supports `Uniform-LoRA`, `M3-v1`, `M3-v2` (variable per-layer MLP ranks: early L0-7 frozen, Late-Z1 L8-13 Q/O+MLP,
         ALL Zone-2 MLPs L14-27 with B1 top-5 N_CF @ r=8 & rest @ r=4, 42 B1 Z2 heads 3-tier, ZERO Zone-3 MLPs!).
  2. EXACT BIT-FOR-BIT STEP-120 SHARED FORK (`--save-resume-step 120`, `--resume-from <path>`):
       - Arm B saves an atomic `resume_step0120.pt` (adapter_state + AdamW optimizer_state + exact SHA-256).
       - Arms C (`Soft-Gated B1-IIT`) and D (`Soft-Gated rnd_head_1 IIT`) resume from the EXACT SAME step-120 state,
         optimizer buffers, cosine LR step, and sample order -> zero trajectory confound across B/C/D!
  3. 4-OPTION SOURCE MARGIN GATING (`--iit-source-margin-gate 2.0`):
       - Triggers Verification IIT ONLY when `u_src_4opt = logit(gold) - max_{c != gold} logit(c) >= m_gate`
         (so IIT never transfers noisy/wrong activations from unsolved items!).
       - Triggers Form-Pathway Debiasing IIT ONLY when `u_tgt_4opt >= m_gate` (never cements wrong target decisions!).
       - Logs exact per-subgroup trigger rates (`ce_row_A`, `ce_row_B`, `step_1`, `step_2`, `step_3`, and all 5 families).
  4. SOFT INTERPOLATION PAIRED WITH SOFT DIRECTIONAL OBJECTIVE:
       - Linearly ramps activation interpolation `alpha: 0.30 -> 0.70` across steps `(iit_start_step + 1) .. total_steps`:
           `h_tgt <- (1 - alpha) * h_tgt + alpha * h_src`
         (100% GC-safe under `use_reentrant=False` across all 3 length regimes `<, ==, >`, verified `<1e-5` in self-test!).
       - Paired with consistent soft directional target (Astra's exact consistency fix):
           `target_u = (1 - alpha) * clamp(u_tgt_orig, -m_max, m_max) + alpha * (y_src * margin_iit)`
           `l_ver = ReLU(y_src * (target_u - u_patch)) + lambda_cap * ReLU(|u_patch| - m_max)^2`
  5. SAME-BATCH SEPARATE MAIN vs IIT GRADIENT TELEMETRY & INDEPENDENT `0.30x` CAP BEFORE GLOBAL CLIPPING:
       - Backprops main task (`SFT + Corridor + Tone-Inv + Z1 Prox`), snapshots exact `g_main` & `norm_main`.
       - Zeroes grads, backprops IIT on the EXACT SAME BATCH, measures exact `norm_iit_raw` & exact same-batch cosine
         similarity `cos(g_main, g_iit)` (global AND Zone-2 only!).
       - Scales `g_iit` by `min(1.0, (iit_grad_ratio_cap * norm_main) / norm_iit_raw)` (default cap `0.30`) BEFORE summing
         `g_main + g_iit_capped` and applying global `clip_grad_norm_(max_norm=2.0)`.
       - Logs `norm_main`, `norm_iit_raw`, `norm_iit_applied`, `cos_main_iit`, `cos_main_iit_z2`, and effective post-clip
         update norms at EVERY step!

CPU Self-Test:
  python experiments/ACL-exp-c3.5/scripts/c35_train.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
C35 = HERE.parents[0]
REPO = C35.parents[1]
C1 = REPO / "experiments" / "ACL-exp-c1"
C3 = REPO / "experiments" / "ACL-exp-c3"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(C3 / "scripts"))
sys.path.insert(0, str(C1 / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-b1" / "scripts"))

import b1_lib as B1L  # noqa: E402
import c1_train as C1T  # noqa: E402
import c3_train as C3T  # noqa: E402
import c35_export_peft_lora as EXP35  # noqa: E402
import c35_modules as M35  # noqa: E402

OPTION_LETTERS = ("A", "B", "C", "D")


def get_readout_4opt_ids(tok) -> dict[str, int]:
    """Return exact token IDs for A/B/C/D after `Final:` matching `B1L.readout_token_ids` identically."""
    a_id, b_id = B1L.readout_token_ids(tok, "Final:", "A", "B")
    c_id, d_id = B1L.readout_token_ids(tok, "Final:", "C", "D")
    out = {"A": int(a_id), "B": int(b_id), "C": int(c_id), "D": int(d_id)}
    assert len(set(out.values())) == 4, f"Non-unique 4-option readout token IDs: {out}"
    return out


def compute_iit_alpha(
    cur_step: int,
    iit_start_step: int,
    total_steps: int,
    alpha_start: float,
    alpha_end: float,
) -> float:
    """Return soft interpolation weight `alpha` (0.0 during warmup <= iit_start_step; linear ramp thereafter)."""
    if cur_step <= iit_start_step:
        return 0.0
    span = max(1, total_steps - iit_start_step - 1)
    prog = min(1.0, max(0.0, (cur_step - iit_start_step - 1) / span))
    return float(alpha_start + prog * (alpha_end - alpha_start))


@torch.no_grad()
def collect_window_readouts_4opt_nograd(
    model,
    window: list[dict],
    pad_token_id: int,
    microbatch: int,
    opt_tok_ids: dict[str, int],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Pass-1 no-grad readout returning BOTH signed `u_theta` AND true 4-option gold margin `u_4opt_gold`."""
    dev = C1T._input_device(model)
    out_u: dict[str, dict[str, float]] = defaultdict(dict)
    out_m4: dict[str, dict[str, float]] = defaultdict(dict)
    opt_vals = list(opt_tok_ids.values())
    was_training = model.training
    model.eval()
    try:
        elig = [ex for ex in window if ex.get("answer_readout_pos") is not None and ex.get("pos_tok") is not None]
        for j in range(0, len(elig), microbatch):
            mb = elig[j : j + microbatch]
            bsz = len(mb)
            max_len = max(int(ex["input_ids"].numel()) for ex in mb)
            input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
            attn_mask = torch.zeros((bsz, max_len), dtype=torch.long)
            for bi, ex in enumerate(mb):
                L = int(ex["input_ids"].numel())
                input_ids[bi, :L] = ex["input_ids"]
                attn_mask[bi, :L] = 1
            logits = model(input_ids=input_ids.to(dev), attention_mask=attn_mask.to(dev), use_cache=False).logits
            for bi, ex in enumerate(mb):
                pos = int(ex["answer_readout_pos"])
                row_l = logits[bi, pos].float()
                u_val = float(row_l[ex["pos_tok"]] - row_l[ex["neg_tok"]])
                out_u[ex["pair_group_id"]][ex["arm"]] = u_val

                ms = int(ex.get("margin_sign", 0))
                if ms != 0:
                    gold_tok = int(ex["pos_tok"] if ms > 0 else ex["neg_tok"])
                    other_tids = [tid for tid in opt_vals if tid != gold_tok]
                    if len(other_tids) == 3:
                        other_max = max(float(row_l[tid]) for tid in other_tids)
                    else:
                        # Defensive fallback if a synthetic tokenizer differs
                        other_tok = int(ex["neg_tok"] if ms > 0 else ex["pos_tok"])
                        other_max = float(row_l[other_tok])
                    out_m4[ex["pair_group_id"]][ex["arm"]] = float(row_l[gold_tok]) - other_max
    finally:
        if was_training:
            model.train()
    return out_u, out_m4


@contextmanager
def differentiable_soft_interchange_forward(
    model,
    src_ex: dict,
    tgt_ex: dict,
    patch_heads: list[tuple[int, int]],
    patch_mlps: list[int],
    head_dim: int,
    alpha: float = 0.5,
    k_max: int = 8,
):
    """GC-safe differentiable soft activation interpolation `h_tgt <- (1 - alpha) * h_tgt + alpha * h_src`."""
    assert 0.0 < alpha <= 1.0, alpha
    dev = C1T._input_device(model)
    p_s = int(src_ex["answer_readout_pos"])
    p_t = int(tgt_ex["answer_readout_pos"])
    k = C3T._aligned_readout_suffix_len(src_ex, tgt_ex, k_max=k_max)
    sl_s = slice(p_s - k + 1, p_s + 1)
    sl_t = slice(p_t - k + 1, p_t + 1)

    heads_by_layer: dict[int, list[int]] = defaultdict(list)
    for l, h in patch_heads:
        heads_by_layer[int(l)].append(int(h))
    mlp_set = {int(l) for l in patch_mlps}

    cached_head_slices: dict[tuple[int, int], torch.Tensor] = {}
    cached_mlp_out: dict[int, torch.Tensor] = {}

    # Pass A: Clean differentiable source forward
    rec_hooks = []
    try:
        for l, hs in heads_by_layer.items():
            o_mod = model.model.layers[l].self_attn.o_proj

            def _make_rec_o(layer_i: int, h_list: list[int]):
                def _hook(_mod, inp):
                    x = inp[0]
                    for h in h_list:
                        cached_head_slices[(layer_i, h)] = x[:, sl_s, h * head_dim : (h + 1) * head_dim]
                return _hook

            rec_hooks.append(o_mod.register_forward_pre_hook(_make_rec_o(l, hs)))

        for l in mlp_set:
            mlp_mod = model.model.layers[l].mlp

            def _make_rec_mlp(layer_i: int):
                def _hook(_mod, _inp, out):
                    t = out[0] if isinstance(out, tuple) else out
                    cached_mlp_out[layer_i] = t[:, sl_s, :]
                return _hook

            rec_hooks.append(mlp_mod.register_forward_hook(_make_rec_mlp(l)))

        src_ids = src_ex["input_ids"][None].to(dev)
        _ = model(input_ids=src_ids, use_cache=False)
    finally:
        for hk in rec_hooks:
            hk.remove()

    one_minus_a = 1.0 - float(alpha)
    a_f = float(alpha)

    @contextmanager
    def patch_target_layer(layer_i: int):
        layer = model.model.layers[layer_i]
        heads = heads_by_layer.get(layer_i, ())
        with ExitStack() as hooks:
            if heads:
                def patch_heads_input(_module, inputs):
                    patched = inputs[0].clone()
                    for h in heads:
                        hs_sl = slice(h * head_dim, (h + 1) * head_dim)
                        src_t = cached_head_slices[layer_i, h].to(patched)
                        if a_f >= 1.0 - 1e-9:
                            patched[:, sl_t, hs_sl] = src_t
                        else:
                            patched[:, sl_t, hs_sl] = one_minus_a * patched[:, sl_t, hs_sl] + a_f * src_t
                    return (patched, *inputs[1:])

                hooks.enter_context(layer.self_attn.o_proj.register_forward_pre_hook(patch_heads_input))
            if layer_i in mlp_set:
                def patch_mlp_output(_module, _inputs, output):
                    tensor = output[0] if isinstance(output, tuple) else output
                    patched = tensor.clone()
                    src_t = cached_mlp_out[layer_i].to(patched)
                    if a_f >= 1.0 - 1e-9:
                        patched[:, sl_t, :] = src_t
                    else:
                        patched[:, sl_t, :] = one_minus_a * patched[:, sl_t, :] + a_f * src_t
                    return (patched, *output[1:]) if isinstance(output, tuple) else patched

                hooks.enter_context(layer.mlp.register_forward_hook(patch_mlp_output))
            yield

    def checkpoint_with_patch(checkpoint, layer_i: int):
        def checkpoint_target(function, *args, **kwargs):
            def patched_forward(*inputs, **forward_kwargs):
                with patch_target_layer(layer_i):
                    return function(*inputs, **forward_kwargs)
            return checkpoint(patched_forward, *args, **kwargs)
        return checkpoint_target

    # Pass B: Soft-interpolated target forward — hooks strictly isolated inside target checkpoint callable!
    with ExitStack() as target:
        for layer_i in sorted(set(heads_by_layer) | mlp_set):
            layer = model.model.layers[layer_i]
            if layer.training and getattr(layer, "gradient_checkpointing", False):
                checkpoint = layer._gradient_checkpointing_func
                target.callback(setattr, layer, "_gradient_checkpointing_func", checkpoint)
                layer._gradient_checkpointing_func = checkpoint_with_patch(checkpoint, layer_i)
            else:
                target.enter_context(patch_target_layer(layer_i))
        tgt_ids = tgt_ex["input_ids"][None].to(dev)
        logits_at = model(input_ids=tgt_ids, use_cache=False).logits[0, p_t].float()
    yield logits_at[tgt_ex["pos_tok"]] - logits_at[tgt_ex["neg_tok"]]


def soft_directional_ver_loss(
    u_patch: torch.Tensor,
    u_tgt_orig: float,
    y_src: float,
    alpha: float,
    margin_iit: float,
    margin_max: float,
    lambda_cap: float,
) -> tuple[torch.Tensor, float, bool, bool]:
    """Soft directional objective mathematically paired with soft interpolation `alpha`.

    `target_u = (1 - alpha) * clamp(u_tgt_orig, -margin_max, margin_max) + alpha * (y_src * margin_iit)`
    At `alpha=0.3`, asks `u_patch` to move 30% of the way toward `y_src * margin_iit` (never an impossible full flip!).
    At `alpha=1.0`, reduces identically to `signed_corridor_loss(u_patch, y_src, margin_iit, margin_max, lambda_cap)`.
    """
    u_tgt_clamped = max(-margin_max, min(margin_max, float(u_tgt_orig)))
    target_u = (1.0 - float(alpha)) * u_tgt_clamped + float(alpha) * (float(y_src) * float(margin_iit))
    floor_pen = F.relu(float(y_src) * (target_u - u_patch))
    cap_ex = F.relu(u_patch.abs() - float(margin_max))
    cap_pen = float(lambda_cap) * (cap_ex * cap_ex)
    return floor_pen + cap_pen, float(target_u), bool(float(floor_pen.detach()) > 0.0), bool(float(cap_ex.detach()) > 0.0)


def compute_and_backward_soft_iit_for_group(
    model,
    group_by_arm: dict[str, dict],
    group_ref_u: dict[str, float],
    group_ref_m4: dict[str, float],
    iit_spec: dict[str, Any],
    head_dim: int,
    step_idx: int,
    group_idx: int,
    n_groups_in_window: int,
    alpha: float,
    source_margin_gate: float,
    lambda_iit_form: float,
    lambda_iit_ver: float,
    margin_iit: float,
    margin_max: float,
    lambda_cap: float,
    k_max: int = 8,
) -> dict[str, float]:
    """Soft-Gated Dual-Pathway IIT backward for one base group."""
    stats = {
        "iit_form_loss": 0.0,
        "iit_ver_loss": 0.0,
        "iit_form_eligible": 0.0,
        "iit_form_triggered": 0.0,
        "iit_ver_eligible": 0.0,
        "iit_ver_triggered": 0.0,
        "iit_form_active": 0.0,
        "iit_ver_active": 0.0,
        "train_iit_ver_dir_hit": 0.0,
        "train_iit_ver_target_hit": 0.0,
        "train_iit_ver_sign_hit": 0.0,
        "train_iit_form_sign_hit": 0.0,
    }
    if alpha <= 0.0:
        return stats

    scale = 1.0 / max(1, n_groups_in_window)
    form_heads = [tuple(x) for x in iit_spec["form_only_heads"]]
    ver_heads = [tuple(x) for x in iit_spec["ver_only_heads"]]
    ver_mlps = [int(x) for x in iit_spec["ver_mlps"]]

    # 1. Form-Pathway Debiasing Interchange (same validity, opposite tone) — gated on known-correct TARGET 4-opt margin!
    if lambda_iit_form > 0.0 and form_heads:
        form_pairs = [
            ("V_auth", "V_hes", +1.0),
            ("F_auth", "F_hes", -1.0),
            ("V_hes", "V_auth", +1.0),
            ("F_hes", "F_auth", -1.0),
        ]
        src_arm, tgt_arm, y_tgt = form_pairs[(step_idx + group_idx) % 4]
        if src_arm in group_by_arm and tgt_arm in group_by_arm and tgt_arm in group_ref_u:
            stats["iit_form_eligible"] = 1.0
            tgt_m4 = float(group_ref_m4.get(tgt_arm, -1e9))
            if tgt_m4 >= source_margin_gate:
                stats["iit_form_triggered"] = 1.0
                with differentiable_soft_interchange_forward(
                    model,
                    src_ex=group_by_arm[src_arm],
                    tgt_ex=group_by_arm[tgt_arm],
                    patch_heads=form_heads,
                    patch_mlps=[],
                    head_dim=head_dim,
                    alpha=alpha,
                    k_max=k_max,
                ) as u_patch:
                    ref_clamped = max(-margin_max, min(margin_max, float(group_ref_u[tgt_arm])))
                    u_ref = torch.tensor(ref_clamped, device=u_patch.device, dtype=torch.float32)
                    l_stay = F.smooth_l1_loss(u_patch, u_ref, beta=2.0)
                    l_corr, f_act, c_act = C3T.signed_corridor_loss(u_patch, y_tgt, margin_iit, margin_max, lambda_cap)
                    l_form = 0.5 * l_stay + 0.5 * l_corr
                    if l_form.requires_grad and float(l_form.detach()) > 0.0:
                        (scale * lambda_iit_form * l_form).backward()
                    stats["iit_form_loss"] = float(l_form.detach())
                    stats["iit_form_active"] = float(f_act or c_act or float(l_stay.detach()) > 0.1)
                    stats["train_iit_form_sign_hit"] = float((y_tgt * float(u_patch.detach())) > 0.0)

    # 2. Verification-Pathway Soft Directional Interchange — gated on known-correct SOURCE 4-opt margin!
    if lambda_iit_ver > 0.0 and (ver_heads or ver_mlps):
        ver_pairs = [
            ("V", "F", +1.0),
            ("F", "V", -1.0),
            ("V_auth", "F_auth", +1.0),
            ("F_auth", "V_auth", -1.0),
            ("V_hes", "F_hes", +1.0),
            ("F_hes", "V_hes", -1.0),
        ]
        src_arm, tgt_arm, y_src = ver_pairs[(step_idx + group_idx) % 6]
        if src_arm in group_by_arm and tgt_arm in group_by_arm and tgt_arm in group_ref_u:
            stats["iit_ver_eligible"] = 1.0
            src_m4 = float(group_ref_m4.get(src_arm, -1e9))
            if src_m4 >= source_margin_gate:
                stats["iit_ver_triggered"] = 1.0
                u_tgt_orig = float(group_ref_u[tgt_arm])
                with differentiable_soft_interchange_forward(
                    model,
                    src_ex=group_by_arm[src_arm],
                    tgt_ex=group_by_arm[tgt_arm],
                    patch_heads=ver_heads,
                    patch_mlps=ver_mlps,
                    head_dim=head_dim,
                    alpha=alpha,
                    k_max=k_max,
                ) as u_patch:
                    l_ver, target_u, f_act, c_act = soft_directional_ver_loss(
                        u_patch,
                        u_tgt_orig=u_tgt_orig,
                        y_src=y_src,
                        alpha=alpha,
                        margin_iit=margin_iit,
                        margin_max=margin_max,
                        lambda_cap=lambda_cap,
                    )
                    if l_ver.requires_grad and float(l_ver.detach()) > 0.0:
                        (scale * lambda_iit_ver * l_ver).backward()
                    u_p_val = float(u_patch.detach())
                    stats["iit_ver_loss"] = float(l_ver.detach())
                    stats["iit_ver_active"] = float(f_act or c_act)
                    stats["train_iit_ver_dir_hit"] = float((y_src * (u_p_val - u_tgt_orig)) > 0.0)
                    stats["train_iit_ver_target_hit"] = float((y_src * (u_p_val - target_u)) >= 0.0)
                    stats["train_iit_ver_sign_hit"] = float((y_src * u_p_val) > 0.0)

    return stats


def combine_main_and_capped_iit_grads(
    model: torch.nn.Module,
    g_main: dict[int, torch.Tensor],
    g_iit: dict[int, torch.Tensor],
    iit_grad_ratio_cap: float,
    use_pcgrad: bool = False,
) -> dict[str, float | None]:
    """Compute exact same-batch grad norms & cosine similarity (global + Zone-2), optionally PCGrad-project out negative
    conflict (`g_iit <- g_iit - (dot / ||g_main||^2) * g_main` when `dot < 0`), cap IIT <= cap * norm_main, sum into `.grad`."""
    z2_param_ids: set[int] = set()
    for _, mod in M35.iter_c3_adapters(model):
        if mod.c3_zone == 2:
            for p in mod.parameters():
                if p.requires_grad:
                    z2_param_ids.add(id(p))

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    norm_main_sq = 0.0
    norm_iit_sq = 0.0
    dot_all = 0.0
    norm_main_z2_sq = 0.0
    norm_iit_z2_sq = 0.0
    dot_z2 = 0.0

    for p in trainable_params:
        pid = id(p)
        gm = g_main.get(pid)
        gi = g_iit.get(pid)
        in_z2 = pid in z2_param_ids
        if gm is not None:
            v = float(gm.float().pow(2).sum())
            norm_main_sq += v
            if in_z2:
                norm_main_z2_sq += v
        if gi is not None:
            v = float(gi.float().pow(2).sum())
            norm_iit_sq += v
            if in_z2:
                norm_iit_z2_sq += v
        if gm is not None and gi is not None:
            d = float((gm.float() * gi.float()).sum())
            dot_all += d
            if in_z2:
                dot_z2 += d

    norm_main = math.sqrt(max(0.0, norm_main_sq))
    norm_iit_raw = math.sqrt(max(0.0, norm_iit_sq))
    cos_all = (dot_all / (norm_main * norm_iit_raw)) if (norm_main > 1e-12 and norm_iit_raw > 1e-12) else None

    norm_main_z2 = math.sqrt(max(0.0, norm_main_z2_sq))
    norm_iit_z2 = math.sqrt(max(0.0, norm_iit_z2_sq))
    cos_z2 = (dot_z2 / (norm_main_z2 * norm_iit_z2)) if (norm_main_z2 > 1e-12 and norm_iit_z2 > 1e-12) else None

    # Optional conflict-free PCGrad orthogonal projection: whenever <g_main, g_iit> < 0, subtract the opposing component!
    pcgrad_projected = 0
    g_iit_eff: dict[int, torch.Tensor] = g_iit
    norm_iit_proj = norm_iit_raw
    cos_after_pcgrad = cos_all
    if use_pcgrad and dot_all < 0.0 and norm_main_sq > 1e-24 and norm_iit_raw > 1e-12:
        pcgrad_projected = 1
        proj_coeff = dot_all / norm_main_sq  # strictly negative
        g_iit_eff = {}
        proj_sq = 0.0
        for p in trainable_params:
            pid = id(p)
            gm = g_main.get(pid)
            gi = g_iit.get(pid)
            if gm is not None and gi is not None:
                gp = (gi.float() - proj_coeff * gm.float()).to(gi.dtype)
            elif gi is not None:
                gp = gi
            elif gm is not None:
                gp = (-proj_coeff * gm.float()).to(gm.dtype)
            else:
                continue
            g_iit_eff[pid] = gp
            proj_sq += float(gp.float().pow(2).sum())
        norm_iit_proj = math.sqrt(max(0.0, proj_sq))
        cos_after_pcgrad = 0.0 if norm_iit_proj > 1e-12 else None

    if iit_grad_ratio_cap > 0.0 and norm_iit_proj > 1e-12 and norm_main > 1e-12:
        iit_cap_scale = min(1.0, (float(iit_grad_ratio_cap) * norm_main) / norm_iit_proj)
    else:
        iit_cap_scale = 1.0
    norm_iit_applied = norm_iit_proj * iit_cap_scale

    for p in trainable_params:
        pid = id(p)
        gm = g_main.get(pid)
        gi = g_iit_eff.get(pid)
        if gm is not None and gi is not None:
            p.grad = gm + gi * iit_cap_scale
        elif gm is not None:
            p.grad = gm
        elif gi is not None:
            p.grad = gi * iit_cap_scale
        else:
            p.grad = None

    return {
        "norm_main": norm_main,
        "norm_iit_raw": norm_iit_raw,
        "norm_iit_proj": norm_iit_proj,
        "norm_iit_applied": norm_iit_applied,
        "iit_cap_scale": iit_cap_scale,
        "pcgrad_projected": pcgrad_projected,
        "cos_main_iit": cos_all,
        "cos_main_iit_z2": cos_z2,
        "cos_after_pcgrad": cos_after_pcgrad,
    }


def _optimizer_state_to_cpu(state_dict: dict[str, Any]) -> dict[str, Any]:
    out_state = {}
    for k, v in state_dict["state"].items():
        out_state[k] = {sk: (sv.detach().cpu().clone() if isinstance(sv, torch.Tensor) else sv) for sk, sv in v.items()}
    return {"state": out_state, "param_groups": state_dict["param_groups"]}


def _move_optimizer_state_to_param_devices(opt: torch.optim.Optimizer) -> None:
    for group in opt.param_groups:
        for p in group["params"]:
            st = opt.state.get(p)
            if not st:
                continue
            for sk, sv in list(st.items()):
                if sk == "step":
                    # PyTorch AdamW (capturable=False) requires `step` tensor to stay on CPU!
                    if isinstance(sv, torch.Tensor) and sv.device.type != "cpu":
                        st[sk] = sv.cpu()
                    continue
                if isinstance(sv, torch.Tensor) and sv.device != p.device:
                    st[sk] = sv.to(p.device)


def save_resume_bundle_atomic(
    path: Path,
    *,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    cur_step: int,
    adapter_sha256: str,
    arch: str,
    model_key: str,
    seed: int,
    total_steps: int,
) -> None:
    """Atomically write full resume bundle (`tmp -> os.replace`) so parallel waiting GPUs never see partial files."""
    adapter_state = {
        k: v.detach().cpu().clone()
        for k, v in model.state_dict().items()
        if (".lora_A" in k or ".lora_B" in k)
    }
    payload = {
        "schema": "cedar.acl_exp_c35.resume_bundle.v1",
        "step": int(cur_step),
        "arch": arch,
        "model_key": model_key,
        "seed": int(seed),
        "total_steps": int(total_steps),
        "adapter_sha256": adapter_sha256,
        "adapter_state": adapter_state,
        "optimizer_state": _optimizer_state_to_cpu(opt.state_dict()),
    }
    tmp_p = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp_p)
    os.replace(tmp_p, path)


def load_resume_bundle(
    path: Path,
    *,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    expected_arch: str,
    expected_seed: int,
) -> dict[str, Any]:
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    assert bundle["arch"] == expected_arch, (bundle["arch"], expected_arch)
    assert int(bundle["seed"]) == int(expected_seed), (bundle["seed"], expected_seed)
    state = bundle["adapter_state"]
    expected = {name for name, p in model.named_parameters() if p.requires_grad}
    missing, extra = expected - state.keys(), state.keys() - expected
    if missing or extra:
        raise ValueError(f"Resume adapter key mismatch: missing={len(missing)}, extra={len(extra)}")
    model.load_state_dict(state, strict=False)
    opt.load_state_dict(bundle["optimizer_state"])
    _move_optimizer_state_to_param_devices(opt)
    return {
        "step": int(bundle["step"]),
        "adapter_sha256": str(bundle["adapter_sha256"]),
    }


def _load_base_meta_map(bases_file: Path | None) -> dict[str, dict[str, Any]]:
    if not bases_file or not bases_file.is_file():
        return {}
    out = {}
    for line in bases_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[r["base_id"]] = {
            "fallacy_class": r.get("fallacy_class"),
            "pair_type": r.get("pair_type"),
            "cell_member": r.get("cell_member"),
            "broken_step": r.get("broken_step"),
        }
    return out


def _encode_c35_records(tok, rows: list[dict], base_meta: dict[str, dict[str, Any]]) -> list[dict]:
    out = []
    for r in rows:
        enc = C1T.encode_sft_example(tok, r, enable_thinking=False, mask_rule_format_ce=False)
        enc["arm"] = r["arm"]
        bm = base_meta.get(r.get("base_id", ""), {})
        enc["fallacy_class"] = r.get("fallacy_class") or bm.get("fallacy_class")
        enc["pair_type"] = r.get("pair_type") or bm.get("pair_type")
        enc["cell_member"] = bm.get("cell_member")
        enc["broken_step"] = bm.get("broken_step")
        out.append(enc)
    return out


def _subgroup_tags_for_group(group_by_arm: dict[str, dict]) -> list[str]:
    ex0 = next(iter(group_by_arm.values()))
    tags = ["all"]
    fc = ex0.get("fallacy_class")
    if fc:
        tags.append(f"family:{fc}")
    cm = ex0.get("cell_member")
    if cm in ("A", "B"):
        tags.append(f"ce_row_{cm}")
    bs = ex0.get("broken_step")
    if bs in (1, 2, 3):
        tags.append(f"step_{bs}")
    return tags


# -------------------------------------------------------------------------------------- Exhaustive CPU Self-Test


def run_selftest_tiny() -> None:
    """Verify ALL 7 critical mathematical & engineering invariants of `c35_train.py`:
      1. Soft directional loss consistency: `alpha=0.3` asks for 30% shift (never impossible full flip), `alpha=1.0` matches C3!
      2. Linear alpha ramp schedule (`<= start_step -> 0.0`, `start+1 -> alpha_start`, `total -> alpha_end`).
      3. 4-option source margin gating (`m4 >= gate` triggers; `m4 < gate` skips cleanly with 0 grad).
      4. Soft interpolation (`alpha=0.5`) GC (`use_reentrant=False`) exactness `< 1e-5` across `<, ==, >` on BOTH `M3-v2` & `Uniform-LoRA`!
      5. Same-batch main/IIT gradient cosine & independent `0.30x` cap BEFORE summing:
         - Opposite-direction huge IIT grad gets exact `cos == -1.0` and exact `norm_iit_applied == 0.30 * norm_main`!
         - Small IIT grad (`< 0.30 * norm_main`) is NEVER amplified (`iit_cap_scale == 1.0`)!
      6. Atomic Step-K resume bundle produces EXACT (`max_diff < 1e-6`) bit-for-bit identical weights to uninterrupted run!
      7. End-to-end tiny M3-v2 Soft-Gated IIT run saves lossless PEFT LoRA (`dw_err < 1e-5`).
    """
    from transformers import Qwen3Config, Qwen3ForCausalLM

    # 1. Soft directional objective math check
    u_p = torch.tensor(-2.0, dtype=torch.float32, requires_grad=True)
    loss_03, tgt_03, f_act, _ = soft_directional_ver_loss(u_p, u_tgt_orig=-5.0, y_src=+1.0, alpha=0.3, margin_iit=3.5, margin_max=10.0, lambda_cap=0.05)
    assert abs(tgt_03 - (-2.45)) < 1e-5, tgt_03
    assert not f_act and float(loss_03.detach()) == 0.0, (loss_03, f_act)  # -2.0 already beats target -2.45!
    u_p2 = torch.tensor(-3.5, dtype=torch.float32, requires_grad=True)
    loss_03b, _, f_act2, _ = soft_directional_ver_loss(u_p2, u_tgt_orig=-5.0, y_src=+1.0, alpha=0.3, margin_iit=3.5, margin_max=10.0, lambda_cap=0.05)
    assert f_act2 and abs(float(loss_03b.detach()) - 1.05) < 1e-5, loss_03b
    # At alpha=1.0, matches signed_corridor_loss identically!
    u_p3 = torch.tensor(-1.5, dtype=torch.float32)
    l_soft1, _, _, _ = soft_directional_ver_loss(u_p3, u_tgt_orig=-5.0, y_src=+1.0, alpha=1.0, margin_iit=3.5, margin_max=10.0, lambda_cap=0.05)
    l_c3, _, _ = C3T.signed_corridor_loss(u_p3, y_sign=+1.0, m_min=3.5, m_max=10.0, lambda_cap=0.05)
    assert abs(float(l_soft1) - float(l_c3)) < 1e-6
    print("[OK] 1. Soft directional objective consistency (alpha=0.3 & alpha=1.0 == C3) verified")

    # 2. Alpha ramp check
    assert compute_iit_alpha(120, 120, 300, 0.30, 0.70) == 0.0
    assert abs(compute_iit_alpha(121, 120, 300, 0.30, 0.70) - 0.30) < 1e-6
    assert abs(compute_iit_alpha(300, 120, 300, 0.30, 0.70) - 0.70) < 1e-6
    print("[OK] 2. Delayed linear alpha ramp schedule verified")

    cfg = Qwen3Config(
        vocab_size=300, hidden_size=64, intermediate_size=128,
        num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=2048, tie_word_embeddings=False,
    )
    tok = C1T._TinyTok()
    opt_ids = get_readout_4opt_ids(tok)
    train_path = C3 / "data" / "c3_train_examples_nothink.jsonl"
    bases_path = C3 / "data" / "c3_train_bases.jsonl"
    base_meta = _load_base_meta_map(bases_path)
    raw_rows = [json.loads(l) for l in train_path.read_text(encoding="utf-8").splitlines()[:20]]
    encoded = _encode_c35_records(tok, raw_rows, base_meta)
    grp0 = encoded[:10]
    grp_by_arm = {e["arm"]: e for e in grp0}

    tiny_m3v2 = {
        "kind": "m3_dac",
        "zones": {"zone1": [0, 1], "zone2": [2, 3], "zone3": [4, 5]},
        "zone1_dense_qo_layers": [1],
        "zone1_dense_qo_rank": 2,
        "mlp_ranks": {"1": 4, "2": 8, "3": 4},
        "head_ranks": {"2:0": 8, "2:2": 4, "3:1": 6, "4:0": 2, "5:3": 2},
    }
    tiny_unif = {
        "kind": "uniform",
        "uniform_rank": 2,
        "target_modules": ["q_proj", "o_proj", "gate_proj", "down_proj"],
        "zones": {"zone1": [0, 1], "zone2": [2, 3], "zone3": [4, 5]},
    }
    tiny_iit = {
        "form_only_heads": [[2, 0], [3, 1]],
        "ver_only_heads": [[2, 2], [4, 0]],
        "hub_heads": [[5, 3]],
        "ver_mlps": [2, 3],
    }

    # 3 & 4. Soft interpolation (alpha=0.5) GC equivalence across (<, ==, >) on BOTH M3-v2 & Uniform-LoRA!
    def _pad_prefix(ex: dict, n_pad: int) -> dict:
        if n_pad == 0:
            return dict(ex)
        pref = ex["input_ids"][:n_pad]
        return {**ex, "input_ids": torch.cat((pref, ex["input_ids"])), "answer_readout_pos": ex["answer_readout_pos"] + n_pad}

    ex_v, ex_f = grp_by_arm["V"], grp_by_arm["F"]
    L_min = min(int(ex_v["input_ids"].numel()), int(ex_f["input_ids"].numel()))
    eq_v = {**ex_v, "input_ids": ex_v["input_ids"][-L_min:], "answer_readout_pos": ex_v["answer_readout_pos"] - (int(ex_v["input_ids"].numel()) - L_min)}
    eq_f = {**ex_f, "input_ids": ex_f["input_ids"][-L_min:], "answer_readout_pos": ex_f["answer_readout_pos"] - (int(ex_f["input_ids"].numel()) - L_min)}
    length_cases = [
        ("src<tgt", eq_v, _pad_prefix(eq_f, 12)),
        ("src==tgt", eq_v, eq_f),
        ("src>tgt", _pad_prefix(eq_v, 14), eq_f),
    ]
    for arch_name, arch_spec in (("M3-v2", tiny_m3v2), ("Uniform-LoRA", tiny_unif)):
        torch.manual_seed(31)
        m_seed = Qwen3ForCausalLM(cfg).float()
        M35.apply_c35_adapters(m_seed, arch_spec)
        with torch.no_grad():
            for n, p in m_seed.named_parameters():
                if "lora_B" in n:
                    p.normal_(0.0, 0.05)
        sd_seed = {k: v.detach().clone() for k, v in m_seed.state_dict().items()}
        max_gc_diff = 0.0
        for cname, s_ex, t_ex in length_cases:
            g_pair = {}
            for use_gc in (False, True):
                m_t = Qwen3ForCausalLM(cfg).float()
                M35.apply_c35_adapters(m_t, arch_spec)
                m_t.load_state_dict(sd_seed)
                if use_gc:
                    m_t.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                    m_t.enable_input_require_grads()
                m_t.train()
                m_t.zero_grad()
                with differentiable_soft_interchange_forward(
                    m_t, s_ex, t_ex,
                    patch_heads=[tuple(x) for x in tiny_iit["ver_only_heads"]],
                    patch_mlps=tiny_iit["ver_mlps"],
                    head_dim=16, alpha=0.5,
                ) as u_p_t:
                    lv, _, _, _ = soft_directional_ver_loss(u_p_t, u_tgt_orig=-4.0, y_src=+1.0, alpha=0.5, margin_iit=8.0, margin_max=15.0, lambda_cap=0.05)
                    lv.backward()
                g_pair[use_gc] = {n: p.grad.detach().clone() for n, p in m_t.named_parameters() if p.requires_grad}
            cdiff = max(float((g_pair[False][n] - g_pair[True][n]).abs().max()) for n in g_pair[False])
            assert cdiff < 1e-5, (arch_name, cname, cdiff)
            max_gc_diff = max(max_gc_diff, cdiff)
        print(f"[OK] 3 & 4. {arch_name:12s} Soft-IIT (alpha=0.5) GC exactness (<, ==, >) verified (max_diff={max_gc_diff:.2e})")

    # 3b. 4-option gating check: gate=+999 skips completely (0 triggers, 0 grad); gate=-999 triggers cleanly!
    torch.manual_seed(37)
    m_g = Qwen3ForCausalLM(cfg).float()
    M35.apply_c35_adapters(m_g, tiny_m3v2)
    u_map, m4_map = collect_window_readouts_4opt_nograd(m_g, grp0, tok.pad_token_id, microbatch=2, opt_tok_ids=opt_ids)
    gid = grp0[0]["pair_group_id"]
    m_g.zero_grad()
    st_skip = compute_and_backward_soft_iit_for_group(
        m_g, grp_by_arm, u_map[gid], m4_map[gid], tiny_iit, head_dim=16,
        step_idx=0, group_idx=0, n_groups_in_window=1, alpha=0.5, source_margin_gate=999.0,
        lambda_iit_form=0.15, lambda_iit_ver=0.25, margin_iit=3.5, margin_max=10.0, lambda_cap=0.05,
    )
    assert st_skip["iit_ver_triggered"] == 0.0 and st_skip["iit_form_triggered"] == 0.0
    assert all(p.grad is None for p in m_g.parameters() if p.requires_grad)

    st_fire = compute_and_backward_soft_iit_for_group(
        m_g, grp_by_arm, u_map[gid], m4_map[gid], tiny_iit, head_dim=16,
        step_idx=0, group_idx=0, n_groups_in_window=1, alpha=0.5, source_margin_gate=-999.0,
        lambda_iit_form=0.15, lambda_iit_ver=0.25, margin_iit=3.5, margin_max=10.0, lambda_cap=0.05,
    )
    assert st_fire["iit_ver_triggered"] == 1.0 and st_fire["iit_form_triggered"] == 1.0
    print("[OK] 3b. 4-option source margin gate (skip at +999, trigger at -999) verified")

    # 5. Same-batch grad cosine & independent 0.30x cap test!
    tps = [p for p in m_g.parameters() if p.requires_grad]
    gm_syn = {id(p): torch.ones_like(p) for p in tps}
    # Case A: huge opposite-direction IIT grad (10x magnitude, cos == -1.0) -> MUST be capped to exact 0.30 * norm_main!
    gi_huge_opp = {id(p): -10.0 * torch.ones_like(p) for p in tps}
    c_a = combine_main_and_capped_iit_grads(m_g, gm_syn, gi_huge_opp, iit_grad_ratio_cap=0.30)
    assert abs(c_a["cos_main_iit"] - (-1.0)) < 1e-5, c_a
    assert abs(c_a["norm_iit_applied"] - 0.30 * c_a["norm_main"]) < 1e-4, c_a
    assert abs(c_a["iit_cap_scale"] - 0.03) < 1e-5, c_a
    # Resulting p.grad must be 1.0 + (-10.0 * 0.03) = +0.70 (main direction preserved!)
    assert abs(float(tps[0].grad.mean()) - 0.70) < 1e-5
    # Case B: tiny aligned IIT grad (0.1x magnitude, cos == +1.0) -> MUST NOT be amplified (`iit_cap_scale == 1.0`)!
    gi_tiny = {id(p): 0.1 * torch.ones_like(p) for p in tps}
    c_b = combine_main_and_capped_iit_grads(m_g, gm_syn, gi_tiny, iit_grad_ratio_cap=0.30)
    assert abs(c_b["cos_main_iit"] - 1.0) < 1e-5 and c_b["iit_cap_scale"] == 1.0, c_b
    # Case C: PCGrad on exact opposite grad -> opposing component completely zeroed, leaving exact g_main!
    c_pc1 = combine_main_and_capped_iit_grads(m_g, gm_syn, gi_huge_opp, iit_grad_ratio_cap=0.30, use_pcgrad=True)
    assert c_pc1["pcgrad_projected"] == 1 and c_pc1["norm_iit_proj"] < 1e-4, c_pc1
    assert abs(float(tps[0].grad.mean()) - 1.0) < 1e-5
    # Case D: PCGrad on diagonal conflicting grad -> preserves orthogonal component with EXACT 0 dot product with g_main!
    gm_d = {id(p): torch.zeros_like(p) for p in tps}
    gi_d = {id(p): torch.zeros_like(p) for p in tps}
    gm_d[id(tps[0])].view(-1)[0] = 1.0
    gi_d[id(tps[0])].view(-1)[0] = -3.0
    gi_d[id(tps[0])].view(-1)[1] = 4.0
    c_pc2 = combine_main_and_capped_iit_grads(m_g, gm_d, gi_d, iit_grad_ratio_cap=0.30, use_pcgrad=True)
    assert c_pc2["pcgrad_projected"] == 1 and abs(c_pc2["cos_main_iit"] - (-0.6)) < 1e-5, c_pc2
    assert abs(c_pc2["norm_iit_proj"] - 4.0) < 1e-4 and abs(c_pc2["norm_iit_applied"] - 0.30) < 1e-4, c_pc2
    assert abs(float(tps[0].grad.view(-1)[0]) - 1.0) < 1e-5 and abs(float(tps[0].grad.view(-1)[1]) - 0.30) < 1e-5
    print("[OK] 5. Same-batch grad cosine, 0.30x cap, AND conflict-free PCGrad orthogonal projection verified")

    # 6. Atomic Step-K resume bundle bit-for-bit equivalence check!
    with tempfile.TemporaryDirectory() as td:
        torch.manual_seed(99)
        m_full = Qwen3ForCausalLM(cfg).float()
        M35.apply_c35_adapters(m_full, tiny_m3v2)
        sd_init = {k: v.detach().clone() for k, v in m_full.state_dict().items()}
        opt_full = torch.optim.AdamW(M35.build_zone_optimizer_groups(m_full, lr=1e-3, lr_z1_mult=0.8), weight_decay=0.01)

        resume_p = Path(td) / "resume_step0001.pt"
        wins = [encoded[:10], encoded[10:20]]
        for s_i, w in enumerate(wins):
            opt_full.zero_grad(set_to_none=True)
            l_mb, _ = C3T.compute_loss_microbatch_c3(m_full, w, tok.pad_token_id, "corridor", 0.5, 0.25, 4.5, 10.0, 0.05)
            (l_mb / len(w)).backward()
            opt_full.step()
            if s_i == 0:
                save_resume_bundle_atomic(
                    resume_p, model=m_full, opt=opt_full, cur_step=1,
                    adapter_sha256="dummy", arch="M3-v2", model_key="qwen3_8b", seed=42, total_steps=2,
                )

        # Now resume a fresh model from step 1 and run step 2!
        m_res = Qwen3ForCausalLM(cfg).float()
        M35.apply_c35_adapters(m_res, tiny_m3v2)
        m_res.load_state_dict(sd_init)
        opt_res = torch.optim.AdamW(M35.build_zone_optimizer_groups(m_res, lr=1e-3, lr_z1_mult=0.8), weight_decay=0.01)
        load_resume_bundle(resume_p, model=m_res, opt=opt_res, expected_arch="M3-v2", expected_seed=42)
        opt_res.zero_grad(set_to_none=True)
        l_mb2, _ = C3T.compute_loss_microbatch_c3(m_res, wins[1], tok.pad_token_id, "corridor", 0.5, 0.25, 4.5, 10.0, 0.05)
        (l_mb2 / len(wins[1])).backward()
        opt_res.step()

        max_res_diff = max(
            float((p1 - p2).abs().max())
            for p1, p2 in zip(m_full.parameters(), m_res.parameters())
            if p1.requires_grad
        )
        assert max_res_diff < 1e-6, max_res_diff

        # 7. Lossless PEFT export on trained M3-v2
        ap = Path(td) / "adapter_state.pt"
        M35.save_c3_adapter_state(m_res, ap)
        meta = EXP35.export_c35_adapter_to_peft(
            state_dict=torch.load(ap, map_location="cpu", weights_only=True),
            arch_spec=tiny_m3v2,
            model_spec={"hidden_size": 64, "num_heads": 4, "head_dim": 16, "intermediate_size": 128, "num_layers": 6},
            scaling={"scaling_mode": "rslora", "base_scale": 2.0, "anchor_rank": 8},
            out_dir=Path(td) / "peft",
            export_dtype=torch.float32,
        )
        assert meta["max_fp32_dw_abs_err"] < 1e-5
    print(f"[OK] 6 & 7. Atomic resume exactness (max_diff={max_res_diff:.2e}) & M3-v2 PEFT export verified")
    print("[OK] ALL c35_train self-tests PASSED!")


# -------------------------------------------------------------------------------------- Main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-tiny", action="store_true")
    ap.add_argument("--model", type=str)
    ap.add_argument("--model-key", choices=("qwen3_8b", "qwen3_14b"), default="qwen3_8b")
    ap.add_argument("--arch", type=str, default="M3-v2", help="Architecture key in c35_components_frozen.json")
    ap.add_argument("--loss-mode", choices=("base", "corridor", "corridor_iit"), default="corridor_iit")
    ap.add_argument("--iit-target", choices=("none", "b1", "rnd_head_1", "rnd_head_2", "rnd_full_1", "rnd_full_2"), default="none")
    ap.add_argument("--components-file", type=Path, default=C35 / "data" / "c35_components_frozen.json")
    ap.add_argument("--train-file", type=Path, default=C3 / "data" / "c3_train_examples_nothink.jsonl")
    ap.add_argument("--train-bases-file", type=Path, default=C3 / "data" / "c3_train_bases.jsonl")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2.5e-5)
    ap.add_argument("--lr-z1-mult", type=float, default=None, help="Defaults to 0.80 for m3_dac, 1.0 for uniform")
    ap.add_argument("--lambda-prox-z1", type=float, default=None, help="Defaults to 1e-4 for m3_dac, 0.0 for uniform")
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=2.0)
    ap.add_argument("--grad-accum", type=int, default=20)
    ap.add_argument("--microbatch", type=int, default=2)
    ap.add_argument("--aux-weight", type=float, default=0.5)
    ap.add_argument("--lambda-margin", type=float, default=0.25)
    ap.add_argument("--margin-min", type=float, default=4.5)
    ap.add_argument("--margin-max", type=float, default=10.0)
    ap.add_argument("--lambda-cap", type=float, default=0.05)
    ap.add_argument("--lambda-inv", type=float, default=0.15)
    # Soft-Gated IIT hyperparameters
    ap.add_argument("--iit-start-step", type=int, default=120, help="Enable IIT strictly AFTER this step (steps 1..start are pure corridor)")
    ap.add_argument("--iit-alpha-start", type=float, default=0.30)
    ap.add_argument("--iit-alpha-end", type=float, default=0.70)
    ap.add_argument("--iit-source-margin-gate", type=float, default=2.0, help="Min 4-option gold margin to trigger IIT")
    ap.add_argument("--iit-grad-ratio-cap", type=float, default=0.30, help="Max ||g_iit|| / ||g_main|| BEFORE global clipping")
    ap.add_argument("--iit-pcgrad", action="store_true", help="Orthogonally project out negative-cosine IIT gradient conflict against g_main")
    ap.add_argument("--lambda-iit-form", type=float, default=0.15)
    ap.add_argument("--lambda-iit-ver", type=float, default=0.25)
    ap.add_argument("--margin-iit", type=float, default=3.5)
    ap.add_argument("--iit-k-max", type=int, default=8)
    # Exact shared step-120 fork controls
    ap.add_argument("--save-resume-step", type=int, default=0, help="If >0, atomically write resume_stepXXXX.pt at this step")
    ap.add_argument("--resume-from", type=Path, default=None, help="Load resume_stepXXXX.pt and continue from step+1")
    ap.add_argument("--scaling-mode", choices=("rslora", "constant"), default="rslora")
    ap.add_argument("--lora-scale", type=float, default=2.0)
    ap.add_argument("--anchor-rank", type=int, default=8)
    ap.add_argument("--save-steps", type=str, default="100,120,200,300")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.selftest_tiny:
        run_selftest_tiny()
        return

    assert args.model and args.out_dir, "--model and --out-dir are required"
    if args.loss_mode == "corridor_iit":
        assert args.iit_target != "none", "--loss-mode corridor_iit requires --iit-target != none"
    else:
        assert args.iit_target == "none", f"--loss-mode {args.loss_mode} expects --iit-target none, got {args.iit_target}"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    comp = json.loads(args.components_file.read_text(encoding="utf-8"))
    model_info = comp["models"][args.model_key]
    arch_spec = model_info["architectures"][args.arch]
    head_dim = int(model_info["model_spec"]["head_dim"])
    iit_spec = model_info["iit_targets"][args.iit_target] if args.iit_target != "none" else None

    lr_z1_mult = args.lr_z1_mult if args.lr_z1_mult is not None else (0.80 if arch_spec["kind"] == "m3_dac" else 1.0)
    lambda_prox_z1 = args.lambda_prox_z1 if args.lambda_prox_z1 is not None else (1e-4 if arch_spec["kind"] == "m3_dac" else 0.0)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    opt_tok_ids = get_readout_4opt_ids(tok)

    base_meta = _load_base_meta_map(args.train_bases_file)
    raw_rows = [json.loads(l) for l in args.train_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        raw_rows = raw_rows[: args.limit]
    encoded = _encode_c35_records(tok, raw_rows, base_meta)

    print(
        f"[c35_train] model={args.model} ({args.model_key}) arch={args.arch} loss={args.loss_mode} "
        f"iit={args.iit_target} start={args.iit_start_step} alpha={args.iit_alpha_start}->{args.iit_alpha_end} "
        f"m_gate={args.iit_source_margin_gate} cap={args.iit_grad_ratio_cap} "
        f"lr={args.lr} lr_z1_mult={lr_z1_mult} prox_z1={lambda_prox_z1} seed={args.seed}"
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    adapter_info = M35.apply_c35_adapters(
        model, arch_spec,
        scaling_mode=args.scaling_mode, base_scale=args.lora_scale, anchor_rank=args.anchor_rank, dropout=0.0,
    )
    param_groups = M35.build_zone_optimizer_groups(model, lr=args.lr, lr_z1_mult=lr_z1_mult, weight_decay=args.weight_decay)
    opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    save_steps = sorted({int(x.strip()) for x in args.save_steps.split(",") if x.strip()})
    all_windows: list[list[dict]] = []
    for ep in range(args.epochs):
        all_windows.extend(C1T.build_atomic_windows(encoded, args.grad_accum, seed=args.seed + ep * 1009))
    total_steps = min(len(all_windows), args.max_steps) if args.max_steps > 0 else len(all_windows)
    warmup_steps = max(1, int(0.10 * total_steps))

    start_step_idx = 0
    resumed_info: dict[str, Any] | None = None
    if args.resume_from is not None:
        resumed_info = load_resume_bundle(
            args.resume_from, model=model, opt=opt, expected_arch=args.arch, expected_seed=args.seed
        )
        start_step_idx = resumed_info["step"]
        print(
            f"[c35_train] RESUMED from {args.resume_from} at step={start_step_idx} "
            f"(parent_sha256={resumed_info['adapter_sha256'][:12]}) -> running steps {start_step_idx + 1}..{total_steps}"
        )
        # Save exact copy of parent checkpoint at step `start_step_idx` for dev selection & auditability!
        parent_ckpt_p = args.out_dir / f"adapter_state_step{start_step_idx:04d}.pt"
        M35.save_c3_adapter_state(model, parent_ckpt_p)

    print(
        f"[c35_train] encoded={len(encoded)}, total_steps={total_steps}, start_idx={start_step_idx}, "
        f"trainable={adapter_info['trainable_params']:,} zones={adapter_info['zone_trainable_params']}"
    )

    model.train()
    logs: list[dict[str, Any]] = []
    saved_checkpoints: dict[str, str] = {}
    zone_norm_trajectory: dict[str, dict[str, float]] = {}
    if resumed_info is not None:
        saved_checkpoints[str(start_step_idx)] = C3T._file_sha256(args.out_dir / f"adapter_state_step{start_step_idx:04d}.pt")
        zone_norm_trajectory[str(start_step_idx)] = M35.zone_delta_norms(model)

    n_clipped = 0
    n_steps_run = 0
    # Subgroup IIT trigger accumulators across the active IIT window
    ver_elig_by_tag: dict[str, int] = defaultdict(int)
    ver_trig_by_tag: dict[str, int] = defaultdict(int)
    form_elig_by_tag: dict[str, int] = defaultdict(int)
    form_trig_by_tag: dict[str, int] = defaultdict(int)

    trainable_params = [p for p in model.parameters() if p.requires_grad]

    for step_idx in range(start_step_idx, total_steps):
        win = all_windows[step_idx]
        cur_step = step_idx + 1
        n_steps_run += 1
        lrf = C1T.lr_factor(step_idx, total_steps, warmup_steps)
        for pg in opt.param_groups:
            base_lr = args.lr * lr_z1_mult if pg.get("name") == "zone1" else args.lr
            pg["lr"] = base_lr * lrf

        opt.zero_grad(set_to_none=True)

        groups_in_win: dict[str, dict[str, dict]] = defaultdict(dict)
        for ex in win:
            groups_in_win[ex["pair_group_id"]][ex["arm"]] = ex
        n_groups = max(1, len(groups_in_win))

        paired_chain_grads: dict[tuple[str, str], float] = {}
        u_map: dict[str, dict[str, float]] = {}
        m4_map: dict[str, dict[str, float]] = {}
        inv_loss_sum = 0.0
        inv_act_sum = 0.0

        # Pass 1: Detached readouts (`u_theta` + true 4-option gold margin `u_4opt_gold`)
        if args.loss_mode in ("corridor", "corridor_iit"):
            u_map, m4_map = collect_window_readouts_4opt_nograd(
                model, win, pad_id, microbatch=args.microbatch, opt_tok_ids=opt_tok_ids
            )
            for gid, gu in u_map.items():
                arm_g, lv_inv, inv_a = C3T.compute_tone_invariance_chain_grads(gu, lambda_inv=args.lambda_inv)
                inv_loss_sum += lv_inv
                inv_act_sum += inv_a
                for a, gv in arm_g.items():
                    paired_chain_grads[(gid, a)] = gv

        # Pass 2: Main Task (`SFT + Corridor + Tone-Invariance + Zone-1 Prox`)
        win_sft = 0.0
        win_marg = 0.0
        win_signed = 0
        win_floor_act = 0
        win_cap_act = 0
        win_n = len(win)
        for j in range(0, win_n, args.microbatch):
            mb = win[j : j + args.microbatch]
            loss_sum, st_mb = C3T.compute_loss_microbatch_c3(
                model, mb, pad_id,
                loss_mode=args.loss_mode,
                aux_weight=args.aux_weight,
                lambda_margin=args.lambda_margin,
                margin_min=args.margin_min if args.loss_mode != "base" else 4.0,
                margin_max=args.margin_max,
                lambda_cap=args.lambda_cap,
                paired_chain_grads=paired_chain_grads if args.loss_mode in ("corridor", "corridor_iit") else None,
                n_groups_in_window=n_groups,
                win_n=win_n,
            )
            (loss_sum / win_n).backward()
            win_sft += st_mb["sft_sum"]
            win_marg += st_mb["margin_sum"]
            win_signed += st_mb["n_signed"]
            win_floor_act += st_mb["n_floor_act"]
            win_cap_act += st_mb["n_cap_act"]

        prox_val = 0.0
        if lambda_prox_z1 > 0.0:
            p_loss = lambda_prox_z1 * M35.zone1_proximal_penalty(model)
            if p_loss.requires_grad:
                p_loss.backward()
            prox_val = float(p_loss.detach())

        # Pass 3: Soft-Gated IIT (strictly AFTER `iit_start_step`), with EXACT separate grad norm & cosine logging!
        iit_alpha = (
            compute_iit_alpha(cur_step, args.iit_start_step, total_steps, args.iit_alpha_start, args.iit_alpha_end)
            if (args.loss_mode == "corridor_iit" and iit_spec is not None)
            else 0.0
        )
        iit_f_sum = 0.0
        iit_v_sum = 0.0
        iit_f_elig = 0.0
        iit_f_trig = 0.0
        iit_v_elig = 0.0
        iit_v_trig = 0.0
        iit_v_dir_hit = 0.0
        iit_v_tgt_hit = 0.0
        iit_v_sign_hit = 0.0
        iit_f_sign_hit = 0.0

        norm_main_sq = sum(float(p.grad.detach().float().pow(2).sum()) for p in trainable_params if p.grad is not None)
        grad_info: dict[str, float | None] = {
            "norm_main": math.sqrt(max(0.0, norm_main_sq)),
            "norm_iit_raw": 0.0,
            "norm_iit_proj": 0.0,
            "norm_iit_applied": 0.0,
            "iit_cap_scale": 1.0,
            "pcgrad_projected": 0,
            "cos_main_iit": None,
            "cos_main_iit_z2": None,
            "cos_after_pcgrad": None,
        }

        if iit_alpha > 0.0:
            # Snapshot exact main-task gradient before zeroing for same-batch IIT measurement!
            g_main = {id(p): p.grad.detach().clone() for p in trainable_params if p.grad is not None}
            opt.zero_grad(set_to_none=True)

            for g_i, gid in enumerate(sorted(groups_in_win.keys())):
                grp_arms = groups_in_win[gid]
                st_iit = compute_and_backward_soft_iit_for_group(
                    model,
                    group_by_arm=grp_arms,
                    group_ref_u=u_map.get(gid, {}),
                    group_ref_m4=m4_map.get(gid, {}),
                    iit_spec=iit_spec,
                    head_dim=head_dim,
                    step_idx=step_idx,
                    group_idx=g_i,
                    n_groups_in_window=n_groups,
                    alpha=iit_alpha,
                    source_margin_gate=args.iit_source_margin_gate,
                    lambda_iit_form=args.lambda_iit_form,
                    lambda_iit_ver=args.lambda_iit_ver,
                    margin_iit=args.margin_iit,
                    margin_max=args.margin_max,
                    lambda_cap=args.lambda_cap,
                    k_max=args.iit_k_max,
                )
                iit_f_sum += st_iit["iit_form_loss"]
                iit_v_sum += st_iit["iit_ver_loss"]
                iit_f_elig += st_iit["iit_form_eligible"]
                iit_f_trig += st_iit["iit_form_triggered"]
                iit_v_elig += st_iit["iit_ver_eligible"]
                iit_v_trig += st_iit["iit_ver_triggered"]
                iit_v_dir_hit += st_iit["train_iit_ver_dir_hit"]
                iit_v_tgt_hit += st_iit["train_iit_ver_target_hit"]
                iit_v_sign_hit += st_iit["train_iit_ver_sign_hit"]
                iit_f_sign_hit += st_iit["train_iit_form_sign_hit"]

                tags = _subgroup_tags_for_group(grp_arms)
                for tg in tags:
                    ver_elig_by_tag[tg] += int(st_iit["iit_ver_eligible"] > 0.5)
                    ver_trig_by_tag[tg] += int(st_iit["iit_ver_triggered"] > 0.5)
                    form_elig_by_tag[tg] += int(st_iit["iit_form_eligible"] > 0.5)
                    form_trig_by_tag[tg] += int(st_iit["iit_form_triggered"] > 0.5)

            g_iit = {id(p): p.grad.detach().clone() for p in trainable_params if p.grad is not None}
            grad_info = combine_main_and_capped_iit_grads(
                model, g_main=g_main, g_iit=g_iit,
                iit_grad_ratio_cap=args.iit_grad_ratio_cap,
                use_pcgrad=args.iit_pcgrad,
            )
            del g_main, g_iit

        # Pre-clip per-zone grad norms & global clipping
        z_grads = M35.zone_grad_norms(model)
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.max_grad_norm))
        was_clipped = int(grad_norm > args.max_grad_norm)
        n_clipped += was_clipped
        global_clip_scale = min(1.0, args.max_grad_norm / max(grad_norm, 1e-12))
        opt.step()

        row_log = {
            "step": cur_step,
            "lr_z23": round(opt.param_groups[-1]["lr"], 8),
            "sft_mean": round(win_sft / win_n, 4),
            "margin_mean": round(win_marg / win_n, 4),
            "floor_act_rate": round(win_floor_act / max(1, win_signed), 3),
            "cap_act_rate": round(win_cap_act / max(1, win_signed), 3),
            "inv_mean": round(inv_loss_sum / n_groups, 4),
            "iit_alpha": round(iit_alpha, 4),
            "iit_form_mean": round(iit_f_sum / max(1.0, iit_f_trig), 4) if iit_f_trig > 0 else 0.0,
            "iit_ver_mean": round(iit_v_sum / max(1.0, iit_v_trig), 4) if iit_v_trig > 0 else 0.0,
            "iit_ver_trig_rate": round(iit_v_trig / max(1.0, iit_v_elig), 3) if iit_v_elig > 0 else 0.0,
            "iit_form_trig_rate": round(iit_f_trig / max(1.0, iit_f_elig), 3) if iit_f_elig > 0 else 0.0,
            "iit_ver_dir_hit": round(iit_v_dir_hit / max(1.0, iit_v_trig), 3) if iit_v_trig > 0 else None,
            "iit_ver_target_hit": round(iit_v_tgt_hit / max(1.0, iit_v_trig), 3) if iit_v_trig > 0 else None,
            "iit_ver_sign_hit": round(iit_v_sign_hit / max(1.0, iit_v_trig), 3) if iit_v_trig > 0 else None,
            "iit_form_sign_hit": round(iit_f_sign_hit / max(1.0, iit_f_trig), 3) if iit_f_trig > 0 else None,
            "norm_main": round(float(grad_info["norm_main"] or 0.0), 4),
            "norm_iit_raw": round(float(grad_info["norm_iit_raw"] or 0.0), 4),
            "norm_iit_proj": round(float(grad_info.get("norm_iit_proj") or 0.0), 4),
            "norm_iit_applied": round(float(grad_info["norm_iit_applied"] or 0.0), 4),
            "iit_cap_scale": round(float(grad_info["iit_cap_scale"] or 1.0), 4),
            "pcgrad_projected": int(grad_info.get("pcgrad_projected") or 0),
            "cos_main_iit": round(float(grad_info["cos_main_iit"]), 4) if grad_info["cos_main_iit"] is not None else None,
            "cos_main_iit_z2": round(float(grad_info["cos_main_iit_z2"]), 4) if grad_info["cos_main_iit_z2"] is not None else None,
            "cos_after_pcgrad": round(float(grad_info["cos_after_pcgrad"]), 4) if grad_info.get("cos_after_pcgrad") is not None else None,
            "eff_main_upd": round(float(grad_info["norm_main"] or 0.0) * global_clip_scale, 4),
            "eff_iit_upd": round(float(grad_info["norm_iit_applied"] or 0.0) * global_clip_scale, 4),
            "prox_z1": round(prox_val, 5),
            "g_z1": round(z_grads["g_z1"], 4),
            "g_z2": round(z_grads["g_z2"], 4),
            "g_z3": round(z_grads["g_z3"], 4),
            "grad_norm": round(grad_norm, 4),
            "clipped": was_clipped,
        }
        logs.append(row_log)
        if cur_step == start_step_idx + 1 or cur_step % 25 == 0 or cur_step == total_steps or cur_step == args.iit_start_step + 1:
            cos_s = f"{row_log['cos_main_iit']:+.3f}" if row_log["cos_main_iit"] is not None else "  n/a"
            print(
                f"[step {cur_step:3d}/{total_steps}] sft={row_log['sft_mean']:.3f} corr={row_log['margin_mean']:.3f} "
                f"a={row_log['iit_alpha']:.2f} trig={row_log['iit_ver_trig_rate']:.2f} "
                f"g_m={row_log['norm_main']:.2f} g_iit={row_log['norm_iit_raw']:.2f}->{row_log['norm_iit_applied']:.2f} "
                f"cos={cos_s} tot={row_log['grad_norm']:.2f}"
            )

        if cur_step in save_steps and cur_step < total_steps:
            ckpt_p = args.out_dir / f"adapter_state_step{cur_step:04d}.pt"
            M35.save_c3_adapter_state(model, ckpt_p)
            ckpt_sha = C3T._file_sha256(ckpt_p)
            saved_checkpoints[str(cur_step)] = ckpt_sha
            zone_norm_trajectory[str(cur_step)] = M35.zone_delta_norms(model)
            if args.save_resume_step > 0 and cur_step == args.save_resume_step:
                res_p = args.out_dir / f"resume_step{cur_step:04d}.pt"
                save_resume_bundle_atomic(
                    res_p, model=model, opt=opt, cur_step=cur_step,
                    adapter_sha256=ckpt_sha, arch=args.arch, model_key=args.model_key,
                    seed=args.seed, total_steps=total_steps,
                )
                print(f"[c35_train] Wrote atomic shared resume bundle -> {res_p} (sha256={ckpt_sha[:12]})")
        elif args.save_resume_step > 0 and cur_step == args.save_resume_step:
            ckpt_p = args.out_dir / f"adapter_state_step{cur_step:04d}.pt"
            M35.save_c3_adapter_state(model, ckpt_p)
            ckpt_sha = C3T._file_sha256(ckpt_p)
            saved_checkpoints[str(cur_step)] = ckpt_sha
            zone_norm_trajectory[str(cur_step)] = M35.zone_delta_norms(model)
            res_p = args.out_dir / f"resume_step{cur_step:04d}.pt"
            save_resume_bundle_atomic(
                res_p, model=model, opt=opt, cur_step=cur_step,
                adapter_sha256=ckpt_sha, arch=args.arch, model_key=args.model_key,
                seed=args.seed, total_steps=total_steps,
            )
            print(f"[c35_train] Wrote atomic shared resume bundle -> {res_p} (sha256={ckpt_sha[:12]})")

    final_ap = args.out_dir / "adapter_state.pt"
    M35.save_c3_adapter_state(model, final_ap)
    final_sha = C3T._file_sha256(final_ap)
    saved_checkpoints[str(total_steps)] = final_sha
    final_norms = M35.zone_delta_norms(model)
    zone_norm_trajectory[str(total_steps)] = final_norms

    # Summarize same-batch gradient telemetry & subgroup IIT trigger rates!
    iit_rows = [r for r in logs if r["iit_alpha"] > 0.0]
    cos_vals = [r["cos_main_iit"] for r in iit_rows if r["cos_main_iit"] is not None]
    cos_z2_vals = [r["cos_main_iit_z2"] for r in iit_rows if r["cos_main_iit_z2"] is not None]
    iit_summary = {
        "n_iit_steps": len(iit_rows),
        "mean_norm_main_during_iit": round(sum(r["norm_main"] for r in iit_rows) / len(iit_rows), 4) if iit_rows else None,
        "mean_norm_iit_raw": round(sum(r["norm_iit_raw"] for r in iit_rows) / len(iit_rows), 4) if iit_rows else None,
        "mean_norm_iit_applied": round(sum(r["norm_iit_applied"] for r in iit_rows) / len(iit_rows), 4) if iit_rows else None,
        "mean_eff_main_upd": round(sum(r["eff_main_upd"] for r in iit_rows) / len(iit_rows), 4) if iit_rows else None,
        "mean_eff_iit_upd": round(sum(r["eff_iit_upd"] for r in iit_rows) / len(iit_rows), 4) if iit_rows else None,
        "frac_steps_iit_capped": round(sum(1 for r in iit_rows if r["iit_cap_scale"] < 0.9999) / len(iit_rows), 4) if iit_rows else None,
        "frac_steps_pcgrad_projected": round(sum(r.get("pcgrad_projected", 0) for r in iit_rows) / len(iit_rows), 4) if iit_rows else None,
        "mean_cos_main_iit": round(sum(cos_vals) / len(cos_vals), 4) if cos_vals else None,
        "median_cos_main_iit": round(sorted(cos_vals)[len(cos_vals) // 2], 4) if cos_vals else None,
        "frac_cos_negative": round(sum(1 for x in cos_vals if x < 0.0) / len(cos_vals), 4) if cos_vals else None,
        "mean_cos_main_iit_z2": round(sum(cos_z2_vals) / len(cos_z2_vals), 4) if cos_z2_vals else None,
        "ver_trigger_by_subgroup": {
            tg: {"triggered": ver_trig_by_tag[tg], "eligible": ver_elig_by_tag[tg], "rate": round(ver_trig_by_tag[tg] / max(1, ver_elig_by_tag[tg]), 4)}
            for tg in sorted(ver_elig_by_tag)
        },
        "form_trigger_by_subgroup": {
            tg: {"triggered": form_trig_by_tag[tg], "eligible": form_elig_by_tag[tg], "rate": round(form_trig_by_tag[tg] / max(1, form_elig_by_tag[tg]), 4)}
            for tg in sorted(form_elig_by_tag)
        },
    }

    scaling_dict = {"scaling_mode": args.scaling_mode, "base_scale": args.lora_scale, "anchor_rank": args.anchor_rank}
    adapter_cfg = {
        "schema": "cedar.acl_exp_c35.adapter_config.v1",
        "base_model": args.model,
        "model_key": args.model_key,
        "arch": args.arch,
        "loss_mode": args.loss_mode,
        "iit_target": args.iit_target,
        "arch_spec": arch_spec,
        "iit_spec": iit_spec,
        "scaling": scaling_dict,
        "trainable_params": adapter_info["trainable_params"],
        "adapter_state_sha256": final_sha,
    }
    (args.out_dir / "adapter_config.json").write_text(json.dumps(adapter_cfg, indent=2) + "\n", encoding="utf-8")

    meta = {
        "schema": "cedar.acl_exp_c35.train_meta.v1",
        "model": args.model,
        "model_key": args.model_key,
        "arch": args.arch,
        "loss_mode": args.loss_mode,
        "iit_target": args.iit_target,
        "seed": args.seed,
        "resumed_from": str(args.resume_from) if args.resume_from else None,
        "resumed_info": resumed_info,
        "train_file": str(args.train_file),
        "train_file_sha256": C3T._file_sha256(args.train_file),
        "n_examples": len(encoded),
        "total_steps": total_steps,
        "steps_executed": n_steps_run,
        "clip_rate": round(n_clipped / max(1, n_steps_run), 4),
        "lora_scaling": scaling_dict,
        "hyperparams": {
            "lr": args.lr,
            "lr_z1_mult": lr_z1_mult,
            "lambda_prox_z1": lambda_prox_z1,
            "weight_decay": args.weight_decay,
            "max_grad_norm": args.max_grad_norm,
            "grad_accum": args.grad_accum,
            "microbatch": args.microbatch,
            "aux_weight": args.aux_weight,
            "lambda_margin": args.lambda_margin,
            "margin_min": args.margin_min,
            "margin_max": args.margin_max,
            "lambda_cap": args.lambda_cap,
            "lambda_inv": args.lambda_inv,
            "iit_start_step": args.iit_start_step,
            "iit_alpha_start": args.iit_alpha_start,
            "iit_alpha_end": args.iit_alpha_end,
            "iit_source_margin_gate": args.iit_source_margin_gate,
            "iit_grad_ratio_cap": args.iit_grad_ratio_cap,
            "iit_pcgrad": bool(args.iit_pcgrad),
            "lambda_iit_form": args.lambda_iit_form,
            "lambda_iit_ver": args.lambda_iit_ver,
            "margin_iit": args.margin_iit,
            "iit_k_max": args.iit_k_max,
            "seed": args.seed,
        },
        "iit_summary": iit_summary,
        "adapter_info": adapter_info,
        "final_zone_delta_norms": final_norms,
        "zone_delta_norm_trajectory": zone_norm_trajectory,
        "adapter_state_sha256": final_sha,
        "saved_checkpoints_sha256": saved_checkpoints,
        "elapsed_seconds": round(time.time() - t0, 1),
        "tail_logs": logs[-20:],
        "all_logs": logs,
    }
    (args.out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(
        f"[c35_train] DONE in {meta['elapsed_seconds']}s | clip_rate={meta['clip_rate']:.2%} "
        f"| sha256={final_sha[:12]} | iit_summary={json.dumps(iit_summary)}"
    )


if __name__ == "__main__":
    main()
