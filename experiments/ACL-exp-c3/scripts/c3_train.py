#!/usr/bin/env python3
"""ACL-exp-c3 Unified Decoupled Trainer: M3-DAC vs Uniform-LoRA x {Base, Corridor, Corridor+B1-IIT, Corridor+Rnd-IIT}.

Key upgrades over ACL-exp-c2 (directly addressing C2 forensic audit & Astra review):
  1. DECOUPLED ARCHITECTURE vs INTERVENTION TARGET:
       - `--arch`: `Uniform-LoRA`, `M3-DAC` (plus `M3-Equal-Rank`, `M3-Shuffled-Rank`, `M3-Random-Mask`)
       - `--loss-mode`: `base`, `corridor`, `corridor_iit`
       - `--iit-target`: `none`, `b1`, `rnd_head_1`, `rnd_head_2`, `rnd_full_1`, `rnd_full_2`
       Enables ALL 8 arms of the 2x4 showdown (`U1..U4`, `M1..M4`) through ONE unified codepath!
  2. SYMMETRIC CORE WEIGHTING (replaces C2's biased `accept_weight=2.0`):
       - Core 6 validity/tone arms (`V, F, V_auth, V_hes, F_auth, F_hes`) receive symmetric weight `1.0` (exact 3:3 V/F!).
       - Auxiliary control arms (`shell_only, AO, S, P0/I0/N/F_src`) receive weight `--aux-weight` (default `0.5`).
  3. SIGNED CONFIDENCE CORRIDOR (`[m_min, m_max]`, default `[4.5, 10.0]`):
       - Triggered on `y * u`: `ReLU(m_min - y * u) + lambda_cap * ReLU(|u| - m_max)^2`.
       - Strongly corrects high-confidence WRONG items (`y*u < 0`), gives ZERO gradient inside `[m_min, m_max]`,
         and smoothly caps extreme logit saturation (`|u| > m_max`), eliminating C2's `-35` logit blow-up!
       - Combined with exact 2-pass cross-tone invariance (`V_auth <-> V_hes`, `F_auth <-> F_hes`, `shell_only <= AO+1`).
  4. ZONE-1 MILD BRIDGE REGULARIZATION (for `M3-DAC`, zero hardcoded freeze schedule):
       - Zero-init LoRA (`lora_B = 0`), lower LR multiplier (`--lr-z1-mult`, default `0.35` on M3-DAC, `1.0` on Uniform),
         and exact differentiable proximal penalty `--lambda-prox-z1 * sum_{m in Z1} ||Delta W_m||_F^2`!
  5. CORRIDOR-GATED GC-SAFE DUAL-PATHWAY IIT:
       - Exact target-only checkpoint replay hook wrapper (`use_reentrant=False` safe across ALL 3 length cases:
         `len(src) < len(tgt)`, `==`, `>` — all 3 asserted to `<1e-5` in `--selftest-tiny`!).
       - Clamps reference `u_ref` to `[-m_max, +m_max]` and caps patched overshoot so IIT stops hammering saturated examples.
  6. RICH PER-ZONE TELEMETRY:
       - Logs pre-clip grad norms (`g_z1, g_z2, g_z3, g_total`), clipping indicator, per-loss active rates,
         training-window post-interchange sign hit rates (`train_iit_ver_sign_hit`, `train_iit_form_sign_hit`),
         and per-zone actual Delta-W norms (`z1_frob, z2_frob, z3_frob`) at every save step (`50,100,200,300`).

CPU Self-Test:
  python experiments/ACL-exp-c3/scripts/c3_train.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
C3 = HERE.parents[0]
REPO = C3.parents[1]
C1 = REPO / "experiments" / "ACL-exp-c1"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(C1 / "scripts"))

import c1_train as C1T  # noqa: E402
import c3_export_peft_lora as EXP3  # noqa: E402
import c3_modules as M3  # noqa: E402

CORE_ARMS = frozenset({"V", "F", "V_auth", "V_hes", "F_auth", "F_hes"})


# -------------------------------------------------------------------------------------- Signed Corridor & 2-Pass Tone Invariance


def signed_corridor_loss(
    u: torch.Tensor,
    y_sign: float,
    m_min: float,
    m_max: float,
    lambda_cap: float,
) -> tuple[torch.Tensor, bool, bool]:
    """Signed confidence corridor loss on scalar readout `u` with target sign `y_sign in {+1, -1}`."""
    floor_pen = F.relu(m_min - y_sign * u)
    cap_ex = F.relu(u.abs() - m_max)
    cap_pen = lambda_cap * (cap_ex * cap_ex)
    return floor_pen + cap_pen, bool(float(floor_pen.detach()) > 0.0), bool(float(cap_ex.detach()) > 0.0)


def compute_tone_invariance_chain_grads(
    group_u: dict[str, float],
    lambda_inv: float,
    shell_margin: float = 1.0,
) -> tuple[dict[str, float], float, float]:
    """Pass-1 detached scalar chain rule for same-validity cross-tone invariance & shell_only pullback."""
    needed = ("V_auth", "V_hes", "F_auth", "F_hes", "shell_only", "AO")
    if lambda_inv <= 0.0 or not all(k in group_u for k in needed):
        return {}, 0.0, 0.0

    u = {k: torch.tensor(float(group_u[k]), dtype=torch.float32, requires_grad=True) for k in needed}
    inv_v = F.smooth_l1_loss(u["V_auth"], u["V_hes"], beta=2.0)
    inv_f = F.smooth_l1_loss(u["F_auth"], u["F_hes"], beta=2.0)
    shell_excess = F.relu(u["shell_only"] - u["AO"].detach() - shell_margin)
    inv_shell = F.smooth_l1_loss(shell_excess, torch.zeros_like(shell_excess), beta=2.0)
    loss_inv = 0.5 * (inv_v + inv_f) + 0.5 * inv_shell

    J = lambda_inv * loss_inv
    if J.requires_grad:
        J.backward()
    grads = {k: (float(v.grad) if v.grad is not None else 0.0) for k, v in u.items()}
    active = float(float(loss_inv.detach()) > 1e-4)
    return grads, float(loss_inv.detach()), active


@torch.no_grad()
def collect_window_readouts_nograd(
    model,
    window: list[dict],
    pad_token_id: int,
    microbatch: int,
) -> dict[str, dict[str, float]]:
    dev = C1T._input_device(model)
    out_u: dict[str, dict[str, float]] = defaultdict(dict)
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
    finally:
        if was_training:
            model.train()
    return out_u


def compute_loss_microbatch_c3(
    model,
    batch: list[dict],
    pad_token_id: int,
    loss_mode: str,
    aux_weight: float,
    lambda_margin: float,
    margin_min: float,
    margin_max: float,
    lambda_cap: float,
    paired_chain_grads: dict[tuple[str, str], float] | None = None,
    n_groups_in_window: int = 1,
    win_n: int = 1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute sum-of-per-example symmetric SFT + base/corridor loss + Pass-2 tone-invariance surrogate."""
    dev = C1T._input_device(model)
    bsz = len(batch)
    max_len = max(int(ex["input_ids"].numel()) for ex in batch)
    input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    attn_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    ce_denoms = torch.ones((bsz,), dtype=torch.float32)

    for bi, ex in enumerate(batch):
        L = int(ex["input_ids"].numel())
        input_ids[bi, :L] = ex["input_ids"]
        labels[bi, :L] = ex["labels"]
        attn_mask[bi, :L] = 1
        ce_denoms[bi] = ex["ce_denom"]

    logits = model(input_ids=input_ids.to(dev), attention_mask=attn_mask.to(dev), use_cache=False).logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous().to(logits.device)
    tok_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(bsz, -1)
    per_ex_sft = tok_losses.sum(dim=1) / ce_denoms.to(logits.device).clamp_min(1.0)

    # Symmetric Core Weighting: 1.0 on all 6 core V/F/tone arms (exact 3:3 balance!), aux_weight on auxiliary arms
    ex_w = torch.tensor(
        [1.0 if ex.get("arm") in CORE_ARMS else aux_weight for ex in batch],
        device=logits.device,
        dtype=per_ex_sft.dtype,
    )
    total_sft = (per_ex_sft * ex_w).sum()
    total_margin = logits.new_zeros((), dtype=torch.float32)
    paired_surrogate = logits.new_zeros((), dtype=torch.float32)

    n_signed = 0
    n_floor_act = 0
    n_cap_act = 0

    for bi, ex in enumerate(batch):
        pos = ex.get("answer_readout_pos")
        if pos is None or ex.get("pos_tok") is None:
            continue
        logits_at = logits[bi, pos].float()
        u_theta = logits_at[ex["pos_tok"]] - logits_at[ex["neg_tok"]]
        ms = int(ex.get("margin_sign", 0))
        w_i = 1.0 if ex.get("arm") in CORE_ARMS else aux_weight

        if ms != 0 and lambda_margin > 0.0:
            n_signed += 1
            if loss_mode == "base":
                hinge = F.relu(margin_min - float(ms) * u_theta)
                n_floor_act += int(float(hinge.detach()) > 0.0)
                total_margin = total_margin + w_i * hinge
            else:
                corr, f_act, c_act = signed_corridor_loss(u_theta, float(ms), margin_min, margin_max, lambda_cap)
                n_floor_act += int(f_act)
                n_cap_act += int(c_act)
                total_margin = total_margin + w_i * corr

        if paired_chain_grads:
            g = paired_chain_grads.get((ex["pair_group_id"], ex["arm"]), 0.0)
            if g != 0.0:
                paired_surrogate = paired_surrogate + (g * (win_n / max(1, n_groups_in_window))) * u_theta

    total_loss = total_sft + lambda_margin * total_margin + paired_surrogate
    stats = {
        "sft_sum": float(total_sft.detach()),
        "margin_sum": float(total_margin.detach()),
        "n_signed": n_signed,
        "n_floor_act": n_floor_act,
        "n_cap_act": n_cap_act,
    }
    return total_loss, stats


# -------------------------------------------------------------------------------------- GC-Safe Differentiable IIT


def _aligned_readout_suffix_len(src_ex: dict, tgt_ex: dict, k_max: int = 8) -> int:
    p_s = int(src_ex["answer_readout_pos"])
    p_t = int(tgt_ex["answer_readout_pos"])
    ids_s = src_ex["input_ids"]
    ids_t = tgt_ex["input_ids"]
    k = 0
    limit = min(k_max, p_s + 1, p_t + 1)
    while k < limit and int(ids_s[p_s - k]) == int(ids_t[p_t - k]):
        k += 1
    return max(1, k)


@contextmanager
def differentiable_interchange_forward(
    model,
    src_ex: dict,
    tgt_ex: dict,
    patch_heads: list[tuple[int, int]],
    patch_mlps: list[int],
    head_dim: int,
    k_max: int = 8,
):
    """GC-safe differentiable interchange forward (exact target-only replay hook isolation for any src/tgt lengths)."""
    dev = C1T._input_device(model)
    p_s = int(src_ex["answer_readout_pos"])
    p_t = int(tgt_ex["answer_readout_pos"])
    k = _aligned_readout_suffix_len(src_ex, tgt_ex, k_max=k_max)
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
                        patched[:, sl_t, hs_sl] = cached_head_slices[layer_i, h].to(patched)
                    return (patched, *inputs[1:])

                hooks.enter_context(layer.self_attn.o_proj.register_forward_pre_hook(patch_heads_input))
            if layer_i in mlp_set:
                def patch_mlp_output(_module, _inputs, output):
                    tensor = output[0] if isinstance(output, tuple) else output
                    patched = tensor.clone()
                    patched[:, sl_t, :] = cached_mlp_out[layer_i].to(patched)
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

    # Pass B: Patched target forward — hooks isolated strictly inside target checkpoint callable!
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


def compute_and_backward_iit_for_group_c3(
    model,
    group_by_arm: dict[str, dict],
    group_ref_u: dict[str, float],
    iit_spec: dict[str, Any],
    head_dim: int,
    step_idx: int,
    group_idx: int,
    n_groups_in_window: int,
    lambda_iit_form: float,
    lambda_iit_ver: float,
    margin_iit: float,
    margin_max: float,
    lambda_cap: float,
    k_max: int = 8,
) -> dict[str, float]:
    """Corridor-gated dual-pathway IIT backward for one base group (works identically on Uniform & M3-DAC!)."""
    stats = {
        "iit_form_loss": 0.0,
        "iit_ver_loss": 0.0,
        "iit_form_active": 0.0,
        "iit_ver_active": 0.0,
        "train_iit_ver_sign_hit": 0.0,
        "train_iit_form_sign_hit": 0.0,
    }
    scale = 1.0 / max(1, n_groups_in_window)

    form_heads = [tuple(x) for x in iit_spec["form_only_heads"]]
    ver_heads = [tuple(x) for x in iit_spec["ver_only_heads"]]
    ver_mlps = [int(x) for x in iit_spec["ver_mlps"]]

    # 1. Form-Pathway Debiasing Interchange (same validity, opposite tone)
    if lambda_iit_form > 0.0 and form_heads:
        form_pairs = [
            ("V_auth", "V_hes", +1.0),
            ("F_auth", "F_hes", -1.0),
            ("V_hes", "V_auth", +1.0),
            ("F_hes", "F_auth", -1.0),
        ]
        src_arm, tgt_arm, y_tgt = form_pairs[(step_idx + group_idx) % 4]
        if src_arm in group_by_arm and tgt_arm in group_by_arm and tgt_arm in group_ref_u:
            with differentiable_interchange_forward(
                model,
                src_ex=group_by_arm[src_arm],
                tgt_ex=group_by_arm[tgt_arm],
                patch_heads=form_heads,
                patch_mlps=[],
                head_dim=head_dim,
                k_max=k_max,
            ) as u_patch:
                # Clamp reference target inside corridor ceiling so IIT never chases extreme saturated logits!
                ref_clamped = max(-margin_max, min(margin_max, float(group_ref_u[tgt_arm])))
                u_ref = torch.tensor(ref_clamped, device=u_patch.device, dtype=torch.float32)
                l_stay = F.smooth_l1_loss(u_patch, u_ref, beta=2.0)
                l_corr, f_act, c_act = signed_corridor_loss(u_patch, y_tgt, margin_iit, margin_max, lambda_cap)
                l_form = 0.5 * l_stay + 0.5 * l_corr
                if l_form.requires_grad and float(l_form.detach()) > 0.0:
                    (scale * lambda_iit_form * l_form).backward()
                stats["iit_form_loss"] = float(l_form.detach())
                stats["iit_form_active"] = float(f_act or c_act or float(l_stay.detach()) > 0.1)
                stats["train_iit_form_sign_hit"] = float((y_tgt * float(u_patch.detach())) > 0.0)

    # 2. Verification-Pathway Counterfactual Flip Interchange (same tone, opposite validity)
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
        if src_arm in group_by_arm and tgt_arm in group_by_arm:
            with differentiable_interchange_forward(
                model,
                src_ex=group_by_arm[src_arm],
                tgt_ex=group_by_arm[tgt_arm],
                patch_heads=ver_heads,
                patch_mlps=ver_mlps,
                head_dim=head_dim,
                k_max=k_max,
            ) as u_patch:
                l_ver, f_act, c_act = signed_corridor_loss(u_patch, y_src, margin_iit, margin_max, lambda_cap)
                if l_ver.requires_grad and float(l_ver.detach()) > 0.0:
                    (scale * lambda_iit_ver * l_ver).backward()
                stats["iit_ver_loss"] = float(l_ver.detach())
                stats["iit_ver_active"] = float(f_act or c_act)
                stats["train_iit_ver_sign_hit"] = float((y_src * float(u_patch.detach())) > 0.0)

    return stats


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _encode_c3_records(tok, rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        enc = C1T.encode_sft_example(tok, r, enable_thinking=False, mask_rule_format_ce=False)
        enc["arm"] = r["arm"]
        out.append(enc)
    return out


# -------------------------------------------------------------------------------------- Exhaustive CPU Self-Test


def run_selftest_tiny() -> None:
    """Verify ALL 6 mathematical & engineering invariants of `c3_train.py`:
      1. Signed Confidence Corridor pulls high-confidence WRONG (`y*u=-15`) toward `+m_min`, gives EXACT 0 inside
         `[m_min, m_max]`, and pulls overconfident RIGHT (`y*u=+20`) back toward `m_max`.
      2. Exact 2-pass tone-invariance chain-rule gradient == direct single-batch autograd (`< 1e-4`).
      3. Decoupled `Uniform-LoRA + B1-IIT` AND `M3-DAC + B1-IIT` both run cleanly with finite grads across all zones.
      4. GC (`use_reentrant=False`) + IIT equivalence (`gc_diff < 1e-5`) across ALL 3 length regimes:
         `len(src) < len(tgt)`, `len(src) == len(tgt)`, and `len(src) > len(tgt)`!
      5. Zone-1 proximal regularization strictly shrinks `||Delta W_{Z1}||_F` relative to unregularized step.
      6. Saved `adapter_state.pt` exports losslessly via `c3_export_peft_lora` (`logit_err < 1e-5`).
    """
    from transformers import Qwen3Config, Qwen3ForCausalLM

    # 1. Signed Confidence Corridor behavior test
    for u_val, y_s, exp_sign_grad, exp_zero in [
        (-15.0, +1.0, -1, False),  # high-conf wrong -> grad < 0 pushes u UP toward +4.5!
        (+15.0, -1.0, +1, False),  # high-conf wrong -> grad > 0 pushes u DOWN toward -4.5!
        (+7.0, +1.0, 0, True),     # inside corridor [4.5, 10.0] -> exact 0 grad!
        (-7.0, -1.0, 0, True),     # inside corridor -> exact 0 grad!
        (+20.0, +1.0, +1, False),  # overconfident right -> grad > 0 pulls u DOWN toward +10.0!
        (-20.0, -1.0, -1, False),  # overconfident right -> grad < 0 pulls u UP toward -10.0!
    ]:
        ut = torch.tensor(u_val, dtype=torch.float32, requires_grad=True)
        loss, _, _ = signed_corridor_loss(ut, y_s, m_min=4.5, m_max=10.0, lambda_cap=0.05)
        loss.backward()
        g = float(ut.grad)
        if exp_zero:
            assert abs(g) == 0.0, (u_val, y_s, g)
        else:
            assert (g > 0) == (exp_sign_grad > 0), (u_val, y_s, g, exp_sign_grad)
    print("[OK] 1. Signed Confidence Corridor sign/zero/cap gradient behavior verified")

    cfg = Qwen3Config(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=2048,
        tie_word_embeddings=False,
    )
    tok = C1T._TinyTok()
    train_path = C3 / "data" / "c3_train_examples_nothink.jsonl"
    raw_rows = [json.loads(l) for l in train_path.read_text(encoding="utf-8").splitlines()[:20]]
    encoded = _encode_c3_records(tok, raw_rows)
    grp0 = encoded[:10]
    assert len({e["pair_group_id"] for e in grp0}) == 1

    tiny_m3 = {
        "kind": "m3_dac",
        "zones": {"zone1": [0, 1], "zone2": [2, 3], "zone3": [4, 5]},
        "zone1_dense_qo_layers": [0, 1],
        "zone1_dense_qo_rank": 2,
        "mlp_layers": [2, 4, 5],
        "mlp_rank": 4,
        "head_ranks": {"2:0": 8, "2:2": 4, "3:1": 6, "4:0": 4, "5:3": 2},
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
        "ver_mlps": [2, 4],
    }

    # 2. Verify exact 2-pass tone-invariance chain rule == direct single-batch autograd on M3-DAC!
    torch.manual_seed(11)
    m_ref = Qwen3ForCausalLM(cfg).float()
    M3.apply_c3_adapters(m_ref, tiny_m3)
    with torch.no_grad():
        for n, p in m_ref.named_parameters():
            if "lora_B" in n:
                p.normal_(0.0, 0.05)
    init_state = {k: v.detach().clone() for k, v in m_ref.state_dict().items()}

    m_ref.train()
    m_ref.zero_grad()
    u_dir = {}
    base_sum = torch.tensor(0.0)
    for ex in grp0:
        l_mb, _ = compute_loss_microbatch_c3(m_ref, [ex], tok.pad_token_id, "corridor", 0.5, 0.25, 4.5, 10.0, 0.05)
        base_sum = base_sum + l_mb
        lg = m_ref(input_ids=ex["input_ids"][None], use_cache=False).logits
        u_dir[ex["arm"]] = lg[0, ex["answer_readout_pos"], ex["pos_tok"]].float() - lg[0, ex["answer_readout_pos"], ex["neg_tok"]].float()
    l_inv_d = (
        0.5 * (F.smooth_l1_loss(u_dir["V_auth"], u_dir["V_hes"], beta=2.0) + F.smooth_l1_loss(u_dir["F_auth"], u_dir["F_hes"], beta=2.0))
        + 0.5 * F.smooth_l1_loss(F.relu(u_dir["shell_only"] - u_dir["AO"].detach() - 1.0), torch.tensor(0.0), beta=2.0)
    )
    (base_sum / len(grp0) + 0.15 * l_inv_d).backward()
    grads_dir = {n: p.grad.detach().clone() for n, p in m_ref.named_parameters() if p.requires_grad}

    m_2p = Qwen3ForCausalLM(cfg).float()
    M3.apply_c3_adapters(m_2p, tiny_m3)
    m_2p.load_state_dict(init_state)
    m_2p.train()
    m_2p.zero_grad()
    u_map = collect_window_readouts_nograd(m_2p, grp0, tok.pad_token_id, microbatch=2)
    gid = grp0[0]["pair_group_id"]
    arm_g, _, _ = compute_tone_invariance_chain_grads(u_map[gid], lambda_inv=0.15)
    cg = {(gid, a): g for a, g in arm_g.items()}
    for j in range(0, len(grp0), 2):
        l_mb, _ = compute_loss_microbatch_c3(
            m_2p, grp0[j : j + 2], tok.pad_token_id, "corridor", 0.5, 0.25, 4.5, 10.0, 0.05,
            paired_chain_grads=cg, n_groups_in_window=1, win_n=len(grp0),
        )
        (l_mb / len(grp0)).backward()
    gdiff = max(float((grads_dir[n] - p.grad).abs().max()) for n, p in m_2p.named_parameters() if p.requires_grad)
    assert gdiff < 1e-4, gdiff
    print(f"[OK] 2. Exact 2-pass corridor + tone-invariance chain-rule equivalence verified (max_diff={gdiff:.2e})")

    # 3 & 4. Test BOTH M3-DAC AND Uniform-LoRA under GC vs non-GC across ALL 3 length regimes (<, ==, >)!
    grp_by_arm_base = {e["arm"]: e for e in grp0}
    for arch_name, arch_spec in (("M3-DAC", tiny_m3), ("Uniform-LoRA", tiny_unif)):
        torch.manual_seed(23)
        m_seed = Qwen3ForCausalLM(cfg).float()
        M3.apply_c3_adapters(m_seed, arch_spec)
        with torch.no_grad():
            for n, p in m_seed.named_parameters():
                if "lora_B" in n:
                    p.normal_(0.0, 0.05)
        sd_seed = {k: v.detach().clone() for k, v in m_seed.state_dict().items()}

        # Build 3 explicit length scenarios: src < tgt, src == tgt, src > tgt!
        def _pad_prefix(ex: dict, n_pad: int) -> dict:
            if n_pad == 0:
                return dict(ex)
            pref = ex["input_ids"][:n_pad]
            return {
                **ex,
                "input_ids": torch.cat((pref, ex["input_ids"])),
                "answer_readout_pos": ex["answer_readout_pos"] + n_pad,
            }

        ex_v = grp_by_arm_base["V"]
        ex_f = grp_by_arm_base["F"]
        L_min = min(int(ex_v["input_ids"].numel()), int(ex_f["input_ids"].numel()))
        # Trim/pad to exact equal length, shorter src, longer src
        eq_v = {**ex_v, "input_ids": ex_v["input_ids"][-L_min:], "answer_readout_pos": ex_v["answer_readout_pos"] - (int(ex_v["input_ids"].numel()) - L_min)}
        eq_f = {**ex_f, "input_ids": ex_f["input_ids"][-L_min:], "answer_readout_pos": ex_f["answer_readout_pos"] - (int(ex_f["input_ids"].numel()) - L_min)}
        length_cases = [
            ("src<tgt", eq_v, _pad_prefix(eq_f, 12)),
            ("src==tgt", eq_v, eq_f),
            ("src>tgt", _pad_prefix(eq_v, 14), eq_f),
        ]
        max_gc_diff_arch = 0.0
        for cname, s_ex, t_ex in length_cases:
            g_pair = {}
            for use_gc in (False, True):
                m_t = Qwen3ForCausalLM(cfg).float()
                M3.apply_c3_adapters(m_t, arch_spec)
                m_t.load_state_dict(sd_seed)
                if use_gc:
                    m_t.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                    m_t.enable_input_require_grads()
                m_t.train()
                m_t.zero_grad()
                with differentiable_interchange_forward(
                    m_t, s_ex, t_ex,
                    patch_heads=[tuple(x) for x in tiny_iit["ver_only_heads"]],
                    patch_mlps=tiny_iit["ver_mlps"],
                    head_dim=16,
                ) as u_p:
                    lv, _, _ = signed_corridor_loss(u_p, +1.0, m_min=10.0, m_max=15.0, lambda_cap=0.05)
                    lv.backward()
                g_pair[use_gc] = {n: p.grad.detach().clone() for n, p in m_t.named_parameters() if p.requires_grad}
                for n, g in g_pair[use_gc].items():
                    assert torch.isfinite(g).all(), (arch_name, cname, use_gc, n)
            cdiff = max(float((g_pair[False][n] - g_pair[True][n]).abs().max()) for n in g_pair[False])
            assert cdiff < 1e-5, (arch_name, cname, cdiff)
            max_gc_diff_arch = max(max_gc_diff_arch, cdiff)
        print(f"[OK] 3 & 4. {arch_name:12s} + IIT GC equivalence across (<, ==, >) verified (max_gc_diff={max_gc_diff_arch:.2e})")

    # 5. Zone-1 proximal regularization strictly shrinks ||Delta W_{Z1}||_F!
    norms_prox = {}
    for lprox in (0.0, 0.5):
        m_p = Qwen3ForCausalLM(cfg).float()
        M3.apply_c3_adapters(m_p, tiny_m3)
        m_p.load_state_dict(init_state)
        opt = torch.optim.SGD(M3.build_zone_optimizer_groups(m_p, lr=0.05, lr_z1_mult=0.5))
        opt.zero_grad()
        if lprox > 0.0:
            (lprox * M3.zone1_proximal_penalty(m_p)).backward()
        opt.step()
        norms_prox[lprox] = M3.zone_delta_norms(m_p)["z1_frob"]
    assert norms_prox[0.5] < norms_prox[0.0], norms_prox
    print(f"[OK] 5. Zone-1 proximal penalty verified: z1_frob {norms_prox[0.0]:.4f} -> {norms_prox[0.5]:.4f}")

    # 6. Save & lossless PEFT export on M3-DAC
    with tempfile.TemporaryDirectory() as td:
        ap = Path(td) / "adapter_state.pt"
        M3.save_c3_adapter_state(m_2p, ap)
        peft_dir = Path(td) / "peft"
        meta = EXP3.export_c3_adapter_to_peft(
            state_dict=torch.load(ap, map_location="cpu", weights_only=True),
            arch_spec=tiny_m3,
            model_spec={"hidden_size": 64, "num_heads": 4, "head_dim": 16, "intermediate_size": 128, "num_layers": 6},
            scaling={"scaling_mode": "rslora", "base_scale": 2.0, "anchor_rank": 8},
            out_dir=peft_dir,
            export_dtype=torch.float32,
        )
        assert meta["max_fp32_dw_abs_err"] < 1e-5
    print("[OK] 6. M3-DAC save & lossless PEFT LoRA export verified")
    print("[OK] ALL c3_train self-tests PASSED!")


# -------------------------------------------------------------------------------------- Main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-tiny", action="store_true")
    ap.add_argument("--model", type=str)
    ap.add_argument("--model-key", choices=("qwen3_8b", "qwen3_14b"), default="qwen3_8b")
    ap.add_argument("--arch", type=str, default="M3-DAC", help="Architecture key in c3_components_frozen.json")
    ap.add_argument("--loss-mode", choices=("base", "corridor", "corridor_iit"), default="corridor_iit")
    ap.add_argument("--iit-target", choices=("none", "b1", "rnd_head_1", "rnd_head_2", "rnd_full_1", "rnd_full_2"), default="none")
    ap.add_argument("--components-file", type=Path, default=C3 / "data" / "c3_components_frozen.json")
    ap.add_argument("--train-file", type=Path, default=C3 / "data" / "c3_train_examples_nothink.jsonl")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2.5e-5)
    ap.add_argument("--lr-z1-mult", type=float, default=None, help="Defaults to 0.35 for m3_dac, 1.0 for uniform")
    ap.add_argument("--lambda-prox-z1", type=float, default=None, help="Defaults to 1e-3 for m3_dac, 0.0 for uniform")
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
    ap.add_argument("--lambda-iit-form", type=float, default=0.15)
    ap.add_argument("--lambda-iit-ver", type=float, default=0.25)
    ap.add_argument("--margin-iit", type=float, default=3.5)
    ap.add_argument("--iit-k-max", type=int, default=8)
    ap.add_argument("--scaling-mode", choices=("rslora", "constant"), default="rslora")
    ap.add_argument("--lora-scale", type=float, default=2.0)
    ap.add_argument("--anchor-rank", type=int, default=8)
    ap.add_argument("--save-steps", type=str, default="50,100,200,300")
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

    lr_z1_mult = args.lr_z1_mult if args.lr_z1_mult is not None else (0.35 if arch_spec["kind"] == "m3_dac" else 1.0)
    lambda_prox_z1 = args.lambda_prox_z1 if args.lambda_prox_z1 is not None else (1e-3 if arch_spec["kind"] == "m3_dac" else 0.0)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    raw_rows = [json.loads(l) for l in args.train_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        raw_rows = raw_rows[: args.limit]
    encoded = _encode_c3_records(tok, raw_rows)

    print(
        f"[c3_train] model={args.model} ({args.model_key}) arch={args.arch} loss={args.loss_mode} "
        f"iit={args.iit_target} lr={args.lr} lr_z1_mult={lr_z1_mult} prox_z1={lambda_prox_z1} seed={args.seed}"
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    adapter_info = M3.apply_c3_adapters(
        model,
        arch_spec,
        scaling_mode=args.scaling_mode,
        base_scale=args.lora_scale,
        anchor_rank=args.anchor_rank,
        dropout=0.0,
    )
    param_groups = M3.build_zone_optimizer_groups(model, lr=args.lr, lr_z1_mult=lr_z1_mult, weight_decay=args.weight_decay)
    opt = torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    save_steps = sorted({int(x.strip()) for x in args.save_steps.split(",") if x.strip()})
    all_windows: list[list[dict]] = []
    for ep in range(args.epochs):
        all_windows.extend(C1T.build_atomic_windows(encoded, args.grad_accum, seed=args.seed + ep * 1009))
    total_steps = min(len(all_windows), args.max_steps) if args.max_steps > 0 else len(all_windows)
    warmup_steps = max(1, int(0.10 * total_steps))
    print(
        f"[c3_train] encoded={len(encoded)}, total_steps={total_steps}, warmup={warmup_steps}, "
        f"trainable={adapter_info['trainable_params']:,} zones={adapter_info['zone_trainable_params']}"
    )

    model.train()
    logs: list[dict[str, Any]] = []
    saved_checkpoints: dict[str, str] = {}
    zone_norm_trajectory: dict[str, dict[str, float]] = {}
    n_clipped = 0

    for step_idx, win in enumerate(all_windows[:total_steps]):
        cur_step = step_idx + 1
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
        inv_loss_sum = 0.0
        inv_act_sum = 0.0

        # Pass 1: Detached readouts for tone invariance + IIT reference
        if args.loss_mode in ("corridor", "corridor_iit"):
            u_map = collect_window_readouts_nograd(model, win, pad_id, microbatch=args.microbatch)
            for gid, gu in u_map.items():
                arm_g, lv_inv, inv_a = compute_tone_invariance_chain_grads(gu, lambda_inv=args.lambda_inv)
                inv_loss_sum += lv_inv
                inv_act_sum += inv_a
                for a, gv in arm_g.items():
                    paired_chain_grads[(gid, a)] = gv

        # Pass 2: Symmetric SFT + Base/Corridor + Tone-Invariance Chain-Rule Surrogate
        win_sft = 0.0
        win_marg = 0.0
        win_signed = 0
        win_floor_act = 0
        win_cap_act = 0
        win_n = len(win)
        for j in range(0, win_n, args.microbatch):
            mb = win[j : j + args.microbatch]
            loss_sum, st_mb = compute_loss_microbatch_c3(
                model,
                mb,
                pad_id,
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

        # Pass 3: Corridor-Gated Dual-Pathway IIT (on BOTH Uniform & M3-DAC when corridor_iit!)
        iit_f_sum = 0.0
        iit_v_sum = 0.0
        iit_f_act = 0.0
        iit_v_act = 0.0
        iit_v_hit = 0.0
        iit_f_hit = 0.0
        if args.loss_mode == "corridor_iit" and iit_spec is not None:
            for g_i, gid in enumerate(sorted(groups_in_win.keys())):
                st_iit = compute_and_backward_iit_for_group_c3(
                    model,
                    group_by_arm=groups_in_win[gid],
                    group_ref_u=u_map.get(gid, {}),
                    iit_spec=iit_spec,
                    head_dim=head_dim,
                    step_idx=step_idx,
                    group_idx=g_i,
                    n_groups_in_window=n_groups,
                    lambda_iit_form=args.lambda_iit_form,
                    lambda_iit_ver=args.lambda_iit_ver,
                    margin_iit=args.margin_iit,
                    margin_max=args.margin_max,
                    lambda_cap=args.lambda_cap,
                    k_max=args.iit_k_max,
                )
                iit_f_sum += st_iit["iit_form_loss"]
                iit_v_sum += st_iit["iit_ver_loss"]
                iit_f_act += st_iit["iit_form_active"]
                iit_v_act += st_iit["iit_ver_active"]
                iit_v_hit += st_iit["train_iit_ver_sign_hit"]
                iit_f_hit += st_iit["train_iit_form_sign_hit"]

        # Zone-1 Proximal Penalty on actual ||Delta W_{Z1}||_F^2
        prox_val = 0.0
        if lambda_prox_z1 > 0.0:
            p_loss = lambda_prox_z1 * M3.zone1_proximal_penalty(model)
            if p_loss.requires_grad:
                p_loss.backward()
            prox_val = float(p_loss.detach())

        # Log pre-clip per-zone grad norms BEFORE clipping!
        z_grads = M3.zone_grad_norms(model)
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=args.max_grad_norm,
            )
        )
        was_clipped = int(grad_norm > args.max_grad_norm)
        n_clipped += was_clipped
        opt.step()

        row_log = {
            "step": cur_step,
            "lr_z23": round(opt.param_groups[-1]["lr"], 8),
            "sft_mean": round(win_sft / win_n, 4),
            "margin_mean": round(win_marg / win_n, 4),
            "floor_act_rate": round(win_floor_act / max(1, win_signed), 3),
            "cap_act_rate": round(win_cap_act / max(1, win_signed), 3),
            "inv_mean": round(inv_loss_sum / n_groups, 4),
            "inv_act_rate": round(inv_act_sum / n_groups, 3),
            "iit_form_mean": round(iit_f_sum / n_groups, 4),
            "iit_ver_mean": round(iit_v_sum / n_groups, 4),
            "iit_form_act_rate": round(iit_f_act / n_groups, 3),
            "iit_ver_act_rate": round(iit_v_act / n_groups, 3),
            "train_iit_ver_sign_hit": round(iit_v_hit / n_groups, 3),
            "train_iit_form_sign_hit": round(iit_f_hit / n_groups, 3),
            "prox_z1": round(prox_val, 5),
            "g_z1": round(z_grads["g_z1"], 4),
            "g_z2": round(z_grads["g_z2"], 4),
            "g_z3": round(z_grads["g_z3"], 4),
            "grad_norm": round(grad_norm, 4),
            "clipped": was_clipped,
        }
        logs.append(row_log)
        if cur_step == 1 or cur_step % 25 == 0 or cur_step == total_steps:
            print(
                f"[step {cur_step:3d}/{total_steps}] sft={row_log['sft_mean']:.3f} corr={row_log['margin_mean']:.3f} "
                f"(flr={row_log['floor_act_rate']:.2f},cap={row_log['cap_act_rate']:.2f}) "
                f"inv={row_log['inv_mean']:.3f} iit_v={row_log['iit_ver_mean']:.3f} "
                f"g=({row_log['g_z1']:.2f}/{row_log['g_z2']:.2f}/{row_log['g_z3']:.2f} tot={row_log['grad_norm']:.2f})"
            )

        if cur_step in save_steps and cur_step < total_steps:
            ckpt_p = args.out_dir / f"adapter_state_step{cur_step:04d}.pt"
            M3.save_c3_adapter_state(model, ckpt_p)
            saved_checkpoints[str(cur_step)] = _file_sha256(ckpt_p)
            zone_norm_trajectory[str(cur_step)] = M3.zone_delta_norms(model)

    final_ap = args.out_dir / "adapter_state.pt"
    M3.save_c3_adapter_state(model, final_ap)
    final_sha = _file_sha256(final_ap)
    saved_checkpoints[str(total_steps)] = final_sha
    final_norms = M3.zone_delta_norms(model)
    zone_norm_trajectory[str(total_steps)] = final_norms

    scaling_dict = {
        "scaling_mode": args.scaling_mode,
        "base_scale": args.lora_scale,
        "anchor_rank": args.anchor_rank,
    }
    adapter_cfg = {
        "schema": "cedar.acl_exp_c3.adapter_config.v1",
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
        "schema": "cedar.acl_exp_c3.train_meta.v1",
        "model": args.model,
        "model_key": args.model_key,
        "arch": args.arch,
        "loss_mode": args.loss_mode,
        "iit_target": args.iit_target,
        "seed": args.seed,
        "train_file": str(args.train_file),
        "train_file_sha256": _file_sha256(args.train_file),
        "n_examples": len(encoded),
        "total_steps": total_steps,
        "clip_rate": round(n_clipped / max(1, total_steps), 4),
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
            "lambda_iit_form": args.lambda_iit_form,
            "lambda_iit_ver": args.lambda_iit_ver,
            "margin_iit": args.margin_iit,
            "iit_k_max": args.iit_k_max,
            "seed": args.seed,
        },
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
        f"[c3_train] DONE in {meta['elapsed_seconds']}s | clip_rate={meta['clip_rate']:.2%} "
        f"| sha256={final_sha[:12]} | norms={final_norms}"
    )


if __name__ == "__main__":
    main()
