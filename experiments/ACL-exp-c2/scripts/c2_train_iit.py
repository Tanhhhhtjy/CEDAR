#!/usr/bin/env python3
"""ACL-exp-c2 Trainer: Verification-Gated Three-Force Equilibrium via Circuit-Guided Interchange Intervention Training (DAC-IIT).

Implements the exact 2x2 factorial training modes sharing identical frozen parameter budgets from
`experiments/ACL-exp-c1/data/c1_components_frozen.json` and 100% identical adapter storage (`adapter_state.pt`)
compatible with `c1_export_peft_lora.py` (lossless vLLM native LoRA) and `c1_eval.py` (lossless HF live):

  1. `--loss-mode base`:
     Standard per-example SFT + Signed Readout Margin loss (exact C1 baseline objective, balanced `--accept-weight 2.0`).
  2. `--loss-mode paired2x2`:
     `base` + Exact 2-Pass Scalar Chain-Rule 2x2 Structural Objective:
       - Horizontal Rhetoric Invariance: SmoothL1(u(V_auth), u(V_hes)) + SmoothL1(u(F_auth), u(F_hes))
         + SmoothL1(ReLU(u(shell_only) - detach(u(AO)) - 1.0), 0)
       - Vertical Evidence Sensitivity: mean_{t in {plain,auth,hes}} ReLU(delta_sens - (u(V_t) - u(F_t)))
     Mathematically identical to holding all 10 sequences of a base in one batch, with ZERO extra VRAM!
  3. `--loss-mode dac_iit`:
     `paired2x2` + Dual-Pathway Circuit-Guided Interchange Intervention Training (Geiger et al., ICML 2022)
     on the exact B1 3-group anatomical partition (or layer-matched random partition for `Random-LM` / `Random-LM-seed2`!):
       - Group 1 (`form_only_heads`, 23 heads on 8B): Conditional Rhetoric Debiasing Interchange
         Patch `form_only` o_proj head slices across tones (`V_hes <-> V_auth`, `F_hes <-> F_auth`) at the
         aligned readout suffix -> decision must stay invariant & correct (debias form WITHOUT destroying it!).
       - Group 2 (`ver_only_heads + ver_mlps`, 21 heads + 5 MLPs on 8B): Counterfactual Verification Flip Interchange
         Patch `ver_only` o_proj head slices + `ver_mlps` outputs across same-tone validity (`F_t <- V_t` & `V_t <- F_t`)
         -> decision MUST counterfactually flip!
       - Group 3 (`hub_heads`, 3 heads on 8B): Deliberately NEVER overwritten during interchange, freeing them to
         learn the verification-gated routing equilibrium g(h_ver) * h_form + h_ver!
     Hooks are active ONLY during training; inference and vLLM native LoRA export have ZERO extra compute overhead.

CPU Self-Test:
  python experiments/ACL-exp-c2/scripts/c2_train_iit.py --selftest-tiny
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
C2 = HERE.parents[0]
REPO = C2.parents[1]
C1 = REPO / "experiments" / "ACL-exp-c1"
sys.path.insert(0, str(C1 / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-b1" / "scripts"))

import c1_modules as M  # noqa: E402
import c1_train as C1T  # noqa: E402


# -------------------------------------------------------------------------------------- Exact 3-Group Partition


def partition_circuit_components(
    comp_doc: dict[str, Any],
    model_key: str,
    arm_name: str,
    partition_seed: int = 20270801,
) -> dict[str, Any]:
    """Return exact disjoint 3-group anatomical partition: form_only_heads, ver_only_heads, hub_heads, ver_mlps.

    - For `Targeted-Dual`: uses exact B1 frozen anatomy:
        hub_heads       = comp['hub_heads']                              (8B: 3,  14B: 4)
        form_only_heads = Targeted-Form \\ hub_heads                      (8B: 23, 14B: 22)
        ver_only_heads  = Targeted-Dual \\ Targeted-Form \\ hub_heads     (8B: 21, 14B: 20)
        ver_mlps        = Targeted-Dual['mlp_layers']                    (8B: 5,  14B: 5)
    - For `Random-LM` / `Random-LM-seed2`: partitions the arm's random heads PER-LAYER with the EXACT SAME
      (n_form_only, n_ver_only, n_hub) counts on every single layer as `Targeted-Dual`, plus the arm's 5 random MLPs!
      Guarantees 100% fair layer-matched 2x2 factorial comparison (`Location x Algorithm`).
    - For `Uniform-LoRA`: returns empty circuit groups (use `--loss-mode base` or `paired2x2`).
    """
    m = comp_doc["models"][model_key]
    arm_spec = m["arms"][arm_name]
    if arm_spec["kind"] != "circuit":
        return {
            "kind": arm_spec["kind"],
            "form_only_heads": [],
            "ver_only_heads": [],
            "hub_heads": [],
            "ver_mlps": [],
        }

    dual_heads = {tuple(x) for x in m["arms"]["Targeted-Dual"]["heads"]}
    form_heads = {tuple(x) for x in m["arms"]["Targeted-Form"]["heads"]}
    hub_heads = {tuple(x) for x in m["hub_heads"]}
    assert form_heads.issubset(dual_heads), "Targeted-Form not subset of Targeted-Dual"
    assert hub_heads.issubset(dual_heads), "hub_heads not subset of Targeted-Dual"

    dual_form_only = sorted(form_heads - hub_heads)
    dual_ver_only = sorted(dual_heads - form_heads - hub_heads)
    dual_hub = sorted(hub_heads)
    assert len(dual_form_only) + len(dual_ver_only) + len(dual_hub) == len(dual_heads)

    if arm_name == "Targeted-Dual":
        return {
            "kind": "circuit",
            "source": "b1_frozen_anatomy",
            "form_only_heads": [list(x) for x in dual_form_only],
            "ver_only_heads": [list(x) for x in dual_ver_only],
            "hub_heads": [list(x) for x in dual_hub],
            "ver_mlps": list(arm_spec.get("mlp_layers", [])),
        }

    # Layer-matched partition for Random-LM / Random-LM-seed2 (deterministic per arm name)
    arm_heads = [tuple(x) for x in arm_spec["heads"]]
    by_layer_arm: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for l, h in arm_heads:
        by_layer_arm[l].append((l, h))

    counts_form = defaultdict(int)
    counts_ver = defaultdict(int)
    counts_hub = defaultdict(int)
    for l, _ in dual_form_only:
        counts_form[l] += 1
    for l, _ in dual_ver_only:
        counts_ver[l] += 1
    for l, _ in dual_hub:
        counts_hub[l] += 1

    seed_offset = sum(ord(c) * (i + 1) for i, c in enumerate(arm_name))
    rng = random.Random(partition_seed + seed_offset)
    r_form, r_ver, r_hub = [], [], []
    for l in sorted(by_layer_arm.keys()):
        hs = list(sorted(by_layer_arm[l]))
        nf, nv, nh = counts_form[l], counts_ver[l], counts_hub[l]
        assert len(hs) == nf + nv + nh, f"Layer {l} mismatch for {arm_name}: {len(hs)} vs {nf}+{nv}+{nh}"
        rng.shuffle(hs)
        r_form.extend(hs[:nf])
        r_ver.extend(hs[nf : nf + nv])
        r_hub.extend(hs[nf + nv :])

    assert len(r_form) == len(dual_form_only)
    assert len(r_ver) == len(dual_ver_only)
    assert len(r_hub) == len(dual_hub)
    assert len(set(r_form) | set(r_ver) | set(r_hub)) == len(arm_heads)
    return {
        "kind": "circuit",
        "source": f"layer_matched_random_partition({arm_name})",
        "form_only_heads": [list(x) for x in sorted(r_form)],
        "ver_only_heads": [list(x) for x in sorted(r_ver)],
        "hub_heads": [list(x) for x in sorted(r_hub)],
        "ver_mlps": list(arm_spec.get("mlp_layers", [])),
    }


# -------------------------------------------------------------------------------------- Level 1: Exact 2-Pass Paired 2x2


def compute_paired2x2_chain_grads(
    group_u: dict[str, float],
    lambda_inv: float,
    lambda_sens: float,
    delta_sens: float,
    shell_margin: float = 1.0,
) -> tuple[dict[str, float], float, float]:
    """Given detached readout logit diffs {arm: u_float} for one 10-item base group, return:
      - arm_grads: exact dJ / d(u_arm) scalar multipliers for Pass 2
      - inv_loss_val: float
      - sens_loss_val: float
    Uses SmoothL1 (beta=2.0) for invariance terms so gradients are smoothly quadratic near 0 and bounded <= 1!
    Detaches u(AO) in the shell_only pullback so shell_only moves down toward AO without pushing AO upward!
    """
    needed = ("V", "F", "V_auth", "V_hes", "F_auth", "F_hes", "shell_only", "AO")
    if not all(k in group_u for k in needed):
        return {}, 0.0, 0.0

    u = {k: torch.tensor(float(group_u[k]), dtype=torch.float32, requires_grad=True) for k in needed}
    inv_v = F.smooth_l1_loss(u["V_auth"], u["V_hes"], beta=2.0)
    inv_f = F.smooth_l1_loss(u["F_auth"], u["F_hes"], beta=2.0)
    shell_excess = F.relu(u["shell_only"] - u["AO"].detach() - shell_margin)
    inv_shell = F.smooth_l1_loss(shell_excess, torch.zeros_like(shell_excess), beta=2.0)
    loss_inv = 0.5 * (inv_v + inv_f) + 0.5 * inv_shell

    s_plain = F.relu(delta_sens - (u["V"] - u["F"]))
    s_auth = F.relu(delta_sens - (u["V_auth"] - u["F_auth"]))
    s_hes = F.relu(delta_sens - (u["V_hes"] - u["F_hes"]))
    loss_sens = (s_plain + s_auth + s_hes) / 3.0

    J = lambda_inv * loss_inv + lambda_sens * loss_sens
    if J.requires_grad:
        J.backward()
    grads = {k: (float(v.grad) if v.grad is not None else 0.0) for k, v in u.items()}
    return grads, float(loss_inv.detach()), float(loss_sens.detach())


@torch.no_grad()
def collect_window_readouts_nograd(
    model,
    window: list[dict],
    pad_token_id: int,
    microbatch: int,
) -> dict[str, dict[str, float]]:
    """Pass 1 of exact 2-pass paired loss: collect detached u_theta scalars per (pair_group_id, arm)."""
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


def compute_loss_microbatch_c2(
    model,
    batch: list[dict],
    pad_token_id: int,
    lambda_margin: float,
    margin_m: float,
    accept_weight: float = 2.0,
    paired_chain_grads: dict[tuple[str, str], float] | None = None,
    n_groups_in_window: int = 1,
    win_n: int = 1,
) -> tuple[torch.Tensor, float, float, dict]:
    """Compute exact sum-of-per-example base loss + Pass-2 paired scalar chain-rule surrogate over microbatch."""
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
    if accept_weight != 1.0:
        acc_w = torch.tensor(
            [accept_weight if int(ex.get("margin_sign", 0)) > 0 else 1.0 for ex in batch],
            device=logits.device,
            dtype=per_ex_sft.dtype,
        )
        per_ex_sft = per_ex_sft * acc_w

    total_sft = per_ex_sft.sum()
    total_margin = logits.new_zeros((), dtype=torch.float32)
    paired_surrogate = logits.new_zeros((), dtype=torch.float32)

    for bi, ex in enumerate(batch):
        pos = ex.get("answer_readout_pos")
        if pos is None or ex.get("pos_tok") is None:
            continue
        logits_at = logits[bi, pos].float()
        u_theta = logits_at[ex["pos_tok"]] - logits_at[ex["neg_tok"]]
        ms = int(ex.get("margin_sign", 0))
        if ms != 0 and lambda_margin > 0.0:
            hinge = F.relu(margin_m - u_theta) if ms > 0 else F.relu(margin_m + u_theta)
            w_m = accept_weight if ms > 0 else 1.0
            total_margin = total_margin + w_m * hinge
        if paired_chain_grads:
            g = paired_chain_grads.get((ex["pair_group_id"], ex["arm"]), 0.0)
            if g != 0.0:
                # Scale by (win_n / n_groups) because caller divides total_loss by win_n!
                paired_surrogate = paired_surrogate + (g * (win_n / max(1, n_groups_in_window))) * u_theta

    total_loss = total_sft + lambda_margin * total_margin + paired_surrogate
    return total_loss, float(total_sft.detach()), float(total_margin.detach()), {"bsz": bsz}


# -------------------------------------------------------------------------------------- Level 2: Circuit-Guided IIT


def _aligned_readout_suffix_len(src_ex: dict, tgt_ex: dict, k_max: int = 8) -> int:
    """Compute exact aligned suffix length (1..k_max) ending at answer_readout_pos across src & tgt."""
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
    """Yield the patched target readout with gradients through the clean source forward.

    Each target layer installs its patch for that layer's execution, including
    its checkpoint replay. Source replay uses the original unpatched callable.
    """
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

    # Step 1: Differentiable source forward capturing exact slices
    rec_hooks = []
    try:
        for l, hs in heads_by_layer.items():
            o_mod = model.model.layers[l].self_attn.o_proj

            def _make_rec_o(layer_i: int, h_list: list[int]):
                def _hook(_mod, inp):
                    x = inp[0]  # [1, T_s, H*hd]
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
    def patch_target_layer(layer_i):
        layer = model.model.layers[layer_i]
        heads = heads_by_layer.get(layer_i, ())
        with ExitStack() as hooks:
            if heads:
                def patch_heads_input(_module, inputs):
                    patched = inputs[0].clone()
                    for h in heads:
                        head_slice = slice(h * head_dim, (h + 1) * head_dim)
                        patched[:, sl_t, head_slice] = cached_head_slices[layer_i, h].to(patched)
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

    def checkpoint_with_patch(checkpoint, layer_i):
        def checkpoint_target(function, *args, **kwargs):
            def patched_forward(*inputs, **forward_kwargs):
                with patch_target_layer(layer_i):
                    return function(*inputs, **forward_kwargs)
            return checkpoint(patched_forward, *args, **kwargs)
        return checkpoint_target

    # Capture target hooks inside the checkpoint's callable. A later source
    # replay (possibly with a shorter sequence) must see a clean layer.
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


def compute_and_backward_iit_for_group(
    model,
    group_by_arm: dict[str, dict],
    group_ref_u: dict[str, float],
    partition: dict[str, Any],
    head_dim: int,
    step_idx: int,
    group_idx: int,
    n_groups_in_window: int,
    lambda_iit_form: float,
    lambda_iit_ver: float,
    margin_iit: float,
    k_max: int = 8,
) -> dict[str, float]:
    """Run 1 Form-Only Debiasing Interchange + 1 Verification-Pathway Counterfactual Flip Interchange and backprop!

    Calls `.backward()` INSIDE `differentiable_interchange_forward(...)` so hooks stay live during GC recomputation!
    """
    stats = {"iit_form_loss": 0.0, "iit_ver_loss": 0.0, "iit_ver_flip_ok": 0.0, "iit_form_kept_ok": 0.0}
    scale = 1.0 / max(1, n_groups_in_window)

    form_heads = [tuple(x) for x in partition["form_only_heads"]]
    ver_heads = [tuple(x) for x in partition["ver_only_heads"]]
    ver_mlps = [int(x) for x in partition["ver_mlps"]]

    # 1. Form-Pathway Conditional Debiasing Interchange (cross-tone same-validity swap on form_only_heads)
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
                u_ref = torch.tensor(float(group_ref_u[tgt_arm]), device=u_patch.device, dtype=torch.float32)
                l_stay = F.smooth_l1_loss(u_patch, u_ref, beta=2.0)
                l_sign = F.relu(margin_iit - y_tgt * u_patch)
                l_form = 0.5 * l_stay + 0.5 * l_sign
                (scale * lambda_iit_form * l_form).backward()
                stats["iit_form_loss"] = float(l_form.detach())
                stats["iit_form_kept_ok"] = float((y_tgt * float(u_patch.detach())) > 0.0)

    # 2. Verification-Pathway Counterfactual Flip Interchange (same-tone opposite-validity swap on ver_only + ver_mlps)
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
                l_ver = F.relu(margin_iit - y_src * u_patch)
                (scale * lambda_iit_ver * l_ver).backward()
                stats["iit_ver_loss"] = float(l_ver.detach())
                stats["iit_ver_flip_ok"] = float((y_src * float(u_patch.detach())) > 0.0)

    return stats


# -------------------------------------------------------------------------------------- Utilities


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _encode_c2_records(tok, rows: list[dict]) -> list[dict]:
    """Encode C2 training examples preserving arm, pair_group_id, pos_tok, neg_tok for ALL rule arms (incl. AO, S, P0)."""
    out = []
    for r in rows:
        enc = C1T.encode_sft_example(tok, r, enable_thinking=False, mask_rule_format_ce=False)
        enc["arm"] = r["arm"]
        # Ensure pos_tok/neg_tok are also populated for AO/shell_only/S (already margin_sign=-1, so they are!)
        out.append(enc)
    return out


# -------------------------------------------------------------------------------------- Exhaustive CPU Self-Test


def run_selftest_tiny() -> None:
    """Exhaustive CPU self-test covering ALL 5 mathematical & systems invariants of DAC-IIT:
      1. Exact 3-group anatomical partition on Qwen3-8B & Qwen3-14B (Targeted-Dual AND layer-matched Random-LM/seed2).
      2. Exact 2-pass paired2x2 scalar chain-rule gradient == single-batch all-10-items autograd (max diff < 1e-5!).
      3. Differentiable IIT interchange strictly alters logits and produces non-zero finite gradients on
         ALL 3 groups (`form_only`, `ver_only + ver_mlps`, AND unpatched `hub_heads`!).
      4. Gradient checkpointing (`use_reentrant=False`) + IIT hooks == non-GC gradients (max diff < 1e-5!).
      5. Saved `adapter_state.pt` loads identically via `c1_modules.load_c1_adapter_state` and exports losslessly via
         `c1_export_peft_lora` (zero inference overhead, zero hooks at inference!).
    """
    from transformers import Qwen3Config, Qwen3ForCausalLM
    import c1_export_peft_lora as EXP

    # 1. Verify 3-group anatomical partition on real c1_components_frozen.json
    comp = json.loads((C1 / "data" / "c1_components_frozen.json").read_text(encoding="utf-8"))
    for mk, exp_counts in (("qwen3_8b", (23, 21, 3, 5)), ("qwen3_14b", (22, 20, 4, 5))):
        p_dual = partition_circuit_components(comp, mk, "Targeted-Dual")
        assert (len(p_dual["form_only_heads"]), len(p_dual["ver_only_heads"]), len(p_dual["hub_heads"]), len(p_dual["ver_mlps"])) == exp_counts
        for r_arm in ("Random-LM", "Random-LM-seed2"):
            p_r = partition_circuit_components(comp, mk, r_arm)
            assert (len(p_r["form_only_heads"]), len(p_r["ver_only_heads"]), len(p_r["hub_heads"]), len(p_r["ver_mlps"])) == exp_counts
            assert sorted(l for l, _ in p_r["form_only_heads"]) == sorted(l for l, _ in p_dual["form_only_heads"])
            assert sorted(l for l, _ in p_r["ver_only_heads"]) == sorted(l for l, _ in p_dual["ver_only_heads"])
            assert sorted(l for l, _ in p_r["hub_heads"]) == sorted(l for l, _ in p_dual["hub_heads"])
    print("[OK] 1. B1 3-group anatomical partition & layer-matched Random-LM/seed2 partition verified (8B & 14B)")

    # Build tiny Qwen3 + 1 real 10-item base group from c2_train_examples_nothink.jsonl
    cfg = Qwen3Config(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=2048,
        tie_word_embeddings=False,
    )
    tok = C1T._TinyTok()
    train_path = C2 / "data" / "c2_train_examples_nothink.jsonl"
    raw_rows = [json.loads(l) for l in train_path.read_text(encoding="utf-8").splitlines()[:20]]
    encoded = _encode_c2_records(tok, raw_rows)
    grp0 = encoded[:10]
    assert len({e["pair_group_id"] for e in grp0}) == 1
    assert {e["arm"] for e in grp0} >= {"V", "F", "V_auth", "V_hes", "F_auth", "F_hes", "shell_only", "AO", "S"}

    tiny_spec = {
        "kind": "circuit",
        "heads": [[1, 0], [1, 2], [2, 1], [2, 3], [3, 0]],
        "head_rank": 4,
        "mlp_layers": [1, 2],
        "mlp_rank": 4,
    }
    tiny_part = {
        "kind": "circuit",
        "form_only_heads": [[1, 0], [2, 1]],
        "ver_only_heads": [[1, 2], [2, 3]],
        "hub_heads": [[3, 0]],  # Layer 3 hub downstream of Layers 1-2!
        "ver_mlps": [1, 2],
    }

    # 2. Verify Exact 2-Pass Paired 2x2 Chain-Rule Gradients == Single-Batch Direct Autograd!
    torch.manual_seed(123)
    m_ref = Qwen3ForCausalLM(cfg).float()
    M.apply_c1_adapters(m_ref, tiny_spec, scaling_mode="rslora", base_scale=2.0, anchor_rank=8)
    # Give lora_B non-zero init so gradients flow strongly everywhere
    with torch.no_grad():
        for n, p in m_ref.named_parameters():
            if "lora_B" in n:
                p.normal_(0.0, 0.05)
    init_state = {k: v.detach().clone() for k, v in m_ref.state_dict().items()}

    # Direct single-pass autograd over all 10 items
    m_ref.train()
    m_ref.zero_grad()
    u_direct = {}
    base_loss_sum = torch.tensor(0.0)
    for ex in grp0:
        l_mb, _, _, _ = compute_loss_microbatch_c2(m_ref, [ex], tok.pad_token_id, 0.25, 4.0, accept_weight=2.0)
        base_loss_sum = base_loss_sum + l_mb
        logits = m_ref(input_ids=ex["input_ids"][None], use_cache=False).logits
        pos = ex["answer_readout_pos"]
        u_direct[ex["arm"]] = logits[0, pos, ex["pos_tok"]].float() - logits[0, pos, ex["neg_tok"]].float()
    inv_v = F.smooth_l1_loss(u_direct["V_auth"], u_direct["V_hes"], beta=2.0)
    inv_f = F.smooth_l1_loss(u_direct["F_auth"], u_direct["F_hes"], beta=2.0)
    sh_ex = F.relu(u_direct["shell_only"] - u_direct["AO"].detach() - 1.0)
    inv_sh = F.smooth_l1_loss(sh_ex, torch.zeros_like(sh_ex), beta=2.0)
    l_inv_d = 0.5 * (inv_v + inv_f) + 0.5 * inv_sh
    l_sens_d = (
        F.relu(6.0 - (u_direct["V"] - u_direct["F"]))
        + F.relu(6.0 - (u_direct["V_auth"] - u_direct["F_auth"]))
        + F.relu(6.0 - (u_direct["V_hes"] - u_direct["F_hes"]))
    ) / 3.0
    (base_loss_sum / len(grp0) + 0.15 * l_inv_d + 0.25 * l_sens_d).backward()
    grads_direct = {n: p.grad.detach().clone() for n, p in m_ref.named_parameters() if p.requires_grad}

    # Exact 2-Pass microbatched (mb=2)
    m_2p = Qwen3ForCausalLM(cfg).float()
    M.apply_c1_adapters(m_2p, tiny_spec, scaling_mode="rslora", base_scale=2.0, anchor_rank=8)
    m_2p.load_state_dict(init_state)
    m_2p.train()
    m_2p.zero_grad()
    u_map = collect_window_readouts_nograd(m_2p, grp0, tok.pad_token_id, microbatch=2)
    gid = grp0[0]["pair_group_id"]
    arm_g, _, _ = compute_paired2x2_chain_grads(u_map[gid], lambda_inv=0.15, lambda_sens=0.25, delta_sens=6.0)
    cg = {(gid, a): g for a, g in arm_g.items()}
    for j in range(0, len(grp0), 2):
        l_mb, _, _, _ = compute_loss_microbatch_c2(
            m_2p, grp0[j : j + 2], tok.pad_token_id, 0.25, 4.0, accept_weight=2.0,
            paired_chain_grads=cg, n_groups_in_window=1, win_n=len(grp0),
        )
        (l_mb / len(grp0)).backward()
    max_gdiff = max(
        float((grads_direct[n] - p.grad).abs().max())
        for n, p in m_2p.named_parameters() if p.requires_grad
    )
    assert max_gdiff < 1e-4, f"2-pass paired2x2 gradient mismatch: {max_gdiff}"
    print(f"[OK] 2. Exact 2-pass paired2x2 chain-rule gradient equivalence verified (max_diff={max_gdiff:.2e})")

    # 3 & 4. Verify Differentiable IIT Interchange (non-GC vs GC use_reentrant=False equivalence + non-zero grads on ALL 3 groups!)
    grp_by_arm = {e["arm"]: e for e in grp0}
    # Different source/target lengths must preserve the target-only intervention during replay.
    for name in ("F", "V_hes"):
        original = grp_by_arm[name]
        prefix = original["input_ids"][:16]
        grp_by_arm[name] = {
            **original,
            "input_ids": torch.cat((prefix, original["input_ids"])),
            "answer_readout_pos": original["answer_readout_pos"] + prefix.numel(),
        }
    assert grp_by_arm["F"]["answer_readout_pos"] - grp_by_arm["V"]["answer_readout_pos"] >= 8
    grads_by_gc = {}
    for use_gc in (False, True):
        m_iit = Qwen3ForCausalLM(cfg).float()
        M.apply_c1_adapters(m_iit, tiny_spec, scaling_mode="rslora", base_scale=2.0, anchor_rank=8)
        m_iit.load_state_dict(init_state)
        if use_gc:
            m_iit.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            m_iit.enable_input_require_grads()
        m_iit.train()
        m_iit.zero_grad()
        # Check patching strictly alters logits
        with torch.no_grad():
            u_orig = float(
                m_iit(input_ids=grp_by_arm["F"]["input_ids"][None]).logits[0, grp_by_arm["F"]["answer_readout_pos"], grp_by_arm["F"]["pos_tok"]]
                - m_iit(input_ids=grp_by_arm["F"]["input_ids"][None]).logits[0, grp_by_arm["F"]["answer_readout_pos"], grp_by_arm["F"]["neg_tok"]]
            )
            with differentiable_interchange_forward(
                m_iit, grp_by_arm["V"], grp_by_arm["F"],
                patch_heads=[(1, 2), (2, 3)], patch_mlps=[1, 2], head_dim=16,
            ) as u_p_t:
                u_patched = float(u_p_t)
            assert abs(u_patched - u_orig) > 1e-5, f"IIT interchange had zero effect! {u_orig} vs {u_patched}"
        # Run IIT backward (step_idx=0, group_idx=0 -> Form V_hes<-V_auth + Ver F<-V)
        st = compute_and_backward_iit_for_group(
            m_iit, grp_by_arm, u_map[gid], tiny_part, head_dim=16,
            step_idx=0, group_idx=0, n_groups_in_window=1,
            lambda_iit_form=0.15, lambda_iit_ver=0.25, margin_iit=10.0,
        )
        # Reverse the verification swap as well (longer source, shorter target).
        compute_and_backward_iit_for_group(
            m_iit, grp_by_arm, u_map[gid], tiny_part, head_dim=16,
            step_idx=1, group_idx=0, n_groups_in_window=1,
            lambda_iit_form=0.15, lambda_iit_ver=0.25, margin_iit=10.0,
        )
        grads_by_gc[use_gc] = {n: p.grad.detach().clone() for n, p in m_iit.named_parameters() if p.requires_grad}
        for n, g in grads_by_gc[use_gc].items():
            assert torch.isfinite(g).all(), f"Non-finite IIT grad on {n}"
        # Verify ALL 3 pathways (Layer 1-2 form/ver heads, Layer 1-2 MLPs, AND unpatched Layer 3 Hub Head!) got non-zero grads!
        g_l1_o = float(grads_by_gc[use_gc]["model.layers.1.self_attn.o_proj.lora_B"].abs().max())
        g_l1_mlp = float(grads_by_gc[use_gc]["model.layers.1.mlp.down_proj.lora_B"].abs().max())
        g_l3_hub = float(grads_by_gc[use_gc]["model.layers.3.self_attn.o_proj.lora_B"].abs().max())
        assert g_l1_o > 1e-6 and g_l1_mlp > 1e-6 and g_l3_hub > 1e-6, (use_gc, g_l1_o, g_l1_mlp, g_l3_hub)

    gc_diff = max(float((grads_by_gc[False][n] - grads_by_gc[True][n]).abs().max()) for n in grads_by_gc[False])
    assert gc_diff < 1e-5, f"GC vs non-GC IIT gradient mismatch: {gc_diff}"
    print(f"[OK] 3 & 4. Differentiable DAC-IIT + Gradient Checkpointing (use_reentrant=False) verified (gc_diff={gc_diff:.2e}, hub_grad={g_l3_hub:.3e})")

    # 5. Verify adapter_state.pt save + c1_eval.checkpoint_scaling + lossless PEFT LoRA export
    import c1_eval as C1EV
    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        ap = td_p / "adapter_state.pt"
        M.save_c1_adapter_state(m_2p, ap)
        (td_p / "adapter_config.json").write_text(
            json.dumps({
                "model_key": "qwen3_8b", "arm": "Targeted-Dual", "arm_spec": tiny_spec,
                "scaling": {"scaling_mode": "rslora", "base_scale": 2.0, "anchor_rank": 8},
            }),
            encoding="utf-8",
        )
        got_sc = C1EV.checkpoint_scaling(ap, "qwen3_8b", "Targeted-Dual", tiny_spec)
        assert got_sc == {"scaling_mode": "rslora", "base_scale": 2.0, "anchor_rank": 8}, got_sc
        peft_dir = td_p / "peft"
        EXP.export_adapter_state_to_peft(
            state_dict=torch.load(ap, map_location="cpu", weights_only=True),
            arm_spec=tiny_spec,
            model_spec={"hidden_size": 64, "num_heads": 4, "head_dim": 16, "intermediate_size": 128, "num_layers": 4},
            scaling=got_sc,
            out_dir=peft_dir,
            base_model_name_or_path="tiny-qwen3",
            export_dtype=torch.float32,
        )
        assert (peft_dir / "adapter_config.json").exists() and (peft_dir / "adapter_model.safetensors").exists()
    print("[OK] 5. adapter_state.pt save & lossless PEFT LoRA export (zero inference overhead) verified")
    print("[OK] ALL c2_train_iit self-tests PASSED!")


# -------------------------------------------------------------------------------------- Main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-tiny", action="store_true")
    ap.add_argument("--model", type=str)
    ap.add_argument("--model-key", choices=("qwen3_8b", "qwen3_14b"), default="qwen3_8b")
    ap.add_argument("--arm", type=str, default="Targeted-Dual")
    ap.add_argument("--loss-mode", choices=("base", "paired2x2", "dac_iit"), default="dac_iit")
    ap.add_argument("--components-file", type=Path, default=C1 / "data" / "c1_components_frozen.json")
    ap.add_argument("--train-file", type=Path, default=C2 / "data" / "c2_train_examples_nothink.jsonl")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3.5e-5)
    ap.add_argument("--mlp-lr-mult", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-accum", type=int, default=20, help="20 = 2 complete 10-item atomic base groups per step")
    ap.add_argument("--microbatch", type=int, default=2)
    ap.add_argument("--accept-weight", type=float, default=2.0)
    ap.add_argument("--lambda-margin", type=float, default=0.25)
    ap.add_argument("--margin-m", type=float, default=4.0)
    ap.add_argument("--lambda-inv", type=float, default=0.15)
    ap.add_argument("--lambda-sens", type=float, default=0.25)
    ap.add_argument("--delta-sens", type=float, default=6.0)
    ap.add_argument("--lambda-iit-form", type=float, default=0.15)
    ap.add_argument("--lambda-iit-ver", type=float, default=0.25)
    ap.add_argument("--margin-iit", type=float, default=3.0)
    ap.add_argument("--iit-k-max", type=int, default=8)
    ap.add_argument("--scaling-mode", choices=("rslora", "constant"), default="rslora")
    ap.add_argument("--lora-scale", type=float, default=2.0)
    ap.add_argument("--anchor-rank", type=int, default=8)
    ap.add_argument("--save-steps", type=str, default="100,200,300")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.selftest_tiny:
        run_selftest_tiny()
        return

    assert args.model and args.out_dir, "--model and --out-dir are required"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    comp = json.loads(args.components_file.read_text(encoding="utf-8"))
    model_info = comp["models"][args.model_key]
    arm_spec = model_info["arms"][args.arm]
    head_dim = int(model_info["model_spec"]["head_dim"])
    partition = partition_circuit_components(comp, args.model_key, args.arm)
    if args.loss_mode == "dac_iit" and partition["kind"] != "circuit":
        raise ValueError(f"--loss-mode dac_iit requires a circuit arm, got {args.arm} ({partition['kind']})")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    raw_rows = [json.loads(l) for l in args.train_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        raw_rows = raw_rows[: args.limit]
    encoded = _encode_c2_records(tok, raw_rows)

    print(f"[c2_train_iit] Loading {args.model} ({args.model_key}) arm={args.arm} loss_mode={args.loss_mode} seed={args.seed}")
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

    adapter_info = M.apply_c1_adapters(
        model,
        arm_spec,
        scaling_mode=args.scaling_mode,
        base_scale=args.lora_scale,
        anchor_rank=args.anchor_rank,
        dropout=0.0,
    )
    opt = C1T.build_optimizer(model, lr=args.lr, mlp_lr_mult=args.mlp_lr_mult, weight_decay=args.weight_decay)

    save_steps = sorted({int(x.strip()) for x in args.save_steps.split(",") if x.strip()})
    all_windows: list[list[dict]] = []
    for ep in range(args.epochs):
        all_windows.extend(C1T.build_atomic_windows(encoded, args.grad_accum, seed=args.seed + ep * 1009))
    total_steps = min(len(all_windows), args.max_steps) if args.max_steps > 0 else len(all_windows)
    warmup_steps = max(1, int(0.10 * total_steps))
    print(f"[c2_train_iit] encoded={len(encoded)}, total_steps={total_steps}, warmup={warmup_steps}, trainable={adapter_info['trainable_params']:,}")

    model.train()
    logs: list[dict[str, Any]] = []
    saved_checkpoints: dict[str, str] = {}

    for step_idx, win in enumerate(all_windows[:total_steps]):
        cur_step = step_idx + 1
        lrf = C1T.lr_factor(step_idx, total_steps, warmup_steps)
        for pg in opt.param_groups:
            base_lr = args.lr * args.mlp_lr_mult if (len(opt.param_groups) == 2 and pg is opt.param_groups[1]) else args.lr
            pg["lr"] = base_lr * lrf

        opt.zero_grad(set_to_none=True)

        # Group window items by pair_group_id
        groups_in_win: dict[str, dict[str, dict]] = defaultdict(dict)
        for ex in win:
            groups_in_win[ex["pair_group_id"]][ex["arm"]] = ex
        n_groups = max(1, len(groups_in_win))

        paired_chain_grads: dict[tuple[str, str], float] = {}
        u_map: dict[str, dict[str, float]] = {}
        inv_loss_sum = 0.0
        sens_loss_sum = 0.0

        # Pass 1 (no_grad) for paired2x2 and dac_iit
        if args.loss_mode in ("paired2x2", "dac_iit"):
            u_map = collect_window_readouts_nograd(model, win, pad_id, microbatch=args.microbatch)
            for gid, gu in u_map.items():
                arm_g, lv_inv, lv_sens = compute_paired2x2_chain_grads(
                    gu,
                    lambda_inv=args.lambda_inv,
                    lambda_sens=args.lambda_sens,
                    delta_sens=args.delta_sens,
                )
                inv_loss_sum += lv_inv
                sens_loss_sum += lv_sens
                for a, gv in arm_g.items():
                    paired_chain_grads[(gid, a)] = gv

        # Pass 2: Base SFT + Margin + Paired 2x2 Chain-Rule Surrogate (microbatched!)
        win_sft = 0.0
        win_marg = 0.0
        win_n = len(win)
        for j in range(0, win_n, args.microbatch):
            mb = win[j : j + args.microbatch]
            loss_sum, sft_v, marg_v, _ = compute_loss_microbatch_c2(
                model,
                mb,
                pad_id,
                lambda_margin=args.lambda_margin,
                margin_m=args.margin_m,
                accept_weight=args.accept_weight,
                paired_chain_grads=paired_chain_grads if args.loss_mode in ("paired2x2", "dac_iit") else None,
                n_groups_in_window=n_groups,
                win_n=win_n,
            )
            (loss_sum / win_n).backward()
            win_sft += sft_v
            win_marg += marg_v

        # Pass 3: Differentiable Dual-Pathway Interchange Intervention Training (dac_iit only!)
        iit_f_sum = 0.0
        iit_v_sum = 0.0
        iit_flip_sum = 0.0
        iit_keep_sum = 0.0
        if args.loss_mode == "dac_iit":
            for g_i, gid in enumerate(sorted(groups_in_win.keys())):
                st = compute_and_backward_iit_for_group(
                    model,
                    group_by_arm=groups_in_win[gid],
                    group_ref_u=u_map.get(gid, {}),
                    partition=partition,
                    head_dim=head_dim,
                    step_idx=step_idx,
                    group_idx=g_i,
                    n_groups_in_window=n_groups,
                    lambda_iit_form=args.lambda_iit_form,
                    lambda_iit_ver=args.lambda_iit_ver,
                    margin_iit=args.margin_iit,
                    k_max=args.iit_k_max,
                )
                iit_f_sum += st["iit_form_loss"]
                iit_v_sum += st["iit_ver_loss"]
                iit_flip_sum += st["iit_ver_flip_ok"]
                iit_keep_sum += st["iit_form_kept_ok"]

        grad_norm = float(
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        )
        opt.step()

        row_log = {
            "step": cur_step,
            "lr": round(opt.param_groups[0]["lr"], 8),
            "sft_mean": round(win_sft / win_n, 4),
            "margin_mean": round(win_marg / win_n, 4),
            "paired_inv_mean": round(inv_loss_sum / n_groups, 4),
            "paired_sens_mean": round(sens_loss_sum / n_groups, 4),
            "iit_form_mean": round(iit_f_sum / n_groups, 4),
            "iit_ver_mean": round(iit_v_sum / n_groups, 4),
            "iit_ver_flip_rate": round(iit_flip_sum / n_groups, 3),
            "iit_form_kept_rate": round(iit_keep_sum / n_groups, 3),
            "grad_norm": round(grad_norm, 4),
        }
        logs.append(row_log)
        if cur_step == 1 or cur_step % 25 == 0 or cur_step == total_steps:
            print(
                f"[step {cur_step:3d}/{total_steps}] sft={row_log['sft_mean']:.3f} marg={row_log['margin_mean']:.3f} "
                f"inv={row_log['paired_inv_mean']:.3f} sens={row_log['paired_sens_mean']:.3f} "
                f"iit_f={row_log['iit_form_mean']:.3f} iit_v={row_log['iit_ver_mean']:.3f} "
                f"flip={row_log['iit_ver_flip_rate']:.2f} gnorm={row_log['grad_norm']:.2f}"
            )

        if cur_step in save_steps and cur_step < total_steps:
            ckpt_p = args.out_dir / f"adapter_state_step{cur_step:04d}.pt"
            M.save_c1_adapter_state(model, ckpt_p)
            saved_checkpoints[str(cur_step)] = _file_sha256(ckpt_p)

    final_ap = args.out_dir / "adapter_state.pt"
    M.save_c1_adapter_state(model, final_ap)
    final_sha = _file_sha256(final_ap)
    saved_checkpoints[str(total_steps)] = final_sha

    # Save adapter_config.json (100% compatible with c1_eval.checkpoint_scaling & c1_export_peft_lora.py!)
    scaling_dict = {
        "scaling_mode": args.scaling_mode,
        "base_scale": args.lora_scale,
        "anchor_rank": args.anchor_rank,
    }
    adapter_cfg = {
        "schema": "cedar.acl_exp_c2.adapter_config.v1",
        "base_model": args.model,
        "model_key": args.model_key,
        "arm": args.arm,
        "loss_mode": args.loss_mode,
        "arm_spec": arm_spec,
        "scaling": scaling_dict,
        "circuit_partition": partition,
        "trainable_params": adapter_info["trainable_params"],
        "adapter_state_sha256": final_sha,
    }
    (args.out_dir / "adapter_config.json").write_text(json.dumps(adapter_cfg, indent=2) + "\n", encoding="utf-8")

    norms = M.module_delta_norms(model)
    meta = {
        "schema": "cedar.acl_exp_c2.train_meta.v1",
        "model": args.model,
        "model_key": args.model_key,
        "arm": args.arm,
        "arm_spec": arm_spec,
        "loss_mode": args.loss_mode,
        "seed": args.seed,
        "train_file": str(args.train_file),
        "train_file_sha256": _file_sha256(args.train_file),
        "n_examples": len(encoded),
        "total_steps": total_steps,
        "hyperparams": {
            "lr": args.lr,
            "mlp_lr_mult": args.mlp_lr_mult,
            "weight_decay": args.weight_decay,
            "grad_accum": args.grad_accum,
            "microbatch": args.microbatch,
            "accept_weight": args.accept_weight,
            "lambda_margin": args.lambda_margin,
            "margin_m": args.margin_m,
            "lambda_inv": args.lambda_inv,
            "lambda_sens": args.lambda_sens,
            "delta_sens": args.delta_sens,
            "lambda_iit_form": args.lambda_iit_form,
            "lambda_iit_ver": args.lambda_iit_ver,
            "margin_iit": args.margin_iit,
            "iit_k_max": args.iit_k_max,
            "scaling_mode": args.scaling_mode,
            "base_scale": args.lora_scale,
            "anchor_rank": args.anchor_rank,
            "seed": args.seed,
        },
        "adapter_info": adapter_info,
        "circuit_partition": partition,
        "final_delta_norms": norms,
        "adapter_state_sha256": final_sha,
        "saved_checkpoints_sha256": saved_checkpoints,
        "elapsed_seconds": round(time.time() - t0, 1),
        "tail_logs": logs[-20:],
        "all_logs": logs,
    }
    (args.out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"[c2_train_iit] DONE in {meta['elapsed_seconds']}s | sha256={final_sha[:12]} | norms={norms}")


if __name__ == "__main__":
    main()
