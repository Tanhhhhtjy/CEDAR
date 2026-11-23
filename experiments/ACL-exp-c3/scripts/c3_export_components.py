#!/usr/bin/env python3
"""Deterministically export frozen 3-zone M3-DAC architectures and decoupled IIT targets for ACL-exp-c3.

100% automated from committed B1 eligible raw tensors (ZERO hardcoded head IDs):
  - experiments/ACL-exp-b1/results/qwen3_8b_eligible/
  - experiments/ACL-exp-b1/results/qwen3_14b_eligible/

Writes:
  experiments/ACL-exp-c3/data/c3_components_frozen.json

Key Design Principles (incorporating C2 Forensic Audit & Astra Review):
  1. Decoupled Trainable Architecture vs IIT Intervention Target:
       - Trainable Architectures:
           * `Uniform-LoRA`     : All layers q_proj, o_proj, gate_proj, down_proj at r=4.
           * `M3-DAC`           : Three-zone mechanistic allocation matching Uniform-LoRA params (~±1%):
               - Zone 1 (Early Bridge): Dense Q/O LoRA r=2 (zero-init, low LR, proximal Delta-W reg).
               - Zone 2+3 MLPs (8 layers, r=8): Top-5 mid N_CF MLPs + Top-3 Zone-3 cross-half-stable dialogue MLPs.
               - Zone 2+3 Heads (per-head tiered ranks): Targeted-Dual union Zone-3 top-7 cross-half-stable |DLA| heads,
                 ranked by cross-half-stable composite causal score into 3 tiers; conflicting-sign heads clamped to lowest
                 conservative rank tier!
           * Pre-registered Wave 3.3 attribution controls:
               - `M3-Equal-Rank`   : Same M3-DAC locations, uniform head rank (isolates tiered budget vs location).
               - `M3-Shuffled-Rank`: Same M3-DAC locations, shuffled head rank tiers (isolates B1 ranking vs heterogeneity).
               - `M3-Random-Mask`  : Same Zone 1 + layer-matched disjoint random heads (identical per-layer rank multiset)
                                     + band-matched disjoint random MLPs.
       - Decoupled IIT Intervention Targets (applicable identically to BOTH `Uniform-LoRA` AND `M3-DAC`!):
           * `b1`         : B1 frozen (form_only, ver_only, hub) excluding conflicting-sign heads + fixed B1 `ver_mlps` (top-5 N_CF).
           * `rnd_head_1` : Layer-matched, count-matched disjoint random heads (seed=2031) with FIXED B1 `ver_mlps`
                            (cleanest unconfounded head-localization control for U3 vs U4 and M3 vs M4!).
           * `rnd_head_2` : Independent second random head mask (seed=2032) for seed=1024 replication.
           * `rnd_full_1` : Disjoint random heads + band-matched disjoint random MLPs (seed=2031) for follow-up MLP ablation.
  2. Hardware / vLLM Native LoRA Safety Guarantee:
       - Verifies that for EVERY layer in EVERY architecture, packed single-layer Q/O rank <= 128 (`vllm_max_lora_rank`).

Usage:
  python experiments/ACL-exp-c3/scripts/c3_export_components.py
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
C3 = HERE.parents[0]
REPO = C3.parents[1]
B1_RES = REPO / "experiments" / "ACL-exp-b1" / "results"
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-b1" / "scripts"))
import b1_lib as L  # noqa: E402


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "qwen3_8b": {
        "repo": "Qwen/Qwen3-8B",
        "num_layers": 36,
        "hidden_size": 4096,
        "num_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 12288,
        "b1_primary_dir": "qwen3_8b_eligible",
        "b1_full_dir": "qwen3_8b",
        "dialog_content_contrast": "C_content_AF",
        # Zone boundaries (inclusive layer indices)
        "zone1_layers": [0, 13],   # 14 early layers -> dense Q/O r=2
        "zone2_layers": [14, 27],  # 14 mid layers
        "zone3_layers": [28, 35],  # 8 top layers
        "uniform_rank": 4,
        "zone1_qo_rank": 2,
        "mlp_rank": 8,
        "top_z3_dla_k": 7,
        "top_z3_dlg_mlp_k": 3,
        # 3-tier causal rank allocation: (count_tier1, r1), (count_tier2, r2), (rest, r3)
        "tier1_count": 10,
        "tier1_rank": 20,
        "tier2_count": 13,
        "tier2_rank": 12,
        "tier3_rank": 6,
        "equal_head_rank": 10,
    },
    "qwen3_14b": {
        "repo": "Qwen/Qwen3-14B",
        "num_layers": 40,
        "hidden_size": 5120,
        "num_heads": 40,
        "num_kv_heads": 8,
        "head_dim": 128,
        "intermediate_size": 17408,
        "b1_primary_dir": "qwen3_14b_eligible",
        "b1_full_dir": "qwen3_14b",
        "dialog_content_contrast": "C_content_CF",
        "zone1_layers": [0, 15],   # 16 early layers -> dense Q/O r=2
        "zone2_layers": [16, 30],  # 15 mid layers
        "zone3_layers": [31, 39],  # 9 top layers
        "uniform_rank": 4,
        "zone1_qo_rank": 2,
        "mlp_rank": 8,
        "top_z3_dla_k": 7,
        "top_z3_dlg_mlp_k": 3,
        # Tuned to match Uniform-LoRA r=4 (~10.49M params) within ~1% with max_layer_rank <= 128
        "tier1_count": 10,
        "tier1_rank": 24,
        "tier2_count": 14,
        "tier2_rank": 14,
        "tier3_rank": 8,
        "equal_head_rank": 13,
    },
}


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _stable_pos(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.minimum(A.clamp(min=0.0), B.clamp(min=0.0))


def _stable_abs(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return torch.where((A * B) > 0.0, torch.minimum(A.abs(), B.abs()), torch.zeros_like(A))


def _count_dense_qo_params(n_layers: int, rank: int, d: int, H: int, hd: int) -> int:
    # q_proj [H*hd, d] + o_proj [d, H*hd]
    return n_layers * rank * ((H * hd + d) + (d + H * hd))


def _count_mlp_params(n_layers: int, rank: int, d: int, d_ff: int) -> int:
    # gate_proj [d_ff, d] + down_proj [d, d_ff]
    return n_layers * 2 * rank * (d + d_ff)


def _count_head_dict_params(head_ranks: dict[str, int], d: int, hd: int) -> int:
    # Per head with rank r_h: Q slice r_h*(d+hd) + O slice r_h*(d+hd) = 2 * r_h * (d+hd)
    return sum(2 * int(r) * (d + hd) for r in head_ranks.values())


def _count_uniform_params(nL: int, rank: int, d: int, H: int, hd: int, d_ff: int) -> int:
    return _count_dense_qo_params(nL, rank, d, H, hd) + _count_mlp_params(nL, rank, d, d_ff)


def _max_packed_layer_rank(
    head_ranks: dict[str, int],
    dense_qo_layers: list[int],
    dense_qo_rank: int,
    mlp_layers: list[int],
    mlp_rank: int,
) -> tuple[int, dict[int, int]]:
    per_layer_qo: dict[int, int] = defaultdict(int)
    for li in dense_qo_layers:
        per_layer_qo[int(li)] += int(dense_qo_rank)
    for key, r in head_ranks.items():
        li = int(key.split(":")[0])
        per_layer_qo[li] += int(r)
    max_qo = max(per_layer_qo.values()) if per_layer_qo else 0
    max_mlp = int(mlp_rank) if mlp_layers else 0
    return max(max_qo, max_mlp), dict(sorted(per_layer_qo.items()))


def _enforce_layer_rank_cap(
    head_ranks: dict[str, int],
    max_cap: int = 128,
) -> dict[str, int]:
    """Verify (and assert) per-layer packed Q/O head rank does not exceed vLLM max_lora_rank."""
    by_l: dict[int, list[str]] = defaultdict(list)
    for k in head_ranks:
        by_l[int(k.split(":")[0])].append(k)
    for li, ks in by_l.items():
        tot = sum(head_ranks[k] for k in ks)
        if tot > max_cap:
            raise AssertionError(
                f"Layer {li} packed Q/O rank {tot} exceeds vLLM cap {max_cap}: "
                f"{[(k, head_ranks[k]) for k in ks]}"
            )
    return head_ranks


def _sample_layer_matched_partition(
    b1_form: list[tuple[int, int]],
    b1_ver: list[tuple[int, int]],
    b1_hub: list[tuple[int, int]],
    H: int,
    exclude: set[tuple[int, int]],
    seed: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    """Sample disjoint random heads with EXACT per-layer counts for (form_only, ver_only, hub)."""
    rng = random.Random(seed)
    c_form: dict[int, int] = defaultdict(int)
    c_ver: dict[int, int] = defaultdict(int)
    c_hub: dict[int, int] = defaultdict(int)
    for li, _ in b1_form:
        c_form[li] += 1
    for li, _ in b1_ver:
        c_ver[li] += 1
    for li, _ in b1_hub:
        c_hub[li] += 1

    all_layers = sorted(set(c_form) | set(c_ver) | set(c_hub))
    r_form, r_ver, r_hub = [], [], []
    for li in all_layers:
        need = c_form[li] + c_ver[li] + c_hub[li]
        pool = [(li, h) for h in range(H) if (li, h) not in exclude]
        assert len(pool) >= need, f"Layer {li}: pool {len(pool)} < needed {need}"
        chosen = rng.sample(pool, need)
        r_form.extend(chosen[: c_form[li]])
        r_ver.extend(chosen[c_form[li] : c_form[li] + c_ver[li]])
        r_hub.extend(chosen[c_form[li] + c_ver[li] :])
    return sorted(r_form), sorted(r_ver), sorted(r_hub)


def build_model_c3_components(alias: str, spec: dict[str, Any]) -> dict[str, Any]:
    pdir = B1_RES / spec["b1_primary_dir"]
    fdir = B1_RES / spec["b1_full_dir"]
    d, H, hd, d_ff, nL = (
        spec["hidden_size"],
        spec["num_heads"],
        spec["head_dim"],
        spec["intermediate_size"],
        spec["num_layers"],
    )
    z1_lo, z1_hi = spec["zone1_layers"]
    z2_lo, z2_hi = spec["zone2_layers"]
    z3_lo, z3_hi = spec["zone3_layers"]
    assert z1_lo == 0 and z2_lo == z1_hi + 1 and z3_lo == z2_hi + 1 and z3_hi == nL - 1
    zone1_layers = list(range(z1_lo, z1_hi + 1))

    dlg_c = spec["dialog_content_contrast"]
    ncf = torch.load(pdir / "atp_C_content_N_CF.pt", map_location="cpu", weights_only=False)
    dlgt = torch.load(pdir / f"atp_{dlg_c}.pt", map_location="cpu", weights_only=False)
    frm = torch.load(pdir / "atp_C_form.pt", map_location="cpu", weights_only=False)
    soc = torch.load(pdir / "atp_C_soc.pt", map_location="cpu", weights_only=False)
    pin = torch.load(pdir / "atp_C_pinpoint.pt", map_location="cpu", weights_only=False)
    dla = torch.load(pdir / "dla_C_pinpoint.pt", map_location="cpu", weights_only=False)

    ncf_full = torch.load(fdir / "atp_C_content_N_CF.pt", map_location="cpu", weights_only=False)
    pin_full = torch.load(fdir / "atp_C_pinpoint.pt", map_location="cpu", weights_only=False)

    # 1. Original B1 Targeted-Dual heads & hub heads
    atp_pin_20 = L.top_k(pin["head_all"], 20)
    atp_form_20 = L.top_k(frm["head_all"], 20)
    atp_soc_12 = L.top_k(soc["head_all"], 12)
    atp_ncf_20 = L.top_k(ncf["head_all"], 20)
    atp_dlg_10 = L.top_k(dlgt["head_all"], 10)

    full_pin_20 = L.top_k(pin_full["head_all"], 20)
    full_ncf_20 = L.top_k(ncf_full["head_all"], 20)
    hub_heads = sorted((set(full_pin_20) & set(full_ncf_20)) | (set(atp_pin_20) & set(atp_ncf_20)))

    form_heads = sorted(set(atp_form_20) | set(atp_soc_12))
    dual_heads = sorted(set(form_heads) | set(atp_ncf_20) | set(atp_dlg_10))

    # 2. Automated Zone-3 Direct Readout Heads (top-k cross-half-stable |DLA_pinpoint| in Zone 3)
    s_dla = _stable_abs(dla["dla_A"], dla["dla_B"])
    dla_z3 = s_dla.clone()
    dla_z3[:z3_lo, :] = 0.0
    top_dla_z3 = L.top_k(dla_z3, spec["top_z3_dla_k"])

    union_heads = sorted(set(dual_heads) | set(top_dla_z3))
    # Assert clean zone separation: zero head-slice overlap with Zone 1 dense Q/O layers
    z1_overlap = [h for h in union_heads if h[0] <= z1_hi]
    assert not z1_overlap, f"Unexpected Zone-1 overlap in union_heads: {z1_overlap}"

    # 3. Automated Zone 2+3 MLPs: Top-5 mid N_CF + Top-3 Zone-3 cross-half-stable dialogue MLPs
    mlp_ncf_5 = sorted(int(li) for li, _ in L.top_k(ncf["mlp_B"].view(nL, 1), 5))
    mlp_dlg_st = _stable_pos(dlgt["mlp_A"], dlgt["mlp_B"]).view(nL, 1)
    mlp_dlg_z3 = mlp_dlg_st.clone()
    mlp_dlg_z3[:z3_lo, :] = 0.0
    for li in mlp_ncf_5:
        mlp_dlg_z3[li, 0] = 0.0
    mlp_dlg_3 = sorted(int(li) for li, _ in L.top_k(mlp_dlg_z3, spec["top_z3_dlg_mlp_k"]))
    m3_mlps = sorted(set(mlp_ncf_5) | set(mlp_dlg_3))
    assert len(m3_mlps) == 5 + spec["top_z3_dlg_mlp_k"]

    # 4. Cross-Half-Stable Composite Importance & Automated Evidence-Conflict Flag
    s_ncf = _stable_pos(ncf["head_A"], ncf["head_B"])
    s_dlg = _stable_pos(dlgt["head_A"], dlgt["head_B"])
    s_frm = _stable_abs(frm["head_A"], frm["head_B"])
    s_soc = _stable_abs(soc["head_A"], soc["head_B"])

    comp_score = (
        s_ncf / (s_ncf.max().clamp(min=1e-8))
        + 0.70 * s_dlg / (s_dlg.max().clamp(min=1e-8))
        + 0.70 * s_frm / (s_frm.max().clamp(min=1e-8))
        + 0.50 * s_soc / (s_soc.max().clamp(min=1e-8))
        + 0.50 * s_dla / (s_dla.max().clamp(min=1e-8))
    )

    head_diagnostics: dict[str, dict[str, Any]] = {}
    conflicting_heads: set[tuple[int, int]] = set()
    for li, h in union_heads:
        dla_a, dla_b, dla_all = float(dla["dla_A"][li, h]), float(dla["dla_B"][li, h]), float(dla["dla_all"][li, h])
        frm_a, frm_b, frm_all = float(frm["head_A"][li, h]), float(frm["head_B"][li, h]), float(frm["head_all"][li, h])
        ncf_a, ncf_b, ncf_all = float(ncf["head_A"][li, h]), float(ncf["head_B"][li, h]), float(ncf["head_all"][li, h])
        dlg_a, dlg_b, dlg_all = float(dlgt["head_A"][li, h]), float(dlgt["head_B"][li, h]), float(dlgt["head_all"][li, h])
        soc_a, soc_b, soc_all = float(soc["head_A"][li, h]), float(soc["head_B"][li, h]), float(soc["head_all"][li, h])

        conflict_reasons = []
        # Rule A: Strong opposite-sign polarity between DLA_pinpoint and AtP_form (e.g. 8B L34H30)
        if (dla_all * frm_all < 0.0) and (float(s_dla[li, h]) >= 1.0) and (float(s_frm[li, h]) >= 0.05):
            conflict_reasons.append("dla_vs_form_opposite_sign")
        # Rule B: Strong opposite-sign polarity between N_CF and dialogue content
        if (ncf_all * dlg_all < 0.0) and (float(_stable_abs(ncf["head_A"], ncf["head_B"])[li, h]) >= 0.05) and (
            float(_stable_abs(dlgt["head_A"], dlgt["head_B"])[li, h]) >= 0.05
        ):
            conflict_reasons.append("ncf_vs_dialog_opposite_sign")
        # Rule C: Cross-half sign flip across ALL five contrasts
        any_stable = any(
            x > 1e-6
            for x in (
                float(s_ncf[li, h]),
                float(s_dlg[li, h]),
                float(s_frm[li, h]),
                float(s_soc[li, h]),
                float(s_dla[li, h]),
            )
        )
        if not any_stable:
            conflict_reasons.append("no_cross_half_stable_signal")

        is_conf = len(conflict_reasons) > 0
        if is_conf:
            conflicting_heads.add((li, h))

        head_diagnostics[f"{li}:{h}"] = {
            "zone": 2 if li <= z2_hi else 3,
            "composite_score": round(float(comp_score[li, h]), 5),
            "is_hub": (li, h) in set(hub_heads),
            "in_dual": (li, h) in set(dual_heads),
            "in_top_z3_dla": (li, h) in set(top_dla_z3),
            "conflicting": is_conf,
            "conflict_reasons": conflict_reasons,
            "ncf_AB": [round(ncf_a, 4), round(ncf_b, 4)],
            "dlg_AB": [round(dlg_a, 4), round(dlg_b, 4)],
            "form_AB": [round(frm_a, 4), round(frm_b, 4)],
            "soc_AB": [round(soc_a, 4), round(soc_b, 4)],
            "dla_AB": [round(dla_a, 3), round(dla_b, 3)],
        }

    # 5. Automated 3-Tier Rank Assignment (conflicting heads clamped to lowest conservative tier!)
    non_conf = sorted(
        [x for x in union_heads if x not in conflicting_heads],
        key=lambda x: (-float(comp_score[x[0], x[1]]), x[0], x[1]),
    )
    t1_n, t2_n = spec["tier1_count"], spec["tier2_count"]
    r1, r2, r3 = spec["tier1_rank"], spec["tier2_rank"], spec["tier3_rank"]

    m3_head_ranks: dict[str, int] = {}
    for idx, (li, h) in enumerate(non_conf):
        r = r1 if idx < t1_n else (r2 if idx < (t1_n + t2_n) else r3)
        m3_head_ranks[f"{li}:{h}"] = r
        head_diagnostics[f"{li}:{h}"]["tier"] = 1 if idx < t1_n else (2 if idx < (t1_n + t2_n) else 3)
        head_diagnostics[f"{li}:{h}"]["rank"] = r
    for li, h in sorted(conflicting_heads):
        m3_head_ranks[f"{li}:{h}"] = r3
        head_diagnostics[f"{li}:{h}"]["tier"] = 3
        head_diagnostics[f"{li}:{h}"]["rank"] = r3

    m3_head_ranks = dict(sorted(m3_head_ranks.items(), key=lambda kv: tuple(int(x) for x in kv[0].split(":"))))
    _enforce_layer_rank_cap(m3_head_ranks, max_cap=128)

    # Compute exact budgets
    unif_r = spec["uniform_rank"]
    unif_params = _count_uniform_params(nL, unif_r, d, H, hd, d_ff)
    z1_r = spec["zone1_qo_rank"]
    mlp_r = spec["mlp_rank"]
    m3_params = (
        _count_dense_qo_params(len(zone1_layers), z1_r, d, H, hd)
        + _count_mlp_params(len(m3_mlps), mlp_r, d, d_ff)
        + _count_head_dict_params(m3_head_ranks, d, hd)
    )
    m3_max_r, m3_layer_qo = _max_packed_layer_rank(m3_head_ranks, zone1_layers, z1_r, m3_mlps, mlp_r)

    # 6. Pre-registered Wave 3.3 Attribution Controls (Equal-Rank, Shuffled-Rank, Random-Mask)
    eq_r = spec["equal_head_rank"]
    m3_eq_head_ranks = {k: eq_r for k in m3_head_ranks}
    _enforce_layer_rank_cap(m3_eq_head_ranks, max_cap=128)
    m3_eq_params = (
        _count_dense_qo_params(len(zone1_layers), z1_r, d, H, hd)
        + _count_mlp_params(len(m3_mlps), mlp_r, d, d_ff)
        + _count_head_dict_params(m3_eq_head_ranks, d, hd)
    )

    # Shuffled-Rank: shuffle rank tiers across the SAME union_heads while keeping per-layer sum <= 128
    rng_shuf = random.Random(2033)
    keys_list = list(m3_head_ranks.keys())
    vals_list = [m3_head_ranks[k] for k in keys_list]
    m3_shuf_head_ranks: dict[str, int] = {}
    for attempt in range(200):
        cand_vals = list(vals_list)
        rng_shuf.shuffle(cand_vals)
        cand = dict(zip(keys_list, cand_vals))
        try:
            _enforce_layer_rank_cap(cand, max_cap=128)
            m3_shuf_head_ranks = cand
            break
        except AssertionError:
            continue
    assert m3_shuf_head_ranks, "Failed to find <=128 shuffled rank permutation"

    # Random-Mask: same Zone 1 r=2, band-matched disjoint random 8 MLPs r=8,
    # and layer-matched disjoint random heads inheriting the EXACT per-layer rank multiset!
    rng_rm = random.Random(2034)
    z2_mlp_pool = [li for li in range(z2_lo, z2_hi + 1) if li not in set(m3_mlps)]
    z3_mlp_pool = [li for li in range(z3_lo, z3_hi + 1) if li not in set(m3_mlps)]
    rm_mlps = sorted(rng_rm.sample(z2_mlp_pool, len(mlp_ncf_5)) + rng_rm.sample(z3_mlp_pool, len(mlp_dlg_3)))
    by_l_ranks: dict[int, list[int]] = defaultdict(list)
    for k, r in m3_head_ranks.items():
        by_l_ranks[int(k.split(":")[0])].append(r)
    union_set = set(union_heads)
    rm_head_ranks: dict[str, int] = {}
    for li in sorted(by_l_ranks):
        rs = by_l_ranks[li]
        pool = [h for h in range(H) if (li, h) not in union_set]
        chosen = sorted(rng_rm.sample(pool, len(rs)))
        for h, r in zip(chosen, rs):
            rm_head_ranks[f"{li}:{h}"] = r
    _enforce_layer_rank_cap(rm_head_ranks, max_cap=128)
    rm_params = (
        _count_dense_qo_params(len(zone1_layers), z1_r, d, H, hd)
        + _count_mlp_params(len(rm_mlps), mlp_r, d, d_ff)
        + _count_head_dict_params(rm_head_ranks, d, hd)
    )
    assert rm_params == m3_params, "M3-Random-Mask param count must match M3-DAC identically!"

    # 7. Decoupled IIT Intervention Targets (shared identically by Uniform AND M3-DAC!)
    # Exclude conflicting-sign heads from aggressive IIT swapping (keep them conservative!)
    b1_hub_iit = sorted(set(hub_heads) - conflicting_heads)
    b1_form_iit = sorted((set(form_heads) - set(hub_heads)) - conflicting_heads)
    b1_ver_iit = sorted((set(dual_heads) - set(form_heads) - set(hub_heads)) - conflicting_heads)
    assert not (set(b1_hub_iit) & set(b1_form_iit))
    assert not (set(b1_hub_iit) & set(b1_ver_iit))
    assert not (set(b1_form_iit) & set(b1_ver_iit))

    # Layer-matched disjoint random IIT head targets with FIXED B1 ver_mlps (Astra's pure head-localization test!)
    r1_form, r1_ver, r1_hub = _sample_layer_matched_partition(
        b1_form_iit, b1_ver_iit, b1_hub_iit, H, exclude=union_set, seed=2031
    )
    r2_form, r2_ver, r2_hub = _sample_layer_matched_partition(
        b1_form_iit, b1_ver_iit, b1_hub_iit, H, exclude=union_set, seed=2032
    )
    # Band-matched disjoint random MLPs for optional full-random MLP+Head ablation
    lo_m, hi_m = max(z2_lo, min(mlp_ncf_5) - 4), min(z2_hi, max(mlp_ncf_5) + 4)
    mlp_band_pool = [li for li in range(lo_m, hi_m + 1) if li not in set(mlp_ncf_5)]
    rnd_mlps_1 = sorted(random.Random(2031).sample(mlp_band_pool, len(mlp_ncf_5)))
    rnd_mlps_2 = sorted(random.Random(2032).sample(mlp_band_pool, len(mlp_ncf_5)))

    iit_targets = {
        "b1": {
            "description": "B1 frozen non-conflicting (form_only, ver_only, hub) heads + B1 verified mid-layer N_CF top-5 MLPs",
            "form_only_heads": [[li, h] for li, h in b1_form_iit],
            "ver_only_heads": [[li, h] for li, h in b1_ver_iit],
            "hub_heads": [[li, h] for li, h in b1_hub_iit],
            "ver_mlps": list(mlp_ncf_5),
        },
        "rnd_head_1": {
            "description": "Primary seed=42 matched random-head IIT (seed=2031, disjoint from B1, FIXED B1 ver_mlps for pure head localization)",
            "form_only_heads": [[li, h] for li, h in r1_form],
            "ver_only_heads": [[li, h] for li, h in r1_ver],
            "hub_heads": [[li, h] for li, h in r1_hub],
            "ver_mlps": list(mlp_ncf_5),
        },
        "rnd_head_2": {
            "description": "Independent seed=1024 matched random-head IIT (seed=2032, disjoint from B1, FIXED B1 ver_mlps)",
            "form_only_heads": [[li, h] for li, h in r2_form],
            "ver_only_heads": [[li, h] for li, h in r2_ver],
            "hub_heads": [[li, h] for li, h in r2_hub],
            "ver_mlps": list(mlp_ncf_5),
        },
        "rnd_full_1": {
            "description": "Follow-up full-random IIT (disjoint random heads + band-matched disjoint random MLPs, seed=2031)",
            "form_only_heads": [[li, h] for li, h in r1_form],
            "ver_only_heads": [[li, h] for li, h in r1_ver],
            "hub_heads": [[li, h] for li, h in r1_hub],
            "ver_mlps": list(rnd_mlps_1),
        },
        "rnd_full_2": {
            "description": "Follow-up full-random IIT seed #2 (seed=2032)",
            "form_only_heads": [[li, h] for li, h in r2_form],
            "ver_only_heads": [[li, h] for li, h in r2_ver],
            "hub_heads": [[li, h] for li, h in r2_hub],
            "ver_mlps": list(rnd_mlps_2),
        },
    }

    architectures = {
        "Uniform-LoRA": {
            "kind": "uniform",
            "description": f"All {nL} layers dense LoRA on q_proj, o_proj, gate_proj, down_proj at r={unif_r}",
            "uniform_rank": unif_r,
            "target_modules": ["q_proj", "o_proj", "gate_proj", "down_proj"],
            "zones": {
                "zone1": [z1_lo, z1_hi],
                "zone2": [z2_lo, z2_hi],
                "zone3": [z3_lo, z3_hi],
            },
            "max_packed_layer_rank": unif_r,
            "trainable_params": unif_params,
        },
        "M3-DAC": {
            "kind": "m3_dac",
            "description": (
                f"Three-Zone Mechanistic Adaptation: Zone 1 L{z1_lo}-{z1_hi} dense Q/O r={z1_r}; "
                f"Zone 2+3 8 MLPs (5 mid N_CF + 3 top dialogue) r={mlp_r}; "
                f"{len(m3_head_ranks)} B1+DLA heads with cross-half-stable 3-tier ranks {r1}/{r2}/{r3} "
                f"(conflicting heads clamped to r={r3})"
            ),
            "zones": {
                "zone1": [z1_lo, z1_hi],
                "zone2": [z2_lo, z2_hi],
                "zone3": [z3_lo, z3_hi],
            },
            "zone1_dense_qo_layers": zone1_layers,
            "zone1_dense_qo_rank": z1_r,
            "mlp_layers": m3_mlps,
            "mlp_ncf_layers": mlp_ncf_5,
            "mlp_dialog_z3_layers": mlp_dlg_3,
            "mlp_rank": mlp_r,
            "head_ranks": m3_head_ranks,
            "conflicting_heads": [[li, h] for li, h in sorted(conflicting_heads)],
            "max_packed_layer_rank": m3_max_r,
            "per_layer_packed_qo_rank": {str(k): v for k, v in m3_layer_qo.items()},
            "trainable_params": m3_params,
        },
        "M3-Equal-Rank": {
            "kind": "m3_dac",
            "description": f"Attribution control: identical M3-DAC components with equal head rank r={eq_r}",
            "zones": {"zone1": [z1_lo, z1_hi], "zone2": [z2_lo, z2_hi], "zone3": [z3_lo, z3_hi]},
            "zone1_dense_qo_layers": zone1_layers,
            "zone1_dense_qo_rank": z1_r,
            "mlp_layers": m3_mlps,
            "mlp_rank": mlp_r,
            "head_ranks": m3_eq_head_ranks,
            "max_packed_layer_rank": _max_packed_layer_rank(m3_eq_head_ranks, zone1_layers, z1_r, m3_mlps, mlp_r)[0],
            "trainable_params": m3_eq_params,
        },
        "M3-Shuffled-Rank": {
            "kind": "m3_dac",
            "description": "Attribution control: identical M3-DAC components with 3-tier head ranks randomly shuffled (seed=2033)",
            "zones": {"zone1": [z1_lo, z1_hi], "zone2": [z2_lo, z2_hi], "zone3": [z3_lo, z3_hi]},
            "zone1_dense_qo_layers": zone1_layers,
            "zone1_dense_qo_rank": z1_r,
            "mlp_layers": m3_mlps,
            "mlp_rank": mlp_r,
            "head_ranks": m3_shuf_head_ranks,
            "max_packed_layer_rank": _max_packed_layer_rank(m3_shuf_head_ranks, zone1_layers, z1_r, m3_mlps, mlp_r)[0],
            "trainable_params": m3_params,
        },
        "M3-Random-Mask": {
            "kind": "m3_dac",
            "description": "Attribution control: same Zone 1 r=2 + layer-matched disjoint random heads (same per-layer rank multiset) + band-matched disjoint random MLPs",
            "zones": {"zone1": [z1_lo, z1_hi], "zone2": [z2_lo, z2_hi], "zone3": [z3_lo, z3_hi]},
            "zone1_dense_qo_layers": zone1_layers,
            "zone1_dense_qo_rank": z1_r,
            "mlp_layers": rm_mlps,
            "mlp_rank": mlp_r,
            "head_ranks": rm_head_ranks,
            "max_packed_layer_rank": _max_packed_layer_rank(rm_head_ranks, zone1_layers, z1_r, rm_mlps, mlp_r)[0],
            "trainable_params": rm_params,
        },
    }

    src_hashes = {}
    for fn in (
        "atp_C_pinpoint.pt",
        "atp_C_soc.pt",
        "atp_C_form.pt",
        "atp_C_content_N_CF.pt",
        f"atp_{dlg_c}.pt",
        "dla_C_pinpoint.pt",
    ):
        p = pdir / fn
        if p.exists():
            src_hashes[f"{spec['b1_primary_dir']}/{fn}"] = _sha256(p)

    return {
        "model_spec": spec,
        "uniform_anchor_params": unif_params,
        "source_b1_sha256": src_hashes,
        "architectures": architectures,
        "iit_targets": iit_targets,
        "head_diagnostics": head_diagnostics,
    }


def main() -> None:
    payload = {
        "schema": "cedar.acl_exp_c3.components.v1",
        "models": {alias: build_model_c3_components(alias, spec) for alias, spec in MODEL_SPECS.items()},
    }
    out = C3 / "data" / "c3_components_frozen.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"[OK] wrote {out} ({out.stat().st_size:,} bytes)")
    for alias, m in payload["models"].items():
        anc = m["uniform_anchor_params"]
        print(f"\n=== {alias} (Uniform r=4 anchor = {anc:,} params) ===")
        for aname, a in m["architectures"].items():
            p = a["trainable_params"]
            rel = (p / anc - 1.0) * 100.0
            mr = a["max_packed_layer_rank"]
            extra = (
                f"Z1={len(a['zone1_dense_qo_layers'])}@r{a['zone1_dense_qo_rank']} "
                f"MLPs={a['mlp_layers']}@r{a['mlp_rank']} "
                f"Heads={len(a['head_ranks'])} max_layer_r={mr}"
                if a["kind"] == "m3_dac"
                else f"all_layers_r={a['uniform_rank']} max_layer_r={mr}"
            )
            print(f"  ARCH {aname:16s}: {p:10,d} ({rel:+5.2f}%) | {extra}")
        m3 = m["architectures"]["M3-DAC"]
        print(f"  Conflicting heads (clamped to lowest tier & excluded from IIT): {m3['conflicting_heads']}")
        for tname, t in m["iit_targets"].items():
            print(
                f"  IIT  {tname:12s}: form={len(t['form_only_heads'])} ver={len(t['ver_only_heads'])} "
                f"hub={len(t['hub_heads'])} mlps={t['ver_mlps']}"
            )


if __name__ == "__main__":
    main()
