#!/usr/bin/env python3
"""Deterministically export frozen ACL-exp-c3.5 architectures (`Uniform-LoRA`, `M3-v1`, `M3-v2`) & decoupled IIT targets.

100% automated from committed B1 eligible raw tensors (ZERO hardcoded head IDs!) + Probe 1 Sub-Projection KL evidence:
  - Stage 1 Probe 1 (`results/stage1/probe1_subproj/SUBPROJECTION_PROBE.md`) revealed:
      1. Zone 3 (`L28-35`) MLPs contributed 0.0 pp on `c3_dev` and IMPROVED Row B when disabled (4->5/30 on m2, 5->6/30 on u2).
      2. Zone 1 (`L0-13`) dense Q/O alone drove 24.2% of WikiText-2 KL on m2 for only 0.5 pp dev / 0.0 pp Step-3,
         whereas Zone-1 MLP drove +10.0 pp dev, +10/36 Row B, and +22.5 pp Step-3 on u2!
      3. Zone 2 (`L14-27`) MLPs are the highest-leverage / lowest-KL component in the entire model
         (+26.7 pp dev, +13/36 Row B, +42.5 pp Step-3 for ONLY 1.4% KL on u2; +30.5 pp dev for 3.5% KL on m2!).
  - Locked `M3-v2` allocation (within ±1% of `Uniform-LoRA` on BOTH 8B and 14B, `max_packed_layer_rank <= 128`):
      * Early Zone 1 (`L0-7` on 8B / `L0-8` on 14B): 100% FROZEN (`0` params -> protects early lexical/syntactic Wiki KL!).
      * Late Zone 1 Bridge (`L8-13` on 8B / `L9-15` on 14B): Dense Q/O `r=2` + Dense MLP `r=4` (captures Row B & Step-3 bridge!).
      * Full Zone 2 Core (`L14-27` on 8B / `L16-30` on 14B):
          - ALL Zone-2 MLPs active: B1 top-5 `N_CF` MLPs `@ r=8` + remaining Zone-2 MLPs `@ r=4`!
          - All B1 Zone-2 heads with cross-half-stable 3-tier ranks (`r1/r2/r3`, conflicting heads clamped to `r3`).
      * Minimal Zone 3 Readout (`L28-35` on 8B / `L31-39` on 14B):
          - ZERO Zone-3 MLPs (`0` params, dropped `[29, 32, 35]`!).
          - B1/DLA Zone-3 heads clamped to conservative `r=4` (keeps Zone-3 `form_only_heads` trainable for B1-IIT at minimal cost).

Writes:
  experiments/ACL-exp-c3.5/data/c35_components_frozen.json

Usage:
  python experiments/ACL-exp-c3.5/scripts/c35_export_components.py
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
C35 = HERE.parents[0]
REPO = C35.parents[1]
C3 = REPO / "experiments" / "ACL-exp-c3"

sys.path.insert(0, str(C3 / "scripts"))
import c3_export_components as EXP3C  # noqa: E402

M3V2_SPECS: dict[str, dict[str, Any]] = {
    "qwen3_8b": {
        "late_z1_layers": [8, 13],  # Upper 6 layers of Zone 1; L0-7 completely frozen!
        "late_z1_qo_rank": 2,
        "late_z1_mlp_rank": 4,
        "z2_top_mlp_rank": 8,       # Top-5 B1 N_CF MLPs
        "z2_rest_mlp_rank": 4,      # Remaining 9 Zone-2 MLPs
        "z3_head_rank": 4,          # Minimal readout tier for Zone-3 B1/DLA heads
        "z2_tier1_count": 10,
        "z2_tier1_rank": 16,
        "z2_tier2_count": 10,
        "z2_tier2_rank": 10,
        "z2_tier3_rank": 6,
        # Stage 3 M3-v3 KL-Surgical specs
        "v3_puremid_z2_top_mlp_rank": 10,
        "v3_puremid_z2_rest_mlp_rank": 6,
        "v3_tight_z1_layers": [10, 13],  # 4 layers MLP-only, ZERO Z1 Q/O, L0-9 frozen!
        "v3_tight_z1_mlp_rank": 4,
        "v3_tight_z2_top_mlp_rank": 10,
        "v3_tight_z2_rest_mlp_rank": 4,
        "v3_z2_tier1_count": 10,
        "v3_z2_tier1_rank": 18,
        "v3_z2_tier2_count": 12,
        "v3_z2_tier2_rank": 10,
        "v3_z2_tier3_rank": 6,
        "v3_z3_form_head_rank": 2,
        # Stage 4 M3-v4 specs (100% pure Zone-2 High-MLP & 2-layer L12-13 Bridge, ZERO Zone-3!)
        "v4_mlpplus_z2_top_mlp_rank": 12,
        "v4_mlpplus_z2_rest_mlp_rank": 8,
        "v4_mlpplus_t1_r": 14,
        "v4_mlpplus_t2_r": 8,
        "v4_mlpplus_t3_r": 4,
        "v4_bridge2_z1_layers": [12, 13],  # ONLY 2 late layers MLP-only @ r=4!
        "v4_bridge2_z1_mlp_rank": 4,
        "v4_bridge2_z2_top_mlp_rank": 10,
        "v4_bridge2_z2_rest_mlp_rank": 6,
        "v4_bridge2_t1_r": 16,
        "v4_bridge2_t2_r": 10,
        "v4_bridge2_t3_r": 6,
    },
    "qwen3_14b": {
        "late_z1_layers": [9, 15],  # Upper 7 layers of Zone 1; L0-8 completely frozen!
        "late_z1_qo_rank": 2,
        "late_z1_mlp_rank": 4,
        "z2_top_mlp_rank": 8,       # Top-5 B1 N_CF MLPs
        "z2_rest_mlp_rank": 4,      # Remaining 10 Zone-2 MLPs
        "z3_head_rank": 4,
        "z2_tier1_count": 10,
        "z2_tier1_rank": 20,
        "z2_tier2_count": 12,
        "z2_tier2_rank": 12,
        "z2_tier3_rank": 6,
        # Stage 3 M3-v3 KL-Surgical specs
        "v3_puremid_z2_top_mlp_rank": 10,
        "v3_puremid_z2_rest_mlp_rank": 6,
        "v3_tight_z1_layers": [11, 15],  # 5 layers MLP-only, ZERO Z1 Q/O, L0-10 frozen!
        "v3_tight_z1_mlp_rank": 4,
        "v3_tight_z2_top_mlp_rank": 10,
        "v3_tight_z2_rest_mlp_rank": 4,
        "v3_z2_tier1_count": 10,
        "v3_z2_tier1_rank": 22,
        "v3_z2_tier2_count": 12,
        "v3_z2_tier2_rank": 12,
        "v3_z2_tier3_rank": 7,
        "v3_z3_form_head_rank": 2,
        # Stage 4 M3-v4 specs (14B mirror)
        "v4_mlpplus_z2_top_mlp_rank": 12,
        "v4_mlpplus_z2_rest_mlp_rank": 8,
        "v4_mlpplus_t1_r": 16,
        "v4_mlpplus_t2_r": 10,
        "v4_mlpplus_t3_r": 5,
        "v4_bridge2_z1_layers": [14, 15],  # ONLY 2 late layers MLP-only @ r=4!
        "v4_bridge2_z1_mlp_rank": 4,
        "v4_bridge2_z2_top_mlp_rank": 10,
        "v4_bridge2_z2_rest_mlp_rank": 6,
        "v4_bridge2_t1_r": 20,
        "v4_bridge2_t2_r": 12,
        "v4_bridge2_t3_r": 6,
    },
}


def _count_mlp_dict_params(mlp_ranks: dict[str, int], d: int, d_ff: int) -> int:
    return sum(2 * int(r) * (d + d_ff) for r in mlp_ranks.values())


def _max_packed_rank_v2(
    head_ranks: dict[str, int],
    dense_qo_layers: list[int],
    dense_qo_rank: int,
    mlp_ranks: dict[str, int],
) -> tuple[int, dict[int, int]]:
    per_layer_qo: dict[int, int] = defaultdict(int)
    for li in dense_qo_layers:
        per_layer_qo[int(li)] += int(dense_qo_rank)
    for k, r in head_ranks.items():
        per_layer_qo[int(k.split(":")[0])] += int(r)
    max_qo = max(per_layer_qo.values(), default=0)
    max_mlp = max((int(v) for v in mlp_ranks.values()), default=0)
    return max(max_qo, max_mlp), dict(sorted(per_layer_qo.items()))


def build_c35_model_entry(alias: str, c3_entry: dict[str, Any], v2_cfg: dict[str, Any]) -> dict[str, Any]:
    ms = c3_entry["model_spec"]
    d, H, hd, d_ff = ms["hidden_size"], ms["num_heads"], ms["head_dim"], ms["intermediate_size"]
    m3v1 = c3_entry["architectures"]["M3-DAC"]
    unif = c3_entry["architectures"]["Uniform-LoRA"]
    hd_diag = c3_entry["head_diagnostics"]

    zones = m3v1["zones"]
    z1_lo, z1_hi = zones["zone1"]
    z2_lo, z2_hi = zones["zone2"]
    z3_lo, z3_hi = zones["zone3"]

    lz1_lo, lz1_hi = v2_cfg["late_z1_layers"]
    assert z1_lo < lz1_lo <= lz1_hi == z1_hi, (zones, v2_cfg["late_z1_layers"])
    late_z1_layers = list(range(lz1_lo, lz1_hi + 1))
    frozen_early_layers = list(range(z1_lo, lz1_lo))

    # 1. Per-layer MLP rank map: Late-Z1 @ r4 + ALL Zone-2 (B1 top-5 N_CF @ r8, rest @ r4) + ZERO Zone-3 MLPs!
    ncf_top5 = set(int(x) for x in m3v1["mlp_ncf_layers"])
    mlp_ranks: dict[str, int] = {}
    for li in late_z1_layers:
        mlp_ranks[str(li)] = int(v2_cfg["late_z1_mlp_rank"])
    for li in range(z2_lo, z2_hi + 1):
        mlp_ranks[str(li)] = int(v2_cfg["z2_top_mlp_rank"] if li in ncf_top5 else v2_cfg["z2_rest_mlp_rank"])

    # 2. Head ranks: Zone-2 heads get cross-half-stable 3-tier ranks; conflicting Zone-2 heads clamped to tier 3;
    #    Zone-3 B1/DLA heads clamped to conservative readout tier `z3_head_rank`.
    conflicting_set = {f"{li}:{h}" for li, h in m3v1["conflicting_heads"]}
    z2_keys = [k for k in m3v1["head_ranks"] if hd_diag[k]["zone"] == 2]
    z3_keys = [k for k in m3v1["head_ranks"] if hd_diag[k]["zone"] == 3]

    z2_non_conf = sorted(
        [k for k in z2_keys if k not in conflicting_set],
        key=lambda k: (-float(hd_diag[k]["composite_score"]), int(k.split(":")[0]), int(k.split(":")[1])),
    )
    t1_n, t2_n = int(v2_cfg["z2_tier1_count"]), int(v2_cfg["z2_tier2_count"])
    r1, r2, r3 = int(v2_cfg["z2_tier1_rank"]), int(v2_cfg["z2_tier2_rank"]), int(v2_cfg["z2_tier3_rank"])
    r_z3 = int(v2_cfg["z3_head_rank"])

    v2_head_ranks: dict[str, int] = {}
    for idx, k in enumerate(z2_non_conf):
        v2_head_ranks[k] = r1 if idx < t1_n else (r2 if idx < (t1_n + t2_n) else r3)
    for k in sorted(k for k in z2_keys if k in conflicting_set):
        v2_head_ranks[k] = r3
    for k in sorted(z3_keys):
        v2_head_ranks[k] = r_z3

    v2_head_ranks = dict(sorted(v2_head_ranks.items(), key=lambda kv: tuple(int(x) for x in kv[0].split(":"))))
    EXP3C._enforce_layer_rank_cap(v2_head_ranks, max_cap=128)

    lz1_qo_r = int(v2_cfg["late_z1_qo_rank"])
    v2_params = (
        EXP3C._count_dense_qo_params(len(late_z1_layers), lz1_qo_r, d, H, hd)
        + _count_mlp_dict_params(mlp_ranks, d, d_ff)
        + EXP3C._count_head_dict_params(v2_head_ranks, d, hd)
    )
    v2_max_r, v2_layer_qo = _max_packed_rank_v2(v2_head_ranks, late_z1_layers, lz1_qo_r, mlp_ranks)
    assert v2_max_r <= 128, (alias, v2_max_r)

    rel_pct = abs(v2_params / unif["trainable_params"] - 1.0) * 100.0
    assert rel_pct <= 2.0, f"{alias} M3-v2 params {v2_params:,} diverge {rel_pct:.2f}% from Uniform {unif['trainable_params']:,}"

    # 3. Also export matched `M3-v2-Random-Mask` control (identical Zone-1 & Zone-2 MLPs, layer-matched disjoint random heads!)
    rng_rm = random.Random(2035)
    by_l_ranks: dict[int, list[int]] = defaultdict(list)
    for k, r in v2_head_ranks.items():
        by_l_ranks[int(k.split(":")[0])].append(r)
    excl_heads = {(int(k.split(":")[0]), int(k.split(":")[1])) for k in v2_head_ranks}
    rm_head_ranks: dict[str, int] = {}
    for li in sorted(by_l_ranks):
        rs = by_l_ranks[li]
        pool = [h for h in range(H) if (li, h) not in excl_heads]
        chosen = sorted(rng_rm.sample(pool, len(rs)))
        for h, r in zip(chosen, rs):
            rm_head_ranks[f"{li}:{h}"] = r
    EXP3C._enforce_layer_rank_cap(rm_head_ranks, max_cap=128)

    # Randomize which 5 Zone-2 MLPs get r8 vs r4 (disjoint from B1 top-5 N_CF)
    z2_non_ncf = [li for li in range(z2_lo, z2_hi + 1) if li not in ncf_top5]
    rm_top5 = set(rng_rm.sample(z2_non_ncf, len(ncf_top5)))
    rm_mlp_ranks = {str(li): int(v2_cfg["late_z1_mlp_rank"]) for li in late_z1_layers}
    for li in range(z2_lo, z2_hi + 1):
        rm_mlp_ranks[str(li)] = int(v2_cfg["z2_top_mlp_rank"] if li in rm_top5 else v2_cfg["z2_rest_mlp_rank"])
    rm_params = (
        EXP3C._count_dense_qo_params(len(late_z1_layers), lz1_qo_r, d, H, hd)
        + _count_mlp_dict_params(rm_mlp_ranks, d, d_ff)
        + EXP3C._count_head_dict_params(rm_head_ranks, d, hd)
    )
    assert rm_params == v2_params

    # Exact zone param breakdown for documentation & audit
    z1_p = EXP3C._count_dense_qo_params(len(late_z1_layers), lz1_qo_r, d, H, hd) + sum(
        2 * int(mlp_ranks[str(li)]) * (d + d_ff) for li in late_z1_layers
    )
    z2_p = sum(2 * int(mlp_ranks[str(li)]) * (d + d_ff) for li in range(z2_lo, z2_hi + 1)) + sum(
        2 * int(r) * (d + hd) for k, r in v2_head_ranks.items() if hd_diag[k]["zone"] == 2
    )
    z3_p = sum(2 * int(r) * (d + hd) for k, r in v2_head_ranks.items() if hd_diag[k]["zone"] == 3)
    assert z1_p + z2_p + z3_p == v2_params

    # 4. Stage 3 KL-Surgical `M3-v3-PureMid`, `M3-v3-PureMid-Random-Mask`, and `M3-v3-TightBridge`!
    # Keep ONLY Zone-3 `form_only_heads` at minimal r=2 so B1-IIT form pathway remains 100% trainable at ~1% cost.
    b1_form_z3_keys = sorted(
        f"{li}:{h}" for li, h in c3_entry["iit_targets"]["b1"]["form_only_heads"] if z3_lo <= int(li) <= z3_hi
    )
    vt1_n, vt2_n = int(v2_cfg["v3_z2_tier1_count"]), int(v2_cfg["v3_z2_tier2_count"])
    vr1, vr2, vr3 = int(v2_cfg["v3_z2_tier1_rank"]), int(v2_cfg["v3_z2_tier2_rank"]), int(v2_cfg["v3_z2_tier3_rank"])
    vr_z3f = int(v2_cfg["v3_z3_form_head_rank"])

    v3_head_ranks: dict[str, int] = {}
    for idx, k in enumerate(z2_non_conf):
        v3_head_ranks[k] = vr1 if idx < vt1_n else (vr2 if idx < (vt1_n + vt2_n) else vr3)
    for k in sorted(k for k in z2_keys if k in conflicting_set):
        v3_head_ranks[k] = vr3
    for k in b1_form_z3_keys:
        v3_head_ranks[k] = vr_z3f
    v3_head_ranks = dict(sorted(v3_head_ranks.items(), key=lambda kv: tuple(int(x) for x in kv[0].split(":"))))
    EXP3C._enforce_layer_rank_cap(v3_head_ranks, max_cap=128)

    # 4a. M3-v3-PureMid: ZERO Zone-1 (L0-13 frozen!), ALL Zone-2 MLPs (top-5 @ r10, rest @ r6), 42 B1 Z2 heads, 5 Z3 form heads @ r2
    pm_mlp_ranks: dict[str, int] = {
        str(li): int(v2_cfg["v3_puremid_z2_top_mlp_rank"] if li in ncf_top5 else v2_cfg["v3_puremid_z2_rest_mlp_rank"])
        for li in range(z2_lo, z2_hi + 1)
    }
    pm_params = _count_mlp_dict_params(pm_mlp_ranks, d, d_ff) + EXP3C._count_head_dict_params(v3_head_ranks, d, hd)
    pm_max_r, pm_layer_qo = _max_packed_rank_v2(v3_head_ranks, [], 0, pm_mlp_ranks)
    assert pm_max_r <= 128 and abs(pm_params / unif["trainable_params"] - 1.0) * 100.0 <= 2.0, (alias, pm_params, pm_max_r)
    pm_z2_p = _count_mlp_dict_params(pm_mlp_ranks, d, d_ff) + sum(
        2 * int(r) * (d + hd) for k, r in v3_head_ranks.items() if hd_diag[k]["zone"] == 2
    )
    pm_z3_p = sum(2 * int(r) * (d + hd) for k, r in v3_head_ranks.items() if hd_diag[k]["zone"] == 3)
    assert pm_z2_p + pm_z3_p == pm_params

    # 4b. M3-v3-PureMid-Random-Mask: IDENTICAL 0 Zone-1 & per-layer rank multiset, disjoint random heads & disjoint random top-5 Z2 MLPs!
    rng_pm_rm = random.Random(2036)
    pm_by_l: dict[int, list[int]] = defaultdict(list)
    for k, r in v3_head_ranks.items():
        pm_by_l[int(k.split(":")[0])].append(r)
    pm_excl = {(int(k.split(":")[0]), int(k.split(":")[1])) for k in v3_head_ranks}
    pm_rm_head_ranks: dict[str, int] = {}
    for li in sorted(pm_by_l):
        rs = pm_by_l[li]
        pool = [h for h in range(H) if (li, h) not in pm_excl]
        chosen = sorted(rng_pm_rm.sample(pool, len(rs)))
        for h, r in zip(chosen, rs):
            pm_rm_head_ranks[f"{li}:{h}"] = r
    EXP3C._enforce_layer_rank_cap(pm_rm_head_ranks, max_cap=128)
    pm_rm_top5 = set(rng_pm_rm.sample(z2_non_ncf, len(ncf_top5)))
    pm_rm_mlp_ranks: dict[str, int] = {
        str(li): int(v2_cfg["v3_puremid_z2_top_mlp_rank"] if li in pm_rm_top5 else v2_cfg["v3_puremid_z2_rest_mlp_rank"])
        for li in range(z2_lo, z2_hi + 1)
    }
    pm_rm_params = _count_mlp_dict_params(pm_rm_mlp_ranks, d, d_ff) + EXP3C._count_head_dict_params(pm_rm_head_ranks, d, hd)
    assert pm_rm_params == pm_params

    # 4c. M3-v3-TightBridge: L0-9 frozen, L10-13 MLP-only @ r4 (ZERO Z1 Q/O!), ALL Zone-2 MLPs (top-5 @ r10, rest @ r4), 42 B1 Z2 heads, 5 Z3 form heads @ r2
    tb_lo, tb_hi = v2_cfg["v3_tight_z1_layers"]
    tb_z1_layers = list(range(tb_lo, tb_hi + 1))
    tb_mlp_ranks: dict[str, int] = {str(li): int(v2_cfg["v3_tight_z1_mlp_rank"]) for li in tb_z1_layers}
    for li in range(z2_lo, z2_hi + 1):
        tb_mlp_ranks[str(li)] = int(v2_cfg["v3_tight_z2_top_mlp_rank"] if li in ncf_top5 else v2_cfg["v3_tight_z2_rest_mlp_rank"])
    tb_params = _count_mlp_dict_params(tb_mlp_ranks, d, d_ff) + EXP3C._count_head_dict_params(v3_head_ranks, d, hd)
    tb_max_r, tb_layer_qo = _max_packed_rank_v2(v3_head_ranks, [], 0, tb_mlp_ranks)
    assert tb_max_r <= 128 and abs(tb_params / unif["trainable_params"] - 1.0) * 100.0 <= 2.0, (alias, tb_params, tb_max_r)
    tb_z1_p = sum(2 * int(tb_mlp_ranks[str(li)]) * (d + d_ff) for li in tb_z1_layers)
    tb_z2_p = sum(2 * int(tb_mlp_ranks[str(li)]) * (d + d_ff) for li in range(z2_lo, z2_hi + 1)) + sum(
        2 * int(r) * (d + hd) for k, r in v3_head_ranks.items() if hd_diag[k]["zone"] == 2
    )
    tb_z3_p = pm_z3_p
    assert tb_z1_p + tb_z2_p + tb_z3_p == tb_params

    # 5. Stage 4 `M3-v4-PureMid-MLP+` (100% Zone 2, ZERO Z1, ZERO Z3, boosted Zone-2 MLPs!) & `M3-v4-Bridge2` (2-layer L12-13 MLP + full PureMid Z2 MLPs, ZERO Z3!)
    def _build_z2_only_heads(tr1: int, tr2: int, tr3: int) -> dict[str, int]:
        hr: dict[str, int] = {}
        for idx, k in enumerate(z2_non_conf):
            hr[k] = tr1 if idx < vt1_n else (tr2 if idx < (vt1_n + vt2_n) else tr3)
        for k in sorted(k for k in z2_keys if k in conflicting_set):
            hr[k] = tr3
        hr = dict(sorted(hr.items(), key=lambda kv: tuple(int(x) for x in kv[0].split(":"))))
        EXP3C._enforce_layer_rank_cap(hr, max_cap=128)
        return hr

    v4_mp_heads = _build_z2_only_heads(int(v2_cfg["v4_mlpplus_t1_r"]), int(v2_cfg["v4_mlpplus_t2_r"]), int(v2_cfg["v4_mlpplus_t3_r"]))
    v4_mp_mlps = {
        str(li): int(v2_cfg["v4_mlpplus_z2_top_mlp_rank"] if li in ncf_top5 else v2_cfg["v4_mlpplus_z2_rest_mlp_rank"])
        for li in range(z2_lo, z2_hi + 1)
    }
    v4_mp_params = _count_mlp_dict_params(v4_mp_mlps, d, d_ff) + EXP3C._count_head_dict_params(v4_mp_heads, d, hd)
    v4_mp_max_r, v4_mp_layer_qo = _max_packed_rank_v2(v4_mp_heads, [], 0, v4_mp_mlps)
    assert v4_mp_max_r <= 128 and abs(v4_mp_params / unif["trainable_params"] - 1.0) * 100.0 <= 2.5, (alias, v4_mp_params, v4_mp_max_r)

    # 5b. Stage 5 Strictly Rank-Matched Random-Mask Controls for `M3-v4-PureMid-MLP+` (Mask #1 & independent Mask #2)!
    def _build_v4_mlpplus_random_mask(mask_seed: int, mask_label: str) -> dict[str, Any]:
        rng_v4 = random.Random(mask_seed)
        by_l: dict[int, list[int]] = defaultdict(list)
        for k, r in v4_mp_heads.items():
            by_l[int(k.split(":")[0])].append(r)
        excl = {(int(k.split(":")[0]), int(k.split(":")[1])) for k in v4_mp_heads}
        rm_heads: dict[str, int] = {}
        for li in sorted(by_l):
            rs = list(by_l[li])
            rng_v4.shuffle(rs)
            pool = [h for h in range(H) if (li, h) not in excl]
            assert len(pool) >= len(rs), (alias, li, len(pool), len(rs))
            chosen = sorted(rng_v4.sample(pool, len(rs)))
            for h, r in zip(chosen, rs):
                rm_heads[f"{li}:{h}"] = r
        rm_heads = dict(sorted(rm_heads.items(), key=lambda kv: tuple(int(x) for x in kv[0].split(":"))))
        EXP3C._enforce_layer_rank_cap(rm_heads, max_cap=128)
        rm_top5 = sorted(rng_v4.sample(z2_non_ncf, len(ncf_top5)))
        rm_top5_set = set(rm_top5)
        rm_mlps: dict[str, int] = {
            str(li): int(v2_cfg["v4_mlpplus_z2_top_mlp_rank"] if li in rm_top5_set else v2_cfg["v4_mlpplus_z2_rest_mlp_rank"])
            for li in range(z2_lo, z2_hi + 1)
        }
        rm_p = _count_mlp_dict_params(rm_mlps, d, d_ff) + EXP3C._count_head_dict_params(rm_heads, d, hd)
        rm_mr, rm_lqo = _max_packed_rank_v2(rm_heads, [], 0, rm_mlps)

        # Strict exact-match invariants demanded by Astra's pre-flight audit!
        assert rm_p == v4_mp_params, (alias, mask_label, rm_p, v4_mp_params)
        assert rm_mr == v4_mp_max_r, (alias, mask_label, rm_mr, v4_mp_max_r)
        assert set(rm_heads).isdisjoint(set(v4_mp_heads)), (alias, mask_label)
        assert rm_top5_set.isdisjoint(ncf_top5), (alias, mask_label)
        assert sorted(rm_mlps.values()) == sorted(v4_mp_mlps.values()), (alias, mask_label)
        for li in sorted(by_l):
            rm_l_rs = sorted(r for k, r in rm_heads.items() if int(k.split(":")[0]) == li)
            assert rm_l_rs == sorted(by_l[li]), (alias, mask_label, li, rm_l_rs, by_l[li])
        return {
            "kind": "m3_dac",
            "description": (
                f"Stage 5 Strictly Rank-Matched Random-Mask Control ({mask_label}, seed={mask_seed}) for M3-v4-PureMid-MLP+: "
                f"IDENTICAL 0 Zone-1, 0 Zone-3, IDENTICAL per-layer Q/O head count & rank multiset inside L{z2_lo}-{z2_hi}, "
                f"IDENTICAL MLP rank multiset ({len(ncf_top5)}x r={v2_cfg['v4_mlpplus_z2_top_mlp_rank']} + "
                f"{len(z2_non_ncf)}x r={v2_cfg['v4_mlpplus_z2_rest_mlp_rank']}), strictly disjoint random head coords & non-B1 top-5 MLP layers"
            ),
            "mask_seed": mask_seed,
            "zones": zones,
            "frozen_early_layers": list(range(z1_lo, z1_hi + 1)),
            "zone1_dense_qo_layers": [],
            "zone1_dense_qo_rank": 0,
            "mlp_layers": sorted(int(k) for k in rm_mlps),
            "mlp_ranks": rm_mlps,
            "mlp_top5_random_layers": rm_top5,
            "mlp_ncf_layers": [],
            "mlp_dialog_z3_layers": [],
            "head_ranks": rm_heads,
            "max_packed_layer_rank": rm_mr,
            "per_layer_packed_qo_rank": {str(k): v for k, v in rm_lqo.items()},
            "zone_trainable_params": {"zone1": 0, "zone2": rm_p, "zone3": 0},
            "trainable_params": rm_p,
        }

    v4_mp_rm1 = _build_v4_mlpplus_random_mask(2037, "Mask1")
    v4_mp_rm2 = _build_v4_mlpplus_random_mask(2038, "Mask2")
    assert set(v4_mp_rm1["head_ranks"]) != set(v4_mp_rm2["head_ranks"])
    assert v4_mp_rm1["mlp_top5_random_layers"] != v4_mp_rm2["mlp_top5_random_layers"]

    b2_lo, b2_hi = v2_cfg["v4_bridge2_z1_layers"]
    b2_z1_layers = list(range(b2_lo, b2_hi + 1))
    v4_b2_heads = _build_z2_only_heads(int(v2_cfg["v4_bridge2_t1_r"]), int(v2_cfg["v4_bridge2_t2_r"]), int(v2_cfg["v4_bridge2_t3_r"]))
    v4_b2_mlps: dict[str, int] = {str(li): int(v2_cfg["v4_bridge2_z1_mlp_rank"]) for li in b2_z1_layers}
    for li in range(z2_lo, z2_hi + 1):
        v4_b2_mlps[str(li)] = int(v2_cfg["v4_bridge2_z2_top_mlp_rank"] if li in ncf_top5 else v2_cfg["v4_bridge2_z2_rest_mlp_rank"])
    v4_b2_params = _count_mlp_dict_params(v4_b2_mlps, d, d_ff) + EXP3C._count_head_dict_params(v4_b2_heads, d, hd)
    v4_b2_max_r, v4_b2_layer_qo = _max_packed_rank_v2(v4_b2_heads, [], 0, v4_b2_mlps)
    assert v4_b2_max_r <= 128 and abs(v4_b2_params / unif["trainable_params"] - 1.0) * 100.0 <= 2.5, (alias, v4_b2_params, v4_b2_max_r)
    b2_z1_p = sum(2 * int(v4_b2_mlps[str(li)]) * (d + d_ff) for li in b2_z1_layers)
    b2_z2_p = v4_b2_params - b2_z1_p

    architectures = {
        "Uniform-LoRA": unif,
        "M3-v1": m3v1,
        "M3-DAC": m3v1,  # Alias for seamless compatibility with C3 scripts
        "M3-v2": {
            "kind": "m3_dac",
            "description": (
                f"Probe-1-Guided Mechanistic Allocation: Early L{z1_lo}-{lz1_lo-1} FROZEN (0 params, protects Wiki KL); "
                f"Late-Z1 L{lz1_lo}-{lz1_hi} dense Q/O r={lz1_qo_r} + MLP r={v2_cfg['late_z1_mlp_rank']}; "
                f"Full Z2 L{z2_lo}-{z2_hi} ALL MLPs (top-5 B1 N_CF @ r={v2_cfg['z2_top_mlp_rank']}, rest @ r={v2_cfg['z2_rest_mlp_rank']}) "
                f"+ {len(z2_keys)} B1 Z2 heads 3-tier {r1}/{r2}/{r3}; "
                f"Z3 ZERO MLPs + {len(z3_keys)} B1/DLA Z3 heads clamped @ r={r_z3}"
            ),
            "zones": zones,
            "frozen_early_layers": frozen_early_layers,
            "zone1_dense_qo_layers": late_z1_layers,
            "zone1_dense_qo_rank": lz1_qo_r,
            "mlp_layers": sorted(int(k) for k in mlp_ranks),
            "mlp_ranks": mlp_ranks,
            "mlp_ncf_layers": sorted(ncf_top5),
            "mlp_dialog_z3_layers": [],
            "head_ranks": v2_head_ranks,
            "conflicting_heads": m3v1["conflicting_heads"],
            "max_packed_layer_rank": v2_max_r,
            "per_layer_packed_qo_rank": {str(k): v for k, v in v2_layer_qo.items()},
            "zone_trainable_params": {"zone1": z1_p, "zone2": z2_p, "zone3": z3_p},
            "trainable_params": v2_params,
        },
        "M3-v2-Random-Mask": {
            "kind": "m3_dac",
            "description": "Matched control for M3-v2: same Late-Z1 + Full-Z2 MLP structure with disjoint random top-5 MLPs and layer-matched disjoint random heads",
            "zones": zones,
            "frozen_early_layers": frozen_early_layers,
            "zone1_dense_qo_layers": late_z1_layers,
            "zone1_dense_qo_rank": lz1_qo_r,
            "mlp_layers": sorted(int(k) for k in rm_mlp_ranks),
            "mlp_ranks": rm_mlp_ranks,
            "head_ranks": rm_head_ranks,
            "max_packed_layer_rank": _max_packed_rank_v2(rm_head_ranks, late_z1_layers, lz1_qo_r, rm_mlp_ranks)[0],
            "trainable_params": rm_params,
        },
        "M3-v3-PureMid": {
            "kind": "m3_dac",
            "description": (
                f"Stage 3 KL-Surgical Pure Mid-Layer Core: ALL Zone 1 L{z1_lo}-{z1_hi} 100% FROZEN (0 params -> targets >=30% KL reduction); "
                f"Full Zone 2 L{z2_lo}-{z2_hi} ALL MLPs (top-5 B1 N_CF @ r={v2_cfg['v3_puremid_z2_top_mlp_rank']}, rest @ r={v2_cfg['v3_puremid_z2_rest_mlp_rank']}) "
                f"+ {len(z2_keys)} B1 Z2 heads 3-tier {vr1}/{vr2}/{vr3}; "
                f"Zone 3 ZERO MLPs + {len(b1_form_z3_keys)} Z3 form_only heads @ r={vr_z3f}"
            ),
            "zones": zones,
            "frozen_early_layers": list(range(z1_lo, z1_hi + 1)),
            "zone1_dense_qo_layers": [],
            "zone1_dense_qo_rank": 0,
            "mlp_layers": sorted(int(k) for k in pm_mlp_ranks),
            "mlp_ranks": pm_mlp_ranks,
            "mlp_ncf_layers": sorted(ncf_top5),
            "mlp_dialog_z3_layers": [],
            "head_ranks": v3_head_ranks,
            "conflicting_heads": m3v1["conflicting_heads"],
            "max_packed_layer_rank": pm_max_r,
            "per_layer_packed_qo_rank": {str(k): v for k, v in pm_layer_qo.items()},
            "zone_trainable_params": {"zone1": 0, "zone2": pm_z2_p, "zone3": pm_z3_p},
            "trainable_params": pm_params,
        },
        "M3-v3-PureMid-Random-Mask": {
            "kind": "m3_dac",
            "description": "Unconfounded attribution control for M3-v3-PureMid: identical 0 Zone-1 & per-layer rank multiset with disjoint random top-5 MLPs and disjoint random heads",
            "zones": zones,
            "frozen_early_layers": list(range(z1_lo, z1_hi + 1)),
            "zone1_dense_qo_layers": [],
            "zone1_dense_qo_rank": 0,
            "mlp_layers": sorted(int(k) for k in pm_rm_mlp_ranks),
            "mlp_ranks": pm_rm_mlp_ranks,
            "head_ranks": pm_rm_head_ranks,
            "max_packed_layer_rank": _max_packed_rank_v2(pm_rm_head_ranks, [], 0, pm_rm_mlp_ranks)[0],
            "zone_trainable_params": {"zone1": 0, "zone2": pm_z2_p, "zone3": pm_z3_p},
            "trainable_params": pm_rm_params,
        },
        "M3-v3-TightBridge": {
            "kind": "m3_dac",
            "description": (
                f"Stage 3 KL-Surgical Tight MLP-Only Bridge: L{z1_lo}-{tb_lo-1} FROZEN + ZERO Z1 Q/O; "
                f"Late-Z1 L{tb_lo}-{tb_hi} MLP-only @ r={v2_cfg['v3_tight_z1_mlp_rank']}; "
                f"Full Zone 2 L{z2_lo}-{z2_hi} ALL MLPs (top-5 B1 N_CF @ r={v2_cfg['v3_tight_z2_top_mlp_rank']}, rest @ r={v2_cfg['v3_tight_z2_rest_mlp_rank']}) "
                f"+ {len(z2_keys)} B1 Z2 heads 3-tier {vr1}/{vr2}/{vr3}; "
                f"Zone 3 ZERO MLPs + {len(b1_form_z3_keys)} Z3 form_only heads @ r={vr_z3f}"
            ),
            "zones": zones,
            "frozen_early_layers": list(range(z1_lo, tb_lo)),
            "zone1_dense_qo_layers": [],
            "zone1_dense_qo_rank": 0,
            "mlp_layers": sorted(int(k) for k in tb_mlp_ranks),
            "mlp_ranks": tb_mlp_ranks,
            "mlp_ncf_layers": sorted(ncf_top5),
            "mlp_dialog_z3_layers": [],
            "head_ranks": v3_head_ranks,
            "conflicting_heads": m3v1["conflicting_heads"],
            "max_packed_layer_rank": tb_max_r,
            "per_layer_packed_qo_rank": {str(k): v for k, v in tb_layer_qo.items()},
            "zone_trainable_params": {"zone1": tb_z1_p, "zone2": tb_z2_p, "zone3": tb_z3_p},
            "trainable_params": tb_params,
        },
        "M3-v4-PureMid-MLP+": {
            "kind": "m3_dac",
            "description": (
                f"Stage 4 100% Pure Zone-2 High-MLP: ZERO Zone-1 L{z1_lo}-{z1_hi}, ZERO Zone-3 L{z3_lo}-{z3_hi}; "
                f"ALL Zone-2 MLPs L{z2_lo}-{z2_hi} boosted (top-5 B1 N_CF @ r={v2_cfg['v4_mlpplus_z2_top_mlp_rank']}, "
                f"rest @ r={v2_cfg['v4_mlpplus_z2_rest_mlp_rank']}) + {len(v4_mp_heads)} B1 Z2 heads 3-tier "
                f"{v2_cfg['v4_mlpplus_t1_r']}/{v2_cfg['v4_mlpplus_t2_r']}/{v2_cfg['v4_mlpplus_t3_r']}"
            ),
            "zones": zones,
            "frozen_early_layers": list(range(z1_lo, z1_hi + 1)),
            "zone1_dense_qo_layers": [],
            "zone1_dense_qo_rank": 0,
            "mlp_layers": sorted(int(k) for k in v4_mp_mlps),
            "mlp_ranks": v4_mp_mlps,
            "mlp_ncf_layers": sorted(ncf_top5),
            "mlp_dialog_z3_layers": [],
            "head_ranks": v4_mp_heads,
            "conflicting_heads": m3v1["conflicting_heads"],
            "max_packed_layer_rank": v4_mp_max_r,
            "per_layer_packed_qo_rank": {str(k): v for k, v in v4_mp_layer_qo.items()},
            "zone_trainable_params": {"zone1": 0, "zone2": v4_mp_params, "zone3": 0},
            "trainable_params": v4_mp_params,
        },
        "M3-v4-PureMid-MLP+-Random-Mask": v4_mp_rm1,
        "M3-v4-PureMid-MLP+-Random-Mask2": v4_mp_rm2,
        "M3-v4-Bridge2": {
            "kind": "m3_dac",
            "description": (
                f"Stage 4 2-Layer Late-Z1 MLP Bridge + Full PureMid Zone-2 MLPs (ZERO Z1 Q/O, ZERO Zone-3): "
                f"L{z1_lo}-{b2_lo-1} FROZEN; L{b2_lo}-{b2_hi} MLP-only @ r={v2_cfg['v4_bridge2_z1_mlp_rank']}; "
                f"ALL Zone-2 MLPs L{z2_lo}-{z2_hi} (top-5 B1 N_CF @ r={v2_cfg['v4_bridge2_z2_top_mlp_rank']}, "
                f"rest @ r={v2_cfg['v4_bridge2_z2_rest_mlp_rank']}) + {len(v4_b2_heads)} B1 Z2 heads 3-tier "
                f"{v2_cfg['v4_bridge2_t1_r']}/{v2_cfg['v4_bridge2_t2_r']}/{v2_cfg['v4_bridge2_t3_r']}"
            ),
            "zones": zones,
            "frozen_early_layers": list(range(z1_lo, b2_lo)),
            "zone1_dense_qo_layers": [],
            "zone1_dense_qo_rank": 0,
            "mlp_layers": sorted(int(k) for k in v4_b2_mlps),
            "mlp_ranks": v4_b2_mlps,
            "mlp_ncf_layers": sorted(ncf_top5),
            "mlp_dialog_z3_layers": [],
            "head_ranks": v4_b2_heads,
            "conflicting_heads": m3v1["conflicting_heads"],
            "max_packed_layer_rank": v4_b2_max_r,
            "per_layer_packed_qo_rank": {str(k): v for k, v in v4_b2_layer_qo.items()},
            "zone_trainable_params": {"zone1": b2_z1_p, "zone2": b2_z2_p, "zone3": 0},
            "trainable_params": v4_b2_params,
        },
    }

    return {
        "model_spec": ms,
        "uniform_anchor_params": c3_entry["uniform_anchor_params"],
        "source_b1_sha256": c3_entry["source_b1_sha256"],
        "architectures": architectures,
        "iit_targets": c3_entry["iit_targets"],
        "head_diagnostics": hd_diag,
    }


def main() -> None:
    c3_frozen = json.loads((C3 / "data" / "c3_components_frozen.json").read_text(encoding="utf-8"))
    models_out = {}
    for alias, v2_cfg in M3V2_SPECS.items():
        models_out[alias] = build_c35_model_entry(alias, c3_frozen["models"][alias], v2_cfg)

    payload = {
        "schema": "cedar.acl_exp_c35.components.v1",
        "derived_from": "experiments/ACL-exp-c3/data/c3_components_frozen.json + Stage 1 Probe 1 (a472922)",
        "models": models_out,
    }
    out = C35 / "data" / "c35_components_frozen.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"[OK] wrote {out} ({out.stat().st_size:,} bytes)")
    for alias, m in models_out.items():
        anc = m["uniform_anchor_params"]
        print(f"\n=== {alias} (Uniform r=4 anchor = {anc:,} params) ===")
        for aname in (
            "Uniform-LoRA", "M3-v1", "M3-v2", "M3-v3-PureMid", "M3-v3-PureMid-Random-Mask",
            "M3-v3-TightBridge", "M3-v4-PureMid-MLP+", "M3-v4-PureMid-MLP+-Random-Mask",
            "M3-v4-PureMid-MLP+-Random-Mask2", "M3-v4-Bridge2",
        ):
            a = m["architectures"][aname]
            p = a["trainable_params"]
            rel = (p / anc - 1.0) * 100.0
            mr = a["max_packed_layer_rank"]
            zp = a.get("zone_trainable_params", {})
            print(f"  {aname:32s}: {p:10,d} ({rel:+5.2f}%) | max_layer_r={mr:3d} | zones={zp}")


if __name__ == "__main__":
    main()
