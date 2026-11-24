#!/usr/bin/env python3
"""Deterministically export Stage 6 frozen `Pinpoint-DLA-HeadsOnly` spec without mutating Stage 5's frozen JSON.

Reads committed `ACL-exp-b1/results/qwen3_8b_eligible/dla_C_pinpoint.pt` (`dla_all`, shape `[36, 32]`) and existing
frozen `data/c35_components_frozen.json` (LEFT 100% UNTOUCHED!) and writes a standalone Stage 6 component file:
  `experiments/ACL-exp-c3.5/data/c35_stage6_components_frozen.json`

Architecture (`Pinpoint-DLA-HeadsOnly`, clearly labeled `Pinpoint-style DLA` per Astra's Stage 5 review):
  - Selects Top-K=64 heads (5.56% of 1,152 heads) by `|dla_all|` from `dla_C_pinpoint.pt`
  - GQA Q/O per-head LoRA at uniform rank `r=13` -> `7,028,736` trainable params
    (`-0.69%` vs `Uniform-LoRA` `7,077,888`, `+0.48%` vs `M3-v4-PureMid-MLP+` `6,994,944`)
  - 100% of MLPs FROZEN (`mlp_ranks = {}`, `zone1_dense_qo_layers = []`)
  - `max_packed_layer_rank = 117 <= 128` (lossless vLLM PEFT LoRA compatible!)

Usage:
  python experiments/ACL-exp-c3.5/scripts/c35_export_stage6_pinpoint.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
C35 = HERE.parents[0]
REPO = C35.parents[1]
B1_RES = REPO / "experiments" / "ACL-exp-b1" / "results"
C35_DATA = C35 / "data"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_pinpoint_dla_arch(
    *,
    dla_path: Path,
    model_spec: dict[str, Any],
    zones: dict[str, list[int]],
    top_k: int,
    head_rank: int,
    unif_params: int,
    v4_params: int,
) -> dict[str, Any]:
    obj = torch.load(dla_path, map_location="cpu", weights_only=False)
    dla = obj["dla_all"].float()
    L, H = int(dla.shape[0]), int(dla.shape[1])
    assert L == int(model_spec["num_layers"]) and H == int(model_spec["num_heads"])
    d = int(model_spec["hidden_size"])
    hd = int(model_spec["head_dim"])

    scored = [
        (abs(float(dla[l, h].item())), float(dla[l, h].item()), l, h)
        for l in range(L)
        for h in range(H)
    ]
    # Deterministic tie-break by (layer, head)
    scored.sort(key=lambda x: (-x[0], x[2], x[3]))
    chosen = scored[:top_k]

    head_ranks = {
        f"{l}:{h}": int(head_rank)
        for _, _, l, h in sorted(chosen, key=lambda x: (x[2], x[3]))
    }
    by_layer = Counter(l for _, _, l, _ in chosen)
    max_packed_rank = max(by_layer.values()) * int(head_rank)
    assert max_packed_rank <= 128, f"max_packed_rank={max_packed_rank} > 128!"

    trainable_params = top_k * 2 * int(head_rank) * (d + hd)
    rel_vs_unif_pct = round((trainable_params / unif_params - 1.0) * 100.0, 2)
    rel_vs_v4_pct = round((trainable_params / v4_params - 1.0) * 100.0, 2)
    assert abs(rel_vs_unif_pct) <= 2.0 and abs(rel_vs_v4_pct) <= 2.0, (rel_vs_unif_pct, rel_vs_v4_pct)

    z1_lo, z1_hi = zones["zone1"]
    z2_lo, z2_hi = zones["zone2"]
    z3_lo, z3_hi = zones["zone3"]
    zone_head_counts = {
        "zone1": sum(1 for _, _, l, _ in chosen if z1_lo <= l <= z1_hi),
        "zone2": sum(1 for _, _, l, _ in chosen if z2_lo <= l <= z2_hi),
        "zone3": sum(1 for _, _, l, _ in chosen if z3_lo <= l <= z3_hi),
    }

    return {
        "kind": "m3_dac",
        "baseline_label": "Pinpoint-style DLA (Heads-Only, MLPs frozen)",
        "selection_source": str(dla_path.relative_to(REPO)).replace("\\", "/"),
        "selection_source_sha256": _sha256_file(dla_path),
        "selection_rule": f"Top-{top_k} heads by |dla_all| on B1 eligible C_pinpoint ({top_k}/{L*H} = {100.0*top_k/(L*H):.2f}% of heads); GQA Q/O LoRA r={head_rank}; all MLPs frozen",
        "zones": zones,
        "zone1_dense_qo_layers": [],
        "zone1_dense_qo_rank": 0,
        "mlp_ranks": {},
        "head_ranks": head_ranks,
        "top_k_heads": top_k,
        "head_rank": head_rank,
        "zone_head_counts": zone_head_counts,
        "max_packed_layer_rank": max_packed_rank,
        "trainable_params": trainable_params,
        "rel_vs_uniform_pct": rel_vs_unif_pct,
        "rel_vs_v4_mlpplus_pct": rel_vs_v4_pct,
    }


def main() -> None:
    base_frozen = C35_DATA / "c35_components_frozen.json"
    base_sha = _sha256_file(base_frozen)
    doc = json.loads(base_frozen.read_text(encoding="utf-8"))

    # Clone full frozen document so Stage 6 scripts can load ANY existing arch + Pinpoint-DLA-HeadsOnly
    out_doc = json.loads(json.dumps(doc))
    out_doc["schema"] = "cedar.acl_exp_c35.stage6_components_frozen.v1"
    out_doc["parent_c35_components_sha256"] = base_sha

    m8 = out_doc["models"]["qwen3_8b"]
    p8 = build_pinpoint_dla_arch(
        dla_path=B1_RES / "qwen3_8b_eligible" / "dla_C_pinpoint.pt",
        model_spec=m8["model_spec"],
        zones=m8["architectures"]["M3-v4-PureMid-MLP+"]["zones"],
        top_k=64,
        head_rank=13,
        unif_params=m8["architectures"]["Uniform-LoRA"]["trainable_params"],
        v4_params=m8["architectures"]["M3-v4-PureMid-MLP+"]["trainable_params"],
    )
    m8["architectures"]["Pinpoint-DLA-HeadsOnly"] = p8

    out_path = C35_DATA / "c35_stage6_components_frozen.json"
    out_path.write_text(json.dumps(out_doc, indent=2) + "\n", encoding="utf-8", newline="\n")
    out_sha = _sha256_file(out_path)
    assert _sha256_file(base_frozen) == base_sha, "Parent c35_components_frozen.json must NEVER change!"

    print(f"[OK] Wrote {out_path.name} (sha256={out_sha[:16]}, parent_sha256={base_sha[:16]})")
    print(
        f"  qwen3_8b Pinpoint-DLA-HeadsOnly: params={p8['trainable_params']:,} "
        f"(vs Unif {p8['rel_vs_uniform_pct']:+.2f}%, vs v4 {p8['rel_vs_v4_mlpplus_pct']:+.2f}%), "
        f"max_packed_r={p8['max_packed_layer_rank']}, zone_heads={p8['zone_head_counts']}"
    )


if __name__ == "__main__":
    main()
