#!/usr/bin/env python3
"""Deterministically export frozen B1 component sets & parameter-budget-matched ranks for ACL-exp-c1.

Reads committed B1 results under experiments/ACL-exp-b1/results/ (using the Turn-1-eligible runs as
primary rankings, with full-run hub-head cross-check) and writes:
  experiments/ACL-exp-c1/data/c1_components_frozen.json

Arms exported for each model (qwen3_14b, qwen3_8b):
  1. Pinpoint-DLA   : DLA(C_pinpoint) top-20 heads (Q/O head-slice LoRA, no MLP)
  2. Pinpoint-AtP   : AtP(C_pinpoint) top-20 heads (Q/O head-slice LoRA, no MLP)
  3. Targeted-Form  : AtP(C_form) top-20 U AtP(C_soc) top-12 heads (Q/O head-slice LoRA, no MLP)
  4. Targeted-Dual  : Targeted-Form U AtP(C_content_N_CF) top-20 U AtP(C_content_CF/AF) top-10 heads
                      (Q/O head-slice LoRA) + C_content_N_CF top-5 mid-layer MLPs (gate+down LoRA)
  5. Random-LM      : Layer-matched random heads & random MLP layers matching Targeted-Dual exactly
  6. Uniform-LoRA   : All layers q_proj, o_proj, gate_proj, down_proj LoRA matched to Dual total params
  7. Hub-Only       : Shared mid-layer routing hub heads only (C_pinpoint & C_content_N_CF intersection)
  8. Dual-wo-Hub    : Targeted-Dual with the shared hub heads frozen

Usage:
  python experiments/ACL-exp-c1/scripts/c1_export_components.py
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
C1 = HERE.parents[0]
REPO = C1.parents[1]
B1_RES = REPO / "experiments" / "ACL-exp-b1" / "results"
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-b1" / "scripts"))
import b1_lib as L  # noqa: E402

MODEL_SPECS = {
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
        # Target Dual ranks chosen to match Uniform-LoRA r=4 within <3% total trainable params
        "dual_head_rank": 18,
        "dual_mlp_rank": 8,
    },
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
        "dual_head_rank": 14,
        "dual_mlp_rank": 8,
    },
}


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load_atp_head(res_dir: Path, contrast: str, half: str = "all") -> torch.Tensor:
    return torch.load(res_dir / f"atp_{contrast}.pt")[f"head_{half}"]


def _load_atp_mlp(res_dir: Path, contrast: str, half: str = "all") -> torch.Tensor:
    return torch.load(res_dir / f"atp_{contrast}.pt")[f"mlp_{half}"]


def _load_dla(res_dir: Path, contrast: str, half: str = "all") -> torch.Tensor:
    b = torch.load(res_dir / f"dla_{contrast}.pt")
    return b.get(f"dla_{half}", b.get("dla_all", b.get(f"dla_{contrast}")))


def count_head_lora_params(n_heads: int, rank: int, d: int, hd: int) -> int:
    # Per head: Q slice [hd, d] -> rank*(hd + d); O slice [d, hd] -> rank*(d + hd)
    return n_heads * 2 * rank * (d + hd)


def count_mlp_lora_params(n_layers: int, rank: int, d: int, d_ff: int) -> int:
    # Per MLP layer: gate_proj [d_ff, d] + down_proj [d, d_ff] -> 2 * rank * (d + d_ff)
    return n_layers * 2 * rank * (d + d_ff)


def count_uniform_lora_params(num_layers: int, rank: int, d: int, H: int, hd: int, d_ff: int) -> int:
    # Per layer: q_proj [H*hd, d] + o_proj [d, H*hd] + gate_proj [d_ff, d] + down_proj [d, d_ff]
    per_layer = rank * ((H * hd + d) + (d + H * hd) + (d_ff + d) + (d + d_ff))
    return num_layers * per_layer


def best_head_rank_for_budget(n_heads: int, budget: int, d: int, hd: int) -> int:
    if n_heads <= 0:
        return 0
    unit = n_heads * 2 * (d + hd)
    r = max(1, round(budget / unit))
    # Cap at head_dim so LoRA remains <= full-slice rank
    return min(r, hd)


def best_uniform_rank_for_budget(num_layers: int, budget: int, d: int, H: int, hd: int, d_ff: int) -> int:
    unit = count_uniform_lora_params(num_layers, 1, d, H, hd, d_ff)
    return max(1, round(budget / unit))


def sort_heads(hs) -> list[list[int]]:
    return [[int(li), int(h)] for li, h in sorted(set((int(a), int(b)) for a, b in hs))]


def build_model_components(alias: str, spec: dict) -> dict:
    pdir = B1_RES / spec["b1_primary_dir"]
    fdir = B1_RES / spec["b1_full_dir"]
    d, H, hd, d_ff, nL = (
        spec["hidden_size"],
        spec["num_heads"],
        spec["head_dim"],
        spec["intermediate_size"],
        spec["num_layers"],
    )

    # Primary rankings from eligible-subset B1 (frozen 97e7cbd)
    dla_pin_20 = L.top_k(_load_dla(pdir, "C_pinpoint", "all"), 20)
    atp_pin_20 = L.top_k(_load_atp_head(pdir, "C_pinpoint", "all"), 20)
    atp_form_20 = L.top_k(_load_atp_head(pdir, "C_form", "all"), 20)
    atp_soc_12 = L.top_k(_load_atp_head(pdir, "C_soc", "all"), 12)
    atp_ncf_20 = L.top_k(_load_atp_head(pdir, "C_content_N_CF", "all"), 20)
    dlg_c = spec["dialog_content_contrast"]
    atp_dlg_10 = L.top_k(_load_atp_head(pdir, dlg_c, "all"), 10)

    # Top-5 mid-layer verification MLPs from C_content_N_CF
    mlp_ncf = _load_atp_mlp(pdir, "C_content_N_CF", "all").view(nL, 1)
    mlp_ncf_5 = sorted(int(li) for li, _ in L.top_k(mlp_ncf, 5))

    # Shared mid-layer hub heads: union of (C_pinpoint & C_content_N_CF top-20) across full + eligible runs
    # On 14B this captures the exact 4 audited hubs [(23,4),(24,17),(24,19),(25,15)]; on 8B [(19,31),(20,8),(21,19)].
    full_pin_20 = L.top_k(_load_atp_head(fdir, "C_pinpoint", "all"), 20)
    full_ncf_20 = L.top_k(_load_atp_head(fdir, "C_content_N_CF", "all"), 20)
    hub_heads = sorted((set(full_pin_20) & set(full_ncf_20)) | (set(atp_pin_20) & set(atp_ncf_20)))

    form_heads = sorted(set(atp_form_20) | set(atp_soc_12))
    dual_heads = sorted(set(form_heads) | set(atp_ncf_20) | set(atp_dlg_10))
    dual_wo_hub_heads = sorted(set(dual_heads) - set(hub_heads))

    # Budget anchor = Targeted-Dual total trainable params
    r_dual_h = spec["dual_head_rank"]
    r_dual_m = spec["dual_mlp_rank"]
    budget = count_head_lora_params(len(dual_heads), r_dual_h, d, hd) + count_mlp_lora_params(len(mlp_ncf_5), r_dual_m, d, d_ff)

    # Random-LM: exact layer-matched random heads & mid-layer-band-matched random MLP layers (disjoint from Dual)
    rnd_heads = L.layer_matched_random(dual_heads, H, seed=2027, exclude=set(dual_heads))
    rng = random.Random(2027)
    lo_m, hi_m = max(0, min(mlp_ncf_5) - 5), min(nL - 1, max(mlp_ncf_5) + 5)
    mlp_pool = [li for li in range(lo_m, hi_m + 1) if li not in set(mlp_ncf_5)]
    rnd_mlps = sorted(rng.sample(mlp_pool, len(mlp_ncf_5)))

    # Parameter-matched ranks for heads-only & uniform arms
    r_pin_dla = best_head_rank_for_budget(len(dla_pin_20), budget, d, hd)
    r_pin_atp = best_head_rank_for_budget(len(atp_pin_20), budget, d, hd)
    r_form = best_head_rank_for_budget(len(form_heads), budget, d, hd)
    r_uniform = best_uniform_rank_for_budget(nL, budget, d, H, hd, d_ff)

    arms = {
        "Targeted-Dual": {
            "kind": "circuit",
            "description": "C_form(20) U C_soc(12) U C_content_N_CF(20) U dialog_content(10) Q/O slices + N_CF top-5 MLP gate/down LoRA",
            "heads": sort_heads(dual_heads),
            "head_rank": r_dual_h,
            "mlp_layers": mlp_ncf_5,
            "mlp_rank": r_dual_m,
            "trainable_params": budget,
        },
        "Targeted-Form": {
            "kind": "circuit",
            "description": "C_form(20) U C_soc(12) Q/O slices only (no MLP; rank-matched to Dual total budget)",
            "heads": sort_heads(form_heads),
            "head_rank": r_form,
            "mlp_layers": [],
            "mlp_rank": 0,
            "trainable_params": count_head_lora_params(len(form_heads), r_form, d, hd),
        },
        "Pinpoint-DLA": {
            "kind": "circuit",
            "description": "Pinpoint-style direct-path DLA(C_pinpoint) top-20 Q/O slices only (no MLP; rank-matched to Dual budget)",
            "heads": sort_heads(dla_pin_20),
            "head_rank": r_pin_dla,
            "mlp_layers": [],
            "mlp_rank": 0,
            "trainable_params": count_head_lora_params(len(dla_pin_20), r_pin_dla, d, hd),
        },
        "Pinpoint-AtP": {
            "kind": "circuit",
            "description": "Strengthened Pinpoint using full-graph AtP(C_pinpoint) top-20 Q/O slices only (no MLP; rank-matched)",
            "heads": sort_heads(atp_pin_20),
            "head_rank": r_pin_atp,
            "mlp_layers": [],
            "mlp_rank": 0,
            "trainable_params": count_head_lora_params(len(atp_pin_20), r_pin_atp, d, hd),
        },
        "Random-LM": {
            "kind": "circuit",
            "description": "Layer-matched random heads & random MLP layers disjoint from Targeted-Dual (exact same param count)",
            "heads": sort_heads(rnd_heads),
            "head_rank": r_dual_h,
            "mlp_layers": rnd_mlps,
            "mlp_rank": r_dual_m,
            "trainable_params": count_head_lora_params(len(rnd_heads), r_dual_h, d, hd)
            + count_mlp_lora_params(len(rnd_mlps), r_dual_m, d, d_ff),
        },
        "Uniform-LoRA": {
            "kind": "uniform",
            "description": "All-layer uniform LoRA on q_proj, o_proj, gate_proj, down_proj (rank-matched to Dual total budget)",
            "uniform_rank": r_uniform,
            "target_modules": ["q_proj", "o_proj", "gate_proj", "down_proj"],
            "trainable_params": count_uniform_lora_params(nL, r_uniform, d, H, hd, d_ff),
        },
        "Hub-Only": {
            "kind": "circuit",
            "description": "Ablation: ONLY the shared mid-layer routing hub heads (Q/O slices, rank=dual_head_rank)",
            "heads": sort_heads(hub_heads),
            "head_rank": r_dual_h,
            "mlp_layers": [],
            "mlp_rank": 0,
            "trainable_params": count_head_lora_params(len(hub_heads), r_dual_h, d, hd),
        },
        "Dual-wo-Hub": {
            "kind": "circuit",
            "description": "Ablation: Targeted-Dual with shared mid-layer hub heads frozen",
            "heads": sort_heads(dual_wo_hub_heads),
            "head_rank": r_dual_h,
            "mlp_layers": mlp_ncf_5,
            "mlp_rank": r_dual_m,
            "trainable_params": count_head_lora_params(len(dual_wo_hub_heads), r_dual_h, d, hd)
            + count_mlp_lora_params(len(mlp_ncf_5), r_dual_m, d, d_ff),
        },
    }

    # Record provenance hashes
    src_hashes = {}
    for fn in (f"atp_C_pinpoint.pt", f"atp_C_soc.pt", f"atp_C_form.pt", f"atp_C_content_N_CF.pt", f"atp_{dlg_c}.pt", "dla_C_pinpoint.pt"):
        p = pdir / fn
        if p.exists():
            src_hashes[f"{spec['b1_primary_dir']}/{fn}"] = _sha256(p)

    return {
        "model_spec": spec,
        "budget_anchor_params": budget,
        "hub_heads": sort_heads(hub_heads),
        "source_b1_sha256": src_hashes,
        "arms": arms,
    }


def main() -> None:
    payload = {
        "schema": "cedar.acl_exp_c1.components.v1",
        "models": {alias: build_model_components(alias, spec) for alias, spec in MODEL_SPECS.items()},
    }
    out = C1 / "data" / "c1_components_frozen.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"[OK] wrote {out}")
    for alias, m in payload["models"].items():
        anchor = m["budget_anchor_params"]
        print(f"\n=== {alias} (anchor={anchor:,} params, hubs={m['hub_heads']}) ===")
        for aname, a in m["arms"].items():
            p = a["trainable_params"]
            rel = (p / anchor - 1.0) * 100.0
            extra = f"heads={len(a['heads'])}@r={a['head_rank']} mlps={a['mlp_layers']}@r={a['mlp_rank']}" if a["kind"] == "circuit" else f"uniform_r={a['uniform_rank']}"
            print(f"  {aname:15s}: {p:10,d} ({rel:+5.1f}%) | {extra}")


if __name__ == "__main__":
    main()
