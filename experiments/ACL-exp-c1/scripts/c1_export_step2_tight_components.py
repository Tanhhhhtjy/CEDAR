#!/usr/bin/env python3
"""Export additive tight-budget component specs for Step 2 without touching c1_components_frozen.json.

Reuses the EXACT frozen head & MLP sets from data/c1_components_frozen.json (zero re-sampling)
and preserves the Standard-budget Head/MLP parameter ratio within +-2.5 pp (eliminating capacity vs
Head/MLP-allocation confounding):
  - qwen3_8b (0.0185% of 8B; MLP share 21.6% vs Standard 19.1%):
      * Targeted-Dual-Tight   : exact 47 B1 heads (r_h=3) + exact 5 B1 MLPs (r_m=2) = 1,518,848 params
      * Random-LM-Tight       : exact 47 mask-seed-2027 heads (r_h=3) + 5 MLPs (r_m=2) = 1,518,848 params
      * Random-LM-seed2-Tight : exact 47 mask-seed-2028 heads (r_h=3) + 5 MLPs (r_m=2) = 1,518,848 params
      * Uniform-LoRA-Tight    : all 36 layers q/o/gate/down (r=1) = 1,769,472 params (-14.16% vs Dual-Tight!)
  - qwen3_14b (0.0161% of 14B; MLP share 18.9% vs Standard 17.2%):
      * Targeted-Dual-Tight   : exact 46 B1 heads (r_h=4) + exact 5 B1 MLPs (r_m=2) = 2,381,824 params
      * Random-LM-Tight       : exact 46 mask-seed-2027 heads (r_h=4) + 5 MLPs (r_m=2) = 2,381,824 params
      * Random-LM-seed2-Tight : exact 46 mask-seed-2028 heads (r_h=4) + 5 MLPs (r_m=2) = 2,381,824 params
      * Uniform-LoRA-Tight    : all 40 layers q/o/gate/down (r=1) = 2,621,440 params (-9.14% vs Dual-Tight!)

Note: Because c1_modules.py uses rsLoRA scaling s(r) = base_scale * sqrt(anchor_rank / r),
initial r * s(r)^2 variance stays invariant when compressing from 6.87M -> 1.68M.

Usage:
  python experiments/ACL-exp-c1/scripts/c1_export_step2_tight_components.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
C1 = HERE.parents[0]
sys.path.insert(0, str(HERE))

import c1_export_components as E  # noqa: E402

TIGHT_RANKS = {
    "qwen3_8b": {"head_rank": 3, "mlp_rank": 2, "uniform_rank": 1},
    "qwen3_14b": {"head_rank": 4, "mlp_rank": 2, "uniform_rank": 1},
}


def main() -> None:
    src_path = C1 / "data" / "c1_components_frozen.json"
    src_sha = hashlib.sha256(src_path.read_bytes()).hexdigest()
    src = json.loads(src_path.read_text(encoding="utf-8"))

    out_models = {}
    for mk, tr in TIGHT_RANKS.items():
        m = src["models"][mk]
        ms = m["model_spec"]
        d, H, hd, d_ff, nL = (
            ms["hidden_size"],
            ms["num_heads"],
            ms["head_dim"],
            ms["intermediate_size"],
            ms["num_layers"],
        )
        r_h, r_m, r_u = tr["head_rank"], tr["mlp_rank"], tr["uniform_rank"]

        arms_out = {}
        for base_name, tight_name in (
            ("Targeted-Dual", "Targeted-Dual-Tight"),
            ("Random-LM", "Random-LM-Tight"),
            ("Random-LM-seed2", "Random-LM-seed2-Tight"),
        ):
            base_arm = m["arms"][base_name]
            n_h = len(base_arm["heads"])
            n_m = len(base_arm["mlp_layers"])
            n_params = E.count_head_lora_params(n_h, r_h, d, hd) + E.count_mlp_lora_params(n_m, r_m, d, d_ff)
            arms_out[tight_name] = {
                "kind": "circuit",
                "description": f"Tight-budget ({r_h}h/{r_m}m) exact subset of {base_name} from c1_components_frozen.json",
                "heads": base_arm["heads"],
                "head_rank": r_h,
                "mlp_layers": base_arm["mlp_layers"],
                "mlp_rank": r_m,
                "trainable_params": n_params,
            }

        u_params = E.count_uniform_lora_params(nL, r_u, d, H, hd, d_ff)
        arms_out["Uniform-LoRA-Tight"] = {
            "kind": "uniform",
            "description": f"Tight-budget all-layer uniform LoRA (r={r_u}) matched to Targeted-Dual-Tight",
            "uniform_rank": r_u,
            "target_modules": ["q_proj", "o_proj", "gate_proj", "down_proj"],
            "trainable_params": u_params,
        }

        dual_p = arms_out["Targeted-Dual-Tight"]["trainable_params"]
        assert arms_out["Random-LM-Tight"]["trainable_params"] == dual_p
        assert arms_out["Random-LM-seed2-Tight"]["trainable_params"] == dual_p
        assert dual_p <= u_params, f"Targeted-Dual-Tight ({dual_p}) must be <= Uniform-LoRA-Tight ({u_params})"
        # Verify MLP parameter share matches Standard Targeted-Dual within +-3.0 pp
        std_arm = m["arms"]["Targeted-Dual"]
        std_mlp_share = E.count_mlp_lora_params(len(std_arm["mlp_layers"]), std_arm["mlp_rank"], d, d_ff) / std_arm["trainable_params"]
        tight_mlp_share = E.count_mlp_lora_params(len(std_arm["mlp_layers"]), r_m, d, d_ff) / dual_p
        assert abs(tight_mlp_share - std_mlp_share) <= 0.03, (
            f"{mk} MLP share drift too large: tight={tight_mlp_share:.3f} vs std={std_mlp_share:.3f}"
        )

        out_models[mk] = {
            "model_spec": ms,
            "budget_anchor_params": dual_p,
            "hub_heads": m["hub_heads"],
            "source_frozen_components_sha256": src_sha,
            "arms": arms_out,
        }

    payload = {
        "schema": "cedar.acl_exp_c1.components_step2_tight.v1",
        "source_components_file": "experiments/ACL-exp-c1/data/c1_components_frozen.json",
        "source_components_sha256": src_sha,
        "models": out_models,
    }
    out_path = C1 / "data" / "c1_components_step2_tight.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    out_sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    print(f"[OK] wrote {out_path} (sha256={out_sha[:16]}...)")
    for mk, m in out_models.items():
        u_p = m["arms"]["Uniform-LoRA-Tight"]["trainable_params"]
        d_p = m["arms"]["Targeted-Dual-Tight"]["trainable_params"]
        pct_diff = 100.0 * (d_p - u_p) / u_p
        print(f"  {mk}: Dual-Tight={d_p:,} vs Uniform-r1={u_p:,} ({pct_diff:+.2f}%), Random-1/2 exact={d_p:,}")


if __name__ == "__main__":
    main()
