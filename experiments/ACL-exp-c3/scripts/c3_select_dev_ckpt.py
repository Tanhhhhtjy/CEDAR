#!/usr/bin/env python3
"""Automated, unbiased `c3_dev` checkpoint selector for ACL-exp-c3 (keeps `c3_blind_confirm` 100% blind!).

Evaluates all saved step checkpoints (`step0050`, `step0100`, `step0200`, `step0300` / final `adapter_state.pt`)
for ONE arm on ONE loaded base model instance (zero model reload overhead, ~45s total on 1 A800!) over `c3_dev`
(180 bases, 6 core arms: `turn1_F, turn1_V, V_hes, V_auth, F_hes, F_auth`).

Selection rule (frozen identically across ALL 8 arms `U1..U4`, `M1..M4`):
  1. Primary criterion: highest `c3_dev` family-weighted 4-cell all-correct rate (`four_cell_family_weighted`).
  2. Tie-breaker (within 1e-6): lowest 4-option gold NLL (`mean_4opt_nll`).
  3. Writes `dev_selection.json` recording ALL step scores + exact selected checkpoint path & SHA-256.

CPU Self-Test:
  python experiments/ACL-exp-c3/scripts/c3_select_dev_ckpt.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
C3 = HERE.parents[0]
REPO = C3.parents[1]

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-a1" / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-b1" / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-c1" / "scripts"))

import c1_eval as C1EV  # noqa: E402
import c3_eval as C3EV  # noqa: E402
import c3_modules as M3  # noqa: E402
import c3_stats as C3S  # noqa: E402
from render_prompts import load_config  # noqa: E402

DEV_ARMS = ["turn1_F", "turn1_V", "V_hes", "V_auth", "F_hes", "F_auth"]


def _discover_checkpoints(ckpt_dir: Path) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    for p in sorted(ckpt_dir.glob("adapter_state_step*.pt")):
        step_tag = p.stem.replace("adapter_state_", "")
        found.append((step_tag, p))
    final_p = ckpt_dir / "adapter_state.pt"
    if final_p.is_file():
        found.append(("final", final_p))
    assert found, f"No adapter_state*.pt found in {ckpt_dir}"
    return found


def select_best_entry(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic selection: highest `four_cell_family_weighted`, tie-break lowest `mean_4opt_nll`."""
    assert entries
    return max(
        entries,
        key=lambda e: (
            round(float(e["four_cell_family_weighted"]), 6),
            -(float(e["mean_4opt_nll"]) if e.get("mean_4opt_nll") is not None else 1e9),
        ),
    )


def run_selftest_tiny() -> None:
    entries = [
        {"tag": "step0050", "four_cell_family_weighted": 0.65, "mean_4opt_nll": 0.40},
        {"tag": "step0100", "four_cell_family_weighted": 0.80, "mean_4opt_nll": 0.30},
        {"tag": "step0200", "four_cell_family_weighted": 0.80, "mean_4opt_nll": 0.18},
        {"tag": "final", "four_cell_family_weighted": 0.78, "mean_4opt_nll": 0.15},
    ]
    best = select_best_entry(entries)
    assert best["tag"] == "step0200", best
    print(f"[OK] c3_select_dev_ckpt self-test PASSED: selected={best['tag']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-tiny", action="store_true")
    ap.add_argument("--model", type=str)
    ap.add_argument("--model-key", choices=["qwen3_8b", "qwen3_14b"], default="qwen3_8b")
    ap.add_argument("--arch", type=str)
    ap.add_argument("--ckpt-dir", type=Path, help="Directory with adapter_state_step*.pt and adapter_state.pt")
    ap.add_argument("--components-file", type=Path, default=C3 / "data" / "c3_components_frozen.json")
    ap.add_argument("--eligible-ids-in", type=Path, default=None, help="Optional frozen M0 eligible_ids_no_think.json")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-dir", type=Path, required=False)
    args = ap.parse_args()

    if args.selftest_tiny:
        run_selftest_tiny()
        return

    assert args.model and args.arch and args.ckpt_dir
    out_dir = args.out_dir or args.ckpt_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    a1_cfg = load_config()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    opt_ids = C1EV.get_cached_option_token_ids(tok)
    dev_rows = [json.loads(l) for l in C3EV.C3_DATASETS["c3_dev"].read_text(encoding="utf-8").splitlines() if l.strip()]

    elig_set: set[str] | None = None
    if args.eligible_ids_in and args.eligible_ids_in.is_file():
        raw_el = json.loads(args.eligible_ids_in.read_text(encoding="utf-8"))
        if "c3_dev" in raw_el:
            elig_set = set(raw_el["c3_dev"])

    comp = json.loads(args.components_file.read_text(encoding="utf-8"))
    arch_spec = comp["models"][args.model_key]["architectures"][args.arch]
    ckpts = _discover_checkpoints(args.ckpt_dir)
    scaling = C3EV._read_scaling(ckpts[0][1])

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    ).eval()
    M3.apply_c3_adapters(model, arch_spec, **scaling)

    entries: list[dict[str, Any]] = []
    for tag, ckpt_path in ckpts:
        M3.load_c3_adapter_state(model, ckpt_path)
        model.eval()
        recs = C3EV.eval_nothink_dataset_batched(model, tok, a1_cfg, dev_rows, DEV_ARMS, opt_ids, batch_size=args.batch_size)
        bm = C3S.compute_base_metrics(recs, eligible_ids=elig_set)
        s = C3S.summarize_base_map(bm)
        sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
        entry = {
            "tag": tag,
            "path": str(ckpt_path),
            "sha256": sha,
            "n_items": s["n_items"],
            "n_families": s["n_families"],
            "four_cell_family_weighted": s["four_cell_family_weighted"],
            "dc_vhes_fauth_fw": s["dc_vhes_fauth_fw"],
            "dc_vauth_fhes_fw": s["dc_vauth_fhes_fw"],
            "mean_4opt_nll": s["mean_4opt_nll"],
            "mean_4opt_brier": s["mean_4opt_brier"],
            "ce_row_B_4cell": (s.get("subgroups", {}).get("ce_row_B") or {}).get("four_cell"),
            "step_3_4cell": (s.get("subgroups", {}).get("step_3") or {}).get("four_cell"),
            "zone_norms": M3.zone_delta_norms(model),
        }
        entries.append(entry)
        print(
            f"[dev_select] {tag:8s} | 4cell={entry['four_cell_family_weighted']:.4f} "
            f"| DC={entry['dc_vhes_fauth_fw']:.3f}/{entry['dc_vauth_fhes_fw']:.3f} "
            f"| RowB={entry['ce_row_B_4cell']} | Step3={entry['step_3_4cell']} | NLL={entry['mean_4opt_nll']:.3f}"
        )

    best = select_best_entry(entries)
    sel_copy = out_dir / "selected_adapter_state.pt"
    shutil.copy2(best["path"], sel_copy)
    # Also copy adapter_config.json / train_meta.json next to selected_adapter_state.pt if out_dir != ckpt_dir
    if out_dir.resolve() != args.ckpt_dir.resolve():
        for fn in ("adapter_config.json", "train_meta.json"):
            src_f = args.ckpt_dir / fn
            if src_f.is_file():
                shutil.copy2(src_f, out_dir / fn)

    report = {
        "schema": "cedar.acl_exp_c3.dev_selection.v1",
        "arch": args.arch,
        "ckpt_dir": str(args.ckpt_dir),
        "selected_tag": best["tag"],
        "selected_path": best["path"],
        "selected_copy": str(sel_copy),
        "selected_sha256": best["sha256"],
        "selected_four_cell_fw": best["four_cell_family_weighted"],
        "candidates": entries,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (out_dir / "dev_selection.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"[OK] Selected {best['tag']} (4cell={best['four_cell_family_weighted']:.4f}, sha={best['sha256'][:12]}) -> {sel_copy}")


if __name__ == "__main__":
    main()
