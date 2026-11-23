#!/usr/bin/env python3
"""Deterministically stage Stage 1 baselines (`M0, u1_base, u2_corridor, m2_corridor`) into `results/stage2/`.

Combines:
  - Fixed `c3_blind_confirm_v2` `no_think` records from `experiments/ACL-exp-c3.5/results/stage1/baselines_v2/<arm>/`
  - Frozen C3 `reference_kl`, `capabilities/summary.json`, `vllm_lora_parity.json`, `train_meta.json`, `dev_selection.json`
    from `experiments/ACL-exp-c3/results/c3_8b_showdown/<arm>/` (zero recomputation of identical checkpoints!)

Usage:
  python experiments/ACL-exp-c3.5/scripts/c35_stage_baselines.py --out-root experiments/ACL-exp-c3.5/results/stage2
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
C35 = HERE.parents[0]
REPO = C35.parents[1]
C3_RES = REPO / "experiments" / "ACL-exp-c3" / "results" / "c3_8b_showdown"
S1_BASE = C35 / "results" / "stage1" / "baselines_v2"


def stage_baselines(out_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    for arm in ("M0", "u1_base", "u2_corridor", "m2_corridor"):
        src_nt = S1_BASE / arm / "no_think"
        assert src_nt.is_dir(), f"Missing Stage 1 baseline dir: {src_nt}"
        dst_arm = out_root / arm
        dst_nt = dst_arm / "no_think"
        dst_nt.mkdir(parents=True, exist_ok=True)

        # Copy blind_v2 no_think records + summary
        for fn in ("records_no_think_c3_blind_confirm_v2.jsonl", "eval_summary_no_think.json"):
            src_f = src_nt / fn
            assert src_f.is_file(), f"Missing {src_f}"
            shutil.copy2(src_f, dst_nt / fn)

        if arm == "M0":
            continue

        c3_arm = C3_RES / arm
        assert c3_arm.is_dir(), f"Missing C3 baseline dir: {c3_arm}"

        # Inject frozen C3 exact reference_kl into Stage 2 eval_summary_no_think.json
        c3_nt_sum = json.loads((c3_arm / "no_think" / "eval_summary_no_think.json").read_text(encoding="utf-8"))
        ref_kl = c3_nt_sum.get("reference_kl")
        assert ref_kl and ref_kl.get("kl_mean") is not None, f"Missing reference_kl in {c3_arm}"
        s2_sum_p = dst_nt / "eval_summary_no_think.json"
        s2_sum = json.loads(s2_sum_p.read_text(encoding="utf-8"))
        s2_sum["reference_kl"] = ref_kl
        s2_sum["reference_kl_source"] = f"experiments/ACL-exp-c3/results/c3_8b_showdown/{arm}/no_think/eval_summary_no_think.json"
        s2_sum_p.write_text(json.dumps(s2_sum, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")

        # Copy frozen C3 capabilities, parity, train_meta, dev_selection, and historical c2 records
        cap_src = c3_arm / "capabilities" / "summary.json"
        assert cap_src.is_file(), f"Missing {cap_src}"
        (dst_arm / "capabilities").mkdir(parents=True, exist_ok=True)
        shutil.copy2(cap_src, dst_arm / "capabilities" / "summary.json")

        for fn in ("vllm_lora_parity.json", "train_meta.json", "dev_selection.json"):
            f_src = c3_arm / fn
            assert f_src.is_file(), f"Missing {f_src}"
            shutil.copy2(f_src, dst_arm / fn)

        for hist_fn in ("records_no_think_c2_confirm_bidir.jsonl", "records_no_think_c2_stealth_stress.jsonl"):
            h_src = c3_arm / "no_think" / hist_fn
            if h_src.is_file():
                shutil.copy2(h_src, dst_nt / hist_fn)

        cap = json.loads(cap_src.read_text(encoding="utf-8"))
        par = json.loads((c3_arm / "vllm_lora_parity.json").read_text(encoding="utf-8"))
        print(
            f"[OK] staged {arm:12s}: kl={ref_kl['kl_mean']:.6f} "
            f"ifeval={cap['ifeval']['n_correct_prompts']} gsm8k={cap['gsm8k']['n_correct']} parity={par['passed']}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", type=Path, default=C35 / "results" / "stage2")
    ap.add_argument("--selftest-tiny", action="store_true")
    args = ap.parse_args()
    if args.selftest_tiny:
        with tempfile.TemporaryDirectory() as td:
            stage_baselines(Path(td))
        print("[OK] c35_stage_baselines self-test PASSED!")
        return
    stage_baselines(args.out_root)


if __name__ == "__main__":
    main()
