#!/usr/bin/env python3
"""Validate the completed CEDAR transport delivery without modifying it."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_RECORD_FIELDS = {
    "model",
    "model_revision",
    "family_id",
    "base_id",
    "arm",
    "intervention",
    "prompt_hash",
    "prompt_text_or_stable_id",
    "answer_order",
    "initial_option",
    "target_option",
    "option_token_ids",
    "all_option_logits",
    "predicted_option",
    "output_state",
    "target_follow",
    "m_general",
    "m_target",
    "m_background",
    "m_specific",
    "trigger_count",
    "correction_count",
    "correction_norm",
    "option_probability_mass",
    "entropy",
    "top_token",
    "truncation_flag",
    "invalid_flag",
    "axis_hash",
    "threshold_hash",
    "layer_range",
    "seed",
    "runner_commit",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def csv_rows(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    records = [json.loads(line) for line in (ROOT / "raw_records.jsonl").read_text(encoding="utf-8").splitlines() if line]
    assert len(records) == 384
    assert {row["model"] for row in records} == {"Qwen-FC", "Llama"}
    assert all(REQUIRED_RECORD_FIELDS <= set(row) for row in records)
    assert sum(bool(row["invalid_flag"]) for row in records) == 0
    assert sum(bool(row["truncation_flag"]) for row in records) == 0
    assert all(row["threshold_hash"] is None and row["runner_commit"] is None for row in records)

    e2a = csv_rows("e2a_family_geometry.csv")
    assert len(e2a) == 12
    assert sum(int(row["n_bases"]) for row in e2a) == 24
    assert abs(sum(float(row["delta_m_general"]) for row in e2a) / 12 - 3.1302083333333335) < 1e-12
    assert len(csv_rows("e2a_family_arm_geometry.csv")) == 96
    assert len(csv_rows("cross_model_family_effects.csv")) == 96
    assert len(csv_rows("cross_model_target_follow.csv")) == 4
    assert len(csv_rows("pair_audit.csv")) == 192
    assert all(row["pair_status"] == "paired" for row in csv_rows("pair_audit.csv"))
    assert len(csv_rows("figure2d_forest_input.csv")) == 4

    summary = {(row["model"], row["estimand"]): row for row in csv_rows("cross_model_summary.csv")}
    assert abs(float(summary[("Qwen-FC", "overall_delta_m_general")]["estimate"]) - 2.0746527777777777) < 1e-12
    assert abs(float(summary[("Llama", "overall_delta_m_general")]["estimate"]) - 1.3910590277777777) < 1e-12
    assert float(summary[("Llama", "overall_delta_m_general")]["ci_low"]) > 0
    assert float(summary[("Qwen-FC", "v_minus_f_continuous_selectivity_change")]["ci_low"]) <= 0 <= float(summary[("Qwen-FC", "v_minus_f_continuous_selectivity_change")]["ci_high"])
    assert float(summary[("Llama", "v_minus_f_continuous_selectivity_change")]["ci_low"]) <= 0 <= float(summary[("Llama", "v_minus_f_continuous_selectivity_change")]["ci_high"])

    for stem in ("e2a_family_geometry", "figure2d_cross_model_transport", "cross_model_paired_family_geometry"):
        for suffix in (".svg", ".pdf", ".png"):
            path = ROOT / "figures" / f"{stem}{suffix}"
            assert path.is_file() and path.stat().st_size > 10_000

    manifest = json.loads((ROOT / "delivery_manifest.json").read_text(encoding="utf-8"))
    for item in manifest["files"]:
        path = ROOT / item["file"]
        assert path.is_file(), path
        assert path.stat().st_size == item["bytes"], path
        assert sha256_file(path) == item["sha256"], path
    assert manifest["route"] == "G-A transport"
    print(json.dumps({"status": "passed", "records": len(records), "route": manifest["route"]}, sort_keys=True))


if __name__ == "__main__":
    main()
