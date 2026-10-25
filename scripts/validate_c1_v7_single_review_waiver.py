#!/usr/bin/env python3
"""Validate the transparent C1/V7 single-review waiver before any GPU run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "datasets" / "m2_2b_formal" / "v7_heldout_confirmation"
EXPECTED_STATUS = "approved_for_execution_with_single_review_protocol_deviation"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate(data_dir: Path) -> dict[str, Any]:
    waiver_path = data_dir / "single_review_waiver.json"
    base_path = data_dir / "m2_2b_v7_heldout48_base.jsonl"
    arms_path = data_dir / "m2_2b_v7_heldout192_arms.jsonl"
    waiver = json.loads(waiver_path.read_text(encoding="utf-8"))
    bases = load_jsonl(base_path)
    arms = load_jsonl(arms_path)

    checks = {
        "status_exact": waiver.get("status") == EXPECTED_STATUS,
        "reviewer_count_one": waiver.get("reviewer_count") == 1,
        "second_reviewer_false": waiver.get("independent_second_reviewer_completed") is False,
        "item_labels_false": waiver.get("item_level_labels_recorded") is False,
        "agreement_unavailable": waiver.get("inter_rater_agreement") is None,
        "kappa_unavailable": waiver.get("cohens_kappa") is None,
        "disclosure_required": waiver.get("disclosure_required") is True,
        "no_model_outputs_seen": waiver.get("model_outputs_seen") is False,
        "pre_gpu_waiver": waiver.get("qwen_gpu_or_capping_run_before_waiver") is False,
        "base_count_48": len(bases) == waiver.get("base_cards") == 48,
        "arm_count_192": len(arms) == waiver.get("arms") == 192,
        "base_ids_unique": len({row.get("base_id") for row in bases}) == 48,
        "arm_ids_unique": len({row.get("id") for row in arms}) == 192,
        "base_hash_bound": sha256(base_path) == waiver.get("base_cards_sha256"),
        "arms_hash_bound": sha256(arms_path) == waiver.get("arms_sha256"),
        "all_four_arms": {
            arm: sum(row.get("arm") == arm for row in arms)
            for arm in ("N", "V", "AO", "S")
        }
        == {"N": 48, "V": 48, "AO": 48, "S": 48},
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "status": "passed" if not failures else "failed",
        "checks": checks,
        "failures": failures,
        "waiver_sha256": sha256(waiver_path),
        "base_sha256": sha256(base_path),
        "arms_sha256": sha256(arms_path),
        "scientific_disclosure": (
            "single data-lead global review; no item-level labels, second reviewer, "
            "inter-rater agreement, or Cohen's kappa"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    report = validate(args.data_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

