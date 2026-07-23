#!/usr/bin/env python3
"""Bind genuinely completed C3 review packets to a single-review attestation."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cedar_m2.checkpoint import atomic_write_json, object_sha256  # noqa: E402
from scripts.c3_portable_hashes import canonical_text_sha256  # noqa: E402

DATA_ROOT = ROOT / "datasets" / "c3_sycobench_bidirectional_selectivity"
REVIEW_ROOT = DATA_ROOT / "review"
VALIDATOR_PATH = DATA_ROOT / "validate_c3_sycobench_bidirectional.py"


def load_validator():
    spec = importlib.util.spec_from_file_location("c3_review_validator", VALIDATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load C3 validator: {VALIDATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use only after the named data lead personally completes both frozen review CSVs."
        )
    )
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--confirm-no-model-outputs", action="store_true")
    parser.add_argument("--confirm-no-gpu-execution", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reviewer_id = args.reviewer_id.strip()
    if not reviewer_id or reviewer_id.lower() in {
        "todo",
        "unknown",
        "placeholder",
        "data_lead_id",
        "reviewer_id",
        "data_lead_internal_xx",
    }:
        raise ValueError("A real internal reviewer ID is required")
    if not args.confirm_no_model_outputs or not args.confirm_no_gpu_execution:
        raise ValueError("Both pre-output and pre-GPU confirmations are required")
    validator = load_validator()
    completed = validator._compare_completed(  # noqa: SLF001
        REVIEW_ROOT / "review_packet32.csv",
        REVIEW_ROOT / "review_completed.csv",
        validator.REVIEW_FIELDS,
        "review_id",
    )
    completed_templates = validator._compare_completed(  # noqa: SLF001
        REVIEW_ROOT / "template_review_packet.csv",
        REVIEW_ROOT / "template_review_completed.csv",
        validator.TEMPLATE_REVIEW_FIELDS,
        "template_id",
    )
    hashes = {
        **validator.review_binding_hashes(),
        "review_completed_semantic_sha256": object_sha256(completed),
        "template_review_completed_semantic_sha256": object_sha256(completed_templates),
        "review_completed_canonical_text_sha256": canonical_text_sha256(
            REVIEW_ROOT / "review_completed.csv"
        ),
        "template_review_completed_canonical_text_sha256": canonical_text_sha256(
            REVIEW_ROOT / "template_review_completed.csv"
        ),
    }
    attestation = {
        "status": "approved",
        "protocol_deviation": False,
        "review_mode": "single_data_lead_review",
        "reviewer_id": reviewer_id,
        "direction_cases_reviewed": 32,
        "template_pairs_reviewed": 4,
        "all_items_passed": True,
        "model_outputs_seen": False,
        "gpu_execution_seen": False,
        "dual_independent_review": False,
        "cohen_kappa": None,
        "limitations": [
            "No independent second reviewer",
            "No inter-annotator agreement estimate",
            "No Cohen kappa",
        ],
        "hashes": hashes,
    }
    output = REVIEW_ROOT / "single_review_attestation.json"
    atomic_write_json(output, attestation)
    gate = validator.validate_review_gate(validator.load_builder())
    atomic_write_json(DATA_ROOT / "HUMAN_REVIEW_GATE.json", gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
