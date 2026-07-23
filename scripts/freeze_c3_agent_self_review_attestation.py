#!/usr/bin/env python3
"""Bind completed C3 agent review packets under an explicit user waiver."""

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
            "Use only after an agent has inspected all 32 cases and four template pairs "
            "under the user's explicit pre-GPU human-review waiver."
        )
    )
    parser.add_argument("--review-agent-id", required=True)
    parser.add_argument("--confirm-user-waiver", action="store_true")
    parser.add_argument("--confirm-no-model-outputs", action="store_true")
    parser.add_argument("--confirm-no-gpu-execution", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    review_agent_id = args.review_agent_id.strip()
    if not review_agent_id or review_agent_id.lower() in {
        "todo",
        "unknown",
        "placeholder",
        "agent_id",
        "review_agent_id",
    }:
        raise ValueError("A concrete review agent ID is required")
    if not (
        args.confirm_user_waiver and args.confirm_no_model_outputs and args.confirm_no_gpu_execution
    ):
        raise ValueError("User-waiver, pre-output, and pre-GPU confirmations are all required")
    validator = load_validator()
    completed = validator._compare_completed(  # noqa: SLF001
        REVIEW_ROOT / "review_packet32.csv",
        REVIEW_ROOT / "agent_review_completed.csv",
        validator.REVIEW_FIELDS,
        "review_id",
    )
    completed_templates = validator._compare_completed(  # noqa: SLF001
        REVIEW_ROOT / "template_review_packet.csv",
        REVIEW_ROOT / "template_agent_review_completed.csv",
        validator.TEMPLATE_REVIEW_FIELDS,
        "template_id",
    )
    hashes = {
        **validator.review_binding_hashes(),
        "review_completed_semantic_sha256": object_sha256(completed),
        "template_review_completed_semantic_sha256": object_sha256(completed_templates),
        "review_completed_canonical_text_sha256": canonical_text_sha256(
            REVIEW_ROOT / "agent_review_completed.csv"
        ),
        "template_review_completed_canonical_text_sha256": canonical_text_sha256(
            REVIEW_ROOT / "template_agent_review_completed.csv"
        ),
    }
    attestation = {
        "status": "approved",
        "protocol_deviation": True,
        "review_mode": "agent_self_review_user_waiver",
        "review_agent_id": review_agent_id,
        "direction_cases_reviewed": 32,
        "template_pairs_reviewed": 4,
        "all_items_passed": True,
        "model_outputs_seen": False,
        "gpu_execution_seen": False,
        "human_review_performed": False,
        "data_lead_review_performed": False,
        "user_review_waived": True,
        "dual_independent_review": False,
        "raw_agreement": None,
        "cohen_kappa": None,
        "user_authorization": {
            "date": "2026-07-23",
            "decision": "human_review_waived_agent_self_review_authorized",
            "quoted_instruction": "不用我审核了，你直接自检一遍然后继续跑",
        },
        "limitations": [
            "No human or data-lead review",
            "No independent reviewer",
            "No inter-annotator agreement estimate",
            "No Cohen kappa",
        ],
        "hashes": hashes,
    }
    output = REVIEW_ROOT / "agent_self_review_attestation.json"
    atomic_write_json(output, attestation)
    gate = validator.validate_review_gate(validator.load_builder())
    atomic_write_json(DATA_ROOT / "REVIEW_GATE.json", gate)
    print(json.dumps(gate, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
