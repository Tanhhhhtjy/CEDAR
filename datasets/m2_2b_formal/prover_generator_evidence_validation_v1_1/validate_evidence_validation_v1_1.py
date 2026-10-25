#!/usr/bin/env python3
"""Independent zero-GPU validator for the Evidence-only amendment."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from evidence_validation_common import (
    ASSET_REPO,
    ASSET_REVISION,
    CAP_EXPERIMENT,
    CAP_LAYERS,
    CONDITIONS,
    EVIDENCE_ARMS,
    EXPECTED_AXIS_SHA256,
    EXPECTED_CAP_CONFIG_SHA256,
    EXPECTED_INPUT_ROWS,
    EXPECTED_RESULT_ROWS,
    FROZEN_CANONICAL_SHA256,
    HERE,
    MODEL_REPO,
    MODEL_REVISION,
    PARENT_RESULT_COMMIT,
    ROOT,
    canonical_sha256,
    canonical_text_sha256,
    read_json,
    select_validation_rows,
    validate_module_selection,
    write_json,
)

TRACKED_RELEASE_FILES = (
    "datasets/m2_2b_formal/prover_generator_evidence_validation_v1_1/EVIDENCE_ONLY_AMENDMENT.json",
    "datasets/m2_2b_formal/prover_generator_evidence_validation_v1_1/GATE_SPEC.json",
    "datasets/m2_2b_formal/prover_generator_evidence_validation_v1_1/HUMAN_REVIEW_CARRY_FORWARD.json",
    "datasets/m2_2b_formal/prover_generator_evidence_validation_v1_1/README.md",
    "datasets/m2_2b_formal/prover_generator_evidence_validation_v1_1/evidence_validation_common.py",
    "datasets/m2_2b_formal/prover_generator_evidence_validation_v1_1/run_cpu_preflight.py",
    "datasets/m2_2b_formal/prover_generator_evidence_validation_v1_1/validate_evidence_validation_v1_1.py",
    "docs/PROVER_GENERATOR_EVIDENCE_VALIDATION_V1_1_PROTOCOL.md",
    "scripts/run_prover_generator_evidence_validation_v1_1.py",
    "scripts/audit_prover_generator_evidence_validation_v1_1.py",
    "scripts/analyze_prover_generator_evidence_validation_v1_1.py",
    "tests/test_prover_generator_evidence_validation_v1_1.py",
)


def _preflight_passed(preflight: dict[str, Any]) -> bool:
    p25 = preflight.get("official_p25", {})
    labels = preflight.get("answer_label_token_ids", {})
    selection = preflight.get("selection", {})
    return bool(
        preflight.get("status") == "passed"
        and preflight.get("gpu_used") is False
        and preflight.get("model") == {"repo": MODEL_REPO, "revision": MODEL_REVISION}
        and selection.get("rows") == EXPECTED_INPUT_ROWS
        and selection.get("families") == 12
        and selection.get("bases") == 24
        and labels == {"A": [32], "B": [33], "C": [34], "D": [35]}
        and isinstance(preflight.get("rendered_prompts_sha256"), str)
        and len(preflight["rendered_prompts_sha256"]) == 64
        and isinstance(preflight.get("tokenized_input_ids_sha256"), str)
        and len(preflight["tokenized_input_ids_sha256"]) == 64
        and p25.get("repo") == ASSET_REPO
        and p25.get("revision") == ASSET_REVISION
        and p25.get("axis_sha256") == EXPECTED_AXIS_SHA256
        and p25.get("capping_config_sha256") == EXPECTED_CAP_CONFIG_SHA256
        and p25.get("experiment") == CAP_EXPERIMENT
        and tuple(p25.get("layers", [])) == CAP_LAYERS
    )


def validate(root: Path = HERE, require_release_gate: bool = False) -> dict[str, Any]:
    module_selection = validate_module_selection()
    rows = select_validation_rows()
    amendment = read_json(root / "EVIDENCE_ONLY_AMENDMENT.json")
    gate_spec = read_json(root / "GATE_SPEC.json")
    review = read_json(root / "HUMAN_REVIEW_CARRY_FORWARD.json")
    preflight = read_json(root / "validation_input_fingerprints.json")

    base_rows = {row["base_id"]: row for row in rows}
    automatic_checks = {
        "frozen_parent_and_calibration_hashes": True,
        "calibration_module_selection_recomputed": module_selection["evidence_module"]["status"] == "selected_for_internal_validation",
        "social_module_excluded": module_selection["social_module"]["status"] == "excluded_from_formal_prover_evaluation",
        "parent_result_commit_frozen": amendment.get("parent_result_commit") == PARENT_RESULT_COMMIT,
        "scope_is_evidence_only": amendment.get("authorization", {}).get("scope") == "evidence_only_internal_validation",
        "single_uninterrupted_matrix": amendment.get("authorization", {}).get("single_uninterrupted_matrix") is True,
        "exact_input_rows": len(rows) == EXPECTED_INPUT_ROWS,
        "exact_result_rows": amendment.get("authorization", {}).get("formal_generations") == EXPECTED_RESULT_ROWS,
        "arm_balance": Counter(row["arm"] for row in rows) == Counter({arm: 24 for arm in EVIDENCE_ARMS}),
        "condition_contract": tuple(amendment.get("authorization", {}).get("conditions", [])) == CONDITIONS,
        "validation_families_unseen": len({row["family_id"] for row in rows}) == 12,
        "validation_bases": len(base_rows) == 24,
        "position_balance": set(Counter(row["position_transition"] for row in base_rows.values()).values()) == {2},
        "variant_transition_balance": set(Counter((row["position_transition"], row["variant"]) for row in base_rows.values()).values()) == {1},
        "thinking_4096": gate_spec.get("execution", {}).get("mode") == "thinking" and gate_spec.get("execution", {}).get("max_new_tokens") == 4096,
        "no_retry_or_repair": gate_spec.get("execution", {}).get("content_retry") is False and gate_spec.get("execution", {}).get("repair") is False,
        "itt_and_bounds": gate_spec.get("analysis", {}).get("itt_primary") is True and gate_spec.get("analysis", {}).get("report_sensitivity_bounds") is True,
        "old_parent_gate_not_rewritten": read_json(ROOT / "datasets/m2_2b_formal/prover_generator_development_v1_1/ZERO_GPU_GATE.json").get("p25_authorized") is False,
        "human_review_carried_without_text_change": review.get("status") == "passed_by_frozen_parent_review" and review.get("language_or_logic_changed") is False and review.get("parent_audit_canonical_sha256") == FROZEN_CANONICAL_SHA256["HUMAN_NATURALIZATION_AUDIT.json"],
    }
    expected_ids_hash = canonical_sha256([row["id"] for row in rows])
    expected_messages_hash = canonical_sha256([{"id": row["id"], "messages": row["messages"]} for row in rows])
    release_checks = {
        "fixed_qwen_and_p25_asset_preflight": _preflight_passed(preflight)
        and preflight.get("selection", {}).get("ids_sha256") == expected_ids_hash
        and preflight.get("selection", {}).get("messages_sha256") == expected_messages_hash
    }
    release_files = {}
    for relative in TRACKED_RELEASE_FILES:
        path = ROOT / relative
        if path.is_file():
            release_files[relative] = {"canonical_text_sha256": canonical_text_sha256(path)}
    automatic_checks["all_release_sources_present"] = len(release_files) == len(TRACKED_RELEASE_FILES)
    release_ready = all(automatic_checks.values()) and all(release_checks.values())

    report = {
        "schema": "cedar.prover_generator.evidence_validation_v1_1.validation.v1",
        "status": "release_ready" if release_ready else "automatic_preflight_passed",
        "passed": all(automatic_checks.values()),
        "release_ready": release_ready,
        "automatic_checks": automatic_checks,
        "release_checks": release_checks,
        "counts": {
            "families": len({row["family_id"] for row in rows}),
            "bases": len(base_rows),
            "input_rows": len(rows),
            "formal_results": EXPECTED_RESULT_ROWS,
            "arms": dict(sorted(Counter(row["arm"] for row in rows).items())),
        },
        "selection": {
            "ids_sha256": expected_ids_hash,
            "messages_sha256": expected_messages_hash,
            "ordered_cell_keys_sha256": canonical_sha256(
                [{"id": row["id"], "condition": condition} for row in rows for condition in CONDITIONS]
            ),
        },
        "module_selection": module_selection,
        "frozen_sources": FROZEN_CANONICAL_SHA256,
        "release_files": release_files,
        "gpu_used": False,
    }
    write_json(root / "independent_validation_report.json", report)
    manifest = {
        "schema": "cedar.prover_generator.evidence_validation_v1_1.manifest.v1",
        "status": "ready_for_paired_evidence_validation" if release_ready else "pending_cpu_preflight",
        "parent_result_commit": PARENT_RESULT_COMMIT,
        "scope": "evidence_only_internal_validation",
        "formal_rows": EXPECTED_RESULT_ROWS,
        "conditions": list(CONDITIONS),
        "arms": list(EVIDENCE_ARMS),
        "selection": report["selection"],
        "input_fingerprints_canonical_sha256": canonical_text_sha256(root / "validation_input_fingerprints.json"),
        "release_files": release_files,
        "p25_authorized_scope": "evidence_only_internal_validation" if release_ready else None,
        "p1_authorized": False,
        "social_formal_prover_authorized": False,
    }
    write_json(root / "manifest.json", manifest)
    gate = {
        "schema": "cedar.prover_generator.evidence_validation_v1_1.gate.v1",
        "gate": "E1-paired-evidence-validation" if release_ready else "pending",
        "may_run_paired_validation": release_ready,
        "p25_authorized_scope": "evidence_only_internal_validation" if release_ready else None,
        "formal_rows": EXPECTED_RESULT_ROWS,
        "p1_authorized": False,
        "social_formal_prover_authorized": False,
        "automatic_checks": automatic_checks,
        "release_checks": release_checks,
        "blockers": [name for name, passed in release_checks.items() if not passed],
    }
    write_json(root / "EVIDENCE_VALIDATION_GATE.json", gate)
    if require_release_gate and not release_ready:
        raise ValueError(f"Evidence validation release gate is not ready: {gate['blockers']}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--require-release-gate", action="store_true")
    args = parser.parse_args()
    result = validate(args.root.resolve(), require_release_gate=args.require_release_gate)
    print(
        {
            "status": result["status"],
            "passed": result["passed"],
            "release_ready": result["release_ready"],
            **result["counts"],
            "gpu_used": False,
        }
    )


if __name__ == "__main__":
    main()
