#!/usr/bin/env python3
"""Deterministic CPU release validator and E2A-only Gate writer."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_e2_phase_validity_choice import balanced_mapping
from e2_common import (
    ANCHOR_CPU_GATE_COMMIT,
    ANCHOR_RESULT_COMMIT,
    ANSWER_CONTRACT,
    EXPECTED_E2A_CELLS,
    FORMAL_CONDITIONS,
    SCHEMA,
    answer_rows,
    base_rows,
    canonical_text_sha256,
    object_sha256,
    read_json,
    read_jsonl,
    require,
    validate_anchor,
    write_json,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

SMOKE_IDS = {
    "answer": [
        "CEDAR-DEV-003-N",
        "CEDAR-DEV-004-R",
        "CEDAR-DEV-007-V",
        "CEDAR-DEV-008-F",
    ],
    "validity": [
        "CEDAR-DEV-003-V-VALIDITY",
        "CEDAR-DEV-004-F-VALIDITY",
        "CEDAR-DEV-007-F-VALIDITY",
        "CEDAR-DEV-008-V-VALIDITY",
    ],
}

RELEASE_PATHS = (
    "docs/E2_PHASE_VALIDITY_CHOICE_PROTOCOL.md",
    "src/cedar_m2/interventions/phase_capping.py",
    "scripts/run_e2_phase_validity_choice.py",
    "scripts/audit_e2_phase_validity_choice.py",
    "scripts/analyze_e2_phase_validity_choice.py",
    "tests/test_e2_phase_validity_choice.py",
    "datasets/m2_2b_formal/e2_phase_validity_choice/README.md",
    "datasets/m2_2b_formal/e2_phase_validity_choice/ANALYSIS_PLAN.json",
    "datasets/m2_2b_formal/e2_phase_validity_choice/CPU_TEST_REPORT.json",
    "datasets/m2_2b_formal/e2_phase_validity_choice/e2_common.py",
    "datasets/m2_2b_formal/e2_phase_validity_choice/build_e2_phase_validity_choice.py",
    "datasets/m2_2b_formal/e2_phase_validity_choice/finalize_human_review.py",
    "datasets/m2_2b_formal/e2_phase_validity_choice/run_cpu_preflight.py",
    "datasets/m2_2b_formal/e2_phase_validity_choice/validate_e2_phase_validity_choice.py",
    "datasets/m2_2b_formal/e2_phase_validity_choice/answer_inputs96.jsonl",
    "datasets/m2_2b_formal/e2_phase_validity_choice/validity_inputs48.jsonl",
    "datasets/m2_2b_formal/e2_phase_validity_choice/validity_mapping.json",
    "datasets/m2_2b_formal/e2_phase_validity_choice/human_review.json",
    "datasets/m2_2b_formal/e2_phase_validity_choice/tokenizer_preflight.json",
    "datasets/m2_2b_formal/e2_phase_validity_choice/build_report.json",
)


def _without_contract(text: str) -> str:
    value = text.rstrip()
    require(value.endswith(ANSWER_CONTRACT), "Parent Answer contract suffix changed")
    return value.removesuffix(ANSWER_CONTRACT).rstrip()


def validate(*, write_release: bool = False) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    failures: list[str] = []

    def check(name: str, callback: Any) -> None:
        try:
            callback()
        except Exception as error:
            checks[name] = False
            failures.append(f"{name}: {error}")
        else:
            checks[name] = True

    anchor: dict[str, Any] = {}

    def anchor_check() -> None:
        nonlocal anchor
        anchor = validate_anchor()

    check("anchor_result_and_audit", anchor_check)
    answers = read_jsonl(HERE / "answer_inputs96.jsonl")
    validity = read_jsonl(HERE / "validity_inputs48.jsonl")
    mapping = read_json(HERE / "validity_mapping.json")
    review = read_json(HERE / "human_review.json")
    preflight = read_json(HERE / "tokenizer_preflight.json")
    test_report = read_json(HERE / "CPU_TEST_REPORT.json")
    analysis_plan = read_json(HERE / "ANALYSIS_PLAN.json")

    def answer_check() -> None:
        expected = answer_rows()
        require(answers == expected, "Answer inputs are not the unchanged frozen view")
        require(
            len(answers) == 96 and len({row["id"] for row in answers}) == 96, "Answer IDs changed"
        )

    check("answer_inputs_unchanged", answer_check)

    def mapping_check() -> None:
        expected = balanced_mapping(base_rows(answers))
        observed = mapping["sufficient_label_by_base"]
        require(observed == expected, "Validity A/B mapping changed")
        require(
            Counter(observed.values()) == Counter({"A": 12, "B": 12}), "Mapping balance changed"
        )
        require(
            mapping["observed_balance"]["balanced_position_transitions"] == 10,
            "Transition optimum changed",
        )
        require(mapping["observed_balance"]["search_space"] == 4096, "Mapping search space changed")

    check("mapping_deterministic_and_optimal", mapping_check)

    def validity_check() -> None:
        require(
            len(validity) == 48 and len({row["id"] for row in validity}) == 48,
            "Validity IDs changed",
        )
        sources = {row["id"]: row for row in answers if row["arm"] in {"V", "F"}}
        for row in validity:
            source = sources[row["source_answer_id"]]
            visible = row["messages"][1]["content"]
            quoted = visible.split("--- BEGIN QUOTED UPDATE MATERIAL ---\n", 1)[1].split(
                "\n--- END QUOTED UPDATE MATERIAL ---", 1
            )[0]
            require(
                quoted == _without_contract(source["messages"][3]["content"]),
                f"{row['id']}: quoted update changed",
            )
            require(
                row["gold_semantic"]
                == ("sufficient" if row["source_arm"] == "V" else "insufficient"),
                f"{row['id']}: gold semantic changed",
            )
            expected_label = row[f"{row['gold_semantic']}_label"]
            require(row["gold_label"] == expected_label, f"{row['id']}: gold label changed")
            question = visible.index("The proposed target")
            require(
                visible.index("A) ", question) < visible.index("B) ", question),
                f"{row['id']}: A/B order changed",
            )
        require(
            Counter(row["gold_label"] for row in validity) == Counter({"A": 24, "B": 24}),
            "Validity gold labels changed",
        )

    check("validity_semantics_and_quote_integrity", validity_check)

    def review_check() -> None:
        require(review.get("status") == "passed", "Primary review is not passed")
        require(review.get("reviewer_count_required") == 1, "Reviewer requirement changed")
        require(
            review.get("reviewed_items") == 48 and review.get("passed_items") == 48,
            "Review count changed",
        )
        require(not review.get("failed_items"), "Review contains failed items")
        require(
            {item["id"] for item in review["items"]} == {row["id"] for row in validity},
            "Reviewed IDs changed",
        )

    check("single_reviewer_validity_audit", review_check)

    def preflight_check() -> None:
        require(
            preflight.get("status") == "passed" and preflight.get("gpu_used") is False,
            "Tokenizer preflight is not passed",
        )
        require(
            preflight["answer"]["source_canonical_text_sha256"]
            == canonical_text_sha256(HERE / "answer_inputs96.jsonl"),
            "Answer preflight is stale",
        )
        require(
            preflight["validity"]["source_canonical_text_sha256"]
            == canonical_text_sha256(HERE / "validity_inputs48.jsonl"),
            "Validity preflight is stale",
        )
        for group in ("answer_labels", "validity_labels"):
            require(
                all(
                    item["single_contextual_token"] for item in preflight[group]["labels"].values()
                ),
                f"{group}: contextual label boundary failed",
            )

    check("tokenizer_and_asset_preflight", preflight_check)

    def tests_check() -> None:
        require(
            test_report.get("status") == "passed_with_disclosed_unrelated_full_suite_failures",
            "CPU test report is not accepted",
        )
        require(
            test_report["relevant_suite"]["passed"] == 28
            and test_report["relevant_suite"]["failed"] == 0,
            "Relevant CPU suite changed",
        )
        require(
            test_report["new_source_lint"]["status"] == "passed",
            "E2 lint is not passed",
        )
        require(
            test_report["runner_prepare"]["cells"] == EXPECTED_E2A_CELLS
            and test_report["runner_prepare"]["gpu_used"] is False,
            "E2 prepare check changed",
        )

    check("relevant_tests_lint_and_prepare", tests_check)

    def analysis_plan_check() -> None:
        require(analysis_plan.get("status") == "frozen_before_gpu", "Analysis plan is not frozen")
        require(
            analysis_plan["analysis_unit"]["independent_cluster"] == "family"
            and analysis_plan["analysis_unit"]["families"] == 12,
            "Analysis unit changed",
        )
        require(
            analysis_plan["gate_t"]["validity_rope_pp"] == 10
            and analysis_plan["gate_t"]["validity_equivalence_probability"] == 0.8,
            "Gate T thresholds changed",
        )
        require(
            analysis_plan["hierarchical_models"]["nonconvergence_route"] == "Gate F",
            "Hierarchical convergence policy changed",
        )

    check("analysis_plan_frozen", analysis_plan_check)

    def release_paths_check() -> None:
        missing = [relative for relative in RELEASE_PATHS if not (ROOT / relative).is_file()]
        require(not missing, f"Missing release paths: {missing}")

    check("release_files_present", release_paths_check)
    passed = all(checks.values()) and not failures
    release_hashes = {
        relative: canonical_text_sha256(ROOT / relative)
        for relative in RELEASE_PATHS
        if (ROOT / relative).is_file()
    }
    manifest = {
        "schema": SCHEMA,
        "status": "cpu_gate_ready" if passed else "failed",
        "anchor_result_commit": ANCHOR_RESULT_COMMIT,
        "anchor_cpu_gate_commit": ANCHOR_CPU_GATE_COMMIT,
        "answer_inputs": 96,
        "validity_inputs": 48,
        "formal_conditions": list(FORMAL_CONDITIONS),
        "formal_cells": EXPECTED_E2A_CELLS,
        "smoke_ids": SMOKE_IDS,
        "mapping_amendment": mapping.get("observed_balance"),
        "release_files": release_hashes,
        "release_files_sha256": object_sha256(release_hashes),
    }
    gate = {
        "schema": SCHEMA,
        "gate": "E2-CPU",
        "status": "passed" if passed else "failed",
        "gpu_authorized": passed,
        "authorized_scope": "E2A_only" if passed else None,
        "authorized_matrix": {
            "smoke": "frozen smoke IDs x engineering conditions",
            "formal": "96 Answer + 48 Validity x uncapped/full_p25/prefill_only/decode_only",
            "formal_cells": EXPECTED_E2A_CELLS,
        },
        "prohibited": [
            "E2B_before_E2A_route_review",
            "activation_patching",
            "gradient_or_JVP",
            "J-Lens",
            "p1",
            "item_selection",
            "prompt_or_label_remapping_after_results",
            "new_cap_optimization",
        ],
        "requires_clean_synced_head": True,
        "requires_exactly_four_visible_gpus": True,
        "manifest_canonical_text_sha256": None,
        "checks": checks,
        "failures": failures,
    }
    report = {
        "schema": SCHEMA,
        "status": "passed" if passed else "failed",
        "gpu_used": False,
        "checks": checks,
        "failures": failures,
        "anchor": anchor,
        "release_ready": passed,
    }
    if write_release:
        write_json(HERE / "manifest.json", manifest)
        gate["manifest_canonical_text_sha256"] = canonical_text_sha256(HERE / "manifest.json")
        write_json(HERE / "E2_CPU_GATE.json", gate)
        write_json(HERE / "validation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", action="store_true")
    args = parser.parse_args()
    report = validate(write_release=args.release)
    print(
        json.dumps(
            {
                "status": report["status"],
                "release_ready": report["release_ready"],
                "failures": report["failures"],
            },
            sort_keys=True,
        )
    )
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
