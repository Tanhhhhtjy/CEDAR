#!/usr/bin/env python3
"""Finalize a user-attested external review without fabricating workbook ratings."""

from __future__ import annotations

import json

from e2_new_salt_common import (
    HERE,
    SCHEMA,
    canonical_sha256,
    file_sha256,
    read_json,
    require,
    write_json,
)


def finalize() -> dict[str, object]:
    packet = read_json(HERE / "HUMAN_REVIEW_PACKET.json")
    attestation = read_json(HERE / "human_review_attestation.json")
    workbook = HERE / packet["workbook"]
    checks = {
        "packet_awaited_real_human_review": packet["status"] == "awaiting_human_review",
        "external_expert_role_attested": attestation["reviewer_role"] == "external_expert",
        "all_24_pairs_reviewed": attestation["pairs_reviewed"] == 24
        and attestation["all_pairs_reviewed"] is True,
        "all_rubric_criteria_attested_passed": attestation[
            "all_prespecified_rubric_criteria_passed"
        ]
        is True,
        "all_pairs_attested_passed": attestation["all_pairs_passed"] is True,
        "review_was_blind_to_qwen_outputs": attestation[
            "qwen_outputs_present_in_review_packet"
        ]
        is False,
        "workbook_raw_hash_bound": file_sha256(workbook)
        == packet["workbook_raw_sha256"]
        == attestation["review_packet_raw_sha256"],
        "workbook_canonical_hash_bound": packet["workbook_canonical_content_sha256"]
        == attestation["review_packet_canonical_content_sha256"],
        "documentation_deviation_disclosed": attestation["protocol_deviation"] is True
        and attestation["deviation_class"] == "review_documentation_not_completed",
        "workbook_not_backfilled": attestation["workbook_completed"] is False
        and attestation["codex_filled_workbook"] is False,
        "no_item_ratings_invented": attestation["per_item_numeric_scores_recorded"] is False
        and attestation["codex_invented_item_ratings"] is False,
        "user_explicitly_requested_gpu": attestation[
            "gpu_execution_explicitly_requested_by_user"
        ]
        is True,
    }
    failed = [name for name, passed in checks.items() if not passed]
    require(not failed, f"Human review attestation failed: {failed}")
    result = {
        "schema": f"{SCHEMA}.human_review_result",
        "status": "passed_with_documentation_deviation",
        "substantive_human_review_passed": True,
        "review_mode": "external_expert_full_review_user_attested",
        "pairs_reviewed": 24,
        "all_pairs_passed": True,
        "all_prespecified_rubric_criteria_passed": True,
        "reviewer_identity_disclosed": False,
        "item_level_numeric_scores_available": False,
        "completed_workbook_available": False,
        "protocol_deviation": {
            "present": True,
            "class": "review_documentation_not_completed",
            "limitation": (
                "No completed item-level review workbook or numeric naturalness "
                "scores are available."
            ),
        },
        "review_packet": {
            "path": packet["workbook"],
            "raw_sha256": packet["workbook_raw_sha256"],
            "canonical_content_sha256": packet["workbook_canonical_content_sha256"],
        },
        "attestation_canonical_sha256": canonical_sha256(attestation),
        "attestation": attestation,
        "codex_filled_human_ratings": False,
        "gpu_used": False,
    }
    write_json(HERE / "human_review_result.json", result)
    return result


def main() -> None:
    result = finalize()
    print(
        json.dumps(
            {
                "status": result["status"],
                "pairs_reviewed": result["pairs_reviewed"],
                "documentation_deviation": result["protocol_deviation"],
                "gpu_used": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
