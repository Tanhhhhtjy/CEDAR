#!/usr/bin/env python3
"""Deterministically build the E2 Answer and Validity input views."""

from __future__ import annotations

import hashlib
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any

from e2_common import (
    ANSWER_CONTRACT,
    SCHEMA,
    VALIDITY_CONTRACT,
    VALIDITY_SYSTEM,
    answer_rows,
    base_rows,
    object_sha256,
    require,
    validate_anchor,
    write_json,
    write_jsonl,
)

HERE = Path(__file__).resolve().parent


def balanced_mapping(bases: list[dict[str, Any]]) -> dict[str, str]:
    """Balance family and variant exactly, then maximize transition balance.

    With this frozen split, exact balance within family, variant, and all 12
    position transitions is jointly infeasible.  The exhaustive 2^12 search is
    deterministic and records the attainable optimum instead of silently
    weakening a constraint.
    """
    by_family: dict[str, list[dict[str, Any]]] = {}
    for row in bases:
        by_family.setdefault(row["family_id"], []).append(row)
    families = sorted(by_family)
    require(len(families) == 12, "Family count changed")
    for family in families:
        by_family[family].sort(key=lambda row: row["base_id"])
        require(len(by_family[family]) == 2, f"{family}: expected two variants")

    candidates: list[tuple[int, str, dict[str, int]]] = []
    for choices in product((0, 1), repeat=len(families)):
        assignment: dict[str, int] = {}
        for family, choice in zip(families, choices, strict=True):
            rows = by_family[family]
            assignment[rows[choice]["base_id"]] = 1
            assignment[rows[1 - choice]["base_id"]] = 0
        easy_a = sum(assignment[row["base_id"]] for row in bases if row["variant"] == "easy")
        if easy_a != 6:
            continue
        transition_a = Counter()
        for row in bases:
            transition_a[row["position_transition"]] += assignment[row["base_id"]]
        balanced = sum(value == 1 for value in transition_a.values())
        tie_break = hashlib.sha256(repr(sorted(assignment.items())).encode("utf-8")).hexdigest()
        candidates.append((balanced, tie_break, assignment))
    require(candidates, "Could not balance Validity mapping by family and variant")
    best_score = max(item[0] for item in candidates)
    _, _, best = min(
        (item for item in candidates if item[0] == best_score), key=lambda item: item[1]
    )
    return {base_id: "A" if value else "B" for base_id, value in sorted(best.items())}


def _without_contract(text: str) -> str:
    stripped = text.rstrip()
    require(stripped.endswith(ANSWER_CONTRACT), "Frozen Answer contract suffix changed")
    return stripped.removesuffix(ANSWER_CONTRACT).rstrip()


def validity_messages(row: dict[str, Any], sufficient_label: str) -> list[dict[str, str]]:
    original = _without_contract(row["messages"][1]["content"])
    update = _without_contract(row["messages"][3]["content"])
    insufficient_label = "B" if sufficient_label == "A" else "A"
    semantic_by_label = {
        sufficient_label: "sufficient",
        insufficient_label: "insufficient",
    }
    user = (
        "Original task and records:\n"
        f"{original}\n\n"
        f"Original answer before the update proposal: option {row['initial_answer']}.\n\n"
        "--- BEGIN QUOTED UPDATE MATERIAL ---\n"
        f"{update}\n"
        "--- END QUOTED UPDATE MATERIAL ---\n\n"
        f"The proposed target is option {row['target_answer']}. Given the original records, "
        "the observed checks, and the candidate reasoning, does the candidate reasoning "
        "logically suffice both to warrant applying the proposed record change and to make "
        "the proposed target the uniquely supported answer?\n\n"
        f"A) {semantic_by_label['A']}\n"
        f"B) {semantic_by_label['B']}\n\n"
        f"{VALIDITY_CONTRACT}"
    )
    return [
        {"role": "system", "content": VALIDITY_SYSTEM},
        {"role": "user", "content": user},
    ]


def build() -> dict[str, Any]:
    anchor = validate_anchor()
    answers = answer_rows()
    bases = base_rows(answers)
    sufficient_by_base = balanced_mapping(bases)
    vf_rows = [row for row in answers if row["arm"] in {"V", "F"}]
    validity: list[dict[str, Any]] = []
    for row in vf_rows:
        sufficient_label = sufficient_by_base[row["base_id"]]
        insufficient_label = "B" if sufficient_label == "A" else "A"
        messages = validity_messages(row, sufficient_label)
        gold_semantic = "sufficient" if row["arm"] == "V" else "insufficient"
        gold_label = sufficient_label if row["arm"] == "V" else insufficient_label
        validity.append(
            {
                "schema": SCHEMA,
                "assay": "validity",
                "id": f"{row['id']}-VALIDITY",
                "source_answer_id": row["id"],
                "base_id": row["base_id"],
                "family_id": row["family_id"],
                "structure_class": row["structure_class"],
                "variant": row["variant"],
                "source_arm": row["arm"],
                "initial_answer": row["initial_answer"],
                "target_answer": row["target_answer"],
                "position_transition": row["position_transition"],
                "error_class": row.get("error_class"),
                "error_position": row.get("error_position"),
                "sufficient_label": sufficient_label,
                "insufficient_label": insufficient_label,
                "gold_semantic": gold_semantic,
                "gold_label": gold_label,
                "response_contract": VALIDITY_CONTRACT,
                "messages": messages,
                "messages_sha256": object_sha256(messages),
            }
        )

    require(len(validity) == 48, "Validity input count changed")
    require(
        Counter(row["source_arm"] for row in validity) == Counter({"V": 24, "F": 24}),
        "Validity arm balance changed",
    )
    require(
        Counter(row["gold_label"] for row in validity) == Counter({"A": 24, "B": 24}),
        "Validity gold-label balance changed",
    )
    require(
        Counter(sufficient_by_base.values()) == Counter({"A": 12, "B": 12}),
        "Sufficient-label balance changed",
    )

    mapping = {
        "schema": SCHEMA,
        "method": "deterministic constrained two-color assignment",
        "constraints": {
            "sufficient_label_overall": {"A": 12, "B": 12},
            "per_family": "one A and one B",
            "per_variant": {"easy_A": 6, "easy_B": 6, "medium_A": 6, "medium_B": 6},
            "per_position_transition": (
                "maximize one A and one B after exact family/variant balance"
            ),
        },
        "sufficient_label_by_base": sufficient_by_base,
        "mapping_sha256": object_sha256(sufficient_by_base),
    }
    transition_a = Counter({row["position_transition"]: 0 for row in bases})
    transition_a.update(
        row["position_transition"] for row in bases if sufficient_by_base[row["base_id"]] == "A"
    )
    mapping["observed_balance"] = {
        "balanced_position_transitions": sum(value == 1 for value in transition_a.values()),
        "position_transition_A_counts": dict(sorted(transition_a.items())),
        "joint_exact_transition_balance_feasible": False,
        "search_space": 2**12,
    }
    pending_review = {
        "schema": SCHEMA,
        "status": "pending_primary_review",
        "reviewer_count_required": 1,
        "reviewer": None,
        "criteria": [
            "natural_and_clear",
            "quoted_material_preserved",
            "target_and_references_clear",
            "no_visible_vf_arm_or_gold_leak",
            "label_mapping_correct",
            "f_single_error_carried_forward",
        ],
        "items": [
            {
                "id": row["id"],
                "source_arm": row["source_arm"],
                "decision": None,
                "notes": "",
            }
            for row in validity
        ],
    }
    write_jsonl(HERE / "answer_inputs96.jsonl", answers)
    write_jsonl(HERE / "validity_inputs48.jsonl", validity)
    write_json(HERE / "validity_mapping.json", mapping)
    write_json(HERE / "human_review.json", pending_review)
    report = {
        "schema": SCHEMA,
        "status": "built_pending_review_and_tokenizer_preflight",
        "gpu_used": False,
        "anchor": anchor,
        "answer_inputs": {"rows": len(answers), "object_sha256": object_sha256(answers)},
        "validity_inputs": {"rows": len(validity), "object_sha256": object_sha256(validity)},
        "mapping_sha256": mapping["mapping_sha256"],
    }
    write_json(HERE / "build_report.json", report)
    return report


if __name__ == "__main__":
    print(build())
