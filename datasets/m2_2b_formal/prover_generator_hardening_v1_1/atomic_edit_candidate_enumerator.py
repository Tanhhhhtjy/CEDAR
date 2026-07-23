"""Blind complete one-record-neighborhood enumeration for v1.1."""

from __future__ import annotations

import copy
import random
from itertools import product
from typing import Any

from hardening_common import (
    apply_edit,
    canonical_json,
    canonical_sha256,
    evaluate_fixed_point,
    fact,
    label_from_evaluation,
    rule,
    stable_seed,
)


def expected_labels(transition: str) -> tuple[str, str]:
    before, after = transition.split("_to_", maxsplit=1)
    return before, after


def _candidate_fact(index: int, item: dict[str, Any]) -> dict[str, Any]:
    return fact(f"candidate_fact_{index:03d}", copy.deepcopy(item), role="candidate")


def _candidate_rules(vocabulary: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for index, (body, head) in enumerate(
        product(vocabulary.get("rule_bodies", []), vocabulary.get("rule_heads", []))
    ):
        result.append(
            rule(
                f"candidate_rule_{index:03d}",
                copy.deepcopy(body),
                copy.deepcopy(head),
                role="candidate",
            )
        )
    return result


def _replacement_after(before: dict[str, Any], template: dict[str, Any]) -> dict[str, Any]:
    after = copy.deepcopy(template)
    after["record_id"] = before["record_id"]
    after["role"] = before.get("role", "structural")
    after["mutable"] = before.get("mutable", False)
    after["is_distractor"] = before.get("is_distractor", False)
    return after


def _edit_semantics(edit: dict[str, Any]) -> dict[str, Any]:
    def strip(record: dict[str, Any] | None) -> dict[str, Any] | None:
        if record is None:
            return None
        if record["kind"] == "fact":
            return {"kind": "fact", "literal": record["literal"]}
        return {"kind": "rule", "body": record["body"], "head": record["head"]}

    return {
        "operation": edit["operation"],
        "before_record_id": edit["before"]["record_id"] if edit.get("before") else None,
        "before": strip(edit.get("before")),
        "after": strip(edit.get("after")),
    }


def enumerate_atomic_edit_candidates(builder_output: dict[str, Any]) -> list[dict[str, Any]]:
    """Enumerate, canonical-sort, and seed-shuffle the complete permitted neighborhood."""

    world = builder_output["initial_world"]
    contract = builder_output["enumeration_contract"]
    operations = set(contract["allowed_operations"])
    vocabulary = builder_output["candidate_vocabulary"]
    mutable_records = [record for record in world["records"] if record.get("mutable")]
    mutable_facts = [record for record in mutable_records if record["kind"] == "fact"]
    mutable_rules = [record for record in mutable_records if record["kind"] == "rule"]
    fact_templates = [
        _candidate_fact(index, item)
        for index, item in enumerate(vocabulary.get("fact_literals", []))
    ]
    rule_templates = _candidate_rules(vocabulary)
    candidates: list[dict[str, Any]] = []

    if "fact_addition" in operations:
        candidates.extend(
            {"operation": "add", "before": None, "after": item, "edit_count": 1}
            for item in fact_templates
        )
    if "fact_deletion" in operations:
        candidates.extend(
            {"operation": "delete", "before": item, "after": None, "edit_count": 1}
            for item in mutable_facts
        )
    if "fact_replacement" in operations:
        for before, template in product(mutable_facts, fact_templates):
            after = _replacement_after(before, template)
            if (
                _edit_semantics({"operation": "replace", "before": before, "after": after})[
                    "before"
                ]
                == _edit_semantics({"operation": "replace", "before": before, "after": after})[
                    "after"
                ]
            ):
                continue
            candidates.append(
                {
                    "operation": "replace",
                    "before": before,
                    "after": after,
                    "edit_count": 1,
                }
            )
    if "rule_addition" in operations:
        candidates.extend(
            {"operation": "add", "before": None, "after": item, "edit_count": 1}
            for item in rule_templates
        )
    if "rule_deletion" in operations:
        candidates.extend(
            {"operation": "delete", "before": item, "after": None, "edit_count": 1}
            for item in mutable_rules
        )
    if "rule_replacement" in operations:
        for before, template in product(mutable_rules, rule_templates):
            after = _replacement_after(before, template)
            if (
                _edit_semantics({"operation": "replace", "before": before, "after": after})[
                    "before"
                ]
                == _edit_semantics({"operation": "replace", "before": before, "after": after})[
                    "after"
                ]
            ):
                continue
            candidates.append(
                {
                    "operation": "replace",
                    "before": before,
                    "after": after,
                    "edit_count": 1,
                }
            )

    deduplicated = {canonical_json(_edit_semantics(item)): item for item in candidates}
    ordered = [deduplicated[key] for key in sorted(deduplicated)]
    maximum = int(contract["max_candidates"])
    if len(ordered) > maximum:
        raise ValueError(
            f"complete legal neighborhood has {len(ordered)} candidates, over {maximum}"
        )
    rng = random.Random(stable_seed(builder_output["seed"], "candidate_order"))
    rng.shuffle(ordered)
    for rank, item in enumerate(ordered, start=1):
        item["candidate_id"] = f"cand-{canonical_sha256(_edit_semantics(item))[:16]}"
        item["rank"] = rank
    return ordered


def evaluate_candidate_neighborhood(
    builder_output: dict[str, Any], candidates: list[dict[str, Any]]
) -> dict[str, Any]:
    world = builder_output["initial_world"]
    query = builder_output["query"]
    expected_initial, expected_updated = expected_labels(builder_output["desired_transition"])
    initial_eval = evaluate_fixed_point(world)
    initial_label = label_from_evaluation(initial_eval, query)
    trace: list[dict[str, Any]] = []
    successes: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            updated = apply_edit(world, candidate)
            evaluation = evaluate_fixed_point(updated)
            label = label_from_evaluation(evaluation, query)
            accepted = initial_label == expected_initial and label == expected_updated
            trace.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "rank": candidate["rank"],
                    "operation": candidate["operation"],
                    "updated_label": label,
                    "accepted": accepted,
                    "edit": copy.deepcopy(candidate),
                }
            )
            if accepted:
                successes.append(
                    {
                        "candidate": copy.deepcopy(candidate),
                        "updated_world": updated,
                        "evaluation": evaluation,
                    }
                )
        except (KeyError, ValueError) as error:
            trace.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "rank": candidate["rank"],
                    "operation": candidate["operation"],
                    "accepted": False,
                    "failure": type(error).__name__,
                }
            )
    return {
        "initial_label": initial_label,
        "expected_initial_label": expected_initial,
        "expected_updated_label": expected_updated,
        "candidate_count": len(candidates),
        "success_count": len(successes),
        "successful_rank": successes[0]["candidate"]["rank"] if len(successes) == 1 else None,
        "trace": trace,
        "successes": successes,
    }
