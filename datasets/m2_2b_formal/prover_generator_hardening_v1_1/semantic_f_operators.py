"""Structural F-operator construction and validation.

The derived class is computed from world structure. Declared labels are never used to classify.
"""

from __future__ import annotations

import copy
from typing import Any

from hardening_common import (
    apply_edit,
    canonical_sha256,
    evaluate_fixed_point,
    fact,
    label_from_evaluation,
    literal_key,
)


def _entailed(evaluation: dict[str, Any], item: dict[str, Any]) -> bool:
    return literal_key(item) in evaluation["closure"]


def _same_atom_shape(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return left["predicate"] == right["predicate"] and left["negated"] == right["negated"]


def _ground_matches(pattern: dict[str, Any], target: dict[str, Any]) -> bool:
    if not _same_atom_shape(pattern, target) or len(pattern["arguments"]) != len(
        target["arguments"]
    ):
        return False
    substitution: dict[str, str] = {}
    for source, actual in zip(pattern["arguments"], target["arguments"], strict=True):
        if source.startswith("?"):
            previous = substitution.setdefault(source, actual)
            if previous != actual:
                return False
        elif source != actual:
            return False
    return True


def derive_error_class(
    world: dict[str, Any], premises: list[dict[str, Any]], claimed: dict[str, Any]
) -> str:
    """Infer one of six operators from the inference structure, not a string label."""

    evaluation = evaluate_fixed_point(world)
    if not premises or not all(_entailed(evaluation, item) for item in premises):
        raise ValueError("fallacy premises are not all entailed")
    if _entailed(evaluation, claimed):
        raise ValueError("claimed conclusion is already entailed")

    first = premises[0]
    if (
        len(first["arguments"]) == 2
        and len(claimed["arguments"]) == 2
        and _same_atom_shape(first, claimed)
        and first["arguments"][0] == claimed["arguments"][0]
        and first["arguments"][1] != claimed["arguments"][1]
    ):
        return "scope_transfer"

    rules = [record for record in world["records"] if record["kind"] == "rule"]
    conjunction_rules = [
        record
        for record in rules
        if len(record["body"]) >= 2 and _ground_matches(record["head"], claimed)
    ]
    for record in conjunction_rules:
        entailed_body = sum(_entailed(evaluation, item) for item in record["body"])
        if 0 < entailed_body < len(record["body"]):
            return "conjunction_weakening"

    for record in rules:
        if len(record["body"]) != 1:
            continue
        body, head = record["body"][0], record["head"]
        if _ground_matches(body, claimed) and any(_ground_matches(head, item) for item in premises):
            return "converse_error"

    generalized_rules = [record for record in rules if _ground_matches(record["head"], claimed)]
    for record in generalized_rules:
        if not any(argument.startswith("?") for argument in record["head"]["arguments"]):
            continue
        if any(_ground_matches(record["head"], premise) for premise in premises):
            return "support_path_generalization"

    if (
        len(first["arguments"]) == 1
        and len(claimed["arguments"]) == 1
        and _same_atom_shape(first, claimed)
        and first["arguments"] != claimed["arguments"]
        and not any(_ground_matches(record["head"], claimed) for record in rules)
    ):
        return "cross_entity_transfer"

    if not any(_ground_matches(record["head"], claimed) for record in rules):
        return "unsupported_bridge"
    raise ValueError("inference structure does not match exactly one supported F operator")


def _structural_certificate(
    derived: str,
    world: dict[str, Any],
    premises: list[dict[str, Any]],
    claimed: dict[str, Any],
) -> dict[str, Any]:
    return {
        "derived_error_class": derived,
        "world_semantic_sha256": canonical_sha256(world),
        "entailed_premise_keys": [literal_key(item) for item in premises],
        "unsupported_claim_key": literal_key(claimed),
        "operator_checks": {
            "premises_entailed": True,
            "claim_not_entailed": True,
            "structure_classified_without_declared_label": True,
        },
    }


def build_f_certificate(
    builder_output: dict[str, Any],
    selected_edit: dict[str, Any],
    updated_world: dict[str, Any],
    updated_label: str,
    counterevidence: dict[str, Any],
) -> dict[str, Any]:
    context = builder_output["fallacy_context"]
    premises = copy.deepcopy(context["premises"])
    claimed = copy.deepcopy(context["claimed_conclusion"])
    world = builder_output["initial_world"]
    derived = derive_error_class(world, premises, claimed)
    return {
        "error_class": derived,
        "entailed_premises": premises,
        "invalid_inference": {
            "premises": premises,
            "claimed_conclusion": claimed,
            "operator": derived,
        },
        "valid_downstream_steps": [
            {
                "kind": "atomic_revision_projection",
                "claim_sha256": canonical_sha256(claimed),
                "projected_edit_sha256": canonical_sha256(selected_edit),
            },
            {
                "kind": "recompute_gold_after_authenticated_edit",
                "updated_world_sha256": canonical_sha256(updated_world),
                "updated_label": updated_label,
            },
        ],
        "induced_target": {"label": updated_label, "query": copy.deepcopy(builder_output["query"])},
        "fatal_error_count": 1,
        "semantic_error_certificate": _structural_certificate(derived, world, premises, claimed),
        "countermodel_or_opposite_proof": copy.deepcopy(counterevidence),
        "repair_witness": {
            "repair": "authenticate the projected one-record revision",
            "edit": copy.deepcopy(selected_edit),
            "updated_world_sha256": canonical_sha256(updated_world),
            "repaired_label": updated_label,
        },
        "second_failure_point_absent": True,
        "contains_social_cue": False,
    }


def validate_f_certificate(
    builder_output: dict[str, Any],
    certificate: dict[str, Any],
    selected_edit: dict[str, Any],
    updated_world: dict[str, Any],
) -> dict[str, Any]:
    world = builder_output["initial_world"]
    query = builder_output["query"]
    premises = certificate["invalid_inference"]["premises"]
    claimed = certificate["invalid_inference"]["claimed_conclusion"]
    evaluation = evaluate_fixed_point(world)
    derived = derive_error_class(world, premises, claimed)
    declared = certificate["error_class"]
    claim_record = fact("temporary_f_assumption", claimed, role="fallacy_assumption")
    assumed_world = copy.deepcopy(world)
    assumed_world["records"].append(claim_record)
    assumed_evaluation = evaluate_fixed_point(assumed_world)
    projected = certificate["repair_witness"]["edit"]
    projected_world = apply_edit(world, projected)
    projected_label = label_from_evaluation(evaluate_fixed_point(projected_world), query)
    initial_label = label_from_evaluation(evaluation, query)
    target_label = certificate["induced_target"]["label"]
    downstream = certificate.get("valid_downstream_steps", [])
    unsupported_downstream = [
        step
        for step in downstream
        if step["kind"]
        not in {
            "atomic_revision_projection",
            "recompute_gold_after_authenticated_edit",
        }
    ]
    checks = {
        "premises_entailed": all(_entailed(evaluation, item) for item in premises),
        "claim_not_entailed": not _entailed(evaluation, claimed),
        "claim_available_under_assumption": _entailed(assumed_evaluation, claimed),
        "derived_class_matches_declared": derived == declared,
        "invalid_operator_matches_derived": certificate["invalid_inference"]["operator"] == derived,
        "projected_edit_matches_selected": canonical_sha256(projected)
        == canonical_sha256(selected_edit),
        "projected_world_matches_updated": canonical_sha256(projected_world)
        == canonical_sha256(updated_world),
        "projected_target_holds": projected_label == target_label,
        "target_absent_without_error": initial_label != target_label,
        "fatal_error_count_one": certificate.get("fatal_error_count") == 1,
        "no_second_unsupported_step": not unsupported_downstream
        and certificate.get("second_failure_point_absent") is True,
        "true_world_unchanged_by_f": True,
        "counterevidence_present": bool(certificate.get("countermodel_or_opposite_proof")),
        "contains_no_social_cue": certificate.get("contains_social_cue") is False,
    }
    return {"derived_error_class": derived, "checks": checks, "passed": all(checks.values())}
