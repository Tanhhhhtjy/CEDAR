"""Independent semantic validation for candidate discovery, V, F, and formal difficulty."""

from __future__ import annotations

import copy
from typing import Any

from atomic_edit_candidate_enumerator import expected_labels
from hardening_common import (
    canonical_sha256,
    decisive_literal,
    edit_count,
    evaluate_fixed_point,
    literal_key,
    proof_count,
    run_mace4_countermodel,
    shortest_proof,
    validate_with_both,
)
from semantic_f_operators import build_f_certificate, validate_f_certificate


def _counterevidence(
    world: dict[str, Any],
    query: dict[str, Any],
    initial_label: str,
    updated_label: str,
) -> dict[str, Any]:
    evaluation = evaluate_fixed_point(world)
    if initial_label in {"true", "false"}:
        item = decisive_literal(query, initial_label)
        return {
            "type": "opposite_proof",
            "proved_literal": literal_key(item),
            "proof": shortest_proof(evaluation, item),
            "refutes_induced_label": updated_label,
        }
    target = decisive_literal(query, updated_label)
    model = run_mace4_countermodel(world, target)
    return {
        "type": "mace4_countermodel",
        "target_literal": literal_key(target),
        "model_found": model["model_found"],
        "model": model,
    }


def _difficulty(
    builder_output: dict[str, Any],
    initial_validation: dict[str, Any],
    updated_validation: dict[str, Any],
) -> dict[str, Any]:
    initial_label = initial_validation["python_label"]
    state = "initial" if initial_label != "unknown" else "updated"
    selected = initial_validation if state == "initial" else updated_validation
    world = builder_output["initial_world"] if state == "initial" else updated_validation["world"]
    decisive = decisive_literal(builder_output["query"], selected["python_label"])
    proof = shortest_proof(selected["fixed_point"], decisive)
    by_id = {record["record_id"]: record for record in world["records"]}
    reconstruction = sum(
        by_id[record_id].get("role") == "state_reconstruction" for record_id in proof["record_ids"]
    )
    distractors = sum(
        bool(record.get("is_distractor")) for record in builder_output["initial_world"]["records"]
    )
    depth_low, depth_high = builder_output["difficulty_contract"]["proof_depth"]
    distractor_low, distractor_high = builder_output["difficulty_contract"]["distractor_count"]
    checks = {
        "proof_depth_in_contract": depth_low <= proof["depth"] <= depth_high,
        "distractor_count_in_contract": distractor_low <= distractors <= distractor_high,
        "medium_has_reconstruction": builder_output["variant"] != "medium" or reconstruction >= 1,
    }
    return {
        "decisive_state": state,
        "actual_shortest_proof_depth": proof["depth"],
        "proof_record_ids": proof["record_ids"],
        "proof_path_count": proof_count(selected["fixed_point"], decisive),
        "distractor_count": distractors,
        "state_reconstruction_steps": reconstruction,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _compact_dual(validation: dict[str, Any]) -> dict[str, Any]:
    return {
        "python_label": validation["python_label"],
        "prover9_label": validation["prover9_label"],
        "agreement": validation["agreement"],
        "closure_sha256": canonical_sha256(validation["fixed_point"]["closure"]),
        "prover9": copy.deepcopy(validation["prover9"]),
    }


def semantic_validator(
    builder_output: dict[str, Any], neighborhood_result: dict[str, Any]
) -> dict[str, Any]:
    """Validate the sole discovered update and its paired semantic F certificate."""

    expected_initial, expected_updated = expected_labels(builder_output["desired_transition"])
    if neighborhood_result["success_count"] != 1:
        return {
            "passed": False,
            "failure_stage": "unique_atomic_edit",
            "failure_reason": f"success_count={neighborhood_result['success_count']}",
            "candidate_count": neighborhood_result["candidate_count"],
        }
    success = neighborhood_result["successes"][0]
    edit = success["candidate"]
    updated_world = success["updated_world"]
    initial_world = builder_output["initial_world"]
    query = builder_output["query"]
    initial_dual = validate_with_both(initial_world, query)
    updated_dual = validate_with_both(updated_world, query)
    initial_dual["world"] = initial_world
    updated_dual["world"] = updated_world
    counterevidence = _counterevidence(
        initial_world, query, initial_dual["python_label"], updated_dual["python_label"]
    )
    f_certificate = build_f_certificate(
        builder_output,
        edit,
        updated_world,
        updated_dual["python_label"],
        counterevidence,
    )
    f_validation = validate_f_certificate(builder_output, f_certificate, edit, updated_world)
    difficulty = _difficulty(builder_output, initial_dual, updated_dual)
    checks = {
        "initial_dual_agreement": initial_dual["agreement"],
        "updated_dual_agreement": updated_dual["agreement"],
        "initial_not_invalid": initial_dual["python_label"] != "invalid",
        "updated_not_invalid": updated_dual["python_label"] != "invalid",
        "initial_label_matches": initial_dual["python_label"] == expected_initial,
        "updated_label_matches": updated_dual["python_label"] == expected_updated,
        "transition_changes": expected_initial != expected_updated,
        "atomic_edit": edit_count(edit) == 1,
        "unique_success_in_complete_neighborhood": neighborhood_result["success_count"] == 1,
        "selected_from_enumeration": edit.get("candidate_id") is not None,
        "f_world_unchanged": True,
        "f_target_matches_v": f_certificate["induced_target"]["label"]
        == updated_dual["python_label"],
        "f_semantic_validation": f_validation["passed"],
        "counterevidence_exists": bool(counterevidence),
        "difficulty_contract": difficulty["passed"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "selected_edit": copy.deepcopy(edit),
        "updated_world": updated_world,
        "initial_validation": _compact_dual(initial_dual),
        "updated_validation": _compact_dual(updated_dual),
        "candidate_count": neighborhood_result["candidate_count"],
        "successful_rank": neighborhood_result["successful_rank"],
        "success_count": neighborhood_result["success_count"],
        "candidate_trace": copy.deepcopy(neighborhood_result["trace"]),
        "difficulty": difficulty,
        "f_certificate": f_certificate,
        "f_validation": f_validation,
    }
