"""Independent zero-GPU validator for the CEDAR development bank."""

from __future__ import annotations

import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from development_common import (
    AO_TEMPLATE,
    ARMS,
    ERROR_CLASSES,
    ERROR_POSITIONS,
    OPTION_LETTERS,
    R_TEMPLATE,
    RESPONSE_CONTRACT,
    SCHEMA,
    S_TEMPLATE,
    VARIANTS,
    canonical_sha256,
    canonical_token_count,
    family_plan,
    literal_key,
    whitespace_token_count,
)
from atomic_edit_candidate_enumerator import (  # type: ignore  # noqa: E402
    enumerate_atomic_edit_candidates,
    evaluate_candidate_neighborhood,
)
from hardening_common import (  # type: ignore  # noqa: E402
    apply_edit,
    evaluate_fixed_point,
    label_from_evaluation,
    validate_with_both,
    write_json,
)

HERE = Path(__file__).resolve().parent


class ValidationError(ValueError):
    pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _validate_source(source: dict[str, Any]) -> dict[str, Any]:
    where = source["attempt_id"]
    built = source["builder_output"]
    validation = source["validation"]
    candidates = enumerate_atomic_edit_candidates(built)
    neighborhood = evaluate_candidate_neighborhood(built, candidates)
    initial = validate_with_both(built["initial_world"], built["query"])
    require(initial["agreement"], f"{where}: initial solver disagreement")
    require(neighborhood["success_count"] == 1, f"{where}: V is not unique")
    selected = neighborhood["successes"][0]["candidate"]
    require(
        canonical_sha256(selected) == canonical_sha256(validation["selected_edit"]),
        f"{where}: selected edit changed",
    )
    updated_world = apply_edit(built["initial_world"], selected)
    updated = validate_with_both(updated_world, built["query"])
    require(updated["agreement"], f"{where}: updated solver disagreement")
    require(
        initial["python_label"] == validation["initial_validation"]["python_label"],
        f"{where}: initial label mismatch",
    )
    require(
        updated["python_label"] == validation["updated_validation"]["python_label"],
        f"{where}: updated label mismatch",
    )
    require(
        initial["python_label"] != updated["python_label"],
        f"{where}: transition does not change",
    )
    require(
        initial["python_label"] != "invalid" and updated["python_label"] != "invalid",
        f"{where}: invalid object world",
    )
    f_validation = validation["f_validation"]
    require(f_validation["passed"], f"{where}: object F certificate failed")
    require(
        f_validation["derived_error_class"] == source["assigned_f_error_class"],
        f"{where}: F class mismatch",
    )
    return {
        "initial_label": initial["python_label"],
        "updated_label": updated["python_label"],
        "candidate_count": len(candidates),
        "successful_rank": neighborhood["successful_rank"],
        "passed": True,
    }


def _validate_meta(base: dict[str, Any]) -> dict[str, Any]:
    where = base["base_id"]
    formal = base["paired_formal"]
    query = formal["query"]
    v = validate_with_both(formal["v"]["world"], query)
    f = validate_with_both(formal["f"]["world"], query)
    require(v["agreement"] and v["python_label"] == "true", f"{where}: meta V failed")
    require(f["agreement"] and f["python_label"] == "unknown", f"{where}: meta F proved")
    cert = formal["f"]["certificate"]
    require(cert["fatal_error_count"] == 1, f"{where}: F fatal count")
    require(cert["second_failure_point_absent"], f"{where}: second F failure")
    require(cert["claim_not_entailed"], f"{where}: F claim is entailed")
    require(cert["downstream_valid_under_single_assumption"], f"{where}: downstream invalid")
    require(cert["warrant_absent_without_invalid_step"], f"{where}: F warrant present")
    require(cert["countermodel"]["model_found"], f"{where}: F countermodel missing")
    evaluation = evaluate_fixed_point(formal["f"]["world"])
    require(
        all(literal_key(item) in evaluation["closure"] for item in formal["pre_error_literals"]),
        f"{where}: pre-error administrative step is unsupported",
    )
    return {
        "v_label": v["python_label"],
        "f_label": f["python_label"],
        "pre_error_step_count": len(formal["pre_error_literals"]),
        "passed": True,
    }


def _validate_base_and_arms(
    base: dict[str, Any], arms: list[dict[str, Any]]
) -> dict[str, Any]:
    where = base["base_id"]
    require(len(arms) == 6, f"{where}: arm count")
    by_arm = {row["arm"]: row for row in arms}
    require(set(by_arm) == set(ARMS), f"{where}: arm names")
    require(set(base["semantic_to_letter"].values()) == set(OPTION_LETTERS), f"{where}: map")
    require(base["target_answer"] != base["initial_answer"], f"{where}: same target")
    require(by_arm["V"]["gold_answer"] == base["target_answer"], f"{where}: V gold")
    for arm in ("N", "R", "F", "AO", "S"):
        require(by_arm[arm]["gold_answer"] == base["initial_answer"], f"{where}: {arm} gold")
    require(
        by_arm["V"]["target_answer"] == by_arm["F"]["target_answer"],
        f"{where}: V/F target",
    )
    require(by_arm["AO"]["target_answer"] == by_arm["S"]["target_answer"], f"{where}: AO/S")
    ao = by_arm["AO"]["followup"]
    social = by_arm["S"]["followup"]
    expected_ao = AO_TEMPLATE.format(target=base["target_answer"])
    expected_s = S_TEMPLATE.format(target=base["target_answer"])
    require(ao == expected_ao and social == expected_s, f"{where}: AO/S template drift")
    require(whitespace_token_count(ao) == whitespace_token_count(social), f"{where}: ws length")
    require(canonical_token_count(ao) == canonical_token_count(social), f"{where}: token length")
    require(
        canonical_token_count(f"{ao} {RESPONSE_CONTRACT}")
        == canonical_token_count(f"{social} {RESPONSE_CONTRACT}"),
        f"{where}: contracted token length",
    )
    require(by_arm["R"]["followup"] == R_TEMPLATE, f"{where}: R drift")
    require("option " not in by_arm["N"]["followup"].lower(), f"{where}: N target hint")
    require("option " not in by_arm["R"]["followup"].lower(), f"{where}: R target hint")
    require(
        all(row["messages"][-1]["content"].endswith(RESPONSE_CONTRACT) for row in arms),
        f"{where}: response contract",
    )
    require(
        all(row["analysis_schema"] == base["analysis_schema"] for row in arms),
        f"{where}: analysis schema",
    )
    return {"passed": True}


def validate(root: Path) -> dict[str, Any]:
    cards = read_jsonl(root / "family_cards24.jsonl")
    sources = read_jsonl(root / "symbolic_sources48.jsonl")
    bases = read_jsonl(root / "development_bases48.jsonl")
    arms = read_jsonl(root / "development_arms288.jsonl")
    audits = read_jsonl(root / "construction_audit48.jsonl")
    tokenizer_report = json.loads(
        (root / "tokenizer_preflight.json").read_text(encoding="utf-8")
    )
    require(len(cards) == 24, "family count")
    require(len(sources) == len(bases) == len(audits) == 48, "base count")
    require(len(arms) == 288, "arm count")
    expected_families = family_plan()
    require(
        [(x["family_id"], x["structure_class"]) for x in cards]
        == [(x["family_id"], x["structure_class"]) for x in expected_families],
        "family plan drift",
    )
    require(len({row["base_id"] for row in bases}) == 48, "duplicate base ID")
    require(len({row["id"] for row in arms}) == 288, "duplicate arm ID")
    source_by_id = {row["attempt_id"]: row for row in sources}
    arms_by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for arm in arms:
        arms_by_base[arm["base_id"]].append(arm)
    source_results = {}
    meta_results = {}
    surface_results = {}
    for base in bases:
        source = source_by_id[base["source_attempt_id"]]
        require(base["source_symbolic_sha256"] == canonical_sha256(source), "source hash")
        source_results[base["base_id"]] = _validate_source(source)
        meta_results[base["base_id"]] = _validate_meta(base)
        surface_results[base["base_id"]] = _validate_base_and_arms(
            base, arms_by_base[base["base_id"]]
        )
    target_counts = Counter(row["target_answer"] for row in bases)
    transition_counts = Counter(row["transition"] for row in bases)
    class_counts = Counter(row["error_class"] for row in bases)
    position_counts = Counter(row["error_position"] for row in bases)
    depths: dict[str, dict[str, int]] = defaultdict(dict)
    for row in bases:
        depths[row["family_id"]][row["variant"]] = row["difficulty"][
            "actual_shortest_proof_depth"
        ]
    checks = {
        "schema": all(row["schema"] == SCHEMA for row in bases + arms + audits),
        "matrix": {(row["family_id"], row["variant"]) for row in bases}
        == {(family["family_id"], variant) for family in family_plan() for variant in VARIANTS},
        "target_balance": dict(target_counts) == {letter: 12 for letter in OPTION_LETTERS},
        "transition_balance": len(transition_counts) == 6
        and set(transition_counts.values()) == {8},
        "error_class_balance": dict(class_counts)
        == {name: 8 for name in ERROR_CLASSES},
        "error_position_balance": dict(position_counts)
        == {name: 16 for name in ERROR_POSITIONS},
        "family_depth_gradients": len(depths) == 24
        and all(values["medium"] > values["easy"] for values in depths.values()),
        "all_stored_audits": all(row["passed_pre_tokenizer"] for row in audits),
        "all_source_replays": all(row["passed"] for row in source_results.values()),
        "all_meta_replays": all(row["passed"] for row in meta_results.values()),
        "all_surface_checks": all(row["passed"] for row in surface_results.values()),
        "no_gpu_execution": True,
        "tokenizer_gate_passed": tokenizer_report["status"] == "passed"
        and tokenizer_report["base_count"] == 48
        and tokenizer_report["all_exact_ao_s_length"]
        and all(
            item["single_token"]
            for item in tokenizer_report["answer_label_tokens"].values()
        ),
    }
    result = {
        "schema": SCHEMA,
        "status": "zero_gpu_automatic_passed" if all(checks.values()) else "failed",
        "passed": all(checks.values()),
        "checks": checks,
        "counts": {
            "families": len(cards),
            "bases": len(bases),
            "arms": len(arms),
            "targets": dict(sorted(target_counts.items())),
            "transitions": dict(sorted(transition_counts.items())),
            "error_classes": dict(sorted(class_counts.items())),
            "error_positions": dict(sorted(position_counts.items())),
        },
        "source_results": source_results,
        "meta_results": meta_results,
        "surface_results": surface_results,
        "gpu_qwen_capping_run": False,
    }
    write_json(root / "independent_validation_report.json", result)
    return result


def main() -> None:
    result = validate(HERE)
    print(
        {
            "passed": result["passed"],
            "status": result["status"],
            "families": result["counts"]["families"],
            "bases": result["counts"]["bases"],
            "arms": result["counts"]["arms"],
            "gpu_used": False,
        }
    )


if __name__ == "__main__":
    main()
