from __future__ import annotations

import copy
import inspect
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HARDENING = ROOT / "datasets" / "m2_2b_formal" / "prover_generator_hardening_v1_1"
if str(HARDENING) not in sys.path:
    sys.path.insert(0, str(HARDENING))

from atomic_edit_candidate_enumerator import (  # noqa: E402
    enumerate_atomic_edit_candidates,
    evaluate_candidate_neighborhood,
)
from hardening_common import (  # noqa: E402
    ERROR_CLASSES,
    MASTER_SEED,
    canonical_json_file_sha256,
    stable_seed,
    write_json,
)
from proof_graph_builder import (  # noqa: E402
    FAMILY_BLUEPRINTS,
    assert_no_answer_channel,
    blueprint_by_id,
    proof_graph_builder,
)
from semantic_f_operators import derive_error_class, validate_f_certificate  # noqa: E402
from semantic_validator import semantic_validator  # noqa: E402

SENTINEL_FAMILIES = ["PGF-01", "PGF-02", "PGF-03", "PGF-04", "PGF-05", "PGF-06"]


def _case(family_id: str, error_class: str, variant: str = "easy"):
    blueprint = blueprint_by_id(family_id)
    seed = stable_seed(MASTER_SEED, "test", family_id, variant, error_class)
    built = proof_graph_builder(blueprint, variant, seed, blueprint["transition"], error_class)
    candidates = enumerate_atomic_edit_candidates(built)
    neighborhood = evaluate_candidate_neighborhood(built, candidates)
    return built, candidates, neighborhood


def test_builder_has_no_intended_edit_output_or_parameter() -> None:
    built, _, _ = _case("PGF-01", ERROR_CLASSES[0])
    assert_no_answer_channel(built)
    parameters = inspect.signature(proof_graph_builder).parameters
    assert "intended_edit" not in parameters
    assert "updated_world" not in parameters


def test_enumerator_has_no_answer_argument_and_is_deterministic() -> None:
    built, first, _ = _case("PGF-07", ERROR_CLASSES[0], "medium")
    second = enumerate_atomic_edit_candidates(built)
    assert inspect.signature(enumerate_atomic_edit_candidates).parameters.keys() == {
        "builder_output"
    }
    assert first == second
    assert 20 <= len(first) <= 100
    assert [item["rank"] for item in first] == list(range(1, len(first) + 1))


def test_all_48_neighborhoods_have_one_unique_edit_and_stable_order() -> None:
    counts = []
    for family_index, blueprint in enumerate(FAMILY_BLUEPRINTS):
        for variant_index, variant in enumerate(("easy", "medium")):
            for replicate_id in range(1, 4):
                error_class = ERROR_CLASSES[
                    (family_index * 3 + variant_index * 3 + replicate_id - 1) % 6
                ]
                seed = stable_seed(MASTER_SEED, blueprint["family_id"], variant, replicate_id)
                built = proof_graph_builder(
                    blueprint, variant, seed, blueprint["transition"], error_class
                )
                candidates = enumerate_atomic_edit_candidates(built)
                result = evaluate_candidate_neighborhood(built, candidates)
                assert result["success_count"] == 1
                assert result["successful_rank"] is not None
                assert candidates == enumerate_atomic_edit_candidates(built)
                counts.append(len(candidates))
    assert len(counts) == 48
    assert sum(value >= 20 for value in counts) / len(counts) >= 0.90


@pytest.mark.parametrize(
    ("family_id", "error_class"), zip(SENTINEL_FAMILIES, ERROR_CLASSES, strict=True)
)
def test_each_f_operator_positive_and_dual_gold(family_id: str, error_class: str) -> None:
    built, _, neighborhood = _case(family_id, error_class)
    result = semantic_validator(built, neighborhood)
    assert result["passed"]
    assert result["f_validation"]["derived_error_class"] == error_class
    assert result["initial_validation"]["agreement"]
    assert result["updated_validation"]["agreement"]


@pytest.mark.parametrize(
    ("family_id", "error_class"), zip(SENTINEL_FAMILIES, ERROR_CLASSES, strict=True)
)
def test_each_f_operator_rejects_unentailed_premise(family_id: str, error_class: str) -> None:
    built, _, _ = _case(family_id, error_class)
    context = built["fallacy_context"]
    forged = copy.deepcopy(context["premises"])
    forged[0]["predicate"] += "forged"
    with pytest.raises(ValueError, match="premises"):
        derive_error_class(built["initial_world"], forged, context["claimed_conclusion"])


def test_declared_error_class_cannot_forge_structural_class() -> None:
    built, _, neighborhood = _case("PGF-01", "cross_entity_transfer")
    result = semantic_validator(built, neighborhood)
    certificate = copy.deepcopy(result["f_certificate"])
    certificate["error_class"] = "unsupported_bridge"
    check = validate_f_certificate(
        built, certificate, result["selected_edit"], result["updated_world"]
    )
    assert not check["passed"]
    assert not check["checks"]["derived_class_matches_declared"]


def test_second_unsupported_downstream_step_fails() -> None:
    built, _, neighborhood = _case("PGF-02", "converse_error")
    result = semantic_validator(built, neighborhood)
    certificate = copy.deepcopy(result["f_certificate"])
    certificate["valid_downstream_steps"].append({"kind": "unsupported_extra_step"})
    check = validate_f_certificate(
        built, certificate, result["selected_edit"], result["updated_world"]
    )
    assert not check["passed"]
    assert not check["checks"]["no_second_unsupported_step"]


def test_easy_medium_formal_depth_contracts() -> None:
    for variant, low, high in (("easy", 2, 3), ("medium", 4, 6)):
        built, _, neighborhood = _case("PGF-08", "scope_transfer", variant)
        result = semantic_validator(built, neighborhood)
        assert result["passed"]
        assert low <= result["difficulty"]["actual_shortest_proof_depth"] <= high


def test_canonical_json_hash_ignores_key_order_and_line_endings(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_bytes(b'{\r\n  "b": 2,\r\n  "a": 1\r\n}\r\n')
    write_json(second, {"a": 1, "b": 2})
    assert canonical_json_file_sha256(first) == canonical_json_file_sha256(second)


def test_no_prohibited_generic_fatal_step_in_v11_source() -> None:
    prohibited = "unsupported" + "_atomic_state_claim"
    for path in HARDENING.glob("*.py"):
        assert prohibited not in path.read_text(encoding="utf-8")
