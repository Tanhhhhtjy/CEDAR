from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "datasets" / "m2_2b_formal" / "prover_generator_feasibility_v1"
sys.path.insert(0, str(DATA_DIR))

from build_prover_generator_feasibility import (  # noqa: E402
    FAMILY_BLUEPRINTS,
    ROUTES,
    common_specs,
)
from formal_horn import (  # noqa: E402
    apply_edit,
    canonical_sha256,
    evaluate_fixed_point,
    fact,
    label_from_evaluation,
    literal,
    opposite,
    read_jsonl,
    rule,
    stable_seed,
)


def test_family_catalog_has_eight_unique_families() -> None:
    assert len(FAMILY_BLUEPRINTS) == 8
    assert len({item["family_id"] for item in FAMILY_BLUEPRINTS}) == 8


def test_common_specs_have_easy_and_medium_for_every_family() -> None:
    specs = common_specs()
    assert len(specs) == 16
    for family in FAMILY_BLUEPRINTS:
        variants = {spec["variant"] for spec in specs if spec["family_id"] == family["family_id"]}
        assert variants == {"easy", "medium"}


def test_transition_coverage() -> None:
    counts = Counter(spec["transition"] for spec in common_specs())
    assert len(counts) == 6
    assert min(counts.values()) >= 2


def test_medium_depth_exceeds_easy_for_every_family() -> None:
    specs = common_specs()
    for family in FAMILY_BLUEPRINTS:
        depths = {
            spec["variant"]: spec["target_proof_depth"]
            for spec in specs
            if spec["family_id"] == family["family_id"]
        }
        assert depths["medium"] > depths["easy"]


def test_fixed_point_preserves_open_world_unknown() -> None:
    query = literal("q", "a")
    world = {
        "constants": ["a"],
        "records": [fact("f1", literal("p", "a"))],
    }
    evaluation = evaluate_fixed_point(world)
    assert label_from_evaluation(evaluation, query) == "unknown"


def test_fixed_point_derives_horn_chain() -> None:
    query = literal("q", "a")
    world = {
        "constants": ["a"],
        "records": [
            fact("f1", literal("p", "a")),
            rule("r1", [literal("p", "?x")], literal("m", "?x")),
            rule("r2", [literal("m", "?x")], query),
        ],
    }
    evaluation = evaluate_fixed_point(world)
    assert label_from_evaluation(evaluation, query) == "true"
    assert evaluation["proofs"]["q(a)"][0]["depth"] == 2


def test_fixed_point_marks_signed_contradiction_invalid() -> None:
    query = literal("q", "a")
    world = {
        "constants": ["a"],
        "records": [fact("f1", query), fact("f2", opposite(query))],
    }
    assert label_from_evaluation(evaluate_fixed_point(world), query) == "invalid"


@pytest.mark.parametrize("operation", ["add", "delete", "replace"])
def test_atomic_edit_operations(operation: str) -> None:
    before = fact("critical", literal("p", "a"))
    after = fact("critical", literal("p", "a", negated=True))
    world = {"constants": ["a"], "records": [before]}
    if operation == "add":
        edit = {"operation": operation, "before": None, "after": fact("new", literal("q", "a"))}
    elif operation == "delete":
        edit = {"operation": operation, "before": before, "after": None}
    else:
        edit = {"operation": operation, "before": before, "after": after}
    updated = apply_edit(world, edit)
    assert updated != world


def test_stable_seed_depends_on_route() -> None:
    a = stable_seed(20260725, "provergen_style", "PGF-01", "easy")
    b = stable_seed(20260725, "reviseqa_style", "PGF-01", "easy")
    assert a != b
    assert a == stable_seed(20260725, "provergen_style", "PGF-01", "easy")


def test_canonical_hash_ignores_mapping_key_order() -> None:
    assert canonical_sha256({"a": 1, "b": 2}) == canonical_sha256({"b": 2, "a": 1})


@pytest.mark.skipif(not (DATA_DIR / "provergen_style16.jsonl").exists(), reason="data not built")
@pytest.mark.parametrize("route", ROUTES)
def test_route_output_retains_sixteen_attempts(route: str) -> None:
    rows = read_jsonl(DATA_DIR / f"{route}16.jsonl")
    assert len(rows) == 16
    assert len({row["spec_id"] for row in rows}) == 16


@pytest.mark.skipif(not (DATA_DIR / "provergen_style16.jsonl").exists(), reason="data not built")
@pytest.mark.parametrize("route", ROUTES)
def test_valid_records_have_atomic_v_and_single_error_f(route: str) -> None:
    rows = read_jsonl(DATA_DIR / f"{route}16.jsonl")
    for row in rows:
        if row["status"] != "valid":
            continue
        assert row["arms"].keys() == {"N", "R", "V", "F"}
        assert row["v_edit"]["edit_count"] == 1
        assert row["arms"]["V"]["target"] == row["arms"]["F"]["induced_target"]
        assert sum(not step["valid"] for step in row["arms"]["F"]["candidate_reasoning_steps"]) == 1
        assert row["arms"]["F"]["second_failure_point_absent"]


@pytest.mark.skipif(
    not (DATA_DIR / "naturalization_sentinel8.jsonl").exists(), reason="data not built"
)
def test_naturalization_has_eight_reverse_mappable_records() -> None:
    rows = read_jsonl(DATA_DIR / "naturalization_sentinel8.jsonl")
    assert len(rows) == 8
    assert all(row["gold_hidden_from_text"] for row in rows)
    assert all(row["reverse_map"]["fatal_error_step"] == 2 for row in rows)


@pytest.mark.skipif(not (DATA_DIR / "manifest.json").exists(), reason="analysis not built")
def test_manifest_contains_no_absolute_paths() -> None:
    manifest = json.loads((DATA_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert all(":" not in entry["path"] for entry in manifest["files"])
    assert manifest["gpu_qwen_capping_run"] is False
