from __future__ import annotations

import functools
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

from cedar_m2.checkpoint import file_sha256, object_sha256
from scripts import build_c2_v8_natural_confirmation as builder

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets" / "c2_v8_natural_social_confirmation"
VALIDATOR_PATH = DATA_ROOT / "validate_c2_v8.py"


@functools.lru_cache(maxsize=1)
def built() -> dict:
    return builder.build(allow_network=False, write=False)


def load_validator():
    spec = importlib.util.spec_from_file_location("c2_v8_validator_test", VALIDATOR_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pinned_source_hash_license_schema_and_duplicates_are_disclosed() -> None:
    result = built()
    assert file_sha256(builder.SOURCE_OFFICIAL) == builder.SOURCE_SHA256
    assert file_sha256(builder.SOURCE_LICENSE) == builder.LICENSE_SHA256
    audit = result["source_audit"]
    assert audit["status"] == "passed"
    assert audit["records"] == 600
    assert audit["normalized_stems"] == 272
    assert audit["unique_exact_question_ordered_options"] == 542
    assert audit["duplicate_option_items"] == 0
    assert audit["source_split_fields_present"] is False
    assert audit["train_dev_leakage_status"] == "not_applicable_evaluation_only_release"


def test_four_option_candidate_tokens_and_official_target_rule() -> None:
    result = built()
    assert result["manifest"]["answer_space"]["token_ids_after_exact_prefix"] == {
        "A": 32,
        "B": 33,
        "C": 34,
        "D": 35,
    }
    assert Counter(base["target_answer"] for base in result["bases"]) == {
        "A": 437,
        "B": 163,
    }
    assert all(base["target_answer"] != base["gold_answer"] for base in result["bases"])


def test_bases_arms_templates_and_pair_identity() -> None:
    result = built()
    bases = result["bases"]
    arms = result["arms"]
    assert len(bases) == 600
    assert len(arms) == 1200
    assert Counter(row["arm"] for row in arms) == {"AO": 600, "S": 600}
    assert Counter(base["template_id"] for base in bases) == {
        0: 150,
        1: 150,
        2: 150,
        3: 150,
    }
    grouped = {}
    for row in arms:
        grouped.setdefault(row["base_id"], {})[row["arm"]] = row
    assert all(set(pair) == {"AO", "S"} for pair in grouped.values())
    assert all(
        pair["AO"]["messages"][:-1] == pair["S"]["messages"][:-1]
        for pair in grouped.values()
    )
    assert max(
        abs(
            pair["S"]["followup_token_count_without_contract"]
            - pair["AO"]["followup_token_count_without_contract"]
        )
        for pair in grouped.values()
    ) == 4


def test_free_smoke_and_review_samples_are_frozen_before_results() -> None:
    result = built()
    assert len(result["free"]) == 240
    free_counts = Counter((row["domain"], row["difficulty"]) for row in result["free"])
    easy_counts = {value for (_, difficulty), value in free_counts.items() if difficulty == "easy"}
    harder_counts = {
        value
        for (_, difficulty), value in free_counts.items()
        if difficulty in {"medium", "hard"}
    }
    assert easy_counts == {6}
    assert harder_counts == {12}
    smoke = result["smoke"]
    assert len(smoke) == 12
    assert {row["domain"] for row in smoke} == set(builder.DOMAINS)
    assert {row["difficulty"] for row in smoke} == set(builder.DIFFICULTIES)
    assert {row["template_id"] for row in smoke} == {0, 1, 2, 3}
    assert {row["target_answer"] for row in smoke} == {"A", "B"}
    ids_a = {row["review_id"] for row in result["review_a"]}
    ids_b = {row["review_id"] for row in result["review_b"]}
    assert len(ids_a) == 120
    assert ids_a == ids_b


def test_repeated_in_memory_build_has_identical_content_hashes() -> None:
    first = built()
    second = builder.build(allow_network=False, write=False)
    keys = (
        "source",
        "bases",
        "arms",
        "free",
        "smoke",
        "review_a",
        "review_b",
        "adjudication_template",
    )
    for key in keys:
        assert object_sha256(first[key]) == object_sha256(second[key])


def test_construction_passes_but_human_review_gate_is_pending() -> None:
    validator = load_validator()
    report = validator.validate_construction(builder)
    assert report["status"] == "passed"
    assert report["gpu_authorized"] is False
    assert report["human_review_gate"] == "pending"
    with pytest.raises(ValueError, match="Human review gate is pending"):
        validator.validate_review_gate()


def test_materialized_manifest_matches_deterministic_content() -> None:
    materialized = json.loads((DATA_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert materialized["content_hashes"] == built()["manifest"]["content_hashes"]
    assert materialized["human_review_gate"].startswith("pending")
    assert materialized["gpu_authorized"] is False
