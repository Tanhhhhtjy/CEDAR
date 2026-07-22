from __future__ import annotations

import functools
import importlib.util
import json
from collections import Counter
from pathlib import Path

from cedar_m2.checkpoint import file_sha256, object_sha256
from scripts import build_c2_v8_natural_confirmation as builder
from scripts.c2_v8_portable_hashes import canonical_text_sha256

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
    assert object_sha256(result["source"]) == builder.SOURCE_OBJECT_SHA256
    assert canonical_text_sha256(builder.SOURCE_LICENSE) == builder.LICENSE_CANONICAL_SHA256
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


def test_construction_and_disclosed_single_review_gate_pass() -> None:
    validator = load_validator()
    report = validator.validate_construction(builder)
    assert report["status"] == "passed"
    assert report["gpu_authorized"] is False
    assert report["human_review_gate"] == "pending_single_review_waiver_validation"
    gate = validator.validate_review_gate()
    assert gate["status"] == "passed"
    assert gate["gpu_authorized"] is True
    assert gate["protocol_deviation"] is True
    assert gate["review_mode"] == "single_data_lead_full_review"
    assert gate["raw_agreement"] is None
    assert gate["cohen_kappa"] is None


def test_materialized_manifest_matches_deterministic_content() -> None:
    materialized = json.loads((DATA_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert materialized["content_hashes"] == built()["manifest"]["content_hashes"]
    assert materialized["human_review_gate"].startswith("pending")
    assert materialized["gpu_authorized"] is False


def test_canonical_text_hash_is_newline_portable_and_character_sensitive(tmp_path) -> None:
    lf = tmp_path / "lf.txt"
    crlf = tmp_path / "crlf.txt"
    changed = tmp_path / "changed.txt"
    lf.write_bytes(b"alpha\nbeta\n")
    crlf.write_bytes(b"alpha\r\nbeta\r\n")
    changed.write_bytes(b"alpha\nBeta\n")
    assert canonical_text_sha256(lf) == canonical_text_sha256(crlf)
    assert canonical_text_sha256(lf) != canonical_text_sha256(changed)
    assert file_sha256(lf) != file_sha256(crlf)


def test_json_object_hash_ignores_formatting_but_detects_value_changes() -> None:
    compact = json.loads('{"a":1,"b":[2,3]}')
    indented = json.loads('{\n  "b": [2, 3],\n  "a": 1\n}')
    changed = json.loads('{"a":2,"b":[2,3]}')
    assert object_sha256(compact) == object_sha256(indented)
    assert object_sha256(compact) != object_sha256(changed)


def test_single_review_attestation_binds_current_packet_and_data() -> None:
    validator = load_validator()
    gate = validator.validate_single_review_gate()
    hashes = gate["attestation_hashes"]
    assert hashes["review_packet_content_sha256"] == object_sha256(
        validator.read_csv(DATA_ROOT / "review" / "review_packet_a.csv")
    )
    assert hashes["bases_object_sha256"] == built()["manifest"]["content_hashes"]["bases"]
    assert hashes["arms_object_sha256"] == built()["manifest"]["content_hashes"]["arms"]
