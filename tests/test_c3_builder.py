from __future__ import annotations

import functools
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from cedar_m2.checkpoint import file_sha256, object_sha256
from scripts import build_c3_sycobench_bidirectional as builder
from scripts.c3_portable_hashes import (
    canonical_text_sha256,
    portable_identity_matches,
    semantic_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets" / "c3_sycobench_bidirectional_selectivity"
VALIDATOR_PATH = DATA_ROOT / "validate_c3_sycobench_bidirectional.py"


@functools.lru_cache(maxsize=1)
def built() -> dict:
    return builder.build(allow_network=False, write=False)


def load_validator():
    spec = importlib.util.spec_from_file_location("c3_validator_test", VALIDATOR_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_source_and_frozen_resource_contract() -> None:
    result = built()
    assert object_sha256(result["source"]) == builder.SOURCE_OBJECT_SHA256
    assert builder.SOURCE_OBJECT_SHA256 == (
        "6c02e880a1f75b1bdae53d42e7417aa5b5cf911d53daf6e8059826ebb3b689bb"
    )
    contract = result["manifest"]["model_contract"]
    assert contract["revision"] == "9216db5781bf21249d130ec9da846c4624c16137"
    assert contract["axis_sha256"] == builder.AXIS_SHA256
    assert contract["capping_config_sha256"] == builder.CAP_CONFIG_SHA256
    assert contract["layers"] == list(range(46, 54))
    assert contract["enable_thinking"] is False


def test_counts_mirrors_and_pair_identity() -> None:
    result = built()
    bases = result["bases"]
    arms = result["arms"]
    assert len(bases) == 600
    assert len(arms) == 2400
    assert Counter(row["direction"] for row in arms) == {
        "harmful": 1200,
        "helpful": 1200,
    }
    assert Counter(row["arm"] for row in arms) == {"AO": 1200, "S": 1200}
    grouped = defaultdict(dict)
    for row in arms:
        grouped[row["base_id"]][(row["direction"], row["arm"])] = row
    assert all(len(grid) == 4 for grid in grouped.values())
    for base in bases:
        grid = grouped[base["base_id"]]
        harmful = grid[("harmful", "AO")]
        helpful = grid[("helpful", "AO")]
        assert harmful["initial_answer"] == helpful["target_answer"] == base["gold_answer"]
        assert harmful["target_answer"] == helpful["initial_answer"] == base["foil_answer"]
        for direction in builder.DIRECTIONS:
            ao = grid[(direction, "AO")]
            social = grid[(direction, "S")]
            assert ao["messages"][:-1] == social["messages"][:-1]
            assert ao["followup_token_count_without_contract"] == 15
            assert social["followup_token_count_without_contract"] == 15
            assert ao["followup_target_token_position"] == social["followup_target_token_position"]


def test_foil_and_template_assignments_are_balanced_before_gpu() -> None:
    bases = built()["bases"]
    assert Counter(base["foil_answer"] for base in bases) == {
        "A": 150,
        "B": 151,
        "C": 150,
        "D": 149,
    }
    assert Counter(base["template_id"] for base in bases) == {
        0: 150,
        1: 150,
        2: 150,
        3: 150,
    }
    for field in ("foil_answer", "template_id"):
        counts = Counter((base["domain"], base["difficulty"], base[field]) for base in bases)
        values = builder.LABELS if field == "foil_answer" else range(4)
        for domain in builder.DOMAINS:
            for difficulty in builder.DIFFICULTIES:
                cell = [counts[(domain, difficulty, value)] for value in values]
                assert max(cell) - min(cell) <= 1


def test_smoke_review_and_template_token_audit_are_frozen() -> None:
    result = built()
    smoke = result["smoke"]
    review = result["review"]
    assert len(smoke) == 96
    assert len({row["base_id"] for row in smoke}) == 24
    assert len({(row["domain"], row["difficulty"]) for row in smoke}) == 24
    assert {row["target_answer"] for row in smoke} == set(builder.LABELS)
    assert {row["template_id"] for row in smoke} == {0, 1, 2, 3}
    assert len(review) == 32
    assert Counter(row["direction"] for row in review) == {
        "harmful": 16,
        "helpful": 16,
    }
    assert len({(row["domain"], row["difficulty"]) for row in review}) == 24
    audit = result["manifest"]["template_token_audit"]
    assert audit["status"] == "passed"
    assert all(
        target["tokens"] == 15
        for template in audit["templates"].values()
        for target in template["targets"].values()
    )


def test_repeated_build_and_materialized_semantic_hashes_match() -> None:
    first = built()
    second = builder.build(allow_network=False, write=False)
    mapping = {
        "bases": "c3_base600.jsonl",
        "arms": "c3_arms2400.jsonl",
        "smoke": "c3_smoke.jsonl",
    }
    for key, filename in mapping.items():
        assert object_sha256(first[key]) == object_sha256(second[key])
        assert semantic_sha256(DATA_ROOT / filename) == object_sha256(first[key])
    manifest = json.loads((DATA_ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["content_hashes"]["bases"] == object_sha256(first["bases"])
    assert manifest["content_hashes"]["arms"] == object_sha256(first["arms"])
    assert manifest["content_hashes"]["review_packet"] == semantic_sha256(
        DATA_ROOT / "review" / "review_packet32.csv"
    )
    assert manifest["content_hashes"]["template_review_packet"] == semantic_sha256(
        DATA_ROOT / "review" / "template_review_packet.csv"
    )
    assert manifest["gpu_authorized"] is False


def test_construction_passes_and_missing_review_files_are_blocked(tmp_path) -> None:
    validator = load_validator()
    report = validator.validate_construction(validator.load_builder())
    assert report["status"] == "passed"
    assert report["gpu_authorized"] is False
    assert report["review_gate"] == "pending_real_single_data_lead_review"
    validator.ATTESTATION_PATH = tmp_path / "single_review_attestation.json"
    validator.REVIEW_COMPLETED_PATH = tmp_path / "review_completed.csv"
    validator.TEMPLATE_COMPLETED_PATH = tmp_path / "template_review_completed.csv"
    validator.ARTIFACT_ROOT = tmp_path / "no-model-artifacts"
    with pytest.raises(ValueError, match="human review is pending"):
        validator.validate_review_gate(validator.load_builder())


def test_cross_platform_hashes_and_semantic_json_are_portable(tmp_path) -> None:
    lf = tmp_path / "lf.jsonl"
    crlf = tmp_path / "crlf.jsonl"
    changed = tmp_path / "changed.jsonl"
    lf.write_bytes(b'{"a":1,"b":2}\n')
    crlf.write_bytes(b'{ "b": 2, "a": 1 }\r\n')
    changed.write_bytes(b'{"a":2,"b":2}\n')
    assert canonical_text_sha256(lf) != canonical_text_sha256(crlf)
    assert semantic_sha256(lf) == semantic_sha256(crlf)
    assert semantic_sha256(lf) != semantic_sha256(changed)
    same_text_crlf = tmp_path / "same_text_crlf.txt"
    same_text_lf = tmp_path / "same_text_lf.txt"
    same_text_lf.write_bytes(b"alpha\nbeta\n")
    same_text_crlf.write_bytes(b"alpha\r\nbeta\r\n")
    assert canonical_text_sha256(same_text_lf) == canonical_text_sha256(same_text_crlf)
    assert file_sha256(same_text_lf) != file_sha256(same_text_crlf)
    expected = {
        "hash_mode": "utf8_lf_canonical_v1",
        "git_blob_id": "same-filtered-blob",
        "canonical_text_sha256": canonical_text_sha256(same_text_lf),
        "semantic_hash_mode": None,
        "semantic_sha256": None,
        "raw_sha256_diagnostic": file_sha256(same_text_lf),
        "bytes_diagnostic": same_text_lf.stat().st_size,
    }
    observed = {
        **expected,
        "raw_sha256_diagnostic": file_sha256(same_text_crlf),
        "bytes_diagnostic": same_text_crlf.stat().st_size,
    }
    assert portable_identity_matches(observed, expected)
