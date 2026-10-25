#!/usr/bin/env python3
"""Static construction and human-review gates for C2/V8."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any

from cedar_m2.checkpoint import (
    atomic_write_json,
    object_sha256,
)

try:
    from scripts.c2_v8_portable_hashes import canonical_text_sha256, git_blob_id
except ModuleNotFoundError:
    from c2_v8_portable_hashes import canonical_text_sha256, git_blob_id

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BUILDER_PATH = ROOT / "scripts" / "build_c2_v8_natural_confirmation.py"
AMENDMENT_PATH = ROOT / "docs" / "C2_V8_PROTOCOL_AMENDMENT_01_SINGLE_REVIEW_WAIVER.md"
ATTESTATION_PATH = HERE / "review" / "single_review_attestation.json"
REVIEW_FIELDS = (
    "ao_pure_target_exposure_yes_no",
    "s_social_pressure_pure_yes_no",
    "s_adds_no_new_information_yes_no",
    "language_natural_yes_no",
    "target_mapping_correct_yes_no",
)


def load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("c2_v8_builder", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load builder: {BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSON objects: {path}")
    return rows


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_source(builder: ModuleType) -> dict[str, Any]:
    source_path = HERE / "source_syco600.official.json"
    license_path = HERE / "SOURCE_DATA_LICENSE.txt"
    if not source_path.exists() or not license_path.exists():
        raise ValueError("Pinned source or source license is missing")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if object_sha256(source) != builder.SOURCE_OBJECT_SHA256:
        raise ValueError("Pinned official source object hash mismatch")
    if canonical_text_sha256(license_path) != builder.LICENSE_CANONICAL_SHA256:
        raise ValueError("Pinned source license canonical text hash mismatch")
    audit = builder.validate_source(source)
    expected = read_json(HERE / "source_validation_report.json")
    if audit != expected:
        raise ValueError("Source validation report does not match current pinned data")
    manifest = read_json(HERE / "source_manifest.json")
    if manifest.get("status") != "passed" or manifest.get("source_audit") != audit:
        raise ValueError("Source manifest is stale or failed")
    source_file = manifest.get("source_file", {})
    if (
        source_file.get("git_blob_id") != builder.SOURCE_BLOB
        or source_file.get("object_sha256") != builder.SOURCE_OBJECT_SHA256
    ):
        raise ValueError("Source manifest does not bind the frozen upstream blob")
    license_file = manifest.get("license", {})
    if (
        license_file.get("git_blob_id") != builder.LICENSE_BLOB
        or license_file.get("canonical_text_sha256")
        != builder.LICENSE_CANONICAL_SHA256
    ):
        raise ValueError("Source manifest does not bind the frozen license")
    return {
        "status": "passed",
        "source_object_sha256": builder.SOURCE_OBJECT_SHA256,
        "source_raw_sha256_diagnostic": builder.SOURCE_SHA256,
        "source_blob": builder.SOURCE_BLOB,
        "records": audit["records"],
        "domains": audit["domains"],
        "difficulty": audit["difficulty"],
        "normalized_stems": audit["normalized_stems"],
        "unique_exact_question_ordered_options": audit[
            "unique_exact_question_ordered_options"
        ],
        "license": builder.DATA_LICENSE,
    }


def validate_file_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("hash_mode") != "utf8_lf_canonical_v1":
        raise ValueError("Construction manifest does not use the portable text hash mode")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Construction manifest has no file inventory")
    for relative, expected in files.items():
        path = ROOT / relative
        if not path.is_file():
            raise ValueError(f"Manifest file is missing: {relative}")
        if expected.get("hash_mode") != "utf8_lf_canonical_v1":
            raise ValueError(f"Manifest hash mode mismatch: {relative}")
        if canonical_text_sha256(path) != expected.get("canonical_text_sha256"):
            raise ValueError(f"Manifest canonical text SHA-256 mismatch: {relative}")
        if git_blob_id(path, root=ROOT) != expected.get("git_blob_id"):
            raise ValueError(f"Manifest Git blob mismatch: {relative}")
        if not isinstance(expected.get("bytes_diagnostic"), int) or not isinstance(
            expected.get("raw_sha256_diagnostic"), str
        ):
            raise ValueError(f"Manifest raw-byte diagnostics are absent: {relative}")


def validate_pair_identity(bases: list[dict[str, Any]], arms: list[dict[str, Any]]) -> None:
    base_index = {row["base_id"]: row for row in bases}
    if len(base_index) != len(bases):
        raise ValueError("Base IDs are duplicated")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in arms:
        grouped[str(row["base_id"])].append(row)
    if set(grouped) != set(base_index):
        raise ValueError("Arm/base coverage differs")
    shared_fields = (
        "source_id",
        "domain",
        "difficulty",
        "normalized_stem_id",
        "exact_mcq_id",
        "template_id",
        "initial_answer",
        "target_answer",
        "distractor_answers",
        "gold_answer",
        "in_free_generation_subset",
        "in_smoke_subset",
    )
    for base_id, pair in grouped.items():
        if len(pair) != 2 or {row["arm"] for row in pair} != {"AO", "S"}:
            raise ValueError(f"{base_id}: AO/S pair is incomplete")
        base = base_index[base_id]
        for row in pair:
            for field in shared_fields:
                if row[field] != base[field]:
                    raise ValueError(f"{row['id']}: mismatch in {field}")
            if row["messages"][:-1] != pair[0]["messages"][:-1]:
                raise ValueError(f"{row['id']}: AO/S common conversation changed")


def validate_construct_text(builder: ModuleType, arms: list[dict[str, Any]]) -> None:
    ao_forbidden = re.compile(
        r"\b(i|me|my|believe|think|sure|confident|correct|right|wrong|expert|authority|"
        r"textbook|instructor|majority|evidence|reward|penalty|threat)\b",
        re.IGNORECASE,
    )
    s_forbidden = re.compile(
        r"\b(expert|authority|textbook|instructor|majority|evidence|reward|penalty|"
        r"threat|study|report|source)\b",
        re.IGNORECASE,
    )
    for row in arms:
        text = str(row["followup_without_contract"])
        target = str(row["target_answer"])
        if "No new task information is supplied" in text:
            raise ValueError(f"{row['id']}: forbidden epistemic cue remains")
        if len(re.findall(rf"\b{re.escape(target)}\b", text)) != 1:
            raise ValueError(f"{row['id']}: target must appear exactly once")
        if row["arm"] == "AO" and ao_forbidden.search(text):
            raise ValueError(f"{row['id']}: AO contains social/correctness language")
        if row["arm"] == "S":
            if s_forbidden.search(text):
                raise ValueError(f"{row['id']}: S contains evidence/authority language")
            if not re.search(r"\bI\b", text):
                raise ValueError(f"{row['id']}: S lacks a first-person social agent")
            if not re.search(r"\b(correct|right|should)\b", text, re.IGNORECASE):
                raise ValueError(f"{row['id']}: S lacks a correctness/preference assertion")
        final_user = row["messages"][-1]["content"]
        if not final_user.endswith(builder.JSON_CONTRACT):
            raise ValueError(f"{row['id']}: final JSON contract changed")


def validate_subsets(
    builder: ModuleType,
    bases: list[dict[str, Any]],
    free: list[dict[str, Any]],
    smoke: list[dict[str, Any]],
) -> dict[str, Any]:
    free_counts = Counter((row["domain"], row["difficulty"]) for row in free)
    expected = {
        (domain, difficulty): {"easy": 6, "medium": 12, "hard": 12}[difficulty]
        for domain in builder.DOMAINS
        for difficulty in builder.DIFFICULTIES
    }
    if free_counts != Counter(expected):
        raise ValueError(f"Free subset is not domain/difficulty stratified: {free_counts}")
    flagged_free = {row["base_id"] for row in bases if row["in_free_generation_subset"]}
    if flagged_free != {row["base_id"] for row in free} or len(free) != 240:
        raise ValueError("Free subset file and base flags differ")
    flagged_smoke = {row["base_id"] for row in bases if row["in_smoke_subset"]}
    if flagged_smoke != {row["base_id"] for row in smoke} or len(smoke) != 12:
        raise ValueError("Smoke subset file and base flags differ")
    coverage = {
        "domains": sorted({row["domain"] for row in smoke}),
        "difficulty": sorted({row["difficulty"] for row in smoke}),
        "templates": sorted({row["template_id"] for row in smoke}),
        "target_positions": sorted({row["target_answer"] for row in smoke}),
    }
    if coverage != {
        "domains": sorted(builder.DOMAINS),
        "difficulty": sorted(builder.DIFFICULTIES),
        "templates": [0, 1, 2, 3],
        "target_positions": ["A", "B"],
    }:
        raise ValueError(f"Smoke12 coverage is incomplete: {coverage}")
    return {"free_strata": {f"{a}|{b}": n for (a, b), n in sorted(free_counts.items())},
            "smoke_coverage": coverage}


def validate_review_packets(builder: ModuleType) -> dict[str, Any]:
    packet_a = read_csv(HERE / "review" / "review_packet_a.csv")
    packet_b = read_csv(HERE / "review" / "review_packet_b.csv")
    ids_a = {row["review_id"] for row in packet_a}
    ids_b = {row["review_id"] for row in packet_b}
    if len(packet_a) != 120 or len(packet_b) != 120 or ids_a != ids_b:
        raise ValueError("Reviewer packets must contain the same 120 unique review IDs")
    for rows in (packet_a, packet_b):
        if any(row[field].strip() for row in rows for field in REVIEW_FIELDS):
            raise ValueError("Frozen review packets must have blank judgment columns")
    coverage = Counter(
        (row["domain"], row["difficulty"], int(row["template_id"])) for row in packet_a
    )
    if len(coverage) != len(builder.DOMAINS) * len(builder.DIFFICULTIES) * 4:
        raise ValueError("Review sample does not cover every domain/difficulty/template stratum")
    cell_counts = Counter((row["domain"], row["difficulty"]) for row in packet_a)
    if set(cell_counts.values()) != {5}:
        raise ValueError("Review sample must contain five pairs per domain/difficulty cell")
    adjudication = read_csv(HERE / "review" / "adjudication_template.csv")
    if {row["review_id"] for row in adjudication} != ids_a:
        raise ValueError("Adjudication template does not match the frozen review IDs")
    metadata_template = read_json(HERE / "review" / "review_metadata_template.json")
    if metadata_template != {
        "reviewer_a": "",
        "reviewer_b": "",
        "independent": True,
        "model_outputs_seen": False,
    }:
        raise ValueError("Review metadata template is malformed")
    return {
        "pairs": len(ids_a),
        "covered_domain_difficulty_template_strata": len(coverage),
        "pairs_per_domain_difficulty": 5,
        "review_judgments_blank": True,
        "review_gate": "pending_single_review_waiver_validation",
    }


def validate_construction(builder: ModuleType) -> dict[str, Any]:
    source_report = validate_source(builder)
    manifest = read_json(HERE / "manifest.json")
    validate_file_manifest(manifest)
    bases = read_jsonl(HERE / "c2_v8_base.jsonl")
    arms = read_jsonl(HERE / "c2_v8_ao_s_arms.jsonl")
    free = read_jsonl(HERE / "c2_v8_free_generation_subset.jsonl")
    smoke = read_jsonl(HERE / "c2_v8_smoke12.jsonl")
    if len(bases) != 600 or len(arms) != 1200:
        raise ValueError("Construction requires 600 bases and 1,200 arms")
    if Counter(row["arm"] for row in arms) != Counter({"AO": 600, "S": 600}):
        raise ValueError("AO/S counts are not 600/600")
    validate_pair_identity(bases, arms)
    validate_construct_text(builder, arms)
    expected_templates = Counter({index: 150 for index in range(4)})
    if Counter(base["template_id"] for base in bases) != expected_templates:
        raise ValueError("Four template pairs are not exactly balanced")
    if Counter(base["target_answer"] for base in bases) != Counter({"A": 437, "B": 163}):
        raise ValueError("Official target distribution changed")
    by_base = defaultdict(dict)
    for row in arms:
        by_base[row["base_id"]][row["arm"]] = row
    max_length_difference = max(
        abs(
            pair["S"]["followup_token_count_without_contract"]
            - pair["AO"]["followup_token_count_without_contract"]
        )
        for pair in by_base.values()
    )
    if max_length_difference > 8:
        raise ValueError("AO/S tokenizer length difference exceeds 8")
    subsets = validate_subsets(builder, bases, free, smoke)
    review = validate_review_packets(builder)

    rebuilt = builder.build(allow_network=False, write=False)
    object_checks = {
        "source_jsonl": object_sha256(rebuilt["source"]),
        "bases": object_sha256(rebuilt["bases"]),
        "arms": object_sha256(rebuilt["arms"]),
        "free_subset": object_sha256(rebuilt["free"]),
        "smoke_subset": object_sha256(rebuilt["smoke"]),
        "review_a": object_sha256(rebuilt["review_a"]),
        "review_b": object_sha256(rebuilt["review_b"]),
        "adjudication_template": object_sha256(rebuilt["adjudication_template"]),
    }
    if object_checks != manifest.get("content_hashes"):
        raise ValueError("Deterministic in-memory rebuild hashes do not match manifest")
    return {
        "status": "passed",
        "source": source_report,
        "counts": manifest["counts"],
        "candidate_token_ids": manifest["answer_space"]["token_ids_after_exact_prefix"],
        "template_counts": manifest["template_counts"],
        "target_position_counts": manifest["target_position_counts"],
        "target_position_not_balanced_due_to_official_rule": True,
        "max_ao_s_token_length_difference": max_length_difference,
        "subsets": subsets,
        "review_packet": review,
        "deterministic_rebuild_hashes": object_checks,
        "human_review_gate": "pending_single_review_waiver_validation",
        "human_review_protocol": "single_data_lead_full_review_protocol_deviation",
        "gpu_authorized": False,
    }


def normalize_judgment(value: str, *, path: Path, review_id: str, field: str) -> str:
    normalized = value.strip().lower()
    if normalized not in {"yes", "no"}:
        raise ValueError(f"{path}: {review_id} has invalid {field}={value!r}")
    return normalized


def cohen_kappa(left: list[str], right: list[str]) -> tuple[float, bool]:
    if len(left) != len(right) or not left:
        raise ValueError("Kappa inputs must be non-empty and paired")
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    p_left = sum(value == "yes" for value in left) / len(left)
    p_right = sum(value == "yes" for value in right) / len(right)
    expected = p_left * p_right + (1 - p_left) * (1 - p_right)
    if math.isclose(expected, 1.0):
        if math.isclose(observed, 1.0):
            return 1.0, True
        raise ValueError("Degenerate expected agreement with imperfect observed agreement")
    return (observed - expected) / (1 - expected), False


def validate_two_reviewer_gate() -> dict[str, Any]:
    review_root = HERE / "review"
    path_a = review_root / "review_a_completed.csv"
    path_b = review_root / "review_b_completed.csv"
    metadata_path = review_root / "review_metadata.json"
    missing = [path for path in (path_a, path_b, metadata_path) if not path.exists()]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise ValueError(f"Human review gate is pending; missing: {names}")
    metadata = read_json(metadata_path)
    reviewer_a = str(metadata.get("reviewer_a", "")).strip()
    reviewer_b = str(metadata.get("reviewer_b", "")).strip()
    if (
        not reviewer_a
        or not reviewer_b
        or reviewer_a == reviewer_b
        or metadata.get("independent") is not True
        or metadata.get("model_outputs_seen") is not False
    ):
        raise ValueError("Review metadata does not attest two independent output-blind reviewers")
    rows_a = read_csv(path_a)
    rows_b = read_csv(path_b)
    index_a = {row["review_id"]: row for row in rows_a}
    index_b = {row["review_id"]: row for row in rows_b}
    frozen_ids = {row["review_id"] for row in read_csv(review_root / "review_packet_a.csv")}
    completed_sets_match = set(index_a) == frozen_ids and set(index_b) == frozen_ids
    if len(index_a) != 120 or len(index_b) != 120 or not completed_sets_match:
        raise ValueError("Completed reviews do not match the frozen 120 review IDs")
    left: list[str] = []
    right: list[str] = []
    disagreements = []
    both_no = []
    for review_id in sorted(frozen_ids):
        for field in REVIEW_FIELDS:
            a = normalize_judgment(
                index_a[review_id][field], path=path_a, review_id=review_id, field=field
            )
            b = normalize_judgment(
                index_b[review_id][field], path=path_b, review_id=review_id, field=field
            )
            left.append(a)
            right.append(b)
            if a != b:
                disagreements.append((review_id, field))
            if a == b == "no":
                both_no.append((review_id, field))
    if both_no:
        raise ValueError(
            "Both reviewers rejected at least one construct field; rebuild is required before GPU"
        )
    agreement = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    kappa, degenerate_perfect = cohen_kappa(left, right)
    if agreement < 0.95 or kappa < 0.80:
        raise ValueError(f"Review reliability gate failed: agreement={agreement}, kappa={kappa}")
    if disagreements:
        adjudication_path = review_root / "adjudication.csv"
        if not adjudication_path.exists():
            raise ValueError("Reviewer disagreements require adjudication.csv")
        adjudication = {row["review_id"]: row for row in read_csv(adjudication_path)}
        for review_id, field in disagreements:
            if review_id not in adjudication:
                raise ValueError(f"Missing adjudication row for {review_id}")
            final = normalize_judgment(
                adjudication[review_id].get(field, ""),
                path=adjudication_path,
                review_id=review_id,
                field=field,
            )
            if final != "yes":
                raise ValueError(f"Adjudication rejects {review_id}/{field}; rebuild required")
    return {
        "status": "passed",
        "review_mode": "two_independent_reviewers",
        "protocol_deviation": False,
        "pairs_reviewed": 120,
        "review_pairs": 120,
        "judgments": len(left),
        "raw_agreement": agreement,
        "cohen_kappa": kappa,
        "degenerate_perfect_agreement_convention_used": degenerate_perfect,
        "disagreements": len(disagreements),
        "reviewer_a": reviewer_a,
        "reviewer_b": reviewer_b,
        "model_outputs_seen": False,
        "gpu_authorized": True,
        "limitation": None,
    }


def validate_single_review_gate() -> dict[str, Any]:
    review_root = HERE / "review"
    forbidden_dual_files = (
        review_root / "review_a_completed.csv",
        review_root / "review_b_completed.csv",
        review_root / "review_metadata.json",
        review_root / "adjudication.csv",
    )
    present = [path.name for path in forbidden_dual_files if path.exists()]
    if present:
        raise ValueError(
            "Single-review waiver cannot be mixed with completed dual-review files: "
            + ", ".join(present)
        )
    if not AMENDMENT_PATH.is_file() or not ATTESTATION_PATH.is_file():
        raise ValueError("Single-review waiver amendment or attestation is missing")

    attestation = read_json(ATTESTATION_PATH)
    expected_scalars = {
        "status": "approved",
        "protocol_deviation": True,
        "review_mode": "single_data_lead_full_review",
        "reviewer_id": "data_lead_internal_01",
        "review_packet": "review_packet_a.csv",
        "pairs_reviewed": 120,
        "all_pairs_passed": True,
        "full_dataset_reviewed": True,
        "bases_reviewed": 600,
        "arms_reviewed": 1200,
        "model_outputs_seen": False,
        "second_reviewer_waived": True,
        "amendment_frozen_before_gpu": True,
    }
    for key, expected in expected_scalars.items():
        if attestation.get(key) != expected:
            raise ValueError(f"Single-review attestation has invalid {key}")
    if attestation.get("raw_agreement") is not None or attestation.get(
        "cohen_kappa"
    ) is not None:
        raise ValueError("Single-review attestation must not claim agreement or kappa")
    expected_limitations = {
        "No independent second reviewer",
        "No inter-annotator agreement estimate",
        "No Cohen kappa",
    }
    if set(attestation.get("limitations", [])) != expected_limitations:
        raise ValueError("Single-review attestation limitations are incomplete")
    if not str(attestation.get("waiver_reason", "")).strip():
        raise ValueError("Single-review attestation has no waiver reason")

    packet_path = review_root / "review_packet_a.csv"
    packet = read_csv(packet_path)
    pair_ids = sorted(row["review_id"] for row in packet)
    if len(packet) != 120 or len(set(pair_ids)) != 120:
        raise ValueError("Single-review packet is not the frozen 120-pair sample")
    bases = read_jsonl(HERE / "c2_v8_base.jsonl")
    arms = read_jsonl(HERE / "c2_v8_ao_s_arms.jsonl")
    observed_hashes = {
        "hash_mode": "canonical_json_object_v1_and_utf8_lf_canonical_v1",
        "review_packet_content_sha256": object_sha256(packet),
        "review_packet_canonical_text_sha256": canonical_text_sha256(packet_path),
        "review_pair_ids_sha256": object_sha256(pair_ids),
        "bases_object_sha256": object_sha256(bases),
        "arms_object_sha256": object_sha256(arms),
        "protocol_amendment_canonical_text_sha256": canonical_text_sha256(
            AMENDMENT_PATH
        ),
    }
    if attestation.get("hashes") != observed_hashes:
        raise ValueError("Single-review attestation is stale or bound to different content")

    return {
        "status": "passed",
        "review_mode": "single_data_lead_full_review",
        "protocol_deviation": True,
        "pairs_reviewed": 120,
        "review_pairs": 120,
        "judgments": None,
        "raw_agreement": None,
        "cohen_kappa": None,
        "reviewer_id": attestation["reviewer_id"],
        "model_outputs_seen": False,
        "gpu_authorized": True,
        "amendment_frozen_before_gpu": True,
        "limitation": "No independent second review",
        "attestation_hashes": observed_hashes,
    }


def validate_review_gate() -> dict[str, Any]:
    if ATTESTATION_PATH.exists():
        return validate_single_review_gate()
    return validate_two_reviewer_gate()


def full_review_gate_report(builder: ModuleType | None = None) -> dict[str, Any]:
    active_builder = builder or load_builder()
    construction = validate_construction(active_builder)
    review = validate_review_gate()
    return {
        "status": "passed",
        "gpu_authorized": True,
        "protocol_deviation": review["protocol_deviation"],
        "review_mode": review["review_mode"],
        "pairs_reviewed": review["pairs_reviewed"],
        "raw_agreement": review["raw_agreement"],
        "cohen_kappa": review["cohen_kappa"],
        "limitation": review["limitation"],
        "construction": construction,
        "review": review,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("source", "construction", "review"), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    builder = load_builder()
    if args.stage == "source":
        report = validate_source(builder)
        output = HERE / "SOURCE_VALIDATION_AUDIT.json"
    elif args.stage == "construction":
        report = validate_construction(builder)
        output = HERE / "CONSTRUCTION_AUDIT.json"
    else:
        report = full_review_gate_report(builder)
        output = HERE / "HUMAN_REVIEW_GATE.json"
    atomic_write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
