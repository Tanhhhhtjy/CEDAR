#!/usr/bin/env python3
"""Static construction and explicitly disclosed review gates for C3."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cedar_m2.checkpoint import atomic_write_json, object_sha256  # noqa: E402
from scripts.c3_portable_hashes import (  # noqa: E402
    canonical_text_sha256,
    file_identity,
    portable_identity_matches,
    read_jsonl,
)

HERE = Path(__file__).resolve().parent
BUILDER_PATH = ROOT / "scripts" / "build_c3_sycobench_bidirectional.py"
PROTOCOL_PATH = ROOT / "docs" / "C3_SYCOBENCH_BIDIRECTIONAL_SELECTIVITY_PROTOCOL.md"
AMENDMENT_PATH = ROOT / "docs" / "C3_PROTOCOL_AMENDMENT_01_AGENT_SELF_REVIEW_WAIVER.md"
ARTIFACT_ROOT = ROOT / "artifacts" / "m2" / "c3_sycobench_bidirectional_selectivity"
REVIEW_ROOT = HERE / "review"
ATTESTATION_PATH = REVIEW_ROOT / "agent_self_review_attestation.json"
REVIEW_COMPLETED_PATH = REVIEW_ROOT / "agent_review_completed.csv"
TEMPLATE_COMPLETED_PATH = REVIEW_ROOT / "template_agent_review_completed.csv"

REVIEW_FIELDS = (
    "ao_pure_target_exposure_yes_no",
    "s_only_adds_social_pressure_yes_no",
    "both_add_no_fact_or_reason_yes_no",
    "direction_mirror_correct_yes_no",
    "language_and_mapping_natural_yes_no",
)
TEMPLATE_REVIEW_FIELDS = (
    "ao_pure_target_exposure_yes_no",
    "s_only_adds_social_pressure_yes_no",
    "both_add_no_fact_or_reason_yes_no",
    "language_natural_yes_no",
)


def load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("c3_builder", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load C3 builder: {BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_source(builder: ModuleType) -> dict[str, Any]:
    source_path = (
        ROOT / "datasets" / "c2_v8_natural_social_confirmation" / "source_syco600.official.json"
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if object_sha256(source) != builder.SOURCE_OBJECT_SHA256:
        raise ValueError("C3 pinned SycoBench source object hash mismatch")
    audit = builder.c2.validate_source(source)
    if audit.get("status") != "passed" or audit.get("records") != 600:
        raise ValueError("C3 pinned SycoBench source validation failed")
    manifest = read_json(HERE / "manifest.json")
    provenance = manifest.get("source", {})
    if (
        provenance.get("object_sha256") != builder.SOURCE_OBJECT_SHA256
        or provenance.get("commit") != builder.c2.SOURCE_COMMIT
        or provenance.get("upstream_git_blob_id") != builder.c2.SOURCE_BLOB
        or provenance.get("source_audit") != audit
    ):
        raise ValueError("C3 source provenance manifest is stale")
    return {
        "status": "passed",
        "source_object_sha256": builder.SOURCE_OBJECT_SHA256,
        "source_commit": builder.c2.SOURCE_COMMIT,
        "source_git_blob_id": builder.c2.SOURCE_BLOB,
        "records": 600,
        "domains": audit["domains"],
        "difficulty": audit["difficulty"],
        "normalized_stems": audit["normalized_stems"],
        "license": builder.c2.DATA_LICENSE,
    }


def validate_file_inventory(manifest: dict[str, Any]) -> None:
    if manifest.get("hash_mode") != "utf8_lf_canonical_v1":
        raise ValueError("C3 manifest text hash mode changed")
    if manifest.get("semantic_hash_mode") != "canonical_json_object_v1":
        raise ValueError("C3 manifest semantic hash mode changed")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("C3 manifest file inventory is missing")
    for relative, expected in files.items():
        path = ROOT / relative
        if not path.is_file():
            raise ValueError(f"C3 manifest file is missing: {relative}")
        observed = file_identity(path, root=ROOT)
        if not portable_identity_matches(observed, expected):
            raise ValueError(f"C3 file identity mismatch: {relative}")


def validate_pairing(
    builder: ModuleType,
    bases: list[dict[str, Any]],
    arms: list[dict[str, Any]],
) -> None:
    base_index = {str(row["base_id"]): row for row in bases}
    if len(base_index) != 600:
        raise ValueError("C3 base IDs are missing or duplicated")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in arms:
        grouped[str(row["base_id"])].append(row)
    if set(grouped) != set(base_index):
        raise ValueError("C3 arm/base coverage differs")
    shared = (
        "source_id",
        "normalized_stem_id",
        "exact_mcq_id",
        "domain",
        "difficulty",
        "template_id",
        "gold_answer",
        "foil_answer",
        "in_smoke_subset",
    )
    for base_id, rows in grouped.items():
        if len(rows) != 4 or Counter((row["direction"], row["arm"]) for row in rows) != Counter(
            {(direction, arm): 1 for direction in builder.DIRECTIONS for arm in builder.ARMS}
        ):
            raise ValueError(f"{base_id}: direction/arm grid is incomplete")
        base = base_index[base_id]
        by_key = {(row["direction"], row["arm"]): row for row in rows}
        for row in rows:
            if any(row[field] != base[field] for field in shared):
                raise ValueError(f"{row['id']}: base metadata mismatch")
            expected_initial, expected_target = builder.direction_mapping(
                base, str(row["direction"])
            )
            if (
                row["initial_answer"] != expected_initial
                or row["target_answer"] != expected_target
                or row["initial_position"] != expected_initial
                or row["target_position"] != expected_target
                or expected_initial == expected_target
            ):
                raise ValueError(f"{row['id']}: harmful/helpful mirror mapping changed")
        harmful = by_key[("harmful", "AO")]
        helpful = by_key[("helpful", "AO")]
        if not (
            harmful["initial_answer"] == helpful["target_answer"] == base["gold_answer"]
            and harmful["target_answer"] == helpful["initial_answer"] == base["foil_answer"]
        ):
            raise ValueError(f"{base_id}: directions are not exact mirrors")
        for direction in builder.DIRECTIONS:
            ao = by_key[(direction, "AO")]
            social = by_key[(direction, "S")]
            paired_fields = (
                "direction_id",
                "initial_answer",
                "target_answer",
                "target_position",
                "initial_position",
                "other_wrong_answers",
                "followup_token_count_without_contract",
                "followup_target_token_position",
            )
            if any(ao[field] != social[field] for field in paired_fields):
                raise ValueError(f"{base_id}/{direction}: AO/S pair metadata differs")
            if ao["messages"][:-1] != social["messages"][:-1]:
                raise ValueError(f"{base_id}/{direction}: AO/S pre-follow-up conversation differs")


def validate_construct_text(builder: ModuleType, arms: list[dict[str, Any]]) -> None:
    ao_forbidden = re.compile(
        r"\b(i|me|my|mine|believe|think|sure|confident|certain|correct|right|wrong|"
        r"expert|authority|evidence|reason|because|reward|threat|majority)\b",
        re.IGNORECASE,
    )
    s_forbidden = re.compile(
        r"\b(expert|authority|evidence|reason|because|reward|threat|majority|study|"
        r"report|source|fact)\b",
        re.IGNORECASE,
    )
    for row in arms:
        text = str(row["followup_without_contract"])
        target = str(row["target_answer"])
        if "No new task information" in text:
            raise ValueError(f"{row['id']}: forbidden epistemic cue appears")
        if len(re.findall(rf"\b{target}\b", text)) != 1:
            raise ValueError(f"{row['id']}: target label must appear exactly once")
        if row["arm"] == "AO" and ao_forbidden.search(text):
            raise ValueError(f"{row['id']}: AO contains social/correctness language")
        if row["arm"] == "S":
            if s_forbidden.search(text):
                raise ValueError(f"{row['id']}: S contains fact/evidence/authority language")
            if not re.search(r"\b(I|my)\b", text, re.IGNORECASE):
                raise ValueError(f"{row['id']}: S lacks a first-person agent")
            if not re.search(r"\b(confident|confidently|firm|certain)\b", text, re.IGNORECASE):
                raise ValueError(f"{row['id']}: S lacks frozen confidence language")
        messages = row.get("messages")
        if (
            not isinstance(messages, list)
            or [message.get("role") for message in messages]
            != ["system", "user", "assistant", "user"]
            or not str(messages[-1].get("content", "")).endswith(builder.JSON_CONTRACT)
        ):
            raise ValueError(f"{row['id']}: conversation schema or JSON contract changed")


def validate_smoke_and_review(
    builder: ModuleType,
    bases: list[dict[str, Any]],
    arms: list[dict[str, Any]],
    smoke: list[dict[str, Any]],
) -> dict[str, Any]:
    flagged = {str(row["base_id"]) for row in bases if row["in_smoke_subset"]}
    if len(flagged) != 24 or {str(row["base_id"]) for row in smoke} != flagged:
        raise ValueError("C3 smoke file and base flags differ")
    expected_smoke = [row for row in arms if row["base_id"] in flagged]
    if smoke != expected_smoke or len(smoke) != 96:
        raise ValueError("C3 smoke rows are not the exact frozen 24-base grid")
    smoke_bases = [row for row in bases if row["base_id"] in flagged]
    cells = {(row["domain"], row["difficulty"]) for row in smoke_bases}
    targets = {row["target_answer"] for row in smoke}
    templates = {int(row["template_id"]) for row in smoke}
    if (
        len(cells) != len(builder.DOMAINS) * len(builder.DIFFICULTIES)
        or targets != set(builder.LABELS)
        or templates != set(range(4))
    ):
        raise ValueError("C3 smoke24 coverage is incomplete")

    review = read_csv(REVIEW_ROOT / "review_packet32.csv")
    template_review = read_csv(REVIEW_ROOT / "template_review_packet.csv")
    if len(review) != 32 or len({row["review_id"] for row in review}) != 32:
        raise ValueError("C3 review packet must contain 32 unique direction cases")
    if Counter(row["direction"] for row in review) != Counter({"harmful": 16, "helpful": 16}):
        raise ValueError("C3 review packet is not direction balanced")
    if len({(row["domain"], row["difficulty"]) for row in review}) != 24:
        raise ValueError("C3 review packet does not cover all domain/difficulty cells")
    if {row["target_answer"] for row in review} != set(builder.LABELS):
        raise ValueError("C3 review packet does not cover A/B/C/D targets")
    if {int(row["template_id"]) for row in review} != set(range(4)):
        raise ValueError("C3 review packet does not cover all templates")
    if any(row[field].strip() for row in review for field in REVIEW_FIELDS):
        raise ValueError("Frozen review packet contains fabricated judgments")
    if len(template_review) != 4 or any(
        row[field].strip() for row in template_review for field in TEMPLATE_REVIEW_FIELDS
    ):
        raise ValueError("Frozen template review packet must contain four blank judgments")
    return {
        "smoke_bases": 24,
        "smoke_records_per_condition": 96,
        "smoke_domain_difficulty_cells": 24,
        "smoke_target_positions": sorted(targets),
        "smoke_templates": sorted(templates),
        "review_direction_cases": 32,
        "review_harmful_helpful": {"harmful": 16, "helpful": 16},
        "review_judgments_blank": True,
        "template_pairs_pending_review": 4,
    }


def review_binding_hashes() -> dict[str, Any]:
    packet_path = REVIEW_ROOT / "review_packet32.csv"
    template_path = REVIEW_ROOT / "template_review_packet.csv"
    bases = read_jsonl(HERE / "c3_base600.jsonl")
    arms = read_jsonl(HERE / "c3_arms2400.jsonl")
    packet = read_csv(packet_path)
    templates = read_csv(template_path)
    return {
        "hash_mode": "canonical_json_object_v1_and_utf8_lf_canonical_v1",
        "review_packet_semantic_sha256": object_sha256(packet),
        "review_packet_canonical_text_sha256": canonical_text_sha256(packet_path),
        "review_ids_sha256": object_sha256(sorted(row["review_id"] for row in packet)),
        "template_packet_semantic_sha256": object_sha256(templates),
        "template_packet_canonical_text_sha256": canonical_text_sha256(template_path),
        "bases_semantic_sha256": object_sha256(bases),
        "arms_semantic_sha256": object_sha256(arms),
        "protocol_canonical_text_sha256": canonical_text_sha256(PROTOCOL_PATH),
        "amendment_canonical_text_sha256": canonical_text_sha256(AMENDMENT_PATH),
    }


def validate_construction(builder: ModuleType) -> dict[str, Any]:
    source = validate_source(builder)
    manifest = read_json(HERE / "manifest.json")
    validate_file_inventory(manifest)
    bases = read_jsonl(HERE / "c3_base600.jsonl")
    arms = read_jsonl(HERE / "c3_arms2400.jsonl")
    smoke = read_jsonl(HERE / "c3_smoke.jsonl")
    if len(bases) != 600 or len(arms) != 2400:
        raise ValueError("C3 construction requires 600 bases and 2,400 arms")
    if Counter(row["direction"] for row in arms) != Counter(
        {"harmful": 1200, "helpful": 1200}
    ) or Counter(row["arm"] for row in arms) != Counter({"AO": 1200, "S": 1200}):
        raise ValueError("C3 direction or arm counts changed")
    validate_pairing(builder, bases, arms)
    validate_construct_text(builder, arms)

    foil_counts = Counter(row["foil_answer"] for row in bases)
    if foil_counts != Counter({"A": 150, "B": 151, "C": 150, "D": 149}):
        raise ValueError(f"C3 foil position distribution changed: {foil_counts}")
    grouped_foils = Counter((row["domain"], row["difficulty"], row["foil_answer"]) for row in bases)
    for domain in builder.DOMAINS:
        for difficulty in builder.DIFFICULTIES:
            counts = [grouped_foils[(domain, difficulty, label)] for label in builder.LABELS]
            if max(counts) - min(counts) > 1:
                raise ValueError(f"C3 foil imbalance in {domain}/{difficulty}")
    templates = Counter(int(row["template_id"]) for row in bases)
    if templates != Counter({0: 150, 1: 150, 2: 150, 3: 150}):
        raise ValueError("C3 template assignment changed")
    cell_templates = Counter(
        (row["domain"], row["difficulty"], int(row["template_id"])) for row in bases
    )
    for domain in builder.DOMAINS:
        for difficulty in builder.DIFFICULTIES:
            counts = [cell_templates[(domain, difficulty, index)] for index in range(4)]
            if max(counts) - min(counts) > 1:
                raise ValueError(f"C3 template imbalance in {domain}/{difficulty}")

    subsets = validate_smoke_and_review(builder, bases, arms, smoke)
    rebuilt_a = builder.build(allow_network=False, write=False)
    rebuilt_b = builder.build(allow_network=False, write=False)
    keys = {
        "source": "source",
        "bases": "bases",
        "arms": "arms",
        "smoke": "smoke",
    }
    rebuilt_hashes = {name: object_sha256(rebuilt_a[key]) for name, key in keys.items()}
    rebuilt_hashes.update(
        {
            "review_packet": object_sha256(builder.csv_semantic_rows(rebuilt_a["review"])),
            "template_review_packet": object_sha256(
                builder.csv_semantic_rows(rebuilt_a["template_review"])
            ),
        }
    )
    if rebuilt_hashes != manifest.get("content_hashes"):
        raise ValueError("C3 deterministic rebuild hashes do not match manifest")
    repeat_keys = (*keys.values(), "review", "template_review")
    if any(object_sha256(rebuilt_a[key]) != object_sha256(rebuilt_b[key]) for key in repeat_keys):
        raise ValueError("C3 repeated in-memory rebuild is not deterministic")
    if any(row["followup_token_count_without_contract"] != 15 for row in arms):
        raise ValueError("C3 frozen AO/S templates are not all 15 tokens")
    if manifest.get("gpu_authorized") is not False:
        raise ValueError("Construction manifest must not authorize GPU before the review gate")
    return {
        "status": "passed",
        "gpu_authorized": False,
        "source": source,
        "counts": manifest["counts"],
        "candidate_token_ids": manifest["model_contract"]["candidate_token_ids"],
        "template_token_audit": manifest["template_token_audit"],
        "template_counts": {str(key): value for key, value in sorted(templates.items())},
        "foil_position_counts": dict(sorted(foil_counts.items())),
        "foil_cell_max_min_difference": 1,
        "template_cell_max_min_difference": 1,
        "subsets": subsets,
        "deterministic_rebuild_hashes": rebuilt_hashes,
        "review_binding_hashes": review_binding_hashes(),
        "review_gate": "separate_amended_review_gate_required",
        "original_review_plan": "single_data_lead_review",
        "amended_review_mode": "agent_self_review_user_waiver",
        "protocol_deviation": True,
        "human_review_performed": False,
        "dual_independent_review": False,
        "cohen_kappa": None,
    }


def _compare_completed(
    frozen_path: Path,
    completed_path: Path,
    judgment_fields: tuple[str, ...],
    id_field: str,
) -> list[dict[str, str]]:
    frozen = read_csv(frozen_path)
    completed = read_csv(completed_path)
    if len(frozen) != len(completed):
        raise ValueError(f"{completed_path.name}: row count differs from frozen packet")
    left = {row[id_field]: row for row in frozen}
    right = {row[id_field]: row for row in completed}
    if set(left) != set(right) or len(left) != len(frozen):
        raise ValueError(f"{completed_path.name}: IDs differ from frozen packet")
    for item_id in left:
        for field, value in left[item_id].items():
            if field in {*judgment_fields, "reviewer_notes"}:
                continue
            if right[item_id].get(field) != value:
                raise ValueError(
                    f"{completed_path.name}: frozen content changed for {item_id}/{field}"
                )
        for field in judgment_fields:
            if right[item_id].get(field, "").strip().lower() != "yes":
                raise ValueError(f"{completed_path.name}: {item_id}/{field} is not approved")
    return completed


def validate_review_gate(
    builder: ModuleType | None = None, *, enforce_pre_artifact: bool = True
) -> dict[str, Any]:
    active_builder = builder or load_builder()
    construction = validate_construction(active_builder)
    if enforce_pre_artifact and ARTIFACT_ROOT.exists() and any(ARTIFACT_ROOT.rglob("*")):
        raise ValueError("C3 review gate must be frozen before any C3 model artifact exists")
    required = (ATTESTATION_PATH, REVIEW_COMPLETED_PATH, TEMPLATE_COMPLETED_PATH)
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise ValueError("C3 amended review gate is pending; missing: " + ", ".join(missing))
    completed = _compare_completed(
        REVIEW_ROOT / "review_packet32.csv",
        REVIEW_COMPLETED_PATH,
        REVIEW_FIELDS,
        "review_id",
    )
    completed_templates = _compare_completed(
        REVIEW_ROOT / "template_review_packet.csv",
        TEMPLATE_COMPLETED_PATH,
        TEMPLATE_REVIEW_FIELDS,
        "template_id",
    )
    attestation = read_json(ATTESTATION_PATH)
    expected = {
        "status": "approved",
        "protocol_deviation": True,
        "review_mode": "agent_self_review_user_waiver",
        "direction_cases_reviewed": 32,
        "template_pairs_reviewed": 4,
        "all_items_passed": True,
        "model_outputs_seen": False,
        "gpu_execution_seen": False,
        "human_review_performed": False,
        "data_lead_review_performed": False,
        "user_review_waived": True,
        "dual_independent_review": False,
        "raw_agreement": None,
        "cohen_kappa": None,
    }
    for key, value in expected.items():
        if attestation.get(key) != value:
            raise ValueError(f"C3 agent-self-review attestation has invalid {key}")
    if not str(attestation.get("review_agent_id", "")).strip():
        raise ValueError("C3 agent-self-review attestation has no review agent ID")
    authorization = attestation.get("user_authorization", {})
    if authorization != {
        "date": "2026-07-23",
        "decision": "human_review_waived_agent_self_review_authorized",
        "quoted_instruction": "不用我审核了，你直接自检一遍然后继续跑",
    }:
        raise ValueError("C3 agent-self-review attestation lacks the exact user waiver")
    limitations = {
        "No human or data-lead review",
        "No independent reviewer",
        "No inter-annotator agreement estimate",
        "No Cohen kappa",
    }
    if set(attestation.get("limitations", [])) != limitations:
        raise ValueError("C3 agent-self-review limitations are incomplete")
    hashes = {
        **review_binding_hashes(),
        "review_completed_semantic_sha256": object_sha256(completed),
        "template_review_completed_semantic_sha256": object_sha256(completed_templates),
        "review_completed_canonical_text_sha256": canonical_text_sha256(REVIEW_COMPLETED_PATH),
        "template_review_completed_canonical_text_sha256": canonical_text_sha256(
            TEMPLATE_COMPLETED_PATH
        ),
    }
    if attestation.get("hashes") != hashes:
        raise ValueError("C3 agent-self-review attestation is stale or bound to different content")
    return {
        "status": "passed",
        "gpu_authorized": True,
        "review_mode": "agent_self_review_user_waiver",
        "protocol_deviation": True,
        "direction_cases_reviewed": 32,
        "template_pairs_reviewed": 4,
        "review_agent_id": attestation["review_agent_id"],
        "model_outputs_seen": False,
        "gpu_execution_seen": False,
        "human_review_performed": False,
        "data_lead_review_performed": False,
        "user_review_waived": True,
        "dual_independent_review": False,
        "raw_agreement": None,
        "cohen_kappa": None,
        "limitations": sorted(limitations),
        "hashes": hashes,
        "construction": construction,
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
        output = HERE / "source_audit.json"
    elif args.stage == "construction":
        report = validate_construction(builder)
        output = HERE / "construction_audit.json"
    else:
        report = validate_review_gate(builder)
        output = HERE / "REVIEW_GATE.json"
    atomic_write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
