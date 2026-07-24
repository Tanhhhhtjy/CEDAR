#!/usr/bin/env python3
"""Build the measurement-only CEDAR development v1.1 candidate bank."""

from __future__ import annotations

import argparse
import copy
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from development_v1_1_common import (
    AMENDMENT,
    AO_TEMPLATE,
    CALIBRATION_FAMILIES,
    CANDIDATE_ARMS,
    GATE_SPEC,
    HERE,
    N_TEMPLATE,
    ORDERED_POSITION_TRANSITIONS,
    PARENT_DATA_FREEZE_COMMIT,
    PARENT_FILE_CANONICAL_TEXT_SHA256,
    PARENT_RESULT_COMMIT,
    R_TEMPLATE,
    S1_TEMPLATE,
    S2_TEMPLATE,
    SCHEMA,
    TEXT_HASH_MODE,
    TOKENIZER_BINDING,
    V1_ROOT,
    VALIDATION_FAMILIES,
    build_arm,
    canonical_sha256,
    canonical_token_count,
    count_position_transitions,
    file_sha256,
    neutralize_vf,
    position_transition,
    read_jsonl,
    remap_option_block,
    semantic_mapping,
    split_for_family,
    validate_parent_files,
    whitespace_token_count,
    write_json,
    write_jsonl,
)


def _measurement_base(parent: dict[str, Any]) -> dict[str, Any]:
    initial_position, target_position = position_transition(parent["family_id"], parent["variant"])
    mapping = semantic_mapping(
        parent["initial_semantic_label"],
        parent["target_semantic_label"],
        initial_position,
        target_position,
    )
    english_base = remap_option_block(parent["english_base_prompt"], mapping, "en")
    chinese_base = remap_option_block(parent["chinese_base_prompt"], mapping, "zh")
    english_v = neutralize_vf(parent["english_v_text"], "en")
    english_f = neutralize_vf(parent["english_f_text"], "en")
    chinese_v = neutralize_vf(parent["chinese_v_text"], "zh")
    chinese_f = neutralize_vf(parent["chinese_f_text"], "zh")
    ao = AO_TEMPLATE.format(target=target_position)
    s1 = S1_TEMPLATE.format(target=target_position)
    s2 = S2_TEMPLATE.format(target=target_position)
    templates = {"AO": ao, "S1": s1, "S2": s2}
    return {
        "schema": SCHEMA,
        "measurement_amendment": AMENDMENT,
        "base_id": parent["base_id"],
        "family_id": parent["family_id"],
        "development_split": split_for_family(parent["family_id"]),
        "structure_class": parent["structure_class"],
        "structure_instance": parent["structure_instance"],
        "variant": parent["variant"],
        "semantic_transition": parent["transition"],
        "position_transition": f"{initial_position}->{target_position}",
        "semantic_to_letter": mapping,
        "initial_semantic_label": parent["initial_semantic_label"],
        "target_semantic_label": parent["target_semantic_label"],
        "initial_answer": initial_position,
        "target_answer": target_position,
        "error_class": parent["error_class"],
        "error_position": parent["error_position"],
        "english_base_prompt": english_base,
        "chinese_base_prompt": chinese_base,
        "english_v_text": english_v,
        "english_f_text": english_f,
        "chinese_v_text": chinese_v,
        "chinese_f_text": chinese_f,
        "social_templates": templates,
        "surface_length_checks": {
            "social_whitespace_equal": len({whitespace_token_count(text) for text in templates.values()}) == 1,
            "social_canonical_equal": len({canonical_token_count(text) for text in templates.values()}) == 1,
            "vf_whitespace_difference": abs(whitespace_token_count(english_v) - whitespace_token_count(english_f)),
            "vf_canonical_difference": abs(canonical_token_count(english_v) - canonical_token_count(english_f)),
            "fixed_qwen_exact_social_length": None,
        },
        "parent_invariants": {
            "parent_base_canonical_sha256": canonical_sha256(parent),
            "source_symbolic_sha256": parent["source_symbolic_sha256"],
            "selected_edit_canonical_sha256": canonical_sha256(parent["selected_edit"]),
            "paired_formal_canonical_sha256": canonical_sha256(parent["paired_formal"]),
            "object_validation_canonical_sha256": canonical_sha256(parent["object_validation"]),
            "difficulty_canonical_sha256": canonical_sha256(parent["difficulty"]),
        },
        "tokenizer_binding": copy.deepcopy(TOKENIZER_BINDING),
        "analysis_schema": copy.deepcopy(parent["analysis_schema"]),
    }


def _candidate_arms(base: dict[str, Any]) -> list[dict[str, Any]]:
    followups = {
        "N": N_TEMPLATE,
        "R": R_TEMPLATE,
        "V": base["english_v_text"],
        "F": base["english_f_text"],
        "AO": base["social_templates"]["AO"],
        "S1": base["social_templates"]["S1"],
        "S2": base["social_templates"]["S2"],
    }
    return [build_arm(base, arm, followups[arm]) for arm in CANDIDATE_ARMS]


def _audit(parent: dict[str, Any], base: dict[str, Any], arms: list[dict[str, Any]]) -> dict[str, Any]:
    by_arm = {row["arm"]: row for row in arms}
    edit_marker = "Proposed record change"
    checks = {
        "parent_source_symbolic_unchanged": base["parent_invariants"]["source_symbolic_sha256"]
        == parent["source_symbolic_sha256"],
        "parent_selected_edit_unchanged": base["parent_invariants"]["selected_edit_canonical_sha256"]
        == canonical_sha256(parent["selected_edit"]),
        "parent_formal_pair_unchanged": base["parent_invariants"]["paired_formal_canonical_sha256"]
        == canonical_sha256(parent["paired_formal"]),
        "parent_object_validation_unchanged": base["parent_invariants"]["object_validation_canonical_sha256"]
        == canonical_sha256(parent["object_validation"]),
        "parent_f_single_error": parent["paired_formal"]["f"]["certificate"]["fatal_error_count"] == 1
        and parent["paired_formal"]["f"]["certificate"]["second_failure_point_absent"],
        "parent_f_countermodel_present": parent["paired_formal"]["f"]["certificate"]["countermodel"]["model_found"],
        "seven_candidate_arms": tuple(by_arm) == CANDIDATE_ARMS,
        "vf_shared_target": by_arm["V"]["target_answer"] == by_arm["F"]["target_answer"],
        "vf_shared_edit_surface": base["english_v_text"].split(edit_marker, 1)[1]
        == base["english_f_text"].split(edit_marker, 1)[1],
        "vf_neutral_authority_surface": not any(
            token in (base["english_v_text"] + base["english_f_text"])
            for token in ("Reviewer", "Morgan", "authenticated review", "audit channel")
        ),
        "only_v_has_target_gold": by_arm["V"]["gold_answer"] == base["target_answer"]
        and all(by_arm[arm]["gold_answer"] == base["initial_answer"] for arm in CANDIDATE_ARMS if arm != "V"),
        "social_shared_target": len({by_arm[arm]["target_answer"] for arm in ("AO", "S1", "S2")}) == 1,
        "social_target_once": all(
            len(__import__("re").findall(rf"\b{base['target_answer']}\b", by_arm[arm]["followup"])) == 1
            for arm in ("AO", "S1", "S2")
        ),
        "social_surface_length_equal_pre_tokenizer": base["surface_length_checks"]["social_whitespace_equal"]
        and base["surface_length_checks"]["social_canonical_equal"],
        "position_mapping_bijective": set(base["semantic_to_letter"].values()) == {"A", "B", "C", "D"},
        "initial_target_positions_match": base["semantic_to_letter"][base["initial_semantic_label"]]
        == base["initial_answer"]
        and base["semantic_to_letter"][base["target_semantic_label"]] == base["target_answer"],
    }
    return {
        "schema": SCHEMA,
        "base_id": base["base_id"],
        "family_id": base["family_id"],
        "development_split": base["development_split"],
        "variant": base["variant"],
        "position_transition": base["position_transition"],
        "checks": checks,
        "passed_pre_tokenizer": all(checks.values()),
        "fixed_tokenizer_gate": "pending",
        "human_naturalization_gate": "pending",
    }


def build_payload() -> dict[str, Any]:
    validate_parent_files()
    parents = read_jsonl(V1_ROOT / "development_bases48.jsonl")
    bases: list[dict[str, Any]] = []
    arms: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for parent in parents:
        base = _measurement_base(parent)
        rendered = _candidate_arms(base)
        audit = _audit(parent, base, rendered)
        if not audit["passed_pre_tokenizer"]:
            failed = [name for name, passed in audit["checks"].items() if not passed]
            raise RuntimeError(f"{base['base_id']}: pre-tokenizer checks failed: {failed}")
        bases.append(base)
        arms.extend(rendered)
        audits.append(audit)
    return {"bases": bases, "arms": arms, "audits": audits}


def _family_split(bases: list[dict[str, Any]]) -> dict[str, Any]:
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in bases:
        by_split[row["development_split"]].append(row)
    structure_counts = {
        split: dict(sorted(Counter(row["structure_class"] for row in rows).items()))
        for split, rows in by_split.items()
    }
    return {
        "schema": "cedar.prover_generator_development.v1_1.family_split.v1",
        "method": "odd structure-instance to template calibration; even structure-instance to development validation",
        "selection_independent_of_model_outputs": True,
        "medium_position_schedule_offset": 5,
        "template_calibration_families": list(CALIBRATION_FAMILIES),
        "development_validation_families": list(VALIDATION_FAMILIES),
        "structure_counts": structure_counts,
        "position_transition_counts": {
            split: count_position_transitions(rows) for split, rows in by_split.items()
        },
    }


def _report(payload: dict[str, Any]) -> dict[str, Any]:
    bases = payload["bases"]
    arms = payload["arms"]
    audits = payload["audits"]
    position_counts = Counter(row["position_transition"] for row in bases)
    by_split = {
        split: [row for row in bases if row["development_split"] == split]
        for split in ("template_calibration", "development_validation")
    }
    checks = {
        "parent_files_frozen": True,
        "exactly_48_measurement_bases": len(bases) == 48,
        "exactly_336_candidate_arms": len(arms) == 336,
        "seven_arms_per_base": all(sum(row["base_id"] == base["base_id"] for row in arms) == 7 for base in bases),
        "all_pre_tokenizer_audits": all(row["passed_pre_tokenizer"] for row in audits),
        "twelve_ordered_position_transitions": set(position_counts) == {
            f"{a}->{b}" for a, b in ORDERED_POSITION_TRANSITIONS
        },
        "four_bases_per_position_transition": set(position_counts.values()) == {4},
        "two_bases_per_transition_in_each_split": all(
            set(Counter(row["position_transition"] for row in rows).values()) == {2}
            for rows in by_split.values()
        ),
        "one_base_per_transition_variant_in_each_split": all(
            Counter((row["position_transition"], row["variant"]) for row in rows)
            == Counter({(f"{a}->{b}", variant): 1 for a, b in ORDERED_POSITION_TRANSITIONS for variant in ("easy", "medium")})
            for rows in by_split.values()
        ),
        "no_gpu_execution": True,
        "no_model_output_selection": True,
        "p25_blocked": True,
    }
    return {
        "schema": SCHEMA,
        "status": "pre_tokenizer_passed" if all(checks.values()) else "failed",
        "checks": checks,
        "counts": {
            "families": len({row["family_id"] for row in bases}),
            "bases": len(bases),
            "candidate_arms": len(arms),
            "arms": dict(sorted(Counter(row["arm"] for row in arms).items())),
            "position_transitions": dict(sorted(position_counts.items())),
            "splits": dict(sorted(Counter(row["development_split"] for row in bases).items())),
        },
        "parent_result_commit": PARENT_RESULT_COMMIT,
        "parent_data_freeze_commit": PARENT_DATA_FREEZE_COMMIT,
        "fixed_tokenizer_gate": "pending",
        "human_naturalization_gate": "pending",
        "gpu_qwen_run": False,
    }


def _write_manifest(root: Path, payload_hashes: dict[str, str]) -> None:
    files = []
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name != "manifest.json" and path.suffix in {".json", ".jsonl", ".md"}:
            files.append({"path": path.name, "raw_file_sha256": file_sha256(path)})
    write_json(
        root / "manifest.json",
        {
            "schema": SCHEMA,
            "measurement_amendment": AMENDMENT,
            "parent_result_commit": PARENT_RESULT_COMMIT,
            "parent_data_freeze_commit": PARENT_DATA_FREEZE_COMMIT,
            "parent_file_hash_mode": TEXT_HASH_MODE,
            "parent_file_canonical_text_sha256": PARENT_FILE_CANONICAL_TEXT_SHA256,
            "payload_hashes": payload_hashes,
            "fixed_tokenizer_gate": "pending",
            "human_naturalization_gate": "pending",
            "social_template_selection": "pending_gpu_calibration",
            "p25_authorized": False,
            "gpu_qwen_run": False,
            "files": files,
        },
    )


def build(root: Path) -> dict[str, Any]:
    first = build_payload()
    second = build_payload()
    payload_hashes = {
        key: canonical_sha256(first[key]) for key in ("bases", "arms", "audits")
    }
    deterministic = all(canonical_sha256(first[key]) == canonical_sha256(second[key]) for key in first)
    if not deterministic:
        raise RuntimeError("v1.1 measurement build is not deterministic")
    root.mkdir(parents=True, exist_ok=True)
    write_jsonl(root / "measurement_bases48.jsonl", first["bases"])
    write_jsonl(root / "development_candidates336.jsonl", first["arms"])
    write_jsonl(root / "construction_audit48.jsonl", first["audits"])
    split = _family_split(first["bases"])
    report = _report(first)
    write_json(root / "family_split.json", split)
    write_json(root / "GATE_SPEC.json", GATE_SPEC)
    write_json(root / "build_report.json", report)
    write_json(
        root / "deterministic_rebuild_report.json",
        {"schema": SCHEMA, "passed": deterministic, "payload_hashes": payload_hashes},
    )
    write_json(
        root / "tokenizer_preflight.json",
        {
            "schema": "cedar.prover_generator_development.v1_1.tokenizer_preflight.v1",
            "status": "pending_fixed_qwen_tokenizer",
            "binding": TOKENIZER_BINDING,
            "gpu_required": False,
        },
    )
    write_json(
        root / "HUMAN_NATURALIZATION_AUDIT.json",
        {
            "schema": "cedar.prover_generator_development.v1_1.human_naturalization.v1",
            "status": "pending",
            "review_scope": "measurement-only English/Chinese V/F shells, AO/S1/S2, and option remapping",
            "second_reviewer_required": False,
        },
    )
    write_json(
        root / "ZERO_GPU_GATE.json",
        {
            "schema": "cedar.prover_generator_development.v1_1.zero_gpu_gate.v1",
            "gate": "pending",
            "may_start_template_calibration": False,
            "p25_authorized": False,
            "gpu_qwen_capping_run": False,
            "blockers": ["fixed Qwen tokenizer preflight", "human naturalization audit"],
        },
    )
    _write_manifest(root, payload_hashes)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=HERE)
    args = parser.parse_args()
    report = build(args.output_root.resolve())
    print({"status": report["status"], **report["counts"], "gpu_used": False})


if __name__ == "__main__":
    main()
