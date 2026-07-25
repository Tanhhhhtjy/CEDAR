#!/usr/bin/env python3
"""Build the frozen 24-base E2 new-salt bank without model execution."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from atomic_edit_candidate_enumerator import (  # type: ignore  # noqa: E402
    enumerate_atomic_edit_candidates,
    evaluate_candidate_neighborhood,
)
from e2_new_salt_common import (
    ANALYSIS_SCHEMA,
    ARMS,
    ERROR_CLASSES,
    ERROR_POSITIONS,
    FAMILY_IDS,
    HERE,
    MASTER_SEED,
    N_TEMPLATE,
    OPTION_LETTERS,
    R_TEMPLATE,
    RESPONSE_CONTRACT,
    ROOT,
    SALT,
    SALT_SHA256,
    SCHEMA,
    TOKENIZER_BINDING,
    VARIANTS,
    arm_row,
    canonical_json,
    canonical_jsonl_sha256,
    canonical_sha256,
    family_plan,
    four_option_prompt,
    neutralize_vf,
    normalized_text,
    object_sha256,
    positioned_formal_pair,
    positioned_render_pair,
    require,
    semantic_mapping,
    stable_seed,
    write_json,
    write_jsonl,
)
from proof_graph_builder import (  # type: ignore  # noqa: E402
    assert_no_answer_channel,
    blueprint_by_id,
    proof_graph_builder,
)
from semantic_validator import semantic_validator  # type: ignore  # noqa: E402

OUTPUT_FILES = (
    "family_plan12.json",
    "symbolic_sources24.jsonl",
    "bases24.jsonl",
    "arms96.jsonl",
    "construction_audit24.jsonl",
    "construction_failures.jsonl",
    "duplicate_audit.json",
    "automatic_validation_report.json",
)


def _write_object_evidence(
    output_root: Path, source_id: str, validation: dict[str, Any]
) -> None:
    for state in ("initial_validation", "updated_validation"):
        for polarity in ("query_result", "opposite_result"):
            write_json(
                output_root / "solver_traces" / f"{source_id}__object_{state}__{polarity}.json",
                validation[state]["prover9"][polarity],
            )
    certificate = validation["f_certificate"]
    write_json(
        output_root / "countermodels" / f"{source_id}__object_counterevidence.json",
        certificate["countermodel_or_opposite_proof"],
    )
    write_json(
        output_root / "f_certificates" / f"{source_id}__object_f_certificate.json",
        certificate,
    )


def _write_meta_evidence(output_root: Path, base_id: str, formal: dict[str, Any]) -> None:
    for arm in ("v", "f"):
        for polarity in ("query_result", "opposite_result"):
            write_json(
                output_root / "solver_traces" / f"{base_id}__meta_{arm}__{polarity}.json",
                formal[arm]["solver"][polarity],
            )
    write_json(
        output_root / "countermodels" / f"{base_id}__meta_f_countermodel.json",
        formal["f"]["certificate"]["countermodel"],
    )
    write_json(
        output_root / "f_certificates" / f"{base_id}__meta_f_certificate.json",
        formal["f"]["certificate"],
    )


def _construct_source(
    plan: dict[str, Any], variant: str, output_root: Path
) -> dict[str, Any]:
    family_id = plan["family_id"]
    structure_class = plan["structure_class"]
    blueprint = blueprint_by_id(structure_class)
    seed = stable_seed(MASTER_SEED, SALT, family_id, variant)
    source_id = f"SRC-{family_id}-{variant[0].upper()}"
    built = proof_graph_builder(
        blueprint,
        variant,
        seed,
        blueprint["transition"],
        plan["error_class"],
    )
    assert_no_answer_channel(built)
    built["confirmation_family_id"] = family_id
    built["construction_salt_sha256"] = SALT_SHA256
    candidates = enumerate_atomic_edit_candidates(built)
    neighborhood = evaluate_candidate_neighborhood(built, candidates)
    validation = semantic_validator(built, neighborhood)
    validation.setdefault("checks", {})["assigned_f_class_matches_structure"] = (
        validation.get("f_validation", {}).get("derived_error_class")
        == plan["error_class"]
    )
    validation["passed"] = bool(
        validation.get("passed")
        and validation["checks"]["assigned_f_class_matches_structure"]
    )
    source = {
        "schema": SCHEMA,
        "architecture": "proof_graph_plus_atomic_edit_certification",
        "attempt_id": source_id,
        "family_id": family_id,
        "structure_class": structure_class,
        "variant": variant,
        "seed": seed,
        "construction_salt": SALT,
        "assigned_f_error_class": plan["error_class"],
        "status": "valid" if validation["passed"] else "failed",
        "builder_output": built,
        "validation": validation,
    }
    _write_object_evidence(output_root, source_id, validation)
    return source


def _render_base(
    source: dict[str, Any], plan: dict[str, Any], base_index: int, output_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    variant = source["variant"]
    variant_plan = plan["variants"][variant]
    validation = source["validation"]
    initial_label = validation["initial_validation"]["python_label"]
    target_label = validation["updated_validation"]["python_label"]
    initial_answer = variant_plan["initial_answer"]
    target_answer = variant_plan["target_answer"]
    mapping = semantic_mapping(initial_label, target_label, initial_answer, target_answer)
    error_position = variant_plan["error_position"]
    render_index = 701 + base_index
    edit_id = f"edit{render_index}"
    formal = positioned_formal_pair(
        source["assigned_f_error_class"], variant, edit_id, error_position
    )
    pair = positioned_render_pair(source, render_index, formal, error_position)
    original_question_en = pair["english_base_prompt"].rsplit("\n\n", 1)[1]
    original_question_zh = pair["chinese_base_prompt"].rsplit("\n\n", 1)[1]
    english_base, english_question = four_option_prompt(
        pair["english_base_prompt"], original_question_en, mapping, "en"
    )
    chinese_base, chinese_question = four_option_prompt(
        pair["chinese_base_prompt"], original_question_zh, mapping, "zh"
    )
    english_v = neutralize_vf(pair["english_v_text"], "en")
    english_f = neutralize_vf(pair["english_f_text"], "en")
    chinese_v = neutralize_vf(pair["chinese_v_text"], "zh")
    chinese_f = neutralize_vf(pair["chinese_f_text"], "zh")
    base_id = f"CEDAR-E2C-{base_index + 1:03d}"
    base = {
        "schema": SCHEMA,
        "base_id": base_id,
        "family_id": source["family_id"],
        "structure_class": source["structure_class"],
        "variant": variant,
        "object_transition": source["builder_output"]["desired_transition"],
        "position_transition": f"{initial_answer}->{target_answer}",
        "seed": source["seed"],
        "construction_salt_sha256": SALT_SHA256,
        "source_attempt_id": source["attempt_id"],
        "source_symbolic_sha256": canonical_sha256(source),
        "semantic_to_letter": mapping,
        "initial_semantic_label": initial_label,
        "target_semantic_label": target_label,
        "initial_answer": initial_answer,
        "target_answer": target_answer,
        "candidate_labels": list(OPTION_LETTERS),
        "error_class": source["assigned_f_error_class"],
        "error_position": error_position,
        "english_base_prompt": english_base,
        "chinese_base_prompt": chinese_base,
        "target_question_en": english_question,
        "target_question_zh": chinese_question,
        "english_v_text": english_v,
        "chinese_v_text": chinese_v,
        "english_f_text": english_f,
        "chinese_f_text": chinese_f,
        "selected_edit": copy.deepcopy(validation["selected_edit"]),
        "object_validation": copy.deepcopy(validation),
        "paired_formal": copy.deepcopy(formal),
        "difficulty": copy.deepcopy(validation["difficulty"]),
        "pair_contract": {
            "reviewer": pair["reviewer"],
            "source_type": pair["source_type"],
            "authentication_status": pair["authentication_status"],
            "speech_act": pair["speech_act"],
            "slot_skeleton": pair["slot_skeleton"],
            "slot_skeleton_sha256": pair["slot_skeleton_sha256"],
            "proposed_edit_sha256": pair["proposed_edit_sha256"],
            "output_contract": RESPONSE_CONTRACT,
            "visible_measurement_shell": "neutralized_evidence_v1_1",
        },
        "analysis_schema": copy.deepcopy(ANALYSIS_SCHEMA),
    }
    followups = {"N": N_TEMPLATE, "R": R_TEMPLATE, "V": english_v, "F": english_f}
    arms = [arm_row(base, arm, followups[arm]) for arm in ARMS]
    _write_meta_evidence(output_root, base_id, formal)
    audit = _audit_base(base, source, pair, arms)
    return base, arms, audit


def _audit_base(
    base: dict[str, Any],
    source: dict[str, Any],
    pair: dict[str, Any],
    arms: list[dict[str, Any]],
) -> dict[str, Any]:
    validation = source["validation"]
    formal = base["paired_formal"]
    by_arm = {row["arm"]: row for row in arms}
    pair_contract = base["pair_contract"]
    checks = {
        "source_valid": source["status"] == "valid" and validation["passed"],
        "initial_target_noninvalid": base["initial_semantic_label"]
        in {"true", "false", "unknown"}
        and base["target_semantic_label"] in {"true", "false", "unknown"},
        "semantic_transition_changes": base["initial_semantic_label"]
        != base["target_semantic_label"],
        "position_transition_changes": base["initial_answer"] != base["target_answer"],
        "unique_atomic_v": validation["success_count"] == 1
        and validation["checks"]["atomic_edit"],
        "object_f_single_error": validation["checks"]["f_semantic_validation"]
        and validation["f_certificate"]["fatal_error_count"] == 1
        and validation["f_certificate"]["second_failure_point_absent"],
        "meta_v_unique_proof": formal["v"]["agreement"]
        and formal["v"]["python_label"] == "true"
        and formal["v"]["proof_count"] == 1,
        "meta_f_nonentailment": formal["f"]["agreement"]
        and formal["f"]["python_label"] == "unknown",
        "meta_f_countermodel": formal["f"]["certificate"]["countermodel"]["model_found"],
        "meta_f_exactly_one_error": formal["f"]["certificate"]["fatal_error_count"] == 1
        and formal["f"]["certificate"]["second_failure_point_absent"],
        "pre_error_steps_entailed": formal["f"]["certificate"].get(
            "pre_error_steps_entailed", True
        ),
        "four_arms": set(by_arm) == set(ARMS),
        "v_only_target_gold": by_arm["V"]["gold_answer"] == base["target_answer"]
        and all(
            by_arm[arm]["gold_answer"] == base["initial_answer"]
            for arm in ("N", "R", "F")
        ),
        "vf_shared_target": by_arm["V"]["target_answer"]
        == by_arm["F"]["target_answer"],
        "vf_shared_edit": pair_contract["proposed_edit_sha256"]
        == canonical_sha256(validation["selected_edit"]),
        "vf_shared_reviewer_source_speech_act": all(
            pair_contract[key] == pair[key]
            for key in ("reviewer", "source_type", "authentication_status", "speech_act")
        ),
        "vf_shared_slot_skeleton": pair_contract["slot_skeleton_sha256"]
        == canonical_sha256(pair_contract["slot_skeleton"]),
        "vf_shared_output_contract": all(
            by_arm[arm]["response_contract"] == RESPONSE_CONTRACT for arm in ("V", "F")
        ),
        "n_r_no_target_exposure": all(
            f"option {base['target_answer'].lower()}" not in by_arm[arm]["followup"].lower()
            for arm in ("N", "R")
        ),
        "option_map_bijective": set(base["semantic_to_letter"].values())
        == set(OPTION_LETTERS),
        "all_prompt_hashes_present": all(row["prompt_canonical_sha256"] for row in arms),
    }
    return {
        "schema": SCHEMA,
        "base_id": base["base_id"],
        "family_id": base["family_id"],
        "variant": base["variant"],
        "position_transition": base["position_transition"],
        "error_class": base["error_class"],
        "error_position": base["error_position"],
        "checks": checks,
        "passed_pre_tokenizer": all(checks.values()),
    }


SURFACE_KEYS = (
    "english_base_prompt",
    "chinese_base_prompt",
    "english_v_text",
    "chinese_v_text",
    "english_f_text",
    "chinese_f_text",
)


def _history_corpus(output_root: Path) -> dict[str, Any]:
    exact: dict[str, list[str]] = {}
    normalized: dict[str, list[str]] = {}
    base_ids: set[str] = set()
    selected_edit_hashes: set[str] = set()
    world_hashes: set[str] = set()
    files_scanned: list[str] = []
    for path in sorted((ROOT / "datasets" / "m2_2b_formal").rglob("*.jsonl")):
        if output_root in path.parents or HERE in path.parents:
            continue
        files_scanned.append(path.relative_to(ROOT).as_posix())
        try:
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        for row_index, row in enumerate(rows, start=1):
            if not isinstance(row, dict):
                continue
            location = f"{path.relative_to(ROOT).as_posix()}:{row_index}"
            if isinstance(row.get("base_id"), str):
                base_ids.add(row["base_id"])
            for key in SURFACE_KEYS:
                value = row.get(key)
                if isinstance(value, str):
                    exact.setdefault(value, []).append(f"{location}:{key}")
                    normalized.setdefault(normalized_text(value), []).append(
                        f"{location}:{key}"
                    )
            messages = row.get("messages")
            if isinstance(messages, list) and all(isinstance(item, dict) for item in messages):
                joined = "\n".join(str(item.get("content", "")) for item in messages)
                exact.setdefault(joined, []).append(f"{location}:messages")
                normalized.setdefault(normalized_text(joined), []).append(
                    f"{location}:messages"
                )
            edit = row.get("selected_edit")
            if isinstance(edit, dict):
                selected_edit_hashes.add(canonical_sha256(edit))
            for candidate in (row.get("builder_output"), row.get("initial_world")):
                if isinstance(candidate, dict):
                    world = candidate.get("initial_world", candidate)
                    if isinstance(world, dict) and "records" in world:
                        world_hashes.add(canonical_sha256(world))
    return {
        "exact": exact,
        "normalized": normalized,
        "base_ids": base_ids,
        "selected_edit_hashes": selected_edit_hashes,
        "world_hashes": world_hashes,
        "files_scanned": files_scanned,
    }


def duplicate_audit(
    output_root: Path,
    bases: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    arms: list[dict[str, Any]],
) -> dict[str, Any]:
    corpus = _history_corpus(output_root)
    exact_matches = []
    normalized_matches = []
    for base in bases:
        for key in SURFACE_KEYS:
            value = base[key]
            if value in corpus["exact"]:
                exact_matches.append(
                    {"base_id": base["base_id"], "field": key, "matches": corpus["exact"][value]}
                )
            normalized_value = normalized_text(value)
            if normalized_value in corpus["normalized"]:
                normalized_matches.append(
                    {
                        "base_id": base["base_id"],
                        "field": key,
                        "matches": corpus["normalized"][normalized_value],
                    }
                )
    for arm in arms:
        joined = "\n".join(item["content"] for item in arm["messages"])
        if joined in corpus["exact"]:
            exact_matches.append(
                {"id": arm["id"], "field": "messages", "matches": corpus["exact"][joined]}
            )
        normalized_joined = normalized_text(joined)
        if normalized_joined in corpus["normalized"]:
            normalized_matches.append(
                {
                    "id": arm["id"],
                    "field": "messages",
                    "matches": corpus["normalized"][normalized_joined],
                }
            )
    id_collisions = sorted({base["base_id"] for base in bases} & corpus["base_ids"])
    edit_collisions = sorted(
        canonical_sha256(base["selected_edit"])
        for base in bases
        if canonical_sha256(base["selected_edit"]) in corpus["selected_edit_hashes"]
    )
    world_collisions = sorted(
        canonical_sha256(source["builder_output"]["initial_world"])
        for source in sources
        if canonical_sha256(source["builder_output"]["initial_world"])
        in corpus["world_hashes"]
    )
    checks = {
        "no_exact_surface_duplicate": not exact_matches,
        "no_normalized_surface_duplicate": not normalized_matches,
        "no_base_id_reuse": not id_collisions,
        "no_selected_edit_reuse": not edit_collisions,
        "no_symbolic_world_reuse": not world_collisions,
        "model_outputs_not_scanned": True,
    }
    return {
        "schema": f"{SCHEMA}.duplicate_audit",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "history_scope": "datasets/m2_2b_formal/**/*.jsonl excluding this bank",
        "history_files_scanned": len(corpus["files_scanned"]),
        "history_file_paths_sha256": object_sha256(corpus["files_scanned"]),
        "history_surface_strings": len(corpus["exact"]),
        "exact_matches": exact_matches,
        "normalized_matches": normalized_matches,
        "base_id_collisions": id_collisions,
        "selected_edit_collisions": edit_collisions,
        "symbolic_world_collisions": world_collisions,
        "qwen_outputs_used": False,
    }


def _report(
    plan: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    bases: list[dict[str, Any]],
    arms: list[dict[str, Any]],
    audits: list[dict[str, Any]],
    duplicates: dict[str, Any],
) -> dict[str, Any]:
    classes = Counter(base["error_class"] for base in bases)
    variants = Counter(base["variant"] for base in bases)
    positions = Counter(base["error_position"] for base in bases)
    per_variant_transitions = {
        variant: Counter(
            base["position_transition"] for base in bases if base["variant"] == variant
        )
        for variant in VARIANTS
    }
    family_variants = Counter((base["family_id"], base["variant"]) for base in bases)
    checks = {
        "salt_hash": object_sha256(SALT) == SALT_SHA256,
        "families_12": len(plan) == len(FAMILY_IDS) == 12,
        "sources_24": len(sources) == 24,
        "bases_24": len(bases) == 24,
        "arms_96": len(arms) == 96,
        "family_variants_complete": family_variants
        == Counter({(family, variant): 1 for family in FAMILY_IDS for variant in VARIANTS}),
        "four_arms_per_base": all(
            sum(row["base_id"] == base["base_id"] for row in arms) == 4 for base in bases
        ),
        "six_f_classes_two_families_each": Counter(
            item["error_class"] for item in plan
        )
        == Counter({name: 2 for name in ERROR_CLASSES}),
        "class_count_four_bases_each": classes == Counter({name: 4 for name in ERROR_CLASSES}),
        "variant_balance": variants == Counter({"easy": 12, "medium": 12}),
        "each_variant_all_12_transitions_once": all(
            len(counter) == 12 and set(counter.values()) == {1}
            for counter in per_variant_transitions.values()
        ),
        "error_positions_balanced": positions == Counter({name: 8 for name in ERROR_POSITIONS}),
        "all_sources_valid": all(source["status"] == "valid" for source in sources),
        "all_pre_tokenizer_audits": all(row["passed_pre_tokenizer"] for row in audits),
        "duplicates_passed": duplicates["status"] == "passed",
        "new_symbolic_hashes_unique": len(
            {canonical_sha256(source["builder_output"]["initial_world"]) for source in sources}
        )
        == 24,
        "new_selected_edits_unique": len(
            {canonical_sha256(base["selected_edit"]) for base in bases}
        )
        == 24,
        "no_qwen_screening": True,
        "zero_gpu_construction": True,
    }
    return {
        "schema": f"{SCHEMA}.automatic_validation",
        "status": "passed" if all(checks.values()) else "failed",
        "checks": checks,
        "counts": {
            "families": len(plan),
            "sources": len(sources),
            "bases": len(bases),
            "arms": len(arms),
            "f_classes": dict(sorted(classes.items())),
            "variants": dict(sorted(variants.items())),
            "error_positions": dict(sorted(positions.items())),
            "transitions_by_variant": {
                variant: dict(sorted(counter.items()))
                for variant, counter in per_variant_transitions.items()
            },
        },
        "tokenizer_gate": "pending",
        "human_review_status": "pending_workbook",
        "qwen_outputs_used": False,
        "gpu_used": False,
    }


def build_dataset(output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    for name in ("solver_traces", "countermodels", "f_certificates"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    plan = family_plan()
    sources: list[dict[str, Any]] = []
    bases: list[dict[str, Any]] = []
    arms: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    base_index = 0
    for family in plan:
        for variant in VARIANTS:
            try:
                source = _construct_source(family, variant, output_root)
                sources.append(source)
                require(source["status"] == "valid", f"{source['attempt_id']}: source invalid")
                base, rendered_arms, audit = _render_base(
                    source, family, base_index, output_root
                )
                require(
                    audit["passed_pre_tokenizer"],
                    f"{base['base_id']}: pre-tokenizer audit failed",
                )
                bases.append(base)
                arms.extend(rendered_arms)
                audits.append(audit)
                base_index += 1
            except Exception as error:
                failures.append(
                    {
                        "family_id": family["family_id"],
                        "variant": variant,
                        "reason_class": type(error).__name__,
                        "reason": str(error),
                        "allowed_rejection_category": "symbolic_or_pairing_gate",
                        "model_behavior_used": False,
                    }
                )
                write_jsonl(output_root / "construction_failures.jsonl", failures)
                raise
    duplicates = duplicate_audit(output_root, bases, sources, arms)
    report = _report(plan, sources, bases, arms, audits, duplicates)
    failed_checks = [name for name, passed in report["checks"].items() if not passed]
    require(
        report["status"] == "passed",
        f"Automatic construction Gate failed: {failed_checks}",
    )
    write_json(output_root / "family_plan12.json", {"schema": SCHEMA, "families": plan})
    write_jsonl(output_root / "symbolic_sources24.jsonl", sources)
    write_jsonl(output_root / "bases24.jsonl", bases)
    write_jsonl(output_root / "arms96.jsonl", arms)
    write_jsonl(output_root / "construction_audit24.jsonl", audits)
    write_jsonl(output_root / "construction_failures.jsonl", failures)
    write_json(output_root / "duplicate_audit.json", duplicates)
    write_json(output_root / "automatic_validation_report.json", report)
    write_json(
        output_root / "tokenizer_preflight.json",
        {
            "schema": f"{SCHEMA}.tokenizer_preflight",
            "status": "pending_fixed_qwen_tokenizer",
            "binding": TOKENIZER_BINDING,
            "gpu_used": False,
        },
    )
    hashes = {
        name: canonical_jsonl_sha256(output_root / name)
        if name.endswith(".jsonl")
        else canonical_sha256(json.loads((output_root / name).read_text(encoding="utf-8")))
        for name in OUTPUT_FILES
    }
    return {
        "plan": plan,
        "sources": sources,
        "bases": bases,
        "arms": arms,
        "audits": audits,
        "failures": failures,
        "duplicates": duplicates,
        "report": report,
        "hashes": hashes,
    }


def verify_determinism(output_root: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
        first = build_dataset(Path(first_dir))
        second = build_dataset(Path(second_dir))
    compared = {
        key: first["hashes"][key] == second["hashes"][key]
        for key in sorted(first["hashes"])
    }
    result = {
        "schema": f"{SCHEMA}.deterministic_rebuild",
        "status": "passed" if all(compared.values()) else "failed",
        "passed": all(compared.values()),
        "two_clean_builds": True,
        "compared": compared,
        "first": first["hashes"],
        "second": second["hashes"],
        "gpu_used": False,
    }
    write_json(output_root / "deterministic_rebuild_report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=HERE)
    parser.add_argument("--skip-determinism", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    result = build_dataset(output_root)
    determinism = (
        {"status": "skipped", "passed": False}
        if args.skip_determinism
        else verify_determinism(output_root)
    )
    print(
        canonical_json(
            {
                "status": result["report"]["status"],
                "families": len(result["plan"]),
                "bases": len(result["bases"]),
                "arms": len(result["arms"]),
                "deterministic": determinism["passed"],
                "human_review_status": "not_started",
                "qwen_outputs_used": False,
                "gpu_used": False,
            }
        )
    )


if __name__ == "__main__":
    main()
