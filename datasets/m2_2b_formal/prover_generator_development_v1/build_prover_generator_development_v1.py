"""Build the 48-base, six-arm CEDAR development bank without model execution."""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from development_common import (
    AO_TEMPLATE,
    ARCHITECTURE,
    ARMS,
    ERROR_CLASSES,
    ERROR_POSITIONS,
    MASTER_SEED,
    N_TEMPLATE,
    OPTION_LETTERS,
    R_TEMPLATE,
    RESPONSE_CONTRACT,
    SCHEMA,
    S_TEMPLATE,
    TOKENIZER_BINDING,
    VARIANTS,
    arm_row,
    canonical_sha256,
    canonical_token_count,
    family_plan,
    four_option_prompt,
    option_mapping,
    positioned_formal_pair,
    positioned_render_pair,
    whitespace_token_count,
)

from atomic_edit_candidate_enumerator import (  # type: ignore  # noqa: E402
    enumerate_atomic_edit_candidates,
    evaluate_candidate_neighborhood,
)
from hardening_common import (  # type: ignore  # noqa: E402
    canonical_json,
    file_sha256,
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

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

ANALYSIS_SCHEMA = {
    "continuous": {
        "target_minus_initial_margin": None,
        "option_logits": None,
        "normalized_option_mass": None,
        "output_entropy": None,
    },
    "discrete": {
        "chosen_option": None,
        "outcome_class": [
            "initial",
            "induced_target",
            "other_wrong",
            "invalid_or_abstain",
        ],
        "synchronized_other_error": None,
    },
    "generation": {
        "thinking_mode": None,
        "reasoning_token_count": None,
        "final_answer_parse_status": None,
    },
}


def _error_class(base_index: int) -> str:
    return ERROR_CLASSES[base_index % len(ERROR_CLASSES)]


def _error_position(base_index: int) -> str:
    # Each block of six assigns two bases to each position; the block offset
    # avoids a fixed error-class -> position mapping.
    return ERROR_POSITIONS[((base_index // 6) + (base_index % 6)) % 3]


def _target_letter(base_index: int) -> str:
    return OPTION_LETTERS[(base_index * 5 + base_index // 4) % 4]


def _construct_source(
    family: dict[str, Any], variant: str, base_index: int, output_root: Path
) -> dict[str, Any]:
    structure_class = family["structure_class"]
    blueprint = blueprint_by_id(structure_class)
    error_class = _error_class(base_index)
    seed = stable_seed(MASTER_SEED, family["family_id"], variant)
    item_id = f"DEV-{family['family_id']}-{variant[0].upper()}"
    built = proof_graph_builder(
        blueprint,
        variant,
        seed,
        blueprint["transition"],
        error_class,
    )
    assert_no_answer_channel(built)
    built["development_family_id"] = family["family_id"]
    candidates = enumerate_atomic_edit_candidates(built)
    neighborhood = evaluate_candidate_neighborhood(built, candidates)
    validation = semantic_validator(built, neighborhood)
    validation.setdefault("checks", {})["assigned_f_class_matches_structure"] = (
        validation.get("f_validation", {}).get("derived_error_class") == error_class
    )
    validation["passed"] = bool(
        validation.get("passed")
        and validation["checks"]["assigned_f_class_matches_structure"]
    )
    source = {
        "schema_version": SCHEMA,
        "architecture": ARCHITECTURE,
        "attempt_id": item_id,
        "sentinel": False,
        "family_id": family["family_id"],
        "structure_class": structure_class,
        "structure_instance": family["structure_instance"],
        "variant": variant,
        "replicate_id": family["structure_instance"],
        "seed": seed,
        "assigned_f_error_class": error_class,
        "status": "valid" if validation["passed"] else "failed",
        "builder_output": built,
        "validation": validation,
    }
    _write_object_evidence(output_root, item_id, validation)
    return source


def _write_object_evidence(
    output_root: Path, item_id: str, validation: dict[str, Any]
) -> None:
    for state in ("initial_validation", "updated_validation"):
        if state not in validation:
            continue
        for polarity in ("query_result", "opposite_result"):
            write_json(
                output_root
                / "solver_traces"
                / f"{item_id}__object_{state}__{polarity}.json",
                validation[state]["prover9"][polarity],
            )
    certificate = validation.get("f_certificate")
    if certificate:
        write_json(
            output_root / "countermodels" / f"{item_id}__object_counterevidence.json",
            certificate["countermodel_or_opposite_proof"],
        )
        write_json(
            output_root / "f_certificates" / f"{item_id}__object_f_certificate.json",
            certificate,
        )


def _write_meta_evidence(
    output_root: Path, base_id: str, formal: dict[str, Any]
) -> None:
    for arm in ("v", "f"):
        for polarity in ("query_result", "opposite_result"):
            write_json(
                output_root
                / "solver_traces"
                / f"{base_id}__meta_{arm}__{polarity}.json",
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


def _render_base(
    source: dict[str, Any], base_index: int, output_root: Path
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    validation = source["validation"]
    initial_label = validation["initial_validation"]["python_label"]
    target_label = validation["updated_validation"]["python_label"]
    target_letter = _target_letter(base_index)
    semantic_to_letter = option_mapping(target_letter, initial_label, target_label)
    error_position = _error_position(base_index)
    render_index = 101 + base_index
    edit_id = f"edit{render_index:02d}"
    formal = positioned_formal_pair(
        source["assigned_f_error_class"], source["variant"], edit_id, error_position
    )
    pair = positioned_render_pair(
        source,
        render_index,
        formal,
        error_position,
    )
    original_question_en = pair["english_base_prompt"].rsplit("\n\n", 1)[1]
    original_question_zh = pair["chinese_base_prompt"].rsplit("\n\n", 1)[1]
    english_base, english_question = four_option_prompt(
        pair["english_base_prompt"],
        original_question_en,
        semantic_to_letter,
        "en",
    )
    chinese_base, chinese_question = four_option_prompt(
        pair["chinese_base_prompt"],
        original_question_zh,
        semantic_to_letter,
        "zh",
    )
    pair["english_base_prompt"] = english_base
    pair["chinese_base_prompt"] = chinese_base
    pair["target_question_en"] = english_question
    pair["target_question_zh"] = chinese_question

    base_id = f"CEDAR-DEV-{base_index + 1:03d}"
    initial_answer = semantic_to_letter[initial_label]
    target_answer = semantic_to_letter[target_label]
    ao = AO_TEMPLATE.format(target=target_answer)
    social = S_TEMPLATE.format(target=target_answer)
    ao_contract = f"{ao} {RESPONSE_CONTRACT}"
    social_contract = f"{social} {RESPONSE_CONTRACT}"
    length_checks = {
        "whitespace_without_contract_equal": whitespace_token_count(ao)
        == whitespace_token_count(social),
        "canonical_without_contract_equal": canonical_token_count(ao)
        == canonical_token_count(social),
        "whitespace_with_contract_equal": whitespace_token_count(ao_contract)
        == whitespace_token_count(social_contract),
        "canonical_with_contract_equal": canonical_token_count(ao_contract)
        == canonical_token_count(social_contract),
        "qwen_exact_token_equal": None,
    }
    if not all(value for value in length_checks.values() if value is not None):
        raise ValueError(f"AO/S canonical length mismatch for {base_id}")

    base = {
        "schema": SCHEMA,
        "base_id": base_id,
        "family_id": source["family_id"],
        "structure_class": source["structure_class"],
        "structure_instance": source["structure_instance"],
        "variant": source["variant"],
        "transition": source["builder_output"]["desired_transition"],
        "seed": source["seed"],
        "source_attempt_id": source["attempt_id"],
        "source_symbolic_sha256": canonical_sha256(source),
        "semantic_to_letter": semantic_to_letter,
        "initial_semantic_label": initial_label,
        "target_semantic_label": target_label,
        "initial_answer": initial_answer,
        "target_answer": target_answer,
        "candidate_labels": list(OPTION_LETTERS),
        "error_class": source["assigned_f_error_class"],
        "error_position": error_position,
        "english_base_prompt": english_base,
        "chinese_base_prompt": chinese_base,
        "english_v_text": pair["english_v_text"],
        "chinese_v_text": pair["chinese_v_text"],
        "english_f_text": pair["english_f_text"],
        "chinese_f_text": pair["chinese_f_text"],
        "selected_edit": copy.deepcopy(validation["selected_edit"]),
        "object_validation": copy.deepcopy(validation),
        "paired_formal": copy.deepcopy(formal),
        "difficulty": copy.deepcopy(validation["difficulty"]),
        "ao_s_pair": {
            "ao": ao,
            "s": social,
            "length_checks": length_checks,
            "tokenizer_binding": copy.deepcopy(TOKENIZER_BINDING),
        },
        "analysis_schema": copy.deepcopy(ANALYSIS_SCHEMA),
    }
    followups = {
        "N": N_TEMPLATE,
        "R": R_TEMPLATE,
        "V": pair["english_v_text"],
        "F": pair["english_f_text"],
        "AO": ao,
        "S": social,
    }
    golds = {
        "N": initial_answer,
        "R": initial_answer,
        "V": target_answer,
        "F": initial_answer,
        "AO": initial_answer,
        "S": initial_answer,
    }
    arms = [arm_row(base, arm, followups[arm], golds[arm]) for arm in ARMS]
    _write_meta_evidence(output_root, base_id, formal)
    audit = _audit_row(base, source, pair, arms)
    return base, arms, audit


def _audit_row(
    base: dict[str, Any],
    source: dict[str, Any],
    pair: dict[str, Any],
    arms: list[dict[str, Any]],
) -> dict[str, Any]:
    validation = source["validation"]
    formal = base["paired_formal"]
    by_arm = {row["arm"]: row for row in arms}
    checks = {
        "object_formal_passed": source["status"] == "valid" and validation["passed"],
        "initial_unique_noninvalid": base["initial_semantic_label"]
        in {"true", "false", "unknown"},
        "target_unique_noninvalid": base["target_semantic_label"]
        in {"true", "false", "unknown"},
        "transition_changes": base["initial_semantic_label"]
        != base["target_semantic_label"],
        "unique_atomic_v": validation["success_count"] == 1
        and validation["checks"]["atomic_edit"],
        "object_f_single_error": validation["checks"]["f_semantic_validation"],
        "meta_v_proved_unique": formal["v"]["agreement"]
        and formal["v"]["python_label"] == "true"
        and formal["v"]["proof_count"] == 1,
        "meta_f_not_proved": formal["f"]["agreement"]
        and formal["f"]["python_label"] == "unknown",
        "meta_f_countermodel": formal["f"]["certificate"]["countermodel"]["model_found"],
        "meta_f_single_error": formal["f"]["certificate"]["fatal_error_count"] == 1
        and formal["f"]["certificate"]["second_failure_point_absent"],
        "pre_error_steps_entailed": formal["f"]["certificate"].get(
            "pre_error_steps_entailed", True
        ),
        "six_arms": set(by_arm) == set(ARMS),
        "v_only_arm_with_target_gold": by_arm["V"]["gold_answer"]
        == base["target_answer"]
        and all(
            by_arm[arm]["gold_answer"] == base["initial_answer"]
            for arm in ("N", "R", "F", "AO", "S")
        ),
        "vf_shared_target": by_arm["V"]["target_answer"] == by_arm["F"]["target_answer"],
        "vf_shared_edit": canonical_sha256(pair["proposed_edit"])
        == canonical_sha256(validation["selected_edit"]),
        "ao_s_target_equal": by_arm["AO"]["target_answer"]
        == by_arm["S"]["target_answer"],
        "ao_s_canonical_length_equal": all(
            value
            for key, value in base["ao_s_pair"]["length_checks"].items()
            if key != "qwen_exact_token_equal"
        ),
        "n_r_no_target_hint": f"option {base['target_answer'].lower()}"
        not in N_TEMPLATE.lower()
        and f"option {base['target_answer'].lower()}" not in R_TEMPLATE.lower(),
        "option_mapping_bijective": set(base["semantic_to_letter"].values())
        == set(OPTION_LETTERS),
    }
    return {
        "schema": SCHEMA,
        "base_id": base["base_id"],
        "family_id": base["family_id"],
        "structure_class": base["structure_class"],
        "variant": base["variant"],
        "transition": base["transition"],
        "target_answer": base["target_answer"],
        "error_class": base["error_class"],
        "error_position": base["error_position"],
        "checks": checks,
        "passed_pre_tokenizer": all(checks.values()),
        "qwen_tokenizer_gate": "pending",
    }


def _family_cards() -> list[dict[str, Any]]:
    cards = []
    for family in family_plan():
        blueprint = blueprint_by_id(family["structure_class"])
        cards.append(
            {
                "schema": SCHEMA,
                **family,
                "transition": blueprint["transition"],
                "structure_signature": blueprint["structure_signature"],
                "allowed_operations": blueprint["allowed_operations"],
                "variants": list(VARIANTS),
                "development_only": True,
            }
        )
    return cards


def build_dataset(output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    for name in ("solver_traces", "countermodels", "f_certificates"):
        (output_root / name).mkdir(parents=True, exist_ok=True)
    cards = _family_cards()
    bases: list[dict[str, Any]] = []
    arms: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    base_index = 0
    for family in family_plan():
        for variant in VARIANTS:
            source = _construct_source(family, variant, base_index, output_root)
            sources.append(source)
            if source["status"] != "valid":
                raise RuntimeError(f"formal construction failed for {source['attempt_id']}")
            base, rendered_arms, audit = _render_base(source, base_index, output_root)
            if not audit["passed_pre_tokenizer"]:
                raise RuntimeError(f"pre-tokenizer audit failed for {base['base_id']}")
            bases.append(base)
            arms.extend(rendered_arms)
            audits.append(audit)
            base_index += 1
    write_jsonl(output_root / "family_cards24.jsonl", cards)
    write_jsonl(output_root / "symbolic_sources48.jsonl", sources)
    write_jsonl(output_root / "development_bases48.jsonl", bases)
    write_jsonl(output_root / "development_arms288.jsonl", arms)
    write_jsonl(output_root / "construction_audit48.jsonl", audits)
    report = _report(cards, bases, arms, audits)
    write_json(output_root / "build_report.json", report)
    write_json(
        output_root / "tokenizer_preflight.json",
        {
            "schema": SCHEMA,
            "status": "pending_fixed_qwen_tokenizer",
            "binding": TOKENIZER_BINDING,
            "all_canonical_pairs_equal": all(
                row["checks"]["ao_s_canonical_length_equal"] for row in audits
            ),
            "gpu_required": False,
        },
    )
    return {
        "cards": cards,
        "sources": sources,
        "bases": bases,
        "arms": arms,
        "audits": audits,
        "report": report,
        "hashes": {
            "cards": canonical_sha256(cards),
            "sources": canonical_sha256(sources),
            "bases": canonical_sha256(bases),
            "arms": canonical_sha256(arms),
            "audits": canonical_sha256(audits),
            "report": canonical_sha256(report),
        },
    }


def _report(
    cards: list[dict[str, Any]],
    bases: list[dict[str, Any]],
    arms: list[dict[str, Any]],
    audits: list[dict[str, Any]],
) -> dict[str, Any]:
    transition_counts = Counter(row["transition"] for row in bases)
    target_counts = Counter(row["target_answer"] for row in bases)
    error_counts = Counter(row["error_class"] for row in bases)
    position_counts = Counter(row["error_position"] for row in bases)
    family_depths = {
        family_id: {
            row["variant"]: row["difficulty"]["actual_shortest_proof_depth"]
            for row in bases
            if row["family_id"] == family_id
        }
        for family_id in sorted({row["family_id"] for row in bases})
    }
    gradients = {
        family_id: values.get("medium", -1) > values.get("easy", 10**9)
        for family_id, values in family_depths.items()
    }
    checks = {
        "exactly_24_families": len(cards) == 24,
        "exactly_48_bases": len(bases) == 48,
        "exactly_288_arms": len(arms) == 288,
        "six_arms_per_base": all(
            sum(row["base_id"] == base["base_id"] for row in arms) == 6 for base in bases
        ),
        "all_pre_tokenizer_audits": all(row["passed_pre_tokenizer"] for row in audits),
        "transition_balance": set(transition_counts.values()) == {8}
        and len(transition_counts) == 6,
        "target_position_balance": dict(target_counts) == {letter: 12 for letter in OPTION_LETTERS},
        "error_class_balance": set(error_counts.values()) == {8}
        and len(error_counts) == 6,
        "error_position_balance": dict(position_counts)
        == {position: 16 for position in ERROR_POSITIONS},
        "all_family_depth_gradients": len(gradients) == 24 and all(gradients.values()),
        "no_gpu_execution": True,
        "no_model_output_selection": True,
    }
    return {
        "schema": SCHEMA,
        "status": "pre_tokenizer_passed" if all(checks.values()) else "failed",
        "checks": checks,
        "counts": {
            "families": len(cards),
            "bases": len(bases),
            "arms": len(arms),
            "transitions": dict(sorted(transition_counts.items())),
            "target_positions": dict(sorted(target_counts.items())),
            "error_classes": dict(sorted(error_counts.items())),
            "error_positions": dict(sorted(position_counts.items())),
        },
        "family_depths": family_depths,
        "family_depth_gradients": gradients,
        "tokenizer_gate": "pending",
        "gpu_qwen_capping_run": False,
    }


def verify_determinism(output_root: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
        first = build_dataset(Path(first_dir))
        second = build_dataset(Path(second_dir))
    compared = {key: first["hashes"][key] == second["hashes"][key] for key in first["hashes"]}
    result = {
        "schema": SCHEMA,
        "passed": all(compared.values()),
        "compared": compared,
        "first": first["hashes"],
        "second": second["hashes"],
    }
    write_json(output_root / "deterministic_rebuild_report.json", result)
    return result


def write_manifest(output_root: Path, determinism: dict[str, Any]) -> None:
    files = []
    for path in sorted(output_root.rglob("*")):
        if not path.is_file() or path.name == "manifest.json" or "__pycache__" in path.parts:
            continue
        files.append(
            {
                "path": path.relative_to(REPO_ROOT).as_posix(),
                "raw_file_sha256": file_sha256(path),
            }
        )
    write_json(
        output_root / "manifest.json",
        {
            "schema": SCHEMA,
            "architecture": ARCHITECTURE,
            "master_seed": MASTER_SEED,
            "protocol": "docs/PROVER_GENERATOR_DEVELOPMENT_BANK_V1_PROTOCOL.md",
            "source_hybrid_stress48_sha256": "9199c8259f53467d5cc10903dcdbb1a2260d74703e5fcfd0a6281ec84deff1ab",
            "source_v1_2_human_review_sha256": "a6c8edba466ef72644547273370a64640bb42a6c841a4db2a1db7d6db4a9549e",
            "deterministic_rebuild_passed": determinism["passed"],
            "tokenizer_gate": "pending",
            "human_naturalization_gate": "pending",
            "gpu_qwen_capping_run": False,
            "files": files,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=HERE)
    parser.add_argument("--skip-determinism", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    result = build_dataset(output_root)
    determinism = (
        {"passed": False, "skipped": True}
        if args.skip_determinism
        else verify_determinism(output_root)
    )
    write_manifest(output_root, determinism)
    print(
        canonical_json(
            {
                "status": result["report"]["status"],
                "families": len(result["cards"]),
                "bases": len(result["bases"]),
                "arms": len(result["arms"]),
                "deterministic": determinism["passed"],
                "tokenizer_gate": "pending",
                "gpu_used": False,
            }
        )
    )


if __name__ == "__main__":
    main()
