"""Build six F sentinels and the frozen 48-attempt v1.1 stress matrix."""

from __future__ import annotations

import argparse
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from atomic_edit_candidate_enumerator import (
    enumerate_atomic_edit_candidates,
    evaluate_candidate_neighborhood,
)
from hardening_common import (
    ARCHITECTURE,
    ERROR_CLASSES,
    MASTER_SEED,
    SCHEMA_VERSION,
    canonical_json,
    canonical_sha256,
    file_sha256,
    locate_solver,
    stable_seed,
    write_json,
    write_jsonl,
)
from proof_graph_builder import (
    FAMILY_BLUEPRINTS,
    assert_no_answer_channel,
    blueprint_by_id,
    proof_graph_builder,
)
from semantic_validator import semantic_validator

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]

GENERATOR_CONFIG = {
    "schema_version": SCHEMA_VERSION,
    "architecture": ARCHITECTURE,
    "master_seed": MASTER_SEED,
    "families": 8,
    "variants": ["easy", "medium"],
    "replicates_per_cell": 3,
    "attempts": 48,
    "candidate_limit": 100,
    "minimum_candidate_pool": 20,
    "minimum_fraction_at_pool_floor": 0.90,
    "external_llm_naturalization": False,
    "gpu_qwen_capping_run": False,
}

F_OPERATOR_CONTRACTS = [
    {
        "error_class": name,
        "fatal_error_count": 1,
        "classification_source": "world_structure_not_declared_label",
        "requires_entailed_premises": True,
        "requires_unentailed_claim": True,
        "requires_atomic_edit_projection": True,
        "requires_counterevidence": True,
        "social_cues_allowed": False,
    }
    for name in ERROR_CLASSES
]


def _error_class_for(family_index: int, variant_index: int, replicate_id: int) -> str:
    matrix_index = family_index * 3 + variant_index * 3 + (replicate_id - 1)
    return ERROR_CLASSES[matrix_index % len(ERROR_CLASSES)]


def _solver_version(executable: Path) -> str:
    completed = subprocess.run(
        [str(executable)], input="", capture_output=True, text=True, timeout=10, check=False
    )
    match = re.search(
        r"(Prover9|Mace4) \(32\) version [^,\n]+", completed.stdout + completed.stderr
    )
    return match.group(0) if match else f"{executable.stem}:version-not-emitted-with-empty-input"


def _externalize_evidence(root: Path, item_id: str, validation: dict[str, Any]) -> None:
    for state in ("initial_validation", "updated_validation"):
        if state not in validation:
            continue
        for polarity in ("query_result", "opposite_result"):
            payload = validation[state]["prover9"][polarity]
            write_json(root / "solver_traces" / f"{item_id}__{state}__{polarity}.json", payload)
    certificate = validation.get("f_certificate")
    if not certificate:
        return
    counterevidence = certificate["countermodel_or_opposite_proof"]
    write_json(root / "countermodels" / f"{item_id}__counterevidence.json", counterevidence)
    write_json(
        root / "semantic_error_certificates" / f"{item_id}__f_certificate.json",
        certificate,
    )


def _construct(
    family_id: str,
    variant: str,
    replicate_id: int,
    seed: int,
    error_class: str,
    *,
    item_id: str,
    sentinel: bool,
    output_root: Path,
) -> dict[str, Any]:
    blueprint = blueprint_by_id(family_id)
    try:
        built = proof_graph_builder(blueprint, variant, seed, blueprint["transition"], error_class)
        assert_no_answer_channel(built)
        candidates = enumerate_atomic_edit_candidates(built)
        neighborhood = evaluate_candidate_neighborhood(built, candidates)
        validation = semantic_validator(built, neighborhood)
        if validation.get("f_validation", {}).get("derived_error_class") != error_class:
            validation["passed"] = False
            validation["checks"]["assigned_f_class_matches_structure"] = False
        else:
            validation["checks"]["assigned_f_class_matches_structure"] = True
        _externalize_evidence(output_root, item_id, validation)
        return {
            "schema_version": SCHEMA_VERSION,
            "architecture": ARCHITECTURE,
            "attempt_id": item_id,
            "sentinel": sentinel,
            "family_id": family_id,
            "variant": variant,
            "replicate_id": replicate_id,
            "seed": seed,
            "assigned_f_error_class": error_class,
            "status": "valid" if validation["passed"] else "failed",
            "builder_output": built,
            "validation": validation,
        }
    except Exception as error:  # preserve every construction failure as data
        return {
            "schema_version": SCHEMA_VERSION,
            "architecture": ARCHITECTURE,
            "attempt_id": item_id,
            "sentinel": sentinel,
            "family_id": family_id,
            "variant": variant,
            "replicate_id": replicate_id,
            "seed": seed,
            "assigned_f_error_class": error_class,
            "status": "failed",
            "failure_stage": "construction_or_validation_exception",
            "failure_type": type(error).__name__,
            "failure_reason": str(error),
        }


def build_sentinels(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sentinel_families = ["PGF-01", "PGF-02", "PGF-03", "PGF-04", "PGF-05", "PGF-06"]
    for family_id, error_class in zip(sentinel_families, ERROR_CLASSES, strict=True):
        seed = stable_seed(MASTER_SEED, "f_sentinel", error_class)
        item_id = f"H11-SENTINEL-{error_class.upper().replace('_', '-')}"
        rows.append(
            _construct(
                family_id,
                "easy",
                0,
                seed,
                error_class,
                item_id=item_id,
                sentinel=True,
                output_root=output_root,
            )
        )
    return rows


def _freeze_payload(output_root: Path, sentinels: list[dict[str, Any]]) -> dict[str, Any]:
    prover9 = locate_solver("prover9")
    mace4 = locate_solver("mace4")
    return {
        "schema_version": SCHEMA_VERSION,
        "architecture": ARCHITECTURE,
        "all_sentinels_passed": all(row["status"] == "valid" for row in sentinels),
        "sentinel_count": len(sentinels),
        "sentinel_canonical_sha256": canonical_sha256(sentinels),
        "family_blueprint_canonical_sha256": canonical_sha256(FAMILY_BLUEPRINTS),
        "f_operator_source_sha256": file_sha256(HERE / "semantic_f_operators.py"),
        "candidate_enumerator_source_sha256": file_sha256(
            HERE / "atomic_edit_candidate_enumerator.py"
        ),
        "schema_canonical_sha256": canonical_sha256(GENERATOR_CONFIG),
        "validator_source_sha256": file_sha256(HERE / "semantic_validator.py"),
        "solver": {
            "prover9": {
                "sha256": file_sha256(prover9),
                "version": _solver_version(prover9),
            },
            "mace4": {"sha256": file_sha256(mace4), "version": _solver_version(mace4)},
        },
        "master_seed": MASTER_SEED,
        "formal_stress_authorized": all(row["status"] == "valid" for row in sentinels),
    }


def build_stress(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family_index, blueprint in enumerate(FAMILY_BLUEPRINTS):
        family_id = blueprint["family_id"]
        for variant_index, variant in enumerate(("easy", "medium")):
            for replicate_id in range(1, 4):
                seed = stable_seed(MASTER_SEED, family_id, variant, replicate_id)
                error_class = _error_class_for(family_index, variant_index, replicate_id)
                item_id = f"H11-{family_id}-{variant[0].upper()}-R{replicate_id}"
                rows.append(
                    _construct(
                        family_id,
                        variant,
                        replicate_id,
                        seed,
                        error_class,
                        item_id=item_id,
                        sentinel=False,
                        output_root=output_root,
                    )
                )
    return rows


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["status"] == "valid"]
    class_counts = Counter(row["assigned_f_error_class"] for row in valid)
    pool_sizes = [row["validation"]["candidate_count"] for row in valid]
    ranks = [row["validation"]["successful_rank"] for row in valid]
    depths: dict[str, dict[str, list[int]]] = {}
    for row in valid:
        depths.setdefault(row["family_id"], {}).setdefault(row["variant"], []).append(
            row["validation"]["difficulty"]["actual_shortest_proof_depth"]
        )
    gradients: dict[str, bool] = {}
    for family_id, variants in depths.items():
        easy = sorted(variants.get("easy", []))
        medium = sorted(variants.get("medium", []))
        gradients[family_id] = bool(
            easy and medium and medium[len(medium) // 2] > easy[len(easy) // 2]
        )
    return {
        "attempt_count": len(rows),
        "valid_count": len(valid),
        "failed_count": len(rows) - len(valid),
        "valid_f_class_counts": dict(sorted(class_counts.items())),
        "candidate_pool": {
            "minimum": min(pool_sizes) if pool_sizes else None,
            "maximum": max(pool_sizes) if pool_sizes else None,
            "at_least_20_count": sum(value >= 20 for value in pool_sizes),
            "at_least_20_fraction": sum(value >= 20 for value in pool_sizes) / len(pool_sizes)
            if pool_sizes
            else 0,
        },
        "successful_rank": {
            "minimum": min(ranks) if ranks else None,
            "maximum": max(ranks) if ranks else None,
            "values": ranks,
        },
        "difficulty_gradients": gradients,
        "all_family_gradients_passed": len(gradients) == 8 and all(gradients.values()),
    }


def build(output_root: Path, *, sentinels_only: bool = False) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    for directory in ("solver_traces", "countermodels", "semantic_error_certificates"):
        (output_root / directory).mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "family_blueprints8_v1_1.jsonl", FAMILY_BLUEPRINTS)
    write_json(output_root / "generator_config.json", GENERATOR_CONFIG)
    write_json(output_root / "f_operator_contracts.json", F_OPERATOR_CONTRACTS)
    sentinels = build_sentinels(output_root)
    write_jsonl(output_root / "f_sentinels6.jsonl", sentinels)
    freeze = _freeze_payload(output_root, sentinels)
    write_json(output_root / "f_sentinel_freeze.json", freeze)
    if not freeze["all_sentinels_passed"]:
        raise RuntimeError("F sentinel freeze failed; formal stress run is prohibited")
    if sentinels_only:
        return {"sentinels": sentinels, "freeze": freeze}
    stress = build_stress(output_root)
    write_jsonl(output_root / "hybrid_stress48.jsonl", stress)
    summary = _summary(stress)
    write_json(output_root / "stress_metrics.json", summary)
    return {"sentinels": sentinels, "freeze": freeze, "stress": stress, "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=HERE)
    parser.add_argument("--sentinels-only", action="store_true")
    args = parser.parse_args()
    result = build(args.output_root.resolve(), sentinels_only=args.sentinels_only)
    print(
        canonical_json(
            {key: len(value) if isinstance(value, list) else value for key, value in result.items()}
        )
    )


if __name__ == "__main__":
    main()
