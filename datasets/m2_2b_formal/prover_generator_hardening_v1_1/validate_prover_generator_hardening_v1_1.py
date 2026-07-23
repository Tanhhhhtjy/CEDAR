"""Independent hard-gate validation for Phase 1.1 generated artifacts."""

from __future__ import annotations

import argparse
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from build_prover_generator_hardening_v1_1 import build
from hardening_common import (
    ERROR_CLASSES,
    canonical_tree_hash,
    file_sha256,
    locate_solver,
    read_jsonl,
    write_json,
)

HERE = Path(__file__).resolve().parent


def _check(condition: bool, name: str, failures: list[str]) -> None:
    if not condition:
        failures.append(name)


def _verify_rebuild() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
        first_root, second_root = Path(first), Path(second)
        build(first_root)
        build(second_root)
        first_hashes = canonical_tree_hash(first_root)
        second_hashes = canonical_tree_hash(second_root)
        paths = sorted(set(first_hashes) | set(second_hashes))
        different = [path for path in paths if first_hashes.get(path) != second_hashes.get(path)]
        return {
            "passed": not different,
            "file_count_first": len(first_hashes),
            "file_count_second": len(second_hashes),
            "different_files": different,
        }


def validate(root: Path, *, verify_rebuild: bool) -> dict[str, Any]:
    failures: list[str] = []
    sentinels = read_jsonl(root / "f_sentinels6.jsonl")
    stress = read_jsonl(root / "hybrid_stress48.jsonl")
    import json

    freeze = json.loads((root / "f_sentinel_freeze.json").read_text(encoding="utf-8"))
    _check(len(sentinels) == 6, "sentinel_count", failures)
    _check(all(row["status"] == "valid" for row in sentinels), "sentinel_pass", failures)
    _check(freeze["all_sentinels_passed"], "sentinel_freeze", failures)
    _check(
        freeze["f_operator_source_sha256"] == file_sha256(root / "semantic_f_operators.py"),
        "f_operator_freeze_hash",
        failures,
    )
    _check(
        freeze["candidate_enumerator_source_sha256"]
        == file_sha256(root / "atomic_edit_candidate_enumerator.py"),
        "candidate_enumerator_freeze_hash",
        failures,
    )
    _check(
        freeze["validator_source_sha256"] == file_sha256(root / "semantic_validator.py"),
        "semantic_validator_freeze_hash",
        failures,
    )
    _check(
        freeze["solver"]["prover9"]["sha256"] == file_sha256(locate_solver("prover9")),
        "prover9_freeze_hash",
        failures,
    )
    _check(
        freeze["solver"]["mace4"]["sha256"] == file_sha256(locate_solver("mace4")),
        "mace4_freeze_hash",
        failures,
    )
    _check(len(stress) == 48, "attempt_count", failures)
    _check(len({row["attempt_id"] for row in stress}) == 48, "attempt_id_uniqueness", failures)
    expected_cells = {
        (family, variant, replicate)
        for family in [f"PGF-{index:02d}" for index in range(1, 9)]
        for variant in ("easy", "medium")
        for replicate in range(1, 4)
    }
    actual_cells = {(row["family_id"], row["variant"], row["replicate_id"]) for row in stress}
    _check(actual_cells == expected_cells, "complete_stress_matrix", failures)

    valid = [row for row in stress if row["status"] == "valid"]
    gold_errors = 0
    unique_v_errors = 0
    f_errors = 0
    pool_sizes: list[int] = []
    ranks: list[int] = []
    depths: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in valid:
        validation = row["validation"]
        builder = row["builder_output"]
        prohibited = {
            "edit",
            "intended_edit",
            "updated_world",
            "successful_rank",
            "correct_record_id",
            "correct_location",
        }
        _check(not prohibited.intersection(builder), f"{row['attempt_id']}:builder_leak", failures)
        initial = validation["initial_validation"]
        updated = validation["updated_validation"]
        if not initial["agreement"] or not updated["agreement"]:
            gold_errors += 1
        if validation["success_count"] != 1 or not validation["checks"]["atomic_edit"]:
            unique_v_errors += 1
        if not validation["f_validation"]["passed"]:
            f_errors += 1
        _check(all(validation["checks"].values()), f"{row['attempt_id']}:hard_check", failures)
        _check(
            validation["f_validation"]["derived_error_class"] == row["assigned_f_error_class"],
            f"{row['attempt_id']}:f_class",
            failures,
        )
        _check(
            "unsupported_atomic_state_claim" not in str(validation["f_certificate"]),
            f"{row['attempt_id']}:generic_fatal_step",
            failures,
        )
        pool_sizes.append(validation["candidate_count"])
        ranks.append(validation["successful_rank"])
        _check(
            len(validation["candidate_trace"]) == validation["candidate_count"],
            f"{row['attempt_id']}:candidate_trace_complete",
            failures,
        )
        depths[row["family_id"]][row["variant"]].append(
            validation["difficulty"]["actual_shortest_proof_depth"]
        )

    class_counts = Counter(row["assigned_f_error_class"] for row in valid)
    _check(len(valid) >= 44, "construction_yield", failures)
    _check(gold_errors == 0, "gold_error_zero", failures)
    _check(unique_v_errors == 0, "unique_atomic_v_100pct", failures)
    _check(f_errors == 0, "semantic_f_100pct", failures)
    _check(
        class_counts == Counter({name: 8 for name in ERROR_CLASSES}),
        "balanced_f_coverage",
        failures,
    )
    _check(
        sum(value >= 20 for value in pool_sizes) / len(stress) >= 0.90,
        "candidate_pool_floor",
        failures,
    )
    _check(all(value <= 100 for value in pool_sizes), "candidate_pool_ceiling", failures)
    gradient_checks = {
        family: median(values["medium"]) > median(values["easy"])
        for family, values in sorted(depths.items())
    }
    _check(
        len(gradient_checks) == 8 and all(gradient_checks.values()), "difficulty_gradient", failures
    )
    rebuild = _verify_rebuild() if verify_rebuild else {"passed": None, "not_run": True}
    if verify_rebuild:
        _check(rebuild["passed"], "deterministic_rebuild", failures)
    return {
        "schema_version": "cedar.prover_generator_hardening.validation.v1_1",
        "passed": not failures,
        "failures": failures,
        "attempt_count": len(stress),
        "valid_count": len(valid),
        "failed_count": len(stress) - len(valid),
        "gold_error_count": gold_errors,
        "unique_atomic_v_error_count": unique_v_errors,
        "semantic_f_error_count": f_errors,
        "f_class_counts": dict(sorted(class_counts.items())),
        "candidate_pool_min": min(pool_sizes) if pool_sizes else None,
        "candidate_pool_max": max(pool_sizes) if pool_sizes else None,
        "successful_rank_min": min(ranks) if ranks else None,
        "successful_rank_max": max(ranks) if ranks else None,
        "difficulty_gradients": gradient_checks,
        "deterministic_rebuild": rebuild,
        "gpu_qwen_capping_run": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--verify-rebuild", action="store_true")
    args = parser.parse_args()
    report = validate(args.root.resolve(), verify_rebuild=args.verify_rebuild)
    write_json(args.root.resolve() / "validation_report.json", report)
    print(report)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
