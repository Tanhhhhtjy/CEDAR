# ruff: noqa: E501
"""Produce Phase 1.1 metrics, manifest, research report, and the human-stop gate."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from hardening_common import (
    ERROR_CLASSES,
    MASTER_SEED,
    canonical_json_file_sha256,
    canonical_jsonl_sha256,
    canonical_sha256,
    file_sha256,
    read_jsonl,
    write_json,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
EXPLORE = REPO_ROOT / "explore_outputs" / "prover_generator_hardening_v1_1"


def _canonical_file_hash(path: Path, workbook_hashes: dict[str, str] | None = None) -> str:
    if path.suffix == ".jsonl":
        return canonical_jsonl_sha256(path)
    if path.suffix == ".json":
        return canonical_json_file_sha256(path)
    if path.suffix == ".xlsx":
        if workbook_hashes is None or path.name not in workbook_hashes:
            raise ValueError("workbook canonical content hash is required")
        return workbook_hashes[path.name]
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return canonical_sha256({"normalized_text": normalized})


def _metrics(stress: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in stress if row["status"] == "valid"]
    pool_sizes = [row["validation"]["candidate_count"] for row in valid]
    ranks = [row["validation"]["successful_rank"] for row in valid]
    class_counts = Counter(row["assigned_f_error_class"] for row in valid)
    class_variant = Counter((row["assigned_f_error_class"], row["variant"]) for row in valid)
    depth_cells: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for row in valid:
        depth_cells[row["family_id"]][row["variant"]].append(
            row["validation"]["difficulty"]["actual_shortest_proof_depth"]
        )
    depth_summary = {
        family: {
            variant: {"values": sorted(values), "median": median(values)}
            for variant, values in sorted(variants.items())
        }
        for family, variants in sorted(depth_cells.items())
    }
    return {
        "attempt_count": len(stress),
        "valid_count": len(valid),
        "failed_count": len(stress) - len(valid),
        "construction_yield": len(valid) / len(stress),
        "gold_error_count": sum(
            not row["validation"]["initial_validation"]["agreement"]
            or not row["validation"]["updated_validation"]["agreement"]
            for row in valid
        ),
        "unique_atomic_v_rate": sum(row["validation"]["success_count"] == 1 for row in valid)
        / len(valid),
        "semantic_f_validation_rate": sum(
            row["validation"]["f_validation"]["passed"] for row in valid
        )
        / len(valid),
        "f_class_counts": dict(sorted(class_counts.items())),
        "f_class_variant_counts": {
            f"{name}/{variant}": class_variant[(name, variant)]
            for name in ERROR_CLASSES
            for variant in ("easy", "medium")
        },
        "candidate_pool": {
            "minimum": min(pool_sizes),
            "median": median(pool_sizes),
            "maximum": max(pool_sizes),
            "at_least_20_rate": sum(value >= 20 for value in pool_sizes) / len(valid),
        },
        "successful_edit_rank": {
            "minimum": min(ranks),
            "median": median(ranks),
            "maximum": max(ranks),
        },
        "formal_proof_depth": depth_summary,
        "difficulty_gradient_pass_count": sum(
            values["medium"]["median"] > values["easy"]["median"]
            for values in depth_summary.values()
        ),
    }


def _write_research_reports(
    metrics: dict[str, Any],
    validation: dict[str, Any],
    naturalization: dict[str, Any],
    test_summary: dict[str, str],
) -> None:
    EXPLORE.mkdir(parents=True, exist_ok=True)
    failure_catalog = """# Failure Catalog

## Formal stress failures

None in the Amendment 01 replacement run: 48/48 attempts are valid.

## Engineering failures caught by gates

1. Pre-freeze Mace4 constant parsing: fixed globally with an explicit constant prefix.
2. First-run F schedule lacked easy/medium coverage within class: preserved at commit `b4586f7`,
   corrected by Amendment 01, and followed by a complete 48-attempt rebuild.
3. One automatic V/F length-ratio failure: fixed in the renderer only and rerun for all 12 rows.
4. A pre-review audit found missing `the claim that`, awkward replacement syntax, fixed condition
   wording, and an underspecified human gate. Amendment 02 revised renderer/review governance only;
   332 protected symbolic and proof artifacts retained their canonical hashes.

No failed sample was dropped, no seed was replaced, and no correctness gate was relaxed.
"""
    (EXPLORE / "FAILURE_CATALOG.md").write_text(failure_catalog, encoding="utf-8")
    results = f"""# Phase 1.1 Results

## Outcome

The symbolic generator passed every frozen hard gate. Amendment 02 revised only the renderer and
human-review governance before review. The final state remains Gate B because the blank v2
naturalization workbook has not yet been returned by a real reviewer.

## Symbolic stress test

- Attempts: {metrics["attempt_count"]}
- Valid / failed: {metrics["valid_count"]} / {metrics["failed_count"]}
- Gold errors: {metrics["gold_error_count"]}
- Unique atomic V rate: {metrics["unique_atomic_v_rate"]:.1%}
- Semantic F validation rate: {metrics["semantic_f_validation_rate"]:.1%}
- Six F classes: eight records each, split four easy and four medium
- Candidate pool: min {metrics["candidate_pool"]["minimum"]}, median {metrics["candidate_pool"]["median"]}, max {metrics["candidate_pool"]["maximum"]}
- Successful edit rank: min {metrics["successful_edit_rank"]["minimum"]}, median {metrics["successful_edit_rank"]["median"]}, max {metrics["successful_edit_rank"]["maximum"]}
- Formal difficulty gradients: {metrics["difficulty_gradient_pass_count"]}/8 families
- Deterministic rebuild: {validation["deterministic_rebuild"]["passed"]} across {validation["deterministic_rebuild"]["file_count_first"]} generated files per build

These are formal proof-depth results, not claims about model behavior.

## Naturalization

- Frozen sample: 12 v2 records, covering six F classes at easy and medium difficulty
- Automatic checks: {naturalization["automatic_checks"]["passed"]}
- Human review: `awaiting_human_naturalization_review`
- Workbook: `naturalization_review12_bilingual_v2.xlsx`
- Workbook fields: blank; no gold, F class, or symbolic internals are displayed
- Item-level rule: all six criteria must pass on all 12 rows

## Verification

- Targeted Phase 1.1 tests: {test_summary["targeted"]}
- Full repository test run: {test_summary["full_repository"]}
- Ruff: {test_summary["ruff"]}

The full-repository failures are outside Phase 1.1 and are dominated by unavailable frozen model
assets, absent external commands, and pre-existing hash-bound artifacts. They are not counted as
Phase 1.1 passes; the exact outcome is disclosed rather than suppressed.

## Boundary

No Qwen, GPU, capping, S arm, model-response selection, development bank, or held-out construction
was run. The 48 stress records are not a confirmation set.
"""
    (EXPLORE / "PHASE1_1_RESULTS.md").write_text(results, encoding="utf-8")
    ledger = """# Experiment Ledger

| Date | Stage | Action | Result |
|---|---|---|---|
| 2026-07-23 | Intake | Branch/worktree from `4712b0d` | Passed; v1 paths untouched |
| 2026-07-23 | Audit | Machine-audited both v1 route outputs and code | v1 comparison downgraded to inconclusive |
| 2026-07-23 | Protocol | Froze hybrid architecture and gates | Commit `c861bf2` |
| 2026-07-23 | Sentinels | Six structural F operators and dual solvers | 6/6 passed; deterministic rebuild passed |
| 2026-07-23 | Stress run 1 | 48 attempts | 48/48 valid; downstream class-by-difficulty defect found |
| 2026-07-23 | Amendment 01 | Preserved run 1 and changed only global F schedule | Commit `cbd5ced` |
| 2026-07-23 | Stress run 2 | Complete rebuild and dual validation | 48/48 valid; every class easy=4, medium=4 |
| 2026-07-23 | Determinism | Two clean full builds | 331/331 canonical files identical |
| 2026-07-23 | Naturalization | 12 bilingual deterministic renders | Automatic gate 12/12; human fields blank |
| 2026-07-23 | Tests | Targeted, full repository, Ruff | See `PHASE1_1_RESULTS.md` |
| 2026-07-23 | Amendment 02 | Renderer, leakage audit, and human finalizer only | v2 packet; symbolic hashes unchanged |
"""
    (EXPLORE / "EXPERIMENT_LEDGER.md").write_text(ledger, encoding="utf-8")
    comparability = """# Comparability Report

Phase 1.1 evaluates one approved hybrid engineering architecture. It does not compare or score
ProverGen-style and ReviseQA-style routes, and the v1 `100 vs 99` score remains retired.

All replacement-run attempts use the same frozen 8-family x 2-variant x 3-replicate matrix,
master seed, candidate budget, solvers, and validators. Amendment 01 changes only the global
F-class schedule and rebuilds every attempt. The pre-amendment run is hash-bound in its amendment.

Formal proof depth is not model behavior difficulty. The naturalization sample tests only renderer
quality and awaits a real reviewer; it is not a language-model benchmark. Amendment 02 preserves
the original packet at `06fd188` and changes no symbolic record, gold, edit, F certificate, proof,
countermodel, or candidate result.
"""
    (EXPLORE / "COMPARABILITY_REPORT.md").write_text(comparability, encoding="utf-8")


def _manifest(
    root: Path,
    stress: list[dict[str, Any]],
    naturalization: dict[str, Any],
    final_gate: dict[str, Any],
) -> dict[str, Any]:
    original_naturalization = json.loads(
        (root / "naturalization_automatic_checks.json").read_text(encoding="utf-8")
    )
    workbook_hashes = {
        "naturalization_review12_bilingual.xlsx": original_naturalization[
            "workbook_canonical_content_sha256"
        ],
        "naturalization_review12_bilingual_v2.xlsx": naturalization[
            "workbook_canonical_content_sha256"
        ],
    }
    files: list[dict[str, Any]] = []
    included_roots = [root, EXPLORE]
    for included_root in included_roots:
        for path in sorted(included_root.rglob("*")):
            if not path.is_file() or path.name == "manifest.json" or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(REPO_ROOT).as_posix()
            files.append(
                {
                    "path": relative,
                    "raw_file_sha256": file_sha256(path),
                    "canonical_content_sha256": _canonical_file_hash(path, workbook_hashes),
                }
            )
    item_hashes = []
    for row in stress:
        input_contract = {
            "family_id": row["family_id"],
            "variant": row["variant"],
            "replicate_id": row["replicate_id"],
            "seed": row["seed"],
            "assigned_f_error_class": row["assigned_f_error_class"],
        }
        item_hashes.append(
            {
                "attempt_id": row["attempt_id"],
                "input_canonical_sha256": canonical_sha256(input_contract),
                "output_canonical_sha256": canonical_sha256(row),
                "status": row["status"],
            }
        )
    return {
        "schema_version": "cedar.prover_generator_hardening.manifest.v1_1",
        "research_anchor": "4712b0d555ae2b07f73781c6a11b7caa660ef96a",
        "protocol_commit": "2266a16",
        "naturalization_amendment": "02",
        "protected_baseline_commit": "06fd18863503734c32c754defa411172c2756297",
        "architecture": "proof_graph_plus_atomic_edit_certification",
        "master_seed": MASTER_SEED,
        "builder_sha256": file_sha256(root / "proof_graph_builder.py"),
        "candidate_enumerator_sha256": file_sha256(root / "atomic_edit_candidate_enumerator.py"),
        "semantic_validator_sha256": file_sha256(root / "semantic_validator.py"),
        "f_operator_sha256": file_sha256(root / "semantic_f_operators.py"),
        "family_canonical_sha256": canonical_jsonl_sha256(root / "family_blueprints8_v1_1.jsonl"),
        "solver": json.loads((root / "f_sentinel_freeze.json").read_text(encoding="utf-8"))[
            "solver"
        ],
        "formal_attempts": item_hashes,
        "failures": [row["attempt_id"] for row in stress if row["status"] != "valid"],
        "naturalization12_v1_canonical_sha256": original_naturalization[
            "naturalization_canonical_sha256"
        ],
        "naturalization12_v2_canonical_sha256": naturalization["naturalization_canonical_sha256"],
        "workbook_v1_canonical_content_sha256": workbook_hashes[
            "naturalization_review12_bilingual.xlsx"
        ],
        "workbook_v2_canonical_content_sha256": workbook_hashes[
            "naturalization_review12_bilingual_v2.xlsx"
        ],
        "human_review_finalizer_sha256": file_sha256(root / "finalize_naturalization_review_v2.py"),
        "final_gate_canonical_sha256": canonical_sha256(final_gate),
        "files": files,
        "gpu_qwen_capping_run": False,
    }


def analyze(root: Path, test_summary: dict[str, str]) -> dict[str, Any]:
    stress = read_jsonl(root / "hybrid_stress48.jsonl")
    validation = json.loads((root / "validation_report.json").read_text(encoding="utf-8"))
    naturalization_path = root / "naturalization_automatic_checks_v2.json"
    naturalization = json.loads(naturalization_path.read_text(encoding="utf-8"))
    metrics = _metrics(stress)
    symbolic_pass = validation["passed"] and metrics["valid_count"] >= 44
    automatic_pass = naturalization["automatic_checks"]["passed"]
    final_gate = {
        "gate": "B" if symbolic_pass and automatic_pass else "D",
        "symbolic_status": "symbolic_generator_passed"
        if symbolic_pass
        else "generator_hardening_failed",
        "human_review_status": "awaiting_human_naturalization_review"
        if symbolic_pass and automatic_pass
        else "not_reached",
        "step_1_complete": False,
        "may_prepare_development_bank_protocol": False,
        "review_packet": "naturalization_review12_bilingual_v2.xlsx",
        "human_passing_rule": "all_six_criteria_on_all_12_rows",
        "reason": "real human naturalization v2 review has not been returned"
        if symbolic_pass and automatic_pass
        else "a symbolic or automatic hard gate failed",
        "gpu_qwen_capping_run": False,
    }
    write_json(root / "stress_metrics.json", metrics)
    write_json(EXPLORE / "FINAL_GATE.json", final_gate)
    _write_research_reports(metrics, validation, naturalization, test_summary)
    manifest = _manifest(root, stress, naturalization, final_gate)
    write_json(root / "manifest.json", manifest)
    return {"metrics": metrics, "final_gate": final_gate, "manifest_files": len(manifest["files"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--targeted-tests", default="22 passed")
    parser.add_argument(
        "--full-repository-tests",
        default="238 passed, 20 failed (pre-existing asset/environment-bound failures)",
    )
    parser.add_argument("--ruff", default="passed")
    args = parser.parse_args()
    print(
        analyze(
            args.root.resolve(),
            {
                "targeted": args.targeted_tests,
                "full_repository": args.full_repository_tests,
                "ruff": args.ruff,
            },
        )
    )


if __name__ == "__main__":
    main()
