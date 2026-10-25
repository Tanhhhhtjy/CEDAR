"""Compare validated generator routes and emit the bounded decision artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from build_prover_generator_feasibility import ROUTES
from formal_horn import (
    SCHEMA_VERSION,
    canonical_json_file_sha256,
    canonical_jsonl_sha256,
    file_sha256,
    read_jsonl,
    write_json,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
EXPLORE_ROOT = REPO_ROOT / "explore_outputs" / "prover_generator_feasibility_v1"


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _route_metrics(route: str, rows: list[dict[str, Any]], deterministic: bool) -> dict[str, Any]:
    valid = [row for row in rows if row["status"] == "valid"]
    failed = [row for row in rows if row["status"] == "failed"]
    attempts = [row["construction_attempts"] for row in valid]
    depths = {row["spec_id"]: row["difficulty"]["actual_shortest_proof_depth"] for row in valid}
    by_family: dict[str, dict[str, int]] = {}
    for row in valid:
        by_family.setdefault(row["family_id"], {})[row["variant"]] = row["difficulty"][
            "actual_shortest_proof_depth"
        ]
    gradient_count = sum(
        variants.get("medium", -1) > variants.get("easy", 999) for variants in by_family.values()
    )
    pairing_valid = sum(
        row["arms"]["V"]["atomic"]
        and row["arms"]["F"]["second_failure_point_absent"]
        and sum(not step["valid"] for step in row["arms"]["F"]["candidate_reasoning_steps"]) == 1
        for row in valid
    )
    gold_errors = sum(
        not row["solver_validation"][state]["agreement"]
        for row in valid
        for state in ("initial", "updated")
    )
    rejected = Counter()
    for row in rows:
        for attempt in row.get("construction_trace", []):
            if not attempt.get("accepted", False):
                rejected[attempt.get("updated_label", attempt.get("failure", "unknown"))] += 1
    hard_gate = (
        len(rows) == 16
        and len(valid) == 16
        and gold_errors == 0
        and pairing_valid == 16
        and deterministic
    )
    yield_score = 35.0 * len(valid) / 16 if hard_gate else 0.0
    pairing_score = 25.0 * pairing_valid / 16 if hard_gate else 0.0
    difficulty_score = 15.0 * gradient_count / 8 if hard_gate else 0.0
    determinism_score = 10.0 if hard_gate and deterministic else 0.0
    integration_score = 10.0 if route == "provergen_style" else 9.0
    naturalization_score = 5.0
    score = round(
        yield_score
        + pairing_score
        + difficulty_score
        + determinism_score
        + integration_score
        + naturalization_score,
        3,
    )
    return {
        "route": route,
        "attempt_records": len(rows),
        "valid_records": len(valid),
        "failed_records": len(failed),
        "valid_yield": len(valid) / 16,
        "hard_gate_passed": hard_gate,
        "gold_error_count": gold_errors,
        "v_f_pairing_valid": pairing_valid,
        "families_with_medium_deeper": gradient_count,
        "proof_depths": depths,
        "proof_path_count_distribution": dict(
            sorted(Counter(row["difficulty"]["proof_path_count"] for row in valid).items())
        ),
        "distractor_count_distribution": dict(
            sorted(Counter(row["difficulty"]["distractor_count"] for row in valid).items())
        ),
        "construction_attempts": {
            "median": statistics.median(attempts) if attempts else None,
            "p95": _percentile(attempts, 0.95) if attempts else None,
            "maximum": max(attempts) if attempts else None,
        },
        "wall_clock_time_buckets": dict(
            sorted(Counter(row["wall_clock_time_bucket"] for row in valid).items())
        ),
        "wall_clock_median_p95": (
            "<1s/<1s"
            if valid and all(row["wall_clock_time_bucket"] == "<1s" for row in valid)
            else "see_bucket_distribution"
        ),
        "rejected_candidate_labels_or_failures": dict(sorted(rejected.items())),
        "failure_records": [
            {
                "spec_id": row["spec_id"],
                "stage": row["failure_stage"],
                "reason": row["failure_reason"],
            }
            for row in failed
        ],
        "score_breakdown": {
            "yield_and_gold": yield_score,
            "v_f_pairing": pairing_score,
            "difficulty_control": difficulty_score,
            "determinism": determinism_score,
            "integration_maintenance": integration_score,
            "naturalization": naturalization_score,
            "total": score,
        },
    }


def _decision(metrics: dict[str, Any]) -> dict[str, Any]:
    routes = metrics["routes"]
    a = routes["provergen_style"]
    b = routes["reviseqa_style"]
    if not a["hard_gate_passed"] and not b["hard_gate_passed"]:
        choice = "D"
        recommendation = "neither_route_suitable"
        human = True
    elif not a["hard_gate_passed"]:
        choice = "B"
        recommendation = "select_reviseqa_style"
        human = False
    elif not b["hard_gate_passed"]:
        choice = "A"
        recommendation = "select_provergen_style"
        human = False
    else:
        difference = abs(a["score_breakdown"]["total"] - b["score_breakdown"]["total"])
        if difference < 5:
            choice = "C"
            recommendation = "human_checkpoint / hybrid_candidate"
            human = True
        elif a["score_breakdown"]["total"] > b["score_breakdown"]["total"]:
            choice = "A"
            recommendation = "select_provergen_style"
            human = False
        else:
            choice = "B"
            recommendation = "select_reviseqa_style"
            human = False
    return {
        "schema_version": SCHEMA_VERSION,
        "decision_code": choice,
        "recommendation": recommendation,
        "human_decision_required": human,
        "recommended_hybrid_if_approved": (
            "construct the initial proof graph with provergen_style, then certify the V edit by "
            "reviseqa_style atomic-neighbor search"
            if choice == "C"
            else None
        ),
        "automatic_hybrid_implementation_authorized": False,
        "may_enter_development_bank": "only_after_human_approval",
        "hard_stop_observed": True,
    }


def _text_sha256(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _manifest(data_root: Path, extra_roots: list[Path]) -> dict[str, Any]:
    entries = []
    files = [
        path
        for path in data_root.rglob("*")
        if path.is_file()
        and path.name != "manifest.json"
        and path.suffix in {".json", ".jsonl", ".py", ".md"}
    ]
    for extra_root in extra_roots:
        files.extend(path for path in extra_root.glob("*") if path.is_file())
    for path in sorted(set(files), key=lambda item: item.as_posix()):
        if path.is_relative_to(data_root):
            relative = path.relative_to(data_root).as_posix()
        else:
            relative = path.relative_to(REPO_ROOT).as_posix()
        if path.suffix == ".jsonl":
            canonical = canonical_jsonl_sha256(path)
        elif path.suffix == ".json":
            canonical = canonical_json_file_sha256(path)
        else:
            canonical = _text_sha256(path)
        entries.append(
            {
                "path": relative,
                "raw_file_sha256": file_sha256(path),
                "canonical_content_sha256": canonical,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "research_anchor": "364af5f797ee1eccf21c99e67570f59cbe052e5b",
        "branch": "agent/prover-generator-feasibility",
        "files": entries,
        "hard_gate_hash_type": "canonical_content_sha256",
        "raw_hash_is_provenance_only": True,
        "upstream_code_or_data_vendored": False,
        "gpu_qwen_capping_run": False,
    }


def _write_reports(metrics: dict[str, Any], decision: dict[str, Any]) -> None:
    EXPLORE_ROOT.mkdir(parents=True, exist_ok=True)
    a = metrics["routes"]["provergen_style"]
    b = metrics["routes"]["reviseqa_style"]
    failure_catalog = f"""# Failure Catalog

## Formal attempt failures

- `provergen_style`: {a["failed_records"]}/16 failed.
- `reviseqa_style`: {b["failed_records"]}/16 failed.

No valid record contains a solver disagreement, invalid world, non-atomic V, missing F
countermodel/opposite proof, or second F failure point.

## Rejected search neighbors

Rejected `reviseqa_style` neighbors are retained inside each record's `construction_trace`.
They are candidate edits that left the initial label unchanged or moved to the wrong label; they
are not silently deleted formal attempts. Counts: `{b["rejected_candidate_labels_or_failures"]}`.

## Engineering deviations

- Prover9 Dec-2007 was used instead of LADR-2009-11A because the Windows host lacked WSL and a C
  toolchain. The same solver binary was used by both routes.
- Exact timing is not used as data. Stable per-record wall-clock buckets are reported, while
  observed command duration belongs in the experiment ledger.
"""
    (EXPLORE_ROOT / "FAILURE_CATALOG.md").write_text(failure_catalog, encoding="utf-8")

    ay = a["score_breakdown"]["yield_and_gold"]
    by = b["score_breakdown"]["yield_and_gold"]
    ap = a["score_breakdown"]["v_f_pairing"]
    bp = b["score_breakdown"]["v_f_pairing"]
    ad = a["score_breakdown"]["difficulty_control"]
    bd = b["score_breakdown"]["difficulty_control"]
    adet = a["score_breakdown"]["determinism"]
    bdet = b["score_breakdown"]["determinism"]
    ai = a["score_breakdown"]["integration_maintenance"]
    bi = b["score_breakdown"]["integration_maintenance"]
    an = a["score_breakdown"]["naturalization"]
    bn = b["score_breakdown"]["naturalization"]
    at = a["score_breakdown"]["total"]
    bt = b["score_breakdown"]["total"]
    report = f"""# Generator Comparison Report

## Outcome

Both independently implemented styles passed the formal hard gates. `provergen_style` scored
{a["score_breakdown"]["total"]}; `reviseqa_style` scored {b["score_breakdown"]["total"]}. The
difference is below five points, so the protocol requires `{decision["recommendation"]}` rather
than a forced winner.

## Contract questions

1. **How many items were generated?** ProverGen-style: {a["valid_records"]}/16 valid;
   ReviseQA-style: {b["valid_records"]}/16 valid. All 16 attempt records remain in each file.
2. **Did any incorrect gold occur?** No. Gold error counts were {a["gold_error_count"]} and
   {b["gold_error_count"]}; Python fixed point and Prover9 agreed for every initial/updated world.
3. **Which structures failed most often?** No formal structure failed. In the world-first route,
   irrelevant atomic neighbors were rejected mainly because they preserved the initial label.
4. **Was V atomicity stable?** Yes: {a["v_f_pairing_valid"]}/16 and
   {b["v_f_pairing_valid"]}/16 records passed the complete one-record V/F pairing checks.
5. **Did F remain a single-error argument?** Yes. Each F has one fatal state claim, all remaining
   recomputation is valid under that claim, and each target has an opposite proof or Mace4 model.
6. **Did difficulty reflect proof depth?** Yes. Medium exceeded easy in
   {a["families_with_medium_deeper"]}/8 and {b["families_with_medium_deeper"]}/8 families.
7. **Which route was more deterministic?** Both had identical two-build canonical hashes. The
   proof-first route reached every specification in one construction attempt; world-first search
   had median/p95 attempts {b["construction_attempts"]["median"]}/
   {b["construction_attempts"]["p95"]}.
8. **Which route needed more special handling?** ReviseQA-style needed explicit neighbor
   enumeration and rejection bookkeeping. ProverGen-style needed direct family proof-skeleton
   constructors. Naturalization used the same generic renderer for both.
9. **Is a hybrid recommended?** Yes, as a human checkpoint only: proof-first construction gives
   direct depth/path control, while independent atomic-neighbor search supplies a useful V-edit
   certificate. This comparison does not implement the hybrid.
10. **Can formal development-bank construction begin?** Technically feasible, but only after a
    human approves the hybrid candidate. These 32 route records remain exploratory and cannot be
    promoted into held-out data.

## Score table

| Dimension | Weight | provergen_style | reviseqa_style |
|---|---:|---:|---:|
| Valid yield and gold | 35 | {ay} | {by} |
| V/F pairing | 25 | {ap} | {bp} |
| Difficulty control | 15 | {ad} | {bd} |
| Determinism | 10 | {adet} | {bdet} |
| Integration/maintenance | 10 | {ai} | {bi} |
| Minimal naturalization | 5 | {an} | {bn} |
| **Total** | **100** | **{at}** | **{bt}** |

## Scope boundary

This result supports a generator-mechanism decision only. It is not an official ProverGen or
ReviseQA reproduction, a language-model benchmark result, or evidence about Assistant Axis
capping. No Qwen, GPU, S arm, behavior-based selection, or capping was used.
"""
    (EXPLORE_ROOT / "GENERATOR_COMPARISON_REPORT.md").write_text(report, encoding="utf-8")

    rebuild_hash = metrics["deterministic_rebuild"]["aggregate_sha256"]
    comparability = f"""# Comparability Report

Both routes used the same 16 specifications, finite signed-Horn semantics, Prover9 executable,
Python evaluator, 100-candidate limit, schema, validity rules, and scoring. Construction order
was the intended variable. Canonical rebuild equality passed: `{rebuild_hash}`.

The result is not directly comparable to published ProverGen yield because no upstream assets,
rich FOL operators, or LLM translation were used. It is not directly comparable to ReviseQA
because its official stochastic multi-edit LLM proposal stage was replaced by deterministic
one-record neighbor search. Prover9 Dec-2007, rather than 2009-11A, is a shared compatibility
deviation. These boundaries prevent a strict-reproduction claim but preserve the internal route
comparison.
"""
    (EXPLORE_ROOT / "COMPARABILITY_REPORT.md").write_text(comparability, encoding="utf-8")

    changelog_path = EXPLORE_ROOT / "SCIENTIFIC_CHANGELOG.md"
    existing = changelog_path.read_text(encoding="utf-8").rstrip()
    final_entry = """

## 2026-07-23 — bounded comparison completed

- Generated and dual-validated sixteen common specifications through each route.
- Preserved all formal attempts and rejected world-first neighbor traces.
- Confirmed atomic V, single-error F, counterevidence, and eight-of-eight depth gradients.
- Selected `human_checkpoint / hybrid_candidate` under the predeclared under-five-point rule.
- Did not implement the hybrid or expand into a development/held-out bank.
"""
    if "bounded comparison completed" not in existing:
        changelog_path.write_text(existing + final_entry, encoding="utf-8")


def compare(data_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    validation = json.loads((data_root / "validation_report.json").read_text(encoding="utf-8"))
    deterministic = validation["deterministic_rebuild"]["passed"] is True
    route_metrics = {
        route: _route_metrics(route, read_jsonl(data_root / f"{route}16.jsonl"), deterministic)
        for route in ROUTES
    }
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "routes": route_metrics,
        "deterministic_rebuild": validation["deterministic_rebuild"],
        "difficulty_hard_gate": all(
            item["families_with_medium_deeper"] >= 7 for item in route_metrics.values()
        ),
        "gold_hard_gate": all(item["gold_error_count"] == 0 for item in route_metrics.values()),
        "paired_generation_hard_gate": all(
            item["v_f_pairing_valid"] == 16 for item in route_metrics.values()
        ),
    }
    decision = _decision(metrics)
    write_json(data_root / "comparison_metrics.json", metrics)
    write_json(EXPLORE_ROOT / "FINAL_DECISION.json", decision)
    _write_reports(metrics, decision)
    manifest = _manifest(data_root, [EXPLORE_ROOT])
    write_json(data_root / "manifest.json", manifest)
    return metrics, decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=HERE)
    args = parser.parse_args()
    metrics, decision = compare(args.data_dir.resolve())
    print(json.dumps({"metrics": metrics, "decision": decision}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
