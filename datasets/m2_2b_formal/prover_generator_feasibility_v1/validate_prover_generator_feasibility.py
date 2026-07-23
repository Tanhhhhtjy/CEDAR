"""Hard validator for the bounded prover-generator feasibility comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_prover_generator_feasibility import (
    FAMILY_BLUEPRINTS,
    MAX_CANDIDATES,
    ROUTES,
    SCHEMA_CONTRACT,
    VARIANTS,
    build,
)
from formal_horn import (
    SCHEMA_VERSION,
    apply_edit,
    canonical_json_file_sha256,
    canonical_jsonl_sha256,
    canonical_sha256,
    decisive_literal,
    edit_count,
    evaluate_fixed_point,
    file_sha256,
    label_from_evaluation,
    literal_key,
    read_jsonl,
    shortest_proof,
    validate_with_both,
    write_json,
)

HERE = Path(__file__).resolve().parent
TRANSITIONS = {
    "true_to_false",
    "false_to_true",
    "true_to_unknown",
    "unknown_to_true",
    "false_to_unknown",
    "unknown_to_false",
}


class ValidationFailure(AssertionError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationFailure(message)


def _canonical_hash(path: Path) -> str:
    if path.suffix == ".jsonl":
        return canonical_jsonl_sha256(path)
    return canonical_json_file_sha256(path)


def _canonical_text_hash(path: Path) -> str:
    normalized = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def audit_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    require(manifest_path.is_file(), "manifest.json is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    repo_root = root.parents[2]
    seen: set[str] = set()
    for entry in manifest["files"]:
        relative = entry["path"]
        require(relative not in seen, f"duplicate manifest path: {relative}")
        seen.add(relative)
        path = repo_root / relative if relative.startswith("explore_outputs/") else root / relative
        require(path.is_file(), f"manifest file missing: {relative}")
        require(file_sha256(path) == entry["raw_file_sha256"], f"raw hash mismatch: {relative}")
        if path.suffix == ".jsonl":
            canonical = canonical_jsonl_sha256(path)
        elif path.suffix == ".json":
            canonical = canonical_json_file_sha256(path)
        else:
            canonical = _canonical_text_hash(path)
        require(
            canonical == entry["canonical_content_sha256"],
            f"canonical hash mismatch: {relative}",
        )
    require(manifest["hard_gate_hash_type"] == "canonical_content_sha256", "wrong hash gate")
    require(manifest["raw_hash_is_provenance_only"] is True, "raw hash used as hard gate")
    return {
        "status": "passed",
        "manifest_entries": len(seen),
        "aggregate_canonical_sha256": canonical_sha256(
            {entry["path"]: entry["canonical_content_sha256"] for entry in manifest["files"]}
        ),
    }


def _file_hash_map(root: Path) -> dict[str, str]:
    files = sorted(
        path for path in root.rglob("*") if path.is_file() and path.suffix in {".json", ".jsonl"}
    )
    return {path.relative_to(root).as_posix(): _canonical_hash(path) for path in files}


def _validate_family_and_specs(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    families = read_jsonl(root / "family_blueprints8.jsonl")
    specs = read_jsonl(root / "common_symbolic_specs16.jsonl")
    require(len(families) == 8, "family count must be 8")
    require(len({item["family_id"] for item in families}) == 8, "family IDs must be unique")
    require(len(specs) == 16, "common specification count must be 16")
    require(
        Counter(spec["family_id"] for spec in specs)
        == Counter({f"PGF-{i:02d}": 2 for i in range(1, 9)}),
        "every family must have exactly two specifications",
    )
    require(
        all(
            {spec["variant"] for spec in specs if spec["family_id"] == family["family_id"]}
            == set(VARIANTS)
            for family in families
        ),
        "every family must have easy and medium variants",
    )
    transition_counts = Counter(spec["transition"] for spec in specs)
    require(set(transition_counts) == TRANSITIONS, "all six directed transitions must appear")
    require(
        all(transition_counts[item] >= 2 for item in TRANSITIONS), "each transition needs two bases"
    )
    for spec in specs:
        depth = spec["target_proof_depth"]
        if spec["variant"] == "easy":
            require(2 <= depth <= 3, f"{spec['spec_id']} easy depth out of range")
            require(2 <= spec["distractor_count"] <= 4, "easy distractor count out of range")
        else:
            require(4 <= depth <= 6, f"{spec['spec_id']} medium depth out of range")
            require(6 <= spec["distractor_count"] <= 10, "medium distractor count out of range")
            require(spec["state_reconstruction_steps"] >= 1, "medium needs reconstruction")
        require(spec["candidate_budget"] == MAX_CANDIDATES, "candidate budget changed")
    require(
        canonical_sha256(families) == canonical_sha256(FAMILY_BLUEPRINTS),
        "generated family blueprints differ from frozen builder catalog",
    )
    return families, specs


def _validate_counterevidence(root: Path, record: dict[str, Any]) -> None:
    evidence = record["arms"]["F"]["countermodel_or_opposite_proof"]
    if evidence["type"] == "opposite_proof":
        require(evidence["proof"] is not None, "opposite proof is missing")
        require(evidence["proof"]["depth"] >= 0, "opposite proof depth invalid")
    elif evidence["type"] == "mace4_countermodel":
        path = root / evidence["countermodel_file"]
        require(path.is_file(), "countermodel file missing")
        payload = json.loads(path.read_text(encoding="utf-8"))
        require(payload["model_found"] is True, "countermodel was not found")
        require(evidence["model_found"] is True, "record countermodel status is false")
    else:
        raise ValidationFailure(f"unknown counterevidence type: {evidence['type']}")


def _validate_record(
    root: Path,
    record: dict[str, Any],
    spec: dict[str, Any],
    family: dict[str, Any],
    *,
    rerun_solver: bool,
) -> dict[str, Any]:
    require(record["schema_version"] == SCHEMA_VERSION, "schema version mismatch")
    require(record["route"] in ROUTES, "route name invalid")
    require(record["candidate_budget"] == MAX_CANDIDATES, "candidate budget mismatch")
    require(0 <= record["construction_attempts"] <= MAX_CANDIDATES, "attempt count invalid")
    if record["status"] == "failed":
        require(bool(record["failure_stage"]), "failed record lacks failure stage")
        require(bool(record["failure_reason"]), "failed record lacks failure reason")
        return {"valid": False}

    require(record["status"] == "valid", "status must be valid or failed")
    require(record["failure_stage"] is None, "valid record has failure stage")
    require(record["failure_reason"] is None, "valid record has failure reason")
    require(record["spec_id"] == spec["spec_id"], "record/spec ID mismatch")
    require(record["family_id"] == family["family_id"], "record/family mismatch")
    require(record["transition"] == spec["transition"], "transition mismatch")
    require(record["structure_signature"] == family["structure_signature"], "structure changed")
    require(record["arms"].keys() == {"N", "R", "V", "F"}, "arms must be exactly N/R/V/F")
    require("S" not in record["arms"], "S is prohibited")

    initial_world = record["symbolic_world_initial"]
    updated_world = record["symbolic_world_updated"]
    query = record["query"]
    edit = record["v_edit"]
    require(edit_count(edit) == 1 and edit["edit_count"] == 1, "V must edit one record")
    require(
        apply_edit(initial_world, edit) == updated_world, "stored updated world is not V(initial)"
    )
    require(record["arms"]["N"]["world_delta"] == [], "N changes the world")
    require(record["arms"]["R"]["world_delta"] == [], "R changes the world")
    require(record["arms"]["F"]["world_delta"] == [], "F changes the world")
    require(record["arms"]["V"]["world_delta"] == [edit], "V arm edit mismatch")
    require(
        "No new information is provided." not in record["arms"]["N"]["instruction"],
        "prohibited N sentence appears",
    )
    require(not record["arms"]["R"]["mentions_target"], "R mentions target")
    require(
        not record["arms"]["R"]["contains_evidence_or_social_cue"],
        "R contains evidence/social cue",
    )

    initial_eval = evaluate_fixed_point(initial_world)
    updated_eval = evaluate_fixed_point(updated_world)
    initial_label = label_from_evaluation(initial_eval, query)
    updated_label = label_from_evaluation(updated_eval, query)
    require(initial_label == record["initial_label"], "fixed-point initial label mismatch")
    require(updated_label == record["updated_label"], "fixed-point updated label mismatch")
    require(initial_label != "invalid" and updated_label != "invalid", "invalid world admitted")
    require(initial_label != updated_label, "V does not change the gold")
    require(f"{initial_label}_to_{updated_label}" == spec["transition"], "transition false")

    if rerun_solver:
        initial_dual = validate_with_both(initial_world, query)
        updated_dual = validate_with_both(updated_world, query)
        require(initial_dual["agreement"], "initial solver disagreement")
        require(updated_dual["agreement"], "updated solver disagreement")
        require(initial_dual["python_label"] == initial_label, "initial rerun label mismatch")
        require(updated_dual["python_label"] == updated_label, "updated rerun label mismatch")
    require(record["solver_validation"]["initial"]["agreement"], "stored initial disagreement")
    require(record["solver_validation"]["updated"]["agreement"], "stored updated disagreement")

    require(record["arms"]["N"]["gold"] == initial_label, "N gold changed")
    require(record["arms"]["R"]["gold"] == initial_label, "R gold changed")
    require(record["arms"]["F"]["gold"] == initial_label, "F gold changed")
    require(record["arms"]["V"]["gold"] == updated_label, "V gold mismatch")
    require(
        record["arms"]["V"]["target"] == record["arms"]["F"]["induced_target"],
        "F target differs from V target",
    )
    require(record["updated_target"] == record["arms"]["V"]["target"], "target mismatch")

    target_literal = decisive_literal(query, updated_label)
    if target_literal is not None:
        without_edit = evaluate_fixed_point(initial_world)
        require(
            literal_key(target_literal) not in without_edit["closure"],
            "updated target has an undisclosed path without V",
        )
    else:
        initial_literal = decisive_literal(query, initial_label)
        require(shortest_proof(initial_eval, initial_literal) is not None, "initial proof missing")

    f_arm = record["arms"]["F"]
    steps = f_arm["candidate_reasoning_steps"]
    invalid_steps = [step for step in steps if not step["valid"]]
    require(len(invalid_steps) == 1, "F must have exactly one invalid step")
    require(invalid_steps[0]["step_id"] == f_arm["fatal_error_step"], "fatal step mismatch")
    require(f_arm["error_class"] == family["fallacy_error_class"], "F class changed")
    require(f_arm["second_failure_point_absent"] is True, "second F failure exists")
    require(not f_arm["contains_social_cue"], "F contains social cue")
    require(
        f_arm["repair_witness"]["edit"] == edit
        and f_arm["repair_witness"]["repaired_label"] == updated_label
        and f_arm["repair_witness"]["target_argument_succeeds"],
        "F repair witness invalid",
    )
    _validate_counterevidence(root, record)

    difficulty = record["difficulty"]
    require(
        difficulty["actual_shortest_proof_depth"] == spec["target_proof_depth"],
        "actual proof depth misses contract",
    )
    require(
        difficulty["distractor_count"] == spec["distractor_count"],
        "distractor count misses contract",
    )
    if spec["variant"] == "easy":
        require(2 <= difficulty["actual_shortest_proof_depth"] <= 3, "easy depth invalid")
    else:
        require(4 <= difficulty["actual_shortest_proof_depth"] <= 6, "medium depth invalid")
        require(difficulty["state_reconstruction_steps"] >= 1, "medium reconstruction absent")
    if record["family_id"] == "PGF-07":
        require(difficulty["proof_path_count"] >= 2, "two-support-path family lacks two proofs")

    for state in ("initial", "updated"):
        for polarity in ("query", "opposite"):
            trace_file = root / record["solver_validation"][state][f"{polarity}_trace_file"]
            require(trace_file.is_file(), f"solver trace missing: {trace_file.name}")
    require(all(record["hard_checks"].values()), "stored hard check is false")
    return {
        "valid": True,
        "proof_depth": difficulty["actual_shortest_proof_depth"],
        "attempts": record["construction_attempts"],
        "time_bucket": record["wall_clock_time_bucket"],
    }


def _validate_routes(
    root: Path,
    families: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    *,
    rerun_solver: bool,
) -> dict[str, Any]:
    by_family = {item["family_id"]: item for item in families}
    by_spec = {item["spec_id"]: item for item in specs}
    result: dict[str, Any] = {}
    depths: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(dict))
    for route in ROUTES:
        rows = read_jsonl(root / f"{route}16.jsonl")
        require(len(rows) == 16, f"{route} must retain exactly 16 attempts")
        require(len({row["spec_id"] for row in rows}) == 16, f"{route} duplicate/missing specs")
        require(set(row["spec_id"] for row in rows) == set(by_spec), "route spec set mismatch")
        summaries = []
        for row in rows:
            summary = _validate_record(
                root,
                row,
                by_spec[row["spec_id"]],
                by_family[row["family_id"]],
                rerun_solver=rerun_solver,
            )
            summaries.append(summary)
            if summary["valid"]:
                depths[route][row["family_id"]][row["variant"]] = summary["proof_depth"]
        valid_count = sum(item["valid"] for item in summaries)
        gradient_count = sum(
            values.get("medium", -1) > values.get("easy", 999) for values in depths[route].values()
        )
        require(gradient_count >= 7, f"{route} difficulty gradient fails")
        result[route] = {
            "attempt_records": len(rows),
            "valid_records": valid_count,
            "failed_records": len(rows) - valid_count,
            "families_with_medium_deeper": gradient_count,
            "attempt_distribution": dict(
                sorted(Counter(item["attempts"] for item in summaries if item["valid"]).items())
            ),
            "time_bucket_distribution": dict(
                sorted(Counter(item["time_bucket"] for item in summaries if item["valid"]).items())
            ),
        }
    return result


def _validate_sentinels(root: Path) -> None:
    rows = read_jsonl(root / "sentinel_results4.jsonl")
    require(len(rows) == 4, "sentinel result count must be four")
    require(all(row["status"] == "valid" for row in rows), "sentinel gate failed")
    require(
        Counter(row["route"] for row in rows) == Counter({route: 2 for route in ROUTES}),
        "sentinel route imbalance",
    )
    freeze = json.loads((root / "sentinel_freeze.json").read_text(encoding="utf-8"))
    require(freeze["sentinel_gate_passed"], "sentinel freeze not passed")
    require(
        freeze["family_card_canonical_sha256"]
        == canonical_jsonl_sha256(root / "family_blueprints8.jsonl"),
        "family hash changed after sentinel freeze",
    )
    require(
        freeze["schema_canonical_sha256"] == canonical_sha256(SCHEMA_CONTRACT),
        "schema hash changed after sentinel freeze",
    )


def _validate_naturalization(root: Path) -> None:
    rows = read_jsonl(root / "naturalization_sentinel8.jsonl")
    require(len(rows) == 8, "naturalization sentinel count must be eight")
    require(
        Counter(row["route"] for row in rows) == Counter({route: 4 for route in ROUTES}),
        "naturalization route imbalance",
    )
    require(
        Counter(row["variant"] for row in rows) == Counter({"easy": 4, "medium": 4}),
        "naturalization difficulty imbalance",
    )
    for row in rows:
        require(row["gold_hidden_from_text"], "naturalization exposes declared gold")
        require("No new information is provided." not in json.dumps(row), "prohibited cue leaked")
        require(row["reverse_map"]["fatal_error_step"] == 2, "F error lost in rendering")


def _validate_no_absolute_paths(root: Path) -> None:
    windows_path = re.compile(r"(?<![A-Za-z])[A-Za-z]:[\\/](?![\\/])")
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".md"}:
            continue
        text = path.read_text(encoding="utf-8")
        require(not windows_path.search(text), f"absolute Windows path in {path.name}")
        require(
            "/home/" not in text and "/Users/" not in text, f"absolute POSIX path in {path.name}"
        )


def _deterministic_rebuild() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="cedar_pgf_rebuild_a_") as first_dir:
        with tempfile.TemporaryDirectory(prefix="cedar_pgf_rebuild_b_") as second_dir:
            first = Path(first_dir)
            second = Path(second_dir)
            build(first)
            build(second)
            first_hashes = _file_hash_map(first)
            second_hashes = _file_hash_map(second)
            require(first_hashes == second_hashes, "two complete builds have different hashes")
            return {
                "passed": True,
                "file_count": len(first_hashes),
                "aggregate_sha256": canonical_sha256(first_hashes),
            }


def validate(root: Path, *, rerun_solver: bool, rebuild_check: bool) -> dict[str, Any]:
    families, specs = _validate_family_and_specs(root)
    _validate_sentinels(root)
    route_summary = _validate_routes(root, families, specs, rerun_solver=rerun_solver)
    _validate_naturalization(root)
    _validate_no_absolute_paths(root)
    rebuild = _deterministic_rebuild() if rebuild_check else {"passed": None, "skipped": True}
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "families": len(families),
        "common_specifications": len(specs),
        "sentinels": 4,
        "routes": route_summary,
        "naturalization_sentinels": 8,
        "dual_solver_rerun": rerun_solver,
        "deterministic_rebuild": rebuild,
    }
    write_json(root / "validation_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=HERE)
    parser.add_argument("--skip-solver-rerun", action="store_true")
    parser.add_argument("--skip-rebuild", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    if args.manifest_only:
        print(json.dumps(audit_manifest(args.data_dir.resolve()), ensure_ascii=False, indent=2))
        return
    report = validate(
        args.data_dir.resolve(),
        rerun_solver=not args.skip_solver_rerun,
        rebuild_check=not args.skip_rebuild,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
