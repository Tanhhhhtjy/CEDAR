#!/usr/bin/env python3
"""Audit frozen C1/V7 smoke and formal runs without changing model outputs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cedar_m2.checkpoint import atomic_write_json, file_sha256

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "m2" / "c1_v7_heldout_confirmation"
LAYERS = tuple(range(46, 54))
LOGIT_TOLERANCE = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "formal", "all"), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_phase(root: Path, phase: str, expected_rows: int) -> dict[str, Any]:
    directory = root / phase
    manifest_path = directory / "manifest.json"
    runtime_path = directory / "runtime.json"
    results_path = directory / "results.jsonl"
    record_runtime_path = directory / "record_runtime.jsonl"
    for path in (manifest_path, runtime_path, results_path, record_runtime_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing C1/V7 phase artifact: {path}")
    manifest = read_json(manifest_path)
    runtime = read_json(runtime_path)
    results = read_jsonl(results_path)
    record_runtime = read_jsonl(record_runtime_path)
    if len(results) != expected_rows or len(record_runtime) != expected_rows:
        raise ValueError(
            f"{phase}: expected {expected_rows} results/checkpoints, "
            f"found {len(results)}/{len(record_runtime)}"
        )
    if runtime.get("complete_results") != expected_rows:
        raise ValueError(f"{phase}: runtime is not complete")
    if runtime.get("results_sha256") != file_sha256(results_path):
        raise ValueError(f"{phase}: results SHA-256 mismatch")
    if runtime.get("record_runtime_sha256") != file_sha256(record_runtime_path):
        raise ValueError(f"{phase}: runtime-record SHA-256 mismatch")
    configuration = manifest.get("configuration", {})
    if configuration.get("phase") != phase:
        raise ValueError(f"{phase}: manifest phase mismatch")
    if configuration.get("generation", {}).get("content_retry") is not False:
        raise ValueError(f"{phase}: content retry must be disabled")
    if configuration.get("generation", {}).get("answer_repair") is not False:
        raise ValueError(f"{phase}: answer repair must be disabled")
    if configuration.get("review_deviation") is None:
        raise ValueError(f"{phase}: missing single-review deviation disclosure")
    return {
        "phase": phase,
        "manifest": manifest,
        "runtime": runtime,
        "results": results,
        "record_runtime": record_runtime,
        "hashes": {
            "manifest": file_sha256(manifest_path),
            "runtime": file_sha256(runtime_path),
            "results": file_sha256(results_path),
            "record_runtime": file_sha256(record_runtime_path),
        },
    }


def by_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed = {str(row.get("id")): row for row in rows}
    if len(indexed) != len(rows) or "None" in indexed:
        raise ValueError("Results need unique non-null IDs")
    return indexed


def abc_logits(row: Mapping[str, Any], field: str) -> dict[str, float] | None:
    payload = row.get(field)
    if not isinstance(payload, Mapping):
        return None
    values = payload.get("logits")
    if not isinstance(values, Mapping) or set(values) != {"A", "B", "C"}:
        return None
    return {choice: float(values[choice]) for choice in ("A", "B", "C")}


def max_logit_difference(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> float:
    left_by_id = by_id(left)
    right_by_id = by_id(right)
    if set(left_by_id) != set(right_by_id):
        raise ValueError("Smoke paths have different IDs")
    differences: list[float] = []
    for item_id in sorted(left_by_id):
        for field in ("actual_answer_logits", "scaffold_answer_logits"):
            left_logits = abc_logits(left_by_id[item_id], field)
            right_logits = abc_logits(right_by_id[item_id], field)
            if left_logits is None and right_logits is None:
                continue
            if left_logits is None or right_logits is None:
                raise ValueError(f"{item_id}: asymmetric {field} availability")
            differences.extend(
                abs(left_logits[choice] - right_logits[choice])
                for choice in ("A", "B", "C")
            )
    return max(differences, default=0.0)


def hook_clean(row: Mapping[str, Any]) -> bool:
    return row.get("hook_counts_before") == row.get("hook_counts_after")


def p25_cap_stats_complete(row: Mapping[str, Any]) -> bool:
    stats = row.get("cap_stats")
    if not isinstance(stats, Mapping):
        return False
    by_layer = stats.get("by_layer")
    if not isinstance(by_layer, Mapping):
        return False
    layer_keys = {str(layer) for layer in LAYERS}
    return (
        set(by_layer) == layer_keys
        and all(
            isinstance(by_layer[str(layer)], Mapping)
            and int(by_layer[str(layer)].get("total_positions", 0)) > 0
            for layer in LAYERS
        )
        and all(
            float(by_layer[str(layer)].get("projection_correction_sum", -1.0))
            >= 0.0
            for layer in LAYERS
        )
    )


def audit_smoke(root: Path) -> dict[str, Any]:
    nohook = load_phase(root, "smoke-nohook", 8)
    monitor = load_phase(root, "smoke-monitor", 8)
    capped = load_phase(root, "smoke-p25", 4)
    nohook_by_id = by_id(nohook["results"])
    monitor_by_id = by_id(monitor["results"])
    paired_ids = sorted(nohook_by_id)
    max_difference = max_logit_difference(nohook["results"], monitor["results"])
    paired_checks = {
        "same_ids": set(nohook_by_id) == set(monitor_by_id),
        "same_responses": all(
            nohook_by_id[item_id].get("response") == monitor_by_id[item_id].get("response")
            for item_id in paired_ids
        ),
        "same_prompt_hashes": all(
            nohook_by_id[item_id].get("prompt_sha256")
            == monitor_by_id[item_id].get("prompt_sha256")
            and nohook_by_id[item_id].get("prompt_token_ids_sha256")
            == monitor_by_id[item_id].get("prompt_token_ids_sha256")
            for item_id in paired_ids
        ),
        "logit_difference_le_1e_4": max_difference <= LOGIT_TOLERANCE,
        "nohook_has_no_telemetry": all(
            row.get("axis_telemetry") is None and row.get("cap_stats") is None
            for row in nohook["results"]
        ),
        "monitor_has_telemetry": all(
            isinstance(row.get("axis_telemetry"), Mapping)
            for row in monitor["results"]
        ),
        "all_hooks_clean": all(
            hook_clean(row)
            for row in [*nohook["results"], *monitor["results"], *capped["results"]]
        ),
        "thinking_integrity": all(
            row.get("thinking_integrity", {}).get("passed") is True
            for row in [*nohook["results"], *monitor["results"], *capped["results"]]
        ),
    }
    paired_checks["p25_stats_complete_layers_46_53"] = all(
        p25_cap_stats_complete(row) for row in capped["results"]
    )
    failures = [name for name, passed in paired_checks.items() if not passed]
    return {
        "status": "passed" if not failures else "failed",
        "checks": paired_checks,
        "failures": failures,
        "max_abc_logit_difference": max_difference,
        "phase_hashes": {
            phase["phase"]: phase["hashes"] for phase in (nohook, monitor, capped)
        },
        "behavior_not_used_as_gate": True,
    }


def comparable_configuration(configuration: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": configuration.get("kind"),
        "scope": configuration.get("scope"),
        "selection": configuration.get("selection"),
        "seed": configuration.get("seed"),
        "dataset": configuration.get("dataset"),
        "model": configuration.get("model"),
        "generation": configuration.get("generation"),
        "axis_telemetry": configuration.get("axis_telemetry"),
        "review_deviation": configuration.get("review_deviation"),
        "git_commit": configuration.get("git", {}).get("commit"),
    }


def audit_formal(root: Path) -> dict[str, Any]:
    uncapped = load_phase(root, "formal-uncapped", 192)
    capped = load_phase(root, "formal-p25", 192)
    left = by_id(uncapped["results"])
    right = by_id(capped["results"])
    ids = sorted(left)
    configuration_left = uncapped["manifest"]["configuration"]
    configuration_right = capped["manifest"]["configuration"]
    checks = {
        "same_ids": set(left) == set(right),
        "same_order": [row["id"] for row in uncapped["results"]]
        == [row["id"] for row in capped["results"]],
        "same_prompts": all(
            left[item_id].get("prompt_sha256") == right[item_id].get("prompt_sha256")
            and left[item_id].get("prompt_token_ids_sha256")
            == right[item_id].get("prompt_token_ids_sha256")
            for item_id in ids
        ),
        "same_row_metadata": all(
            all(
                left[item_id].get(field) == right[item_id].get(field)
                for field in (
                    "base_id",
                    "family_id",
                    "arm",
                    "initial_answer",
                    "target_answer",
                    "distractor_answer",
                    "gold_answer",
                )
            )
            for item_id in ids
        ),
        "conditions_exact": configuration_left.get("condition") == "uncapped"
        and configuration_right.get("condition") == "p25",
        "configuration_equal_except_condition_and_phase": comparable_configuration(
            configuration_left
        )
        == comparable_configuration(configuration_right),
        "all_hooks_clean": all(
            hook_clean(row) for row in [*uncapped["results"], *capped["results"]]
        ),
        "all_telemetry_present": all(
            isinstance(row.get("axis_telemetry"), Mapping)
            for row in [*uncapped["results"], *capped["results"]]
        ),
        "uncapped_has_no_cap_stats": all(
            row.get("cap_stats") is None for row in uncapped["results"]
        ),
        "p25_has_cap_stats": all(
            isinstance(row.get("cap_stats"), Mapping) for row in capped["results"]
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "status": "passed" if not failures else "failed",
        "checks": checks,
        "failures": failures,
        "phase_hashes": {
            phase["phase"]: phase["hashes"] for phase in (uncapped, capped)
        },
        "invalid_or_truncated_retained": True,
        "content_retries": 0,
        "review_deviation_disclosed": True,
    }


def main() -> None:
    args = parse_args()
    reports: dict[str, Any] = {}
    if args.stage in {"smoke", "all"}:
        reports["smoke"] = audit_smoke(args.output_root)
        atomic_write_json(args.output_root / "SMOKE_AUDIT.json", reports["smoke"])
    if args.stage in {"formal", "all"}:
        reports["formal"] = audit_formal(args.output_root)
        atomic_write_json(
            args.output_root / "CROSS_CONDITION_AUDIT.json", reports["formal"]
        )
    print(json.dumps(reports, ensure_ascii=False, indent=2, sort_keys=True))
    if any(report["status"] != "passed" for report in reports.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
