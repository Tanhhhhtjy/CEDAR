#!/usr/bin/env python3
"""Audit C2/V8 smoke and paired formal artifacts without model execution."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from cedar_m2.checkpoint import atomic_write_json, file_sha256

try:
    from scripts.c2_v8_portable_hashes import canonical_text_sha256
except ModuleNotFoundError:
    from c2_v8_portable_hashes import canonical_text_sha256

try:
    from scripts.run_c2_v8_natural_confirmation import (
        classify,
        lenient_answer,
        strict_answer,
        thinking_integrity,
    )
except ModuleNotFoundError:
    from run_c2_v8_natural_confirmation import (
        classify,
        lenient_answer,
        strict_answer,
        thinking_integrity,
    )

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "m2" / "c2_v8_natural_social_confirmation"
LAYERS = tuple(range(46, 54))
LABELS = ("A", "B", "C", "D")
LABEL_TOKEN_IDS = {"A": 32, "B": 33, "C": 34, "D": 35}
LOGIT_TOLERANCE = 1e-4
IDENTITY_FIELDS = (
    "base_id",
    "source_id",
    "domain",
    "difficulty",
    "normalized_stem_id",
    "exact_mcq_id",
    "template_id",
    "arm",
    "initial_answer",
    "target_answer",
    "distractor_answers",
    "gold_answer",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", choices=("direct", "free", "all"), required=True)
    parser.add_argument("--stage", choices=("smoke", "formal", "all"), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSON objects: {path}")
    return rows


def load_phase(root: Path, phase: str, expected_rows: int) -> dict[str, Any]:
    directory = root / phase
    paths = {
        "manifest": directory / "manifest.json",
        "runtime": directory / "runtime.json",
        "results": directory / "results.jsonl",
        "record_runtime": directory / "record_runtime.jsonl",
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing C2/V8 phase artifact: {path}")
    manifest = read_json(paths["manifest"])
    runtime = read_json(paths["runtime"])
    results = read_jsonl(paths["results"])
    envelopes = read_jsonl(paths["record_runtime"])
    if len(results) != expected_rows or len(envelopes) != expected_rows:
        raise ValueError(
            f"{phase}: expected {expected_rows} results/checkpoints, "
            f"found {len(results)}/{len(envelopes)}"
        )
    if runtime.get("complete_results") != expected_rows or runtime.get("content_retries") != 0:
        raise ValueError(f"{phase}: runtime is incomplete or reports content retries")
    if runtime.get("results_canonical_text_sha256") != canonical_text_sha256(
        paths["results"]
    ):
        raise ValueError(f"{phase}: results hash mismatch")
    if runtime.get("record_runtime_canonical_text_sha256") != canonical_text_sha256(
        paths["record_runtime"]
    ):
        raise ValueError(f"{phase}: checkpoint-materialization hash mismatch")
    for index, (result, envelope) in enumerate(zip(results, envelopes, strict=True)):
        if envelope.get("index") != index or envelope.get("result") != result:
            raise ValueError(f"{phase}: checkpoint/result mismatch at row {index}")
    configuration = manifest.get("configuration", {})
    if configuration.get("phase") != phase:
        raise ValueError(f"{phase}: manifest phase mismatch")
    if configuration.get("measurement", {}).get("content_retry") is not False:
        raise ValueError(f"{phase}: content retry is not frozen off")
    if configuration.get("measurement", {}).get("answer_repair") is not False:
        raise ValueError(f"{phase}: answer repair is not frozen off")
    human = configuration.get("human_review", {})
    if human.get("status") != "passed" or human.get("gpu_authorized") is not True:
        raise ValueError(f"{phase}: human review authorization is absent")
    if any(
        row.get("phase") != phase
        or row.get("track") != configuration.get("track")
        or row.get("condition") != configuration.get("condition")
        or row.get("instrumentation") != configuration.get("instrumentation")
        for row in results
    ):
        raise ValueError(f"{phase}: result metadata does not match the run manifest")
    return {
        "phase": phase,
        "manifest": manifest,
        "runtime": runtime,
        "results": results,
        "hashes": {name: canonical_text_sha256(path) for name, path in paths.items()},
        "raw_hashes_diagnostic": {name: file_sha256(path) for name, path in paths.items()},
    }


def by_id(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    output = {str(row.get("id")): row for row in rows}
    if len(output) != len(rows) or "None" in output:
        raise ValueError("Results require unique, non-null IDs")
    return output


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def option_logits(record: Mapping[str, Any], field: str) -> dict[str, float] | None:
    payload = record.get(field)
    if not isinstance(payload, Mapping):
        return None
    values = payload.get("logits")
    if not isinstance(values, Mapping) or set(values) != set(LABELS):
        return None
    output = {label: float(values[label]) for label in LABELS}
    return output if all(finite(value) for value in output.values()) else None


def max_logit_difference(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> float:
    left_index = by_id(left)
    right_index = by_id(right)
    if set(left_index) != set(right_index):
        raise ValueError("Paired smoke paths have different IDs")
    differences: list[float] = []
    for item_id in sorted(left_index):
        for field in fields:
            a = option_logits(left_index[item_id], field)
            b = option_logits(right_index[item_id], field)
            if a is None and b is None:
                continue
            if a is None or b is None:
                raise ValueError(f"{item_id}: asymmetric {field} availability")
            differences.extend(abs(a[label] - b[label]) for label in LABELS)
    return max(differences, default=0.0)


def hook_clean(record: Mapping[str, Any]) -> bool:
    return record.get("hook_counts_before") == record.get("hook_counts_after")


def diagnostics_valid(record: Mapping[str, Any], field: str) -> bool:
    payload = record.get(field)
    if payload is None:
        return field != "answer_logits"
    if not isinstance(payload, Mapping) or option_logits(record, field) is None:
        return False
    logits = {label: float(payload["logits"][label]) for label in LABELS}
    initial = str(record.get("initial_answer"))
    target = str(record.get("target_answer"))
    probabilities = payload.get("full_vocab_probabilities", {})
    conditional = payload.get("conditional_probabilities", {})
    if set(probabilities) != set(LABELS) or set(conditional) != set(LABELS):
        return False
    if initial not in LABELS or target not in LABELS:
        return False
    return (
        payload.get("option_argmax") in LABELS
        and payload.get("option_argmax") == max(LABELS, key=lambda label: logits[label])
        and isinstance(payload.get("top_token_is_option"), bool)
        and finite(payload.get("option_mass"))
        and 0.0 <= float(payload["option_mass"]) <= 1.000001
        and math.isclose(
            float(payload["option_mass"]),
            sum(float(probabilities[label]) for label in LABELS),
            rel_tol=1e-7,
            abs_tol=1e-7,
        )
        and math.isclose(
            sum(float(conditional[label]) for label in LABELS),
            1.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
        and math.isclose(
            float(payload["target_minus_initial"]),
            logits[target] - logits[initial],
            rel_tol=1e-7,
            abs_tol=1e-7,
        )
        and finite(payload.get("full_vocab_entropy"))
    )


def telemetry_valid(record: Mapping[str, Any], track: str) -> bool:
    telemetry = record.get("axis_telemetry")
    if not isinstance(telemetry, Mapping):
        return False
    phases = telemetry.get("phases")
    if not isinstance(phases, Mapping):
        return False
    expected_phases = (
        {"conversation_prefill": int(record["conversation_prefill_tokens"]),
         "assistant_prefix_context": int(record["assistant_prefix_context_tokens"]),
         "answer_step": int(record["answer_step_tokens"])}
        if track == "direct"
        else {"prefill": int(record["prompt_tokens"]),
              "decode": max(int(record["response_tokens"]) - 1, 0)}
    )
    if set(phases) != set(expected_phases):
        return False
    for phase, expected_count in expected_phases.items():
        payload = phases.get(phase, {})
        projections = payload.get("assistant_axis_projection", {})
        exposure = payload.get("reference_cap_exposure", {}).get("p25", {})
        if set(projections) != {str(layer) for layer in LAYERS}:
            return False
        if set(exposure) != {str(layer) for layer in LAYERS}:
            return False
        for layer in LAYERS:
            projection = projections[str(layer)]
            item = exposure[str(layer)]
            numeric = (
                projection.get("mean"),
                projection.get("std"),
                projection.get("min"),
                projection.get("max"),
                item.get("trigger_rate"),
                item.get("correction_sum"),
                item.get("mean_correction_all_positions"),
            )
            if int(projection.get("count", -1)) != expected_count:
                return False
            if int(item.get("count", -1)) != expected_count:
                return False
            if not all(finite(value) for value in numeric):
                return False
    return True


def p25_stats_valid(record: Mapping[str, Any], track: str) -> bool:
    stats = record.get("cap_stats")
    split = record.get("actual_cap_stats_by_phase")
    if not isinstance(stats, Mapping) or not isinstance(split, Mapping):
        return False
    by_layer = stats.get("by_layer", {})
    if set(by_layer) != {str(layer) for layer in LAYERS}:
        return False
    expected_per_layer = (
        int(record["prompt_tokens"])
        if track == "direct"
        else int(record["prompt_tokens"]) + max(int(record["response_tokens"]) - 1, 0)
    )
    for layer in LAYERS:
        item = by_layer[str(layer)]
        if int(item.get("total_positions", -1)) != expected_per_layer:
            return False
        if not all(
            finite(item.get(field))
            for field in (
                "trigger_rate",
                "projection_correction_sum",
                "mean_projection_correction",
            )
        ):
            return False
        phase_total = phase_triggered = 0
        phase_correction = 0.0
        expected_phases = (
            ("conversation_prefill", "assistant_prefix_context", "answer_step")
            if track == "direct"
            else ("prefill", "decode")
        )
        if set(split) != set(expected_phases):
            return False
        for phase in expected_phases:
            phase_item = split.get(phase, {}).get(str(layer), {})
            phase_total += int(phase_item.get("total_positions", -10_000))
            phase_triggered += int(phase_item.get("triggered_positions", -10_000))
            phase_correction += float(phase_item.get("projection_correction_sum", math.nan))
        if (
            phase_total != int(item["total_positions"])
            or phase_triggered != int(item["triggered_positions"])
            or not math.isclose(
                phase_correction,
                float(item["projection_correction_sum"]),
                rel_tol=1e-5,
                abs_tol=1e-3,
            )
        ):
            return False
    return True


def parser_fields_valid(record: Mapping[str, Any]) -> bool:
    categories = {"initial", "target", "other_wrong", "invalid_or_truncated"}
    response = str(record.get("response", ""))
    truncated = record.get("truncated") is True
    strict, strict_error = strict_answer(response, truncated=truncated)
    lenient, lenient_error = lenient_answer(response)
    expected_strict_category = classify(record, strict, truncated)
    expected_lenient_category = classify(record, lenient, truncated)
    boundary = record.get("boundary", {})
    if boundary.get("valid_answer_choice") is True and not isinstance(
        record.get("actual_answer_logits"), Mapping
    ):
        return False
    if boundary.get("valid_thinking_boundary") is True and not isinstance(
        record.get("scaffold_answer_logits"), Mapping
    ):
        return False
    return (
        record.get("strict_category") in categories
        and record.get("lenient_category") in categories
        and record.get("strict_answer") == strict
        and record.get("strict_parse_error") == strict_error
        and record.get("strict_category") == expected_strict_category
        and record.get("lenient_answer") == lenient
        and record.get("lenient_parse_error") == lenient_error
        and record.get("lenient_category") == expected_lenient_category
        and record.get("thinking_integrity") == thinking_integrity(response)
        and record.get("finish_reason") in {"length", "eos_token", "other_stop"}
        and isinstance(record.get("thinking_integrity"), Mapping)
        and isinstance(boundary, Mapping)
    )


def same_pair_inputs(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        all(left.get(field) == right.get(field) for field in IDENTITY_FIELDS)
        and left.get("prompt") == right.get("prompt")
        and left.get("prompt_sha256") == right.get("prompt_sha256")
        and left.get("prompt_token_ids_sha256") == right.get("prompt_token_ids_sha256")
    )


def comparable_configuration(configuration: Mapping[str, Any]) -> dict[str, Any]:
    measurement = dict(configuration.get("measurement", {}))
    return {
        "kind": configuration.get("kind"),
        "scope": configuration.get("scope"),
        "track": configuration.get("track"),
        "selection": configuration.get("selection"),
        "seed": configuration.get("seed"),
        "dataset": configuration.get("dataset"),
        "model": configuration.get("model"),
        "measurement": measurement,
        "axis_telemetry": configuration.get("axis_telemetry"),
        "capping": configuration.get("capping"),
        "human_review": configuration.get("human_review"),
        "git_commit": configuration.get("git", {}).get("commit"),
    }


def audit_smoke(root: Path, track: str) -> dict[str, Any]:
    expected = 24
    nohook = load_phase(root, f"smoke-{track}-nohook", expected)
    monitor = load_phase(root, f"smoke-{track}-monitor", expected)
    capped = load_phase(root, f"smoke-{track}-p25", expected)
    nohook_index = by_id(nohook["results"])
    monitor_index = by_id(monitor["results"])
    capped_index = by_id(capped["results"])
    ids = [row["id"] for row in nohook["results"]]
    fields = ("answer_logits",) if track == "direct" else (
        "actual_answer_logits",
        "scaffold_answer_logits",
    )
    max_difference = max_logit_difference(nohook["results"], monitor["results"], fields)
    checks = {
        "same_ids_and_order": (
            ids == [row["id"] for row in monitor["results"]]
            and ids == [row["id"] for row in capped["results"]]
            and set(nohook_index) == set(monitor_index) == set(capped_index)
        ),
        "same_inputs_nohook_monitor": all(
            same_pair_inputs(nohook_index[item_id], monitor_index[item_id]) for item_id in ids
        ),
        "nohook_monitor_output_identity": (
            all(
                nohook_index[item_id].get("response") == monitor_index[item_id].get("response")
                for item_id in ids
            )
            if track == "free"
            else True
        ),
        "nohook_monitor_logit_difference_le_1e_4": max_difference <= LOGIT_TOLERANCE,
        "candidate_token_ids_frozen": all(
            row.get("candidate_token_ids", LABEL_TOKEN_IDS) == LABEL_TOKEN_IDS
            for row in [*nohook["results"], *monitor["results"], *capped["results"]]
        ),
        "nohook_has_no_telemetry_or_cap": all(
            row.get("axis_telemetry") is None
            and row.get("cap_stats") is None
            and row.get("actual_cap_stats_by_phase") is None
            for row in nohook["results"]
        ),
        "monitor_telemetry_complete": all(
            telemetry_valid(row, track) and row.get("cap_stats") is None
            for row in monitor["results"]
        ),
        "p25_telemetry_and_actual_stats_complete": all(
            telemetry_valid(row, track) and p25_stats_valid(row, track)
            for row in capped["results"]
        ),
        "all_hooks_clean": all(
            hook_clean(row)
            for row in [*nohook["results"], *monitor["results"], *capped["results"]]
        ),
        "diagnostics_well_formed": all(
            (
                diagnostics_valid(row, "answer_logits")
                if track == "direct"
                else (
                    parser_fields_valid(row)
                    and diagnostics_valid(row, "actual_answer_logits")
                    and diagnostics_valid(row, "scaffold_answer_logits")
                )
            )
            for row in [*nohook["results"], *monitor["results"], *capped["results"]]
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "status": "passed" if not failures else "failed",
        "track": track,
        "checks": checks,
        "failures": failures,
        "max_option_logit_difference": max_difference,
        "trigger_or_behavior_direction_not_used_as_gate": True,
        "phase_hashes": {
            phase["phase"]: phase["hashes"] for phase in (nohook, monitor, capped)
        },
    }


def audit_formal(root: Path, track: str) -> dict[str, Any]:
    expected = 1200 if track == "direct" else 480
    uncapped = load_phase(root, f"{track}-uncapped", expected)
    capped = load_phase(root, f"{track}-p25", expected)
    left = by_id(uncapped["results"])
    right = by_id(capped["results"])
    ids = [row["id"] for row in uncapped["results"]]
    left_config = uncapped["manifest"]["configuration"]
    right_config = capped["manifest"]["configuration"]
    checks = {
        "same_ids_and_order": (
            set(left) == set(right) and ids == [row["id"] for row in capped["results"]]
        ),
        "same_prompt_token_and_metadata": all(
            same_pair_inputs(left[item_id], right[item_id]) for item_id in ids
        ),
        "conditions_exact": (
            left_config.get("condition") == "uncapped"
            and right_config.get("condition") == "p25"
            and left_config.get("monitor_enabled") is True
            and right_config.get("monitor_enabled") is True
        ),
        "configuration_equal_except_phase_and_condition": (
            comparable_configuration(left_config) == comparable_configuration(right_config)
        ),
        "uncapped_telemetry_complete_no_actual_cap": all(
            telemetry_valid(row, track)
            and row.get("cap_stats") is None
            and row.get("actual_cap_stats_by_phase") is None
            for row in uncapped["results"]
        ),
        "p25_telemetry_and_actual_stats_complete": all(
            telemetry_valid(row, track) and p25_stats_valid(row, track)
            for row in capped["results"]
        ),
        "all_hooks_clean": all(
            hook_clean(row) for row in [*uncapped["results"], *capped["results"]]
        ),
        "all_records_retained": len(left) == expected and len(right) == expected,
        "diagnostics_well_formed": all(
            (
                diagnostics_valid(row, "answer_logits")
                if track == "direct"
                else (
                    parser_fields_valid(row)
                    and diagnostics_valid(row, "actual_answer_logits")
                    and diagnostics_valid(row, "scaffold_answer_logits")
                )
            )
            for row in [*uncapped["results"], *capped["results"]]
        ),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "status": "passed" if not failures else "failed",
        "track": track,
        "checks": checks,
        "failures": failures,
        "phase_hashes": {
            phase["phase"]: phase["hashes"] for phase in (uncapped, capped)
        },
        "content_retries": 0,
        "invalid_or_truncated_retained": track == "free",
    }


def maybe_write_combined(root: Path, stage: str) -> dict[str, Any] | None:
    names = (
        ("SMOKE_DIRECT_AUDIT.json", "SMOKE_FREE_AUDIT.json", "SMOKE_AUDIT.json")
        if stage == "smoke"
        else (
            "DIRECT_CROSS_CONDITION_AUDIT.json",
            "FREE_CROSS_CONDITION_AUDIT.json",
            "CROSS_CONDITION_AUDIT.json",
        )
    )
    paths = [root / name for name in names[:2]]
    if not all(path.is_file() for path in paths):
        return None
    reports = {"direct": read_json(paths[0]), "free": read_json(paths[1])}
    combined = {
        "status": (
            "passed"
            if all(report.get("status") == "passed" for report in reports.values())
            else "failed"
        ),
        "stage": stage,
        "tracks": reports,
        "both_tracks_complete": True,
    }
    atomic_write_json(root / names[2], combined)
    return combined


def main() -> None:
    args = parse_args()
    tracks = ("direct", "free") if args.track == "all" else (args.track,)
    stages = ("smoke", "formal") if args.stage == "all" else (args.stage,)
    output: dict[str, Any] = {}
    for stage in stages:
        for track in tracks:
            report = (
                audit_smoke(args.output_root, track)
                if stage == "smoke"
                else audit_formal(args.output_root, track)
            )
            key = f"{stage}_{track}"
            output[key] = report
            filename = (
                f"SMOKE_{track.upper()}_AUDIT.json"
                if stage == "smoke"
                else f"{track.upper()}_CROSS_CONDITION_AUDIT.json"
            )
            atomic_write_json(args.output_root / filename, report)
        combined = maybe_write_combined(args.output_root, stage)
        if combined is not None:
            output[f"{stage}_combined"] = combined
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    if any(report.get("status") != "passed" for report in output.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
