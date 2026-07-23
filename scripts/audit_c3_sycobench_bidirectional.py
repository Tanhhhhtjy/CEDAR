#!/usr/bin/env python3
"""Audit C3 smoke and formal direct-logit artifacts without model execution."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cedar_m2.checkpoint import atomic_write_json, file_sha256  # noqa: E402
from scripts.c3_portable_hashes import canonical_text_sha256  # noqa: E402

DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "m2" / "c3_sycobench_bidirectional_selectivity"
LAYERS = tuple(range(46, 54))
LABELS = ("A", "B", "C", "D")
LABEL_TOKEN_IDS = {"A": 32, "B": 33, "C": 34, "D": 35}
MODEL_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
AXIS_SHA256 = "a207fe7a36563280b7b29010880aa0082bd8e3113c141cb4a2eed6b46c140211"
CAP_CONFIG_SHA256 = "6aec1220487473aaeab80b05d5d960ac54b5dd9080b51ac4bb0bbd1f4330db24"
LOGIT_TOLERANCE = 1e-4
FLOAT32_PROBABILITY_TOLERANCE = 64 * 2**-23
IDENTITY_FIELDS = (
    "direction_id",
    "base_id",
    "source_id",
    "normalized_stem_id",
    "exact_mcq_id",
    "domain",
    "difficulty",
    "direction",
    "arm",
    "template_id",
    "gold_answer",
    "foil_answer",
    "initial_answer",
    "target_answer",
    "initial_position",
    "target_position",
    "other_wrong_answers",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", choices=("smoke", "uncapped", "p25", "formal", "all"), required=True
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
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
            raise FileNotFoundError(f"Missing C3 phase artifact: {path}")
    manifest = read_json(paths["manifest"])
    runtime = read_json(paths["runtime"])
    results = read_jsonl(paths["results"])
    envelopes = read_jsonl(paths["record_runtime"])
    if len(results) != expected_rows or len(envelopes) != expected_rows:
        raise ValueError(
            f"{phase}: expected {expected_rows}, found {len(results)}/{len(envelopes)}"
        )
    if runtime.get("complete_results") != expected_rows or runtime.get("content_retries") != 0:
        raise ValueError(f"{phase}: runtime incomplete or content retries are nonzero")
    if runtime.get("results_canonical_text_sha256") != canonical_text_sha256(paths["results"]):
        raise ValueError(f"{phase}: results canonical hash mismatch")
    if runtime.get("record_runtime_canonical_text_sha256") != canonical_text_sha256(
        paths["record_runtime"]
    ):
        raise ValueError(f"{phase}: record-runtime canonical hash mismatch")
    for index, (result, envelope) in enumerate(zip(results, envelopes, strict=True)):
        if envelope.get("index") != index or envelope.get("result") != result:
            raise ValueError(f"{phase}: checkpoint materialization differs at row {index}")
    configuration = manifest.get("configuration", {})
    human = configuration.get("human_review", {})
    measurement = configuration.get("measurement", {})
    capping = configuration.get("capping", {})
    axis_telemetry = configuration.get("axis_telemetry", {})
    if (
        configuration.get("phase") != phase
        or configuration.get("track") != "direct"
        or configuration.get("model", {}).get("revision") != MODEL_REVISION
        or measurement.get("enable_thinking") is not False
        or measurement.get("free_generation") is not False
        or measurement.get("single_next_token_forward") is not True
        or measurement.get("candidate_token_ids") != LABEL_TOKEN_IDS
        or measurement.get("batch_size") != 1
        or measurement.get("content_retry") is not False
        or measurement.get("answer_repair") is not False
        or axis_telemetry.get("layers") != list(LAYERS)
        or capping.get("axis_sha256") != AXIS_SHA256
        or capping.get("capping_config_sha256") != CAP_CONFIG_SHA256
        or capping.get("experiment") != "layers_46:54-p0.25"
        or set(capping.get("layers", {})) != {str(layer) for layer in LAYERS}
        or human.get("status") != "passed"
        or human.get("gpu_authorized") is not True
        or human.get("review_mode") != "single_data_lead_review"
        or human.get("dual_independent_review") is not False
        or human.get("cohen_kappa") is not None
    ):
        raise ValueError(f"{phase}: run manifest violates C3 execution/review contract")
    if any(
        row.get("phase") != phase
        or row.get("condition") != configuration.get("condition")
        or row.get("instrumentation") != configuration.get("instrumentation")
        for row in results
    ):
        raise ValueError(f"{phase}: record condition metadata differs from manifest")
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
        raise ValueError("C3 results require unique non-null IDs")
    return output


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def option_logits(record: Mapping[str, Any]) -> dict[str, float] | None:
    payload = record.get("answer_logits")
    if not isinstance(payload, Mapping):
        return None
    values = payload.get("logits")
    if not isinstance(values, Mapping) or set(values) != set(LABELS):
        return None
    output = {label: float(values[label]) for label in LABELS}
    return output if all(finite(value) for value in output.values()) else None


def diagnostics_valid(record: Mapping[str, Any]) -> bool:
    payload = record.get("answer_logits")
    logits = option_logits(record)
    if not isinstance(payload, Mapping) or logits is None:
        return False
    initial = str(record.get("initial_answer"))
    target = str(record.get("target_answer"))
    probabilities = payload.get("full_vocab_probabilities", {})
    conditional = payload.get("conditional_probabilities", {})
    if (
        initial not in LABELS
        or target not in LABELS
        or set(probabilities) != set(LABELS)
        or set(conditional) != set(LABELS)
    ):
        return False
    aliases = {
        "A": record.get("logit_A"),
        "B": record.get("logit_B"),
        "C": record.get("logit_C"),
        "D": record.get("logit_D"),
    }
    option_mass = float(payload.get("option_mass", math.nan))
    target_margin = logits[target] - logits[initial]
    top_id = payload.get("full_vocab_top_token_id")
    return (
        all(math.isclose(float(aliases[label]), logits[label], abs_tol=0.0) for label in LABELS)
        and payload.get("option_argmax") == max(LABELS, key=lambda label: logits[label])
        and record.get("option_argmax") == payload.get("option_argmax")
        and math.isclose(float(payload.get("target_minus_initial")), target_margin, abs_tol=1e-7)
        and math.isclose(float(record.get("target_initial_margin")), target_margin, abs_tol=1e-7)
        and math.isclose(float(record.get("target_logit")), logits[target], abs_tol=1e-7)
        and math.isclose(float(record.get("initial_logit")), logits[initial], abs_tol=1e-7)
        and 0.0 <= option_mass <= 1.0 + FLOAT32_PROBABILITY_TOLERANCE
        and math.isclose(
            option_mass,
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
        and finite(payload.get("full_vocab_entropy"))
        and finite(payload.get("full_vocab_top_probability"))
        and 0.0 <= float(payload["full_vocab_top_probability"]) <= 1.0
        and isinstance(top_id, int)
        and isinstance(payload.get("top_token_is_option"), bool)
        and payload.get("top_token_is_option") == (top_id in set(LABEL_TOKEN_IDS.values()))
        and record.get("top_token_is_option") == payload.get("top_token_is_option")
        and record.get("full_vocab_top_token") == payload.get("full_vocab_top_token_text")
        and math.isclose(
            float(record.get("full_vocab_top_probability")),
            float(payload["full_vocab_top_probability"]),
            abs_tol=0.0,
        )
        and record.get("candidate_token_ids") == LABEL_TOKEN_IDS
    )


def hook_clean(record: Mapping[str, Any]) -> bool:
    return record.get("hook_counts_before") == record.get("hook_counts_after")


def telemetry_valid(record: Mapping[str, Any]) -> bool:
    telemetry = record.get("axis_telemetry")
    summary = record.get("telemetry_summary")
    if not isinstance(telemetry, Mapping) or not isinstance(summary, Mapping):
        return False
    phases = telemetry.get("phases")
    expected = {
        "conversation_prefill": int(record["conversation_prefill_tokens"]),
        "assistant_prefix_context": int(record["assistant_prefix_context_tokens"]),
        "answer_step": 1,
    }
    if not isinstance(phases, Mapping) or set(phases) != set(expected):
        return False
    for phase, expected_count in expected.items():
        payload = phases[phase]
        projections = payload.get("assistant_axis_projection", {})
        exposure = payload.get("reference_cap_exposure", {}).get("p25", {})
        if set(projections) != {str(layer) for layer in LAYERS} or set(exposure) != {
            str(layer) for layer in LAYERS
        }:
            return False
        for layer in LAYERS:
            projection = projections[str(layer)]
            item = exposure[str(layer)]
            if (
                int(projection.get("count", -1)) != expected_count
                or int(item.get("count", -1)) != expected_count
                or not all(
                    finite(value)
                    for value in (
                        projection.get("mean"),
                        projection.get("std"),
                        projection.get("min"),
                        projection.get("max"),
                        item.get("trigger_rate"),
                        item.get("correction_sum"),
                        item.get("mean_correction_all_positions"),
                    )
                )
            ):
                return False
    for field in (
        "trigger_rate_by_layer",
        "correction_sum_by_layer",
        "mean_correction_by_layer",
    ):
        values = summary.get(field)
        if not isinstance(values, Mapping) or set(values) != {str(layer) for layer in LAYERS}:
            return False
        if not all(finite(value) for value in values.values()):
            return False
        if record.get(field) != values:
            return False
    for field in ("prefill_trigger", "answer_step_trigger"):
        item = summary.get(field)
        if not isinstance(item, Mapping) or not all(
            finite(item.get(name))
            for name in ("trigger_rate", "correction_sum", "mean_correction_per_position")
        ):
            return False
        if record.get(field) != item:
            return False
    return True


def p25_stats_valid(record: Mapping[str, Any]) -> bool:
    stats = record.get("cap_stats")
    split = record.get("actual_cap_stats_by_phase")
    summary = record.get("telemetry_summary")
    if (
        not isinstance(stats, Mapping)
        or not isinstance(split, Mapping)
        or not isinstance(summary, Mapping)
        or summary.get("mode") != "actual_p25"
    ):
        return False
    by_layer = stats.get("by_layer", {})
    if set(by_layer) != {str(layer) for layer in LAYERS}:
        return False
    expected_positions = int(record["prompt_tokens"])
    for layer in LAYERS:
        item = by_layer[str(layer)]
        if int(item.get("total_positions", -1)) != expected_positions:
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
        total = triggered = 0
        correction = 0.0
        for phase in ("conversation_prefill", "assistant_prefix_context", "answer_step"):
            phase_item = split.get(phase, {}).get(str(layer), {})
            total += int(phase_item.get("total_positions", -10000))
            triggered += int(phase_item.get("triggered_positions", -10000))
            correction += float(phase_item.get("projection_correction_sum", math.nan))
        if (
            total != int(item["total_positions"])
            or triggered != int(item["triggered_positions"])
            or not math.isclose(
                correction,
                float(item["projection_correction_sum"]),
                rel_tol=1e-5,
                abs_tol=1e-3,
            )
        ):
            return False
    return True


def mapping_valid(record: Mapping[str, Any]) -> bool:
    direction = record.get("direction")
    if direction == "harmful":
        expected = (record.get("gold_answer"), record.get("foil_answer"))
    elif direction == "helpful":
        expected = (record.get("foil_answer"), record.get("gold_answer"))
    else:
        return False
    return (
        (record.get("initial_answer"), record.get("target_answer")) == expected
        and record.get("initial_position") == expected[0]
        and record.get("target_position") == expected[1]
        and expected[0] != expected[1]
    )


def same_inputs(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        all(left.get(field) == right.get(field) for field in IDENTITY_FIELDS)
        and left.get("prompt") == right.get("prompt")
        and left.get("prompt_text_sha256") == right.get("prompt_text_sha256")
        and left.get("prompt_token_sha256") == right.get("prompt_token_sha256")
    )


def max_logit_difference(
    left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]
) -> float:
    a = by_id(left)
    b = by_id(right)
    if set(a) != set(b):
        raise ValueError("C3 paired phases have different IDs")
    differences = []
    for item_id in a:
        left_logits = option_logits(a[item_id])
        right_logits = option_logits(b[item_id])
        if left_logits is None or right_logits is None:
            raise ValueError(f"{item_id}: missing option logits")
        differences.extend(abs(left_logits[label] - right_logits[label]) for label in LABELS)
    return max(differences, default=0.0)


def comparable_configuration(configuration: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in configuration.items()
        if key not in {"phase", "condition", "instrumentation"}
    }


def audit_smoke(root: Path) -> dict[str, Any]:
    nohook = load_phase(root, "smoke-nohook", 96)
    monitor = load_phase(root, "smoke-monitor", 96)
    capped = load_phase(root, "smoke-p25", 96)
    phases = (nohook, monitor, capped)
    indexes = [by_id(phase["results"]) for phase in phases]
    ids = [row["id"] for row in nohook["results"]]
    max_difference = max_logit_difference(nohook["results"], monitor["results"])
    checks = {
        "all_96_records_present": all(len(phase["results"]) == 96 for phase in phases),
        "same_ids_and_order": all(
            ids == [row["id"] for row in phase["results"]] for phase in phases[1:]
        ),
        "same_inputs_across_conditions": all(
            same_inputs(indexes[0][item_id], indexes[index][item_id])
            for item_id in ids
            for index in (1, 2)
        ),
        "nohook_monitor_logits_within_1e_4": max_difference <= LOGIT_TOLERANCE,
        "mapping_and_grid_correct": all(
            mapping_valid(row) for phase in phases for row in phase["results"]
        )
        and all(
            Counter((row["direction"], row["arm"]) for row in phase["results"])
            == Counter(
                {
                    ("harmful", "AO"): 24,
                    ("harmful", "S"): 24,
                    ("helpful", "AO"): 24,
                    ("helpful", "S"): 24,
                }
            )
            for phase in phases
        ),
        "diagnostics_well_formed": all(
            diagnostics_valid(row) for phase in phases for row in phase["results"]
        ),
        "nohook_has_no_hooks_or_cap": all(
            row.get("axis_telemetry") is None
            and row.get("cap_stats") is None
            and row.get("actual_cap_stats_by_phase") is None
            and row.get("telemetry_summary") is None
            for row in nohook["results"]
        ),
        "monitor_has_counterfactual_telemetry_only": all(
            telemetry_valid(row)
            and row.get("telemetry_summary", {}).get("mode") == "uncapped_counterfactual_p25"
            and row.get("cap_stats") is None
            and row.get("actual_cap_stats_by_phase") is None
            for row in monitor["results"]
        ),
        "p25_has_complete_finite_telemetry": all(
            telemetry_valid(row) and p25_stats_valid(row) for row in capped["results"]
        ),
        "all_hooks_clean": all(hook_clean(row) for phase in phases for row in phase["results"]),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "status": "passed" if not failures else "failed",
        "stage": "smoke",
        "checks": checks,
        "failures": failures,
        "max_nohook_monitor_option_logit_difference": max_difference,
        "frozen_tolerance": LOGIT_TOLERANCE,
        "behavior_or_trigger_direction_used_as_gate": False,
        "phase_hashes": {phase["phase"]: phase["hashes"] for phase in phases},
    }


def audit_condition(root: Path, condition: str) -> dict[str, Any]:
    if condition not in {"uncapped", "p25"}:
        raise ValueError(condition)
    phase = load_phase(root, f"formal-{condition}", 2400)
    results = phase["results"]
    ids = [row["id"] for row in results]
    checks = {
        "all_2400_unique_records_present": len(set(ids)) == len(ids) == 2400,
        "complete_direction_arm_grid": Counter((row["direction"], row["arm"]) for row in results)
        == Counter(
            {
                ("harmful", "AO"): 600,
                ("harmful", "S"): 600,
                ("helpful", "AO"): 600,
                ("helpful", "S"): 600,
            }
        ),
        "mapping_correct": all(mapping_valid(row) for row in results),
        "diagnostics_well_formed": all(diagnostics_valid(row) for row in results),
        "telemetry_complete": all(telemetry_valid(row) for row in results),
        "condition_specific_cap_state": (
            all(
                row.get("telemetry_summary", {}).get("mode") == "uncapped_counterfactual_p25"
                and row.get("cap_stats") is None
                and row.get("actual_cap_stats_by_phase") is None
                for row in results
            )
            if condition == "uncapped"
            else all(p25_stats_valid(row) for row in results)
        ),
        "all_hooks_clean": all(hook_clean(row) for row in results),
        "content_retries_zero": phase["runtime"].get("content_retries") == 0,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "status": "passed" if not failures else "failed",
        "stage": condition,
        "checks": checks,
        "failures": failures,
        "phase_hashes": {phase["phase"]: phase["hashes"]},
        "content_retries": 0,
    }


def audit_cross_condition(root: Path) -> dict[str, Any]:
    uncapped = load_phase(root, "formal-uncapped", 2400)
    capped = load_phase(root, "formal-p25", 2400)
    left = by_id(uncapped["results"])
    right = by_id(capped["results"])
    ids = [row["id"] for row in uncapped["results"]]
    left_config = uncapped["manifest"]["configuration"]
    right_config = capped["manifest"]["configuration"]
    checks = {
        "same_ids_and_order": set(left) == set(right)
        and ids == [row["id"] for row in capped["results"]],
        "same_prompt_token_and_metadata": all(
            same_inputs(left[item_id], right[item_id]) for item_id in ids
        ),
        "conditions_exact": left_config.get("condition") == "uncapped"
        and right_config.get("condition") == "p25"
        and left_config.get("instrumentation") == "uncapped"
        and right_config.get("instrumentation") == "p25"
        and left_config.get("monitor_enabled") is True
        and right_config.get("monitor_enabled") is True,
        "configuration_equal_except_phase_condition_instrumentation": comparable_configuration(
            left_config
        )
        == comparable_configuration(right_config),
        "uncapped_telemetry_no_actual_cap": all(
            telemetry_valid(row)
            and row.get("cap_stats") is None
            and row.get("actual_cap_stats_by_phase") is None
            for row in uncapped["results"]
        ),
        "p25_telemetry_and_cap_complete": all(
            telemetry_valid(row) and p25_stats_valid(row) for row in capped["results"]
        ),
        "all_hooks_clean": all(
            hook_clean(row) for row in [*uncapped["results"], *capped["results"]]
        ),
        "all_records_retained": len(left) == len(right) == 2400,
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "status": "passed" if not failures else "failed",
        "stage": "cross_condition",
        "checks": checks,
        "failures": failures,
        "phase_hashes": {
            uncapped["phase"]: uncapped["hashes"],
            capped["phase"]: capped["hashes"],
        },
        "content_retries": 0,
        "free_generation_or_truncation_present": False,
    }


def write_report(root: Path, name: str, report: dict[str, Any]) -> None:
    atomic_write_json(root / name, report)


def main() -> None:
    args = parse_args()
    output: dict[str, Any] = {}
    if args.stage in {"smoke", "all"}:
        output["smoke"] = audit_smoke(args.output_root)
        write_report(args.output_root, "SMOKE_AUDIT.json", output["smoke"])
    if args.stage in {"uncapped", "formal", "all"}:
        output["uncapped"] = audit_condition(args.output_root, "uncapped")
        write_report(args.output_root, "UNCAPPED_AUDIT.json", output["uncapped"])
    if args.stage in {"p25", "formal", "all"}:
        output["p25"] = audit_condition(args.output_root, "p25")
        write_report(args.output_root, "P25_AUDIT.json", output["p25"])
        output["cross_condition"] = audit_cross_condition(args.output_root)
        write_report(args.output_root, "CROSS_CONDITION_AUDIT.json", output["cross_condition"])
    print(json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True))
    if any(report.get("status") != "passed" for report in output.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
