#!/usr/bin/env python3
"""Audit Step 5 uncapped smoke or formal thinking/nonthinking cells."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from cedar_m2.checkpoint import atomic_write_json, file_sha256, object_sha256

try:
    from scripts import run_prover_generator_step5_uncapped as runner
except ImportError:
    import run_prover_generator_step5_uncapped as runner

MODES = ("thinking", "nonthinking")
OUTCOMES = ("initial", "induced_target", "other_wrong", "invalid_or_abstain")
FLOAT_TOLERANCE = 2e-6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("smoke", "formal"), required=True)
    parser.add_argument("--output-root", type=Path, default=runner.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def diagnostics_valid(record: Mapping[str, Any]) -> bool:
    payload = record.get("answer_logits")
    if payload is None:
        return record.get("chosen_option") is None
    if not isinstance(payload, Mapping):
        return False
    logits = payload.get("logits")
    probabilities = payload.get("full_vocab_probabilities")
    conditional = payload.get("conditional_probabilities")
    if not all(isinstance(value, Mapping) for value in (logits, probabilities, conditional)):
        return False
    if any(set(value) != set(runner.LABELS) for value in (logits, probabilities, conditional)):
        return False
    if not all(
        finite(value)
        for values in (logits, probabilities, conditional)
        for value in values.values()
    ):
        return False
    initial = str(record.get("initial_answer"))
    target = str(record.get("target_answer"))
    option_mass = float(payload.get("option_mass", math.nan))
    return (
        initial in runner.LABELS
        and target in runner.LABELS
        and payload.get("option_argmax")
        == max(runner.LABELS, key=lambda label: float(logits[label]))
        and 0.0 <= option_mass <= 1.0 + FLOAT_TOLERANCE
        and math.isclose(
            option_mass,
            sum(float(probabilities[label]) for label in runner.LABELS),
            rel_tol=1e-7,
            abs_tol=1e-7,
        )
        and math.isclose(
            sum(float(conditional[label]) for label in runner.LABELS),
            1.0,
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
        and math.isclose(
            float(payload.get("target_minus_initial", math.nan)),
            float(logits[target]) - float(logits[initial]),
            rel_tol=1e-7,
            abs_tol=1e-7,
        )
        and finite(payload.get("full_vocab_entropy"))
    )


def expected_outcome(record: Mapping[str, Any]) -> str:
    answer = record.get("chosen_option")
    if answer is None:
        return "invalid_or_abstain"
    if answer == record.get("initial_answer"):
        return "initial"
    if answer == record.get("target_answer"):
        return "induced_target"
    return "other_wrong"


def load_cell(
    output_root: Path,
    scope: str,
    mode: str,
    source_rows: Sequence[Mapping[str, Any]],
    failures: list[str],
) -> dict[str, Any] | None:
    name = runner.run_name(scope, mode)
    run_dir = output_root / name
    paths = {
        "manifest": run_dir / "manifest.json",
        "runtime": run_dir / "runtime.json",
        "results": run_dir / "results.jsonl",
        "record_runtime": run_dir / "record_runtime.jsonl",
    }
    absent = [str(path) for path in paths.values() if not path.is_file()]
    if absent:
        failures.append(f"{name}: missing files {absent}")
        return None
    manifest = runner.read_json(paths["manifest"])
    runtime = runner.read_json(paths["runtime"])
    results = runner.read_jsonl(paths["results"])
    runtime_rows = runner.read_jsonl(paths["record_runtime"])
    expected = runner.EXPECTED_ROWS[scope]
    expected_ids = [str(row["id"]) for row in source_rows]
    configuration = manifest.get("configuration", {})

    require(len(results) == expected, f"{name}: result count", failures)
    require(len(runtime_rows) == expected, f"{name}: runtime row count", failures)
    require(
        manifest.get("fingerprint") == object_sha256(configuration),
        f"{name}: run manifest fingerprint",
        failures,
    )
    require(
        [row.get("id") for row in results] == expected_ids,
        f"{name}: result ID order",
        failures,
    )
    require(runtime.get("complete_results") == expected, f"{name}: incomplete runtime", failures)
    require(runtime.get("content_retries") == 0, f"{name}: content retry", failures)
    require(runtime.get("answer_repairs") == 0, f"{name}: answer repair", failures)
    require(
        runtime.get("rendered_prompts_sha256")
        == runner.EXPECTED_RENDERED_PROMPTS_SHA256[(scope, mode)],
        f"{name}: frozen rendered prompts",
        failures,
    )
    require(
        runtime.get("tokenized_input_ids_sha256")
        == runner.EXPECTED_TOKENIZED_INPUT_IDS_SHA256[(scope, mode)],
        f"{name}: frozen tokenized inputs",
        failures,
    )
    require(
        runtime.get("results_sha256") == file_sha256(paths["results"]),
        f"{name}: results hash",
        failures,
    )
    require(
        runtime.get("record_runtime_sha256") == file_sha256(paths["record_runtime"]),
        f"{name}: record runtime hash",
        failures,
    )
    require(configuration.get("scope") == scope, f"{name}: scope metadata", failures)
    require(configuration.get("condition") == "uncapped", f"{name}: not uncapped", failures)
    require(
        configuration.get("runner_amendment") == runner.RUNNER_AMENDMENT,
        f"{name}: runner amendment identity",
        failures,
    )
    require(configuration.get("seed") == runner.SEED, f"{name}: seed", failures)
    require(
        configuration.get("mode") == {"name": mode, **asdict(runner.MODES[mode])},
        f"{name}: mode contract",
        failures,
    )
    require(
        configuration.get("dataset", {}).get("selected_ids_sha256") == object_sha256(expected_ids),
        f"{name}: selected ID hash",
        failures,
    )
    require(
        configuration.get("dataset", {}).get("data_freeze_commit") == runner.DATA_FREEZE_COMMIT,
        f"{name}: data freeze commit",
        failures,
    )
    require(
        configuration.get("git", {}).get("commit")
        == configuration.get("git", {}).get("upstream_commit"),
        f"{name}: runner commit not synced",
        failures,
    )
    require(
        configuration.get("git", {}).get("data_freeze_is_ancestor") is True,
        f"{name}: data freeze ancestry",
        failures,
    )

    for index, (record, source) in enumerate(zip(results, source_rows, strict=False)):
        item = f"{name}[{index}]"
        require(record.get("id") == source.get("id"), f"{item}: source ID", failures)
        require(record.get("scope") == scope, f"{item}: scope", failures)
        require(record.get("mode") == mode, f"{item}: mode", failures)
        require(record.get("condition") == "uncapped", f"{item}: condition", failures)
        require(record.get("seed") == runner.SEED, f"{item}: seed", failures)
        require(record.get("content_retry_count") == 0, f"{item}: retry", failures)
        require(record.get("answer_repair_performed") is False, f"{item}: repair", failures)
        require(record.get("chosen_option") in {*runner.LABELS, None}, f"{item}: answer", failures)
        require(record.get("outcome_class") in OUTCOMES, f"{item}: outcome", failures)
        require(
            record.get("outcome_class") == expected_outcome(record),
            f"{item}: outcome mismatch",
            failures,
        )
        require(
            record.get("gold_correct")
            == (record.get("chosen_option") == record.get("gold_answer")),
            f"{item}: gold correctness",
            failures,
        )
        require(diagnostics_valid(record), f"{item}: answer diagnostics", failures)
        if record.get("answer_logits") is not None:
            require(
                isinstance(record.get("selected_answer_logit_index"), int),
                f"{item}: selected answer-logit index",
                failures,
            )
            require(
                record.get("selected_answer_logit_index")
                in record.get("answer_logit_capture_indices", []),
                f"{item}: selected answer-logit index not captured",
                failures,
            )
            if record.get("chosen_option") is not None:
                require(
                    record.get("answer_boundary", {}).get("answer_choice_index")
                    == record.get("selected_answer_logit_index"),
                    f"{item}: answer boundary/logit alignment",
                    failures,
                )
                require(
                    not record.get("answer_boundary", {}).get("errors"),
                    f"{item}: answer boundary errors",
                    failures,
                )
        require(
            isinstance(record.get("reasoning_token_count"), int)
            and int(record["reasoning_token_count"]) >= 0,
            f"{item}: reasoning token count",
            failures,
        )
    return {
        "name": name,
        "run_dir": str(run_dir),
        "configuration": configuration,
        "runtime": runtime,
        "results": results,
        "hashes": {key: file_sha256(path) for key, path in paths.items()},
    }


def normalized_configuration(configuration: Mapping[str, Any]) -> dict[str, Any]:
    value = deepcopy(dict(configuration))
    value.pop("mode", None)
    return value


def main() -> None:
    args = parse_args()
    failures: list[str] = []
    warnings: list[str] = []
    all_rows = runner.validate_inputs()
    source_rows = runner.select_rows(all_rows, args.scope)
    if args.scope == "formal" and not runner.passed_audit(args.output_root / "SMOKE_AUDIT.json"):
        failures.append("formal audit requires a passed combined smoke audit")

    cells = {
        mode: load_cell(args.output_root, args.scope, mode, source_rows, failures) for mode in MODES
    }
    if all(cells.values()):
        thinking = cells["thinking"]
        nonthinking = cells["nonthinking"]
        assert thinking is not None and nonthinking is not None
        require(
            normalized_configuration(thinking["configuration"])
            == normalized_configuration(nonthinking["configuration"]),
            "thinking/nonthinking configurations differ beyond the frozen mode field",
            failures,
        )
        require(
            thinking["configuration"]["git"]["commit"]
            == nonthinking["configuration"]["git"]["commit"],
            "thinking/nonthinking runner commit differs",
            failures,
        )
        require(
            thinking["runtime"].get("tokenized_input_ids_sha256")
            != nonthinking["runtime"].get("tokenized_input_ids_sha256"),
            "thinking toggle did not change frozen rendered inputs",
            failures,
        )
        for mode, cell in cells.items():
            assert cell is not None
            results = cell["results"]
            parse_failures = [
                row["id"] for row in results if row.get("final_answer_parse_status") != "passed"
            ]
            missing_logits = [row["id"] for row in results if row.get("answer_logits") is None]
            if parse_failures:
                warnings.append(f"{mode}: {len(parse_failures)} strict parse failures")
            if missing_logits:
                warnings.append(f"{mode}: {len(missing_logits)} missing answer-logit captures")

        if args.scope == "smoke":
            thinking_results = thinking["results"]
            nonthinking_results = nonthinking["results"]
            require(
                any(row.get("reasoning_token_count", 0) > 0 for row in thinking_results),
                "thinking smoke produced no reasoning tokens",
                failures,
            )
            require(
                all(row.get("reasoning_token_count") == 0 for row in nonthinking_results),
                "nonthinking smoke produced reasoning tokens",
                failures,
            )
            require(
                any(row.get("answer_logits") is not None for row in thinking_results),
                "thinking smoke produced no direct-margin capture",
                failures,
            )
            require(
                any(row.get("answer_logits") is not None for row in nonthinking_results),
                "nonthinking smoke produced no direct-margin capture",
                failures,
            )

    report = {
        "schema": "cedar.prover_generator.step5.cross_mode_audit.v1",
        "status": "passed" if not failures else "failed",
        "scope": args.scope,
        "condition": "uncapped",
        "expected_rows_per_mode": runner.EXPECTED_ROWS[args.scope],
        "expected_arms": {arm: runner.EXPECTED_ROWS[args.scope] // 6 for arm in runner.ARMS},
        "data_freeze_commit": runner.DATA_FREEZE_COMMIT,
        "failures": failures,
        "warnings": warnings,
        "cells": {
            mode: (
                None
                if cell is None
                else {
                    "run_dir": cell["run_dir"],
                    "results": len(cell["results"]),
                    "arm_counts": dict(Counter(row["arm"] for row in cell["results"])),
                    "outcome_counts": dict(
                        Counter(row["outcome_class"] for row in cell["results"])
                    ),
                    "parse_failures": sum(
                        row["final_answer_parse_status"] != "passed" for row in cell["results"]
                    ),
                    "missing_answer_logits": sum(
                        row["answer_logits"] is None for row in cell["results"]
                    ),
                    "runner_commit": cell["configuration"]["git"]["commit"],
                    "hashes": cell["hashes"],
                }
            )
            for mode, cell in cells.items()
        },
    }
    default_name = "SMOKE_AUDIT.json" if args.scope == "smoke" else "FORMAL_CROSS_MODE_AUDIT.json"
    output = args.report or args.output_root / default_name
    atomic_write_json(output, report)
    print(json.dumps({"status": report["status"], "report": str(output)}, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
