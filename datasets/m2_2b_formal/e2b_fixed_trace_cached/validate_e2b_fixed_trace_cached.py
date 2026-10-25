#!/usr/bin/env python3
"""CPU release Gate for the single E2B cached replay amendment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V1_DATA = HERE.parent / "e2b_fixed_trace_cross"
E2A_DATA = HERE.parent / "e2_phase_validity_choice"
for path in (HERE, V1_DATA, E2A_DATA, ROOT / "src", ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_e2b_fixed_trace_cross import _load_tokenizer  # noqa: E402
from e2b_cached_common import (  # noqa: E402
    AMENDMENT,
    ATTENTION_IMPLEMENTATION,
    BASE_RESULT_COMMIT,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    CAP_EXPERIMENT,
    CAP_LAYERS,
    EXPECTED_AXIS_SHA256,
    EXPECTED_CAP_CONFIG_SHA256,
    EXPECTED_FORMAL_CELLS,
    EXPECTED_PROMPTS,
    EXPECTED_SMOKE_CELLS,
    EXPECTED_TRACES,
    FORWARD_CONDITIONS,
    LABELS,
    MODEL_DTYPE,
    MODEL_REPO,
    MODEL_REVISION,
    REPLAY_TOLERANCE,
    SOURCE_MANIFEST,
    SOURCE_MATRIX,
    SOURCE_TRACES,
    TRACE_SOURCES,
    canonical_text_sha256,
    object_sha256,
    ordered_cells,
    read_json,
    read_jsonl,
    require,
    v1_smoke_replay_diagnostic,
    validate_v1_failure_evidence,
    write_json,
)

ANALYSIS_PLAN = ROOT / "docs/E2B_FIXED_TRACE_CROSS_ANALYSIS_PLAN.json"
DIAGNOSTIC = HERE / "V1_SMOKE_REPLAY_DIAGNOSTIC.json"
CPU_TEST_REPORT = HERE / "CPU_TEST_REPORT.json"

RELEASE_PATHS = (
    "docs/E2B_FIXED_TRACE_CACHED_REPLAY_AMENDMENT.md",
    "datasets/m2_2b_formal/e2b_fixed_trace_cached/README.md",
    "datasets/m2_2b_formal/e2b_fixed_trace_cached/e2b_cached_common.py",
    "datasets/m2_2b_formal/e2b_fixed_trace_cached/validate_e2b_fixed_trace_cached.py",
    "datasets/m2_2b_formal/e2b_fixed_trace_cached/V1_SMOKE_REPLAY_DIAGNOSTIC.json",
    "datasets/m2_2b_formal/e2b_fixed_trace_cached/CPU_TEST_REPORT.json",
    "scripts/run_e2b_fixed_trace_cached.py",
    "scripts/audit_e2b_fixed_trace_cached.py",
    "scripts/analyze_e2b_fixed_trace_cached.py",
    "tests/test_e2b_fixed_trace_cached.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=Path(os.environ.get("HF_HOME", "/data1/WuMingrui/HF_HOME")),
    )
    return parser.parse_args()


def validate(hf_home: Path) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    failures: list[str] = []

    def check(name: str, callback: Any) -> None:
        try:
            callback()
        except Exception as error:
            checks[name] = False
            failures.append(f"{name}: {error}")
        else:
            checks[name] = True

    v1_evidence: dict[str, Any] = {}
    diagnostic: dict[str, Any] = {}
    asset_metadata: dict[str, Any] = {}

    def amendment_check() -> None:
        require(AMENDMENT.is_file(), "Cached replay amendment is missing")
        text = AMENDMENT.read_text(encoding="utf-8")
        for contract in (
            "1e-4",
            "384 formal cells",
            "20,000 bootstrap draws",
            "seed `20260726`",
            "only replay-engine amendment",
        ):
            require(contract in text, f"Amendment contract missing: {contract}")

    check("cached_replay_amendment", amendment_check)

    def v1_failure_check() -> None:
        nonlocal v1_evidence
        v1_evidence = validate_v1_failure_evidence()

    check("v1_failure_retained", v1_failure_check)

    def diagnostic_check() -> None:
        nonlocal diagnostic
        diagnostic = v1_smoke_replay_diagnostic()
        require(diagnostic["conclusion"] == "not_common_shift_only", "Diagnostic changed")
        require(
            diagnostic["summary"]["all"]["centered_max_abs"]["max_abs"] == 0.28125,
            "Centered replay error changed",
        )
        require(
            diagnostic["summary"]["all"]["m_general"]["max_abs"] == 0.375,
            "m_general replay error changed",
        )
        write_json(DIAGNOSTIC, diagnostic)

    check("zero_gpu_v1_replay_decomposition", diagnostic_check)

    traces = read_jsonl(SOURCE_TRACES)
    matrix = read_jsonl(SOURCE_MATRIX)
    source_manifest = read_json(SOURCE_MANIFEST)
    plan = read_json(ANALYSIS_PLAN)

    def source_matrix_check() -> None:
        require(len(traces) == EXPECTED_TRACES, "Source trace count changed")
        require(len({row["trace_id"] for row in traces}) == EXPECTED_TRACES, "Trace IDs collide")
        require(
            len({row["source_id"] for row in traces}) == EXPECTED_PROMPTS,
            "Prompt count changed",
        )
        require(
            Counter(row["trace_source"] for row in traces)
            == Counter({"e2a_uncapped": 96, "e2a_full_p25": 96}),
            "Trace-source balance changed",
        )
        expected = ordered_cells(
            traces, scope="formal", smoke_ids=source_manifest["smoke_prompt_ids"]
        )
        observed_keys = [(row["trace_id"], row["forward_condition"]) for row in matrix]
        expected_keys = [(trace["trace_id"], condition) for trace, condition in expected]
        require(len(matrix) == EXPECTED_FORMAL_CELLS, "Formal matrix count changed")
        require(observed_keys == expected_keys, "Formal matrix order changed")
        require(
            len(
                ordered_cells(
                    traces,
                    scope="smoke",
                    smoke_ids=source_manifest["smoke_prompt_ids"],
                )
            )
            == EXPECTED_SMOKE_CELLS,
            "Smoke matrix changed",
        )

    check("source_traces_and_matrices_unchanged", source_matrix_check)

    def tokenizer_check() -> None:
        tokenizer = _load_tokenizer(hf_home)
        mapping = {
            label: tokenizer.encode(label, add_special_tokens=False) for label in LABELS
        }
        require(all(len(ids) == 1 for ids in mapping.values()), "A/B/C/D are not single tokens")
        for trace in traces:
            require(
                all(trace["candidate_token_ids"][label] == mapping[label][0] for label in LABELS),
                f"Candidate-token mapping changed: {trace['trace_id']}",
            )
            require(
                trace["trace_prefix_tokens"] == trace["selected_answer_logit_index"],
                f"Trace boundary changed: {trace['trace_id']}",
            )

    check("tokenizer_and_trace_boundaries", tokenizer_check)

    def assets_check() -> None:
        nonlocal asset_metadata
        import run_e2_phase_validity_choice as e2run

        _caps, asset_metadata = e2run.load_p25_assets(hf_home)
        require(asset_metadata["axis_sha256"] == EXPECTED_AXIS_SHA256, "Axis hash changed")
        require(
            asset_metadata["capping_config_sha256"] == EXPECTED_CAP_CONFIG_SHA256,
            "Cap config hash changed",
        )
        require(asset_metadata["experiment"] == CAP_EXPERIMENT, "Cap experiment changed")
        require(tuple(map(int, asset_metadata["layers"])) == CAP_LAYERS, "Cap layers changed")

    check("model_axis_cap_and_thresholds", assets_check)

    def analysis_contract_check() -> None:
        require(plan["formal_cells"] == EXPECTED_FORMAL_CELLS, "Analysis cells changed")
        require(plan["trace_sources"] == list(TRACE_SOURCES), "Trace sources changed")
        require(plan["conditions"] == list(FORWARD_CONDITIONS), "Conditions changed")
        require(plan["analysis_unit"]["families"] == 12, "Family count changed")
        require(plan["analysis_unit"]["independent_cluster"] == "family", "Cluster changed")
        require(plan["bootstrap"]["draws"] == BOOTSTRAP_DRAWS, "Bootstrap draws changed")
        require(plan["bootstrap"]["seed"] == BOOTSTRAP_SEED, "Bootstrap seed changed")
        require(REPLAY_TOLERANCE == 1e-4, "Replay tolerance changed")
        require(
            plan["primary_estimands"]
            == [
                "delta_m_general_full_p25_minus_uncapped_given_uncapped_trace",
                "delta_m_general_full_p25_minus_uncapped_given_full_p25_trace",
            ],
            "Primary estimands changed",
        )

    check("scientific_analysis_contract_unchanged", analysis_contract_check)

    def cpu_tests_check() -> None:
        report = read_json(CPU_TEST_REPORT)
        require(report.get("status") == "passed", "CPU test report not passed")
        require(report["tests"]["failed"] == 0, "Targeted tests failed")
        require(report["lint"]["status"] == "passed", "Ruff failed")
        require(report["prepare"]["cells"] == EXPECTED_FORMAL_CELLS, "Prepare count changed")
        require(report["prepare"]["gpu_used"] is False, "Prepare used GPU")

    check("targeted_tests_lint_and_prepare", cpu_tests_check)

    def release_files_check() -> None:
        missing = [relative for relative in RELEASE_PATHS if not (ROOT / relative).is_file()]
        require(not missing, f"Missing release files: {missing}")

    check("release_files_present", release_files_check)

    passed = all(checks.values()) and not failures
    release_hashes = {
        relative: canonical_text_sha256(ROOT / relative)
        for relative in RELEASE_PATHS
        if (ROOT / relative).is_file()
    }
    source_hashes = {
        "frozen_traces192.jsonl": canonical_text_sha256(SOURCE_TRACES),
        "formal_matrix384.jsonl": canonical_text_sha256(SOURCE_MATRIX),
        "v1_manifest.json": canonical_text_sha256(SOURCE_MANIFEST),
        "frozen_analysis_plan.json": canonical_text_sha256(ANALYSIS_PLAN),
    }
    manifest = {
        "schema": "cedar.e2b_fixed_trace_cached.manifest.v1_1",
        "status": "cpu_gate_ready_cached_v1_1" if passed else "failed",
        "base_result_commit": BASE_RESULT_COMMIT,
        "model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "dtype": MODEL_DTYPE,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
        },
        "trace_sources": list(TRACE_SOURCES),
        "forward_conditions": list(FORWARD_CONDITIONS),
        "prompt_rows": EXPECTED_PROMPTS,
        "frozen_traces": EXPECTED_TRACES,
        "formal_cells": EXPECTED_FORMAL_CELLS,
        "smoke_prompt_ids": source_manifest["smoke_prompt_ids"],
        "smoke_cells": EXPECTED_SMOKE_CELLS,
        "replay_engine": "model.generate_kv_cache_forced_tokens",
        "capture_before_force": True,
        "natural_generation": False,
        "replay_tolerance": REPLAY_TOLERANCE,
        "source_hashes": source_hashes,
        "release_files": release_hashes,
        "release_files_sha256": object_sha256(release_hashes),
        "v1_failure_evidence": v1_evidence,
        "v1_smoke_diagnostic_sha256": canonical_text_sha256(DIAGNOSTIC),
        "assets": {
            "axis_sha256": asset_metadata.get("axis_sha256"),
            "capping_config_sha256": asset_metadata.get("capping_config_sha256"),
            "experiment": asset_metadata.get("experiment"),
            "layers": asset_metadata.get("layers"),
        },
        "bootstrap": {"unit": "family", "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED},
    }
    write_json(HERE / "manifest.json", manifest)
    gate = {
        "schema": "cedar.e2b_fixed_trace_cached.cpu_gate.v1_1",
        "gate": "E2B-fixed-trace-cached-replay-one-chance",
        "status": "passed" if passed else "failed",
        "gpu_authorized": passed,
        "authorized_scope": "E2B_cached_v1_1_only" if passed else None,
        "authorized_matrix": {
            "smoke_cells": EXPECTED_SMOKE_CELLS,
            "formal_cells": EXPECTED_FORMAL_CELLS,
            "trace_sources": list(TRACE_SOURCES),
            "forward_conditions": list(FORWARD_CONDITIONS),
        },
        "formal_requires_smoke_replay_16_of_16": True,
        "replay_tolerance": REPLAY_TOLERANCE,
        "one_chance_amendment": True,
        "prohibited": [
            "tolerance_relaxation",
            "anchor_replacement",
            "trace_selection_or_replacement",
            "content_retry",
            "answer_repair",
            "validity_assay",
            "prefill_only",
            "decode_only",
            "p1",
            "activation_patching",
            "J-Lens",
            "MUSE",
            "SAE",
            "gradient_or_JVP",
        ],
        "requires_clean_synced_head": True,
        "requires_exactly_four_visible_gpus": True,
        "manifest_canonical_lf_sha256": canonical_text_sha256(HERE / "manifest.json"),
        "checks": checks,
        "failures": failures,
    }
    write_json(HERE / "CPU_GATE.json", gate)
    validation = {
        "schema": "cedar.e2b_fixed_trace_cached.validation_report.v1_1",
        "status": "passed" if passed else "failed",
        "gpu_used": False,
        "checks": checks,
        "failures": failures,
        "gate": gate,
    }
    write_json(HERE / "validation_report.json", validation)
    return validation


def main() -> None:
    args = parse_args()
    report = validate(args.hf_home.resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
