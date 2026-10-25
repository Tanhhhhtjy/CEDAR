#!/usr/bin/env python3
"""Deterministic CPU validator and GPU authorization Gate for frozen E2B."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from build_e2b_fixed_trace_cross import _load_tokenizer, build
from e2b_common import (
    ANALYSIS_PLAN,
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
    MODEL_REPO,
    MODEL_REVISION,
    ROOT,
    TRACE_SOURCES,
    canonical_text_sha256,
    object_sha256,
    ordered_cells,
    read_json,
    read_jsonl,
    require,
    smoke_prompt_ids,
    validate_anchor,
    validate_frozen_protocol_hashes,
    write_json,
)

HERE = Path(__file__).resolve().parent

RELEASE_PATHS = (
    "docs/E2B_FIXED_TRACE_CROSS_PROTOCOL.md",
    "docs/E2B_FIXED_TRACE_CROSS_ANALYSIS_PLAN.json",
    "docs/E2_PROTOCOL_FREEZE_MANIFEST.json",
    "datasets/m2_2b_formal/e2b_fixed_trace_cross/README.md",
    "datasets/m2_2b_formal/e2b_fixed_trace_cross/e2b_common.py",
    "datasets/m2_2b_formal/e2b_fixed_trace_cross/build_e2b_fixed_trace_cross.py",
    "datasets/m2_2b_formal/e2b_fixed_trace_cross/validate_e2b_fixed_trace_cross.py",
    "datasets/m2_2b_formal/e2b_fixed_trace_cross/frozen_traces192.jsonl",
    "datasets/m2_2b_formal/e2b_fixed_trace_cross/formal_matrix384.jsonl",
    "datasets/m2_2b_formal/e2b_fixed_trace_cross/build_report.json",
    "datasets/m2_2b_formal/e2b_fixed_trace_cross/CPU_TEST_REPORT.json",
    "scripts/run_e2b_fixed_trace_cross.py",
    "scripts/audit_e2b_fixed_trace_cross.py",
    "scripts/analyze_e2b_fixed_trace_cross.py",
    "tests/test_e2b_fixed_trace_cross.py",
    "explore_outputs/e2b_fixed_trace_cross/RUN_LOG.md",
    "explore_outputs/e2b_fixed_trace_cross/CHANGESET.md",
    "explore_outputs/e2b_fixed_trace_cross/SCIENTIFIC_CHANGELOG.md",
    "explore_outputs/e2b_fixed_trace_cross/status.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf-home", type=Path, default=Path(os.environ.get("HF_HOME", "/data1/WuMingrui/HF_HOME"))
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

    protocol_hashes: dict[str, str] = {}
    anchor: dict[str, Any] = {}
    asset_metadata: dict[str, Any] = {}

    def protocol_check() -> None:
        nonlocal protocol_hashes
        protocol_hashes = validate_frozen_protocol_hashes()

    check("frozen_protocol_hashes", protocol_check)

    def anchor_check() -> None:
        nonlocal anchor
        anchor = validate_anchor()

    check("e2a_anchor", anchor_check)

    tokenizer = _load_tokenizer(hf_home)
    traces = read_jsonl(HERE / "frozen_traces192.jsonl")
    matrix = read_jsonl(HERE / "formal_matrix384.jsonl")
    report = read_json(HERE / "build_report.json")
    test_report = read_json(HERE / "CPU_TEST_REPORT.json")
    plan = read_json(ANALYSIS_PLAN)

    def deterministic_build_check() -> None:
        with tempfile.TemporaryDirectory(prefix="cedar-e2b-build-a-") as first_dir:
            first = Path(first_dir)
            first_report = build(first, tokenizer)
            with tempfile.TemporaryDirectory(prefix="cedar-e2b-build-b-") as second_dir:
                second = Path(second_dir)
                second_report = build(second, tokenizer)
                for filename in (
                    "frozen_traces192.jsonl",
                    "formal_matrix384.jsonl",
                    "build_report.json",
                ):
                    require(
                        canonical_text_sha256(first / filename)
                        == canonical_text_sha256(second / filename),
                        f"Two clean E2B builds differ: {filename}",
                    )
                    require(
                        canonical_text_sha256(first / filename)
                        == canonical_text_sha256(HERE / filename),
                        f"Checked-in E2B build is stale: {filename}",
                    )
                require(first_report == second_report == report, "Build reports differ")

    check("two_clean_deterministic_builds", deterministic_build_check)

    def trace_check() -> None:
        require(len(traces) == EXPECTED_TRACES, "Trace count changed")
        require(len({row["trace_id"] for row in traces}) == EXPECTED_TRACES, "Trace IDs collide")
        require(
            Counter(row["trace_source"] for row in traces)
            == Counter({"e2a_uncapped": 96, "e2a_full_p25": 96}),
            "Trace-source counts changed",
        )
        require(
            len({row["source_id"] for row in traces}) == EXPECTED_PROMPTS,
            "Prompt count changed",
        )
        for source_id in {row["source_id"] for row in traces}:
            paired = [row for row in traces if row["source_id"] == source_id]
            require(
                {row["trace_source"] for row in paired} == set(TRACE_SOURCES),
                f"{source_id}: missing trace source",
            )
        require(
            all(row["trace_prefix_tokens"] == row["selected_answer_logit_index"] for row in traces),
            "Trace cut index changed",
        )
        require(
            all(
                row["generated_label_token_id"]
                == row["candidate_token_ids"][row["generated_label"]]
                for row in traces
            ),
            "Generated answer-label token mismatch",
        )
        require(
            all(
                len(tokenizer.encode(label, add_special_tokens=False)) == 1
                and tokenizer.encode(label, add_special_tokens=False)[0]
                == row["candidate_token_ids"][label]
                for row in traces
                for label in ("A", "B", "C", "D")
            ),
            "Contextual candidate token mapping changed",
        )

    check("trace_identity_and_candidate_tokens", trace_check)

    def matrix_check() -> None:
        expected = ordered_cells(
            traces, scope="formal", smoke_ids=report["smoke_prompt_ids"]
        )
        expected_keys = [
            (trace["trace_id"], condition) for trace, condition in expected
        ]
        observed_keys = [
            (row["trace_id"], row["forward_condition"]) for row in matrix
        ]
        require(len(matrix) == EXPECTED_FORMAL_CELLS, "Formal matrix count changed")
        require(observed_keys == expected_keys, "Formal matrix order or identity changed")
        require(
            Counter(row["forward_condition"] for row in matrix)
            == Counter({"uncapped": 192, "full_p25": 192}),
            "Forward condition counts changed",
        )
        prompt_rows = [
            next(trace for trace in traces if trace["source_id"] == source_id)
            for source_id in sorted({trace["source_id"] for trace in traces})
        ]
        selected = smoke_prompt_ids(prompt_rows)
        require(selected == report["smoke_prompt_ids"], "Smoke selection changed")
        smoke = ordered_cells(traces, scope="smoke", smoke_ids=selected)
        require(len(smoke) == EXPECTED_SMOKE_CELLS, "Smoke matrix changed")

    check("formal_and_smoke_matrices", matrix_check)

    def assets_check() -> None:
        nonlocal asset_metadata
        import sys

        scripts = ROOT / "scripts"
        e2a_data = ROOT / "datasets/m2_2b_formal/e2_phase_validity_choice"
        src = ROOT / "src"
        for path in (scripts, e2a_data, src):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
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

    def analysis_plan_check() -> None:
        require(plan.get("formal_cells") == EXPECTED_FORMAL_CELLS, "Analysis cells changed")
        require(plan.get("trace_sources") == list(TRACE_SOURCES), "Trace sources changed")
        require(plan.get("conditions") == list(FORWARD_CONDITIONS), "Conditions changed")
        require(plan["analysis_unit"]["families"] == 12, "Family count changed")
        require(plan["analysis_unit"]["independent_cluster"] == "family", "Cluster changed")
        require(plan["bootstrap"]["draws"] == BOOTSTRAP_DRAWS, "Bootstrap draws changed")
        require(plan["bootstrap"]["seed"] == BOOTSTRAP_SEED, "Bootstrap seed changed")
        require(
            plan["primary_estimands"]
            == [
                "delta_m_general_full_p25_minus_uncapped_given_uncapped_trace",
                "delta_m_general_full_p25_minus_uncapped_given_full_p25_trace",
            ],
            "Primary estimands changed",
        )

    check("analysis_plan_frozen", analysis_plan_check)

    def tests_check() -> None:
        require(test_report.get("status") == "passed", "CPU test report is not passed")
        require(test_report["tests"]["failed"] == 0, "Targeted tests failed")
        require(test_report["lint"]["status"] == "passed", "Ruff failed")
        require(test_report["prepare"]["cells"] == EXPECTED_FORMAL_CELLS, "Prepare changed")
        require(test_report["prepare"]["gpu_used"] is False, "Prepare used GPU")

    check("targeted_tests_lint_and_prepare", tests_check)

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
    data_hashes = {
        filename: canonical_text_sha256(HERE / filename)
        for filename in ("frozen_traces192.jsonl", "formal_matrix384.jsonl")
    }
    manifest = {
        "schema": "cedar.e2b_fixed_trace_cross.manifest.v1",
        "status": "cpu_gate_ready" if passed else "failed",
        "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
        "trace_sources": list(TRACE_SOURCES),
        "forward_conditions": list(FORWARD_CONDITIONS),
        "prompt_rows": EXPECTED_PROMPTS,
        "frozen_traces": EXPECTED_TRACES,
        "formal_cells": EXPECTED_FORMAL_CELLS,
        "smoke_prompt_ids": report["smoke_prompt_ids"],
        "smoke_cells": EXPECTED_SMOKE_CELLS,
        "data_hashes": data_hashes,
        "release_files": release_hashes,
        "release_files_sha256": object_sha256(release_hashes),
        "frozen_protocol_hashes": protocol_hashes,
        "anchor": anchor,
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
        "schema": "cedar.e2b_fixed_trace_cross.cpu_gate.v1",
        "gate": "E2B-fixed-trace-cross",
        "status": "passed" if passed else "failed",
        "gpu_authorized": passed,
        "authorized_scope": "E2B_only" if passed else None,
        "authorized_matrix": {
            "smoke_cells": EXPECTED_SMOKE_CELLS,
            "formal_cells": EXPECTED_FORMAL_CELLS,
            "trace_sources": list(TRACE_SOURCES),
            "forward_conditions": list(FORWARD_CONDITIONS),
        },
        "prohibited": [
            "new_generation",
            "validity_assay",
            "prefill_only",
            "decode_only",
            "p1",
            "trace_selection_or_replacement",
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
        "schema": "cedar.e2b_fixed_trace_cross.validation_report.v1",
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
