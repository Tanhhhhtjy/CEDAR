#!/usr/bin/env python3
"""Build and validate the zero-GPU RG1-P Stage A release gate."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from transformers import AutoConfig

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = ROOT / "src"
E2A_DATA = HERE.parent / "e2_phase_validity_choice"
for path in (HERE, SRC, E2A_DATA, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_e2_phase_validity_choice as e2run  # noqa: E402
from rg1p_common import (  # noqa: E402
    ALL_CONDITIONS,
    ATTENTION_IMPLEMENTATION,
    BASE_RESULT_COMMIT,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    BRANCH,
    CAP_EXPERIMENT,
    CAP_LAYERS,
    CAPTURE_LAYERS,
    ENGINEERING_CORRECTION_ID,
    EXPECTED_AXIS_SHA256,
    EXPECTED_CAP_CONFIG_SHA256,
    EXPECTED_FORMAL_CELLS,
    EXPECTED_NOOP_FORMAL,
    EXPECTED_OFFICIAL_FORMAL,
    EXPECTED_RANDOM_FORMAL,
    EXPECTED_SMOKE_CELLS,
    INITIAL_CPU_GATE_COMMIT,
    LAYER_BOOTSTRAP_DRAWS,
    MODEL_DTYPE,
    MODEL_REPO,
    MODEL_REVISION,
    NOOP_CONDITION,
    PRIMARY_TRACE_SOURCE,
    PROBE_PERMUTATIONS,
    PROTOCOL,
    RANDOM_CONDITIONS,
    RANDOM_ORTHOGONAL_TOLERANCE,
    RANDOM_SEEDS,
    RANDOM_UNIT_TOLERANCE,
    REPORTING_LAYERS,
    SOURCE_ANALYSIS,
    SOURCE_AUDIT,
    SOURCE_MANIFEST,
    SOURCE_MATRIX,
    SOURCE_RESULTS,
    SOURCE_TRACES,
    TRACE_SOURCES,
    build_random_directions,
    canonical_text_sha256,
    file_sha256,
    matrix_cells,
    object_sha256,
    read_json,
    read_jsonl,
    require,
    smoke_prompt_ids,
    tensor_sha256,
    write_json,
    write_jsonl,
)

from cedar_m2.assets.capping_config import (  # noqa: E402
    download_qwen_capping_assets,
    load_axis,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args()


def command(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(args), cwd=ROOT, check=check, capture_output=True, text=True)


def git_checks() -> dict[str, Any]:
    head = command("git", "rev-parse", "HEAD").stdout.strip()
    branch = command("git", "branch", "--show-current").stdout.strip()
    ancestor = command("git", "merge-base", "--is-ancestor", BASE_RESULT_COMMIT, head, check=False)
    require(ancestor.returncode == 0, "HEAD is not based on the frozen E2B result")
    require(branch == BRANCH, f"Expected branch {BRANCH}, found {branch}")
    return {
        "head": head,
        "branch": branch,
        "base_result_commit": BASE_RESULT_COMMIT,
        "base_is_ancestor": True,
        "dirty_during_gate_build": bool(command("git", "status", "--porcelain").stdout),
        "gpu_requires_clean_synced_commit": True,
    }


def validate_sources() -> dict[str, Any]:
    traces = read_jsonl(SOURCE_TRACES)
    matrix = read_jsonl(SOURCE_MATRIX)
    results = read_jsonl(SOURCE_RESULTS)
    audit = read_json(SOURCE_AUDIT)
    analysis = read_json(SOURCE_ANALYSIS)
    source_manifest = read_json(SOURCE_MANIFEST)
    require(len(traces) == 192, "Frozen E2B trace count changed")
    require(len(matrix) == 384, "Frozen E2B matrix count changed")
    require(len(results) == 384, "Frozen E2B result count changed")
    require(audit.get("status") == "passed", "Frozen E2B audit is not passed")
    require(audit.get("replay_gate", {}).get("passed") is True, "E2B replay gate changed")
    require(audit.get("replay_gate", {}).get("checked") == 192, "E2B replay count changed")
    require(analysis.get("status") == "complete", "Frozen E2B analysis is incomplete")
    require(
        Counter(row["trace_source"] for row in traces) == {"e2a_uncapped": 96, "e2a_full_p25": 96},
        "Frozen trace-source balance changed",
    )
    require(len({row["source_id"] for row in traces}) == 96, "Prompt count changed")
    require(len({row["family_id"] for row in traces}) == 12, "Family count changed")
    require(
        Counter(row["arm"] for row in traces) == {arm: 48 for arm in "NRVF"}, "Arm balance changed"
    )
    return {
        "frozen_traces192.jsonl": canonical_text_sha256(SOURCE_TRACES),
        "formal_matrix384.jsonl": canonical_text_sha256(SOURCE_MATRIX),
        "e2b_manifest.json": canonical_text_sha256(SOURCE_MANIFEST),
        "e2b_results.jsonl": canonical_text_sha256(SOURCE_RESULTS),
        "e2b_audit.json": canonical_text_sha256(SOURCE_AUDIT),
        "e2b_analysis.json": canonical_text_sha256(SOURCE_ANALYSIS),
        "e2b_result_commit": BASE_RESULT_COMMIT,
        "e2b_replay_engine": source_manifest["replay_engine"],
    }


def validate_forward_path_equivalence() -> dict[str, Any]:
    """Prove that the smoke resume did not alter any GPU forward operator."""
    names = {
        "TerminalBlockCollector",
        "OfficialScheduleCapture",
        "ScheduleMatchedRandomInjector",
        "run_one",
    }

    def hashes(source: str) -> dict[str, str]:
        found: dict[str, str] = {}
        for node in ast.parse(source).body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
                payload = ast.dump(node, annotate_fields=True, include_attributes=False)
                found[node.name] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        require(set(found) == names, "RG1-P forward definition missing")
        return found

    relative = "scripts/run_rg1p_stagea.py"
    initial = command(
        "git", "show", f"{INITIAL_CPU_GATE_COMMIT}:{relative}", check=False
    )
    require(initial.returncode == 0, "Initial RG1-P runner is unavailable")
    current = (ROOT / relative).read_text(encoding="utf-8")
    initial_hashes = hashes(initial.stdout)
    current_hashes = hashes(current)
    require(initial_hashes == current_hashes, "GPU forward path changed during lens correction")
    return {
        "status": "passed",
        "initial_cpu_gate_commit": INITIAL_CPU_GATE_COMMIT,
        "definition_ast_sha256": current_hashes,
    }


def build_release(hf_home: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    traces = read_jsonl(SOURCE_TRACES)
    smoke_ids = smoke_prompt_ids(traces)
    smoke = matrix_cells(traces, scope="smoke", smoke_ids=smoke_ids)
    formal = matrix_cells(traces, scope="formal", smoke_ids=smoke_ids)
    write_jsonl(HERE / "smoke_matrix72.jsonl", smoke)
    write_jsonl(HERE / "formal_matrix864.jsonl", formal)

    require(len(smoke) == EXPECTED_SMOKE_CELLS, "Smoke matrix count changed")
    require(len(formal) == EXPECTED_FORMAL_CELLS, "Formal matrix count changed")
    formal_counts = Counter(row["cell_kind"] for row in formal)
    require(
        formal_counts
        == {
            "official": EXPECTED_OFFICIAL_FORMAL,
            "engineering_noop": EXPECTED_NOOP_FORMAL,
            "random_reference": EXPECTED_RANDOM_FORMAL,
        },
        "Formal official/no-op/random count changed",
    )
    random_rows = [row for row in formal if row["cell_kind"] == "random_reference"]
    require(
        {row["trace_source"] for row in random_rows} == {PRIMARY_TRACE_SOURCE},
        "Random references escaped treatment-naive traces",
    )
    require(
        Counter(row["forward_condition"] for row in random_rows)
        == {condition: 96 for condition in RANDOM_CONDITIONS},
        "Random condition balance changed",
    )
    noop_rows = [row for row in formal if row["cell_kind"] == "engineering_noop"]
    require(len(noop_rows) == EXPECTED_NOOP_FORMAL, "No-op formal count changed")
    require(
        {row["trace_source"] for row in noop_rows} == {PRIMARY_TRACE_SOURCE}
        and {row["forward_condition"] for row in noop_rows} == {NOOP_CONDITION},
        "No-op controls escaped treatment-naive uncapped replay",
    )
    caps, cap_metadata = e2run.load_p25_assets(hf_home.resolve())
    model_config = AutoConfig.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        cache_dir=e2run.resolve_hub_cache(hf_home.resolve()),
        local_files_only=True,
    )
    require(model_config.num_hidden_layers == 64, "Model depth changed")
    require(
        CAPTURE_LAYERS == tuple(range(model_config.num_hidden_layers)),
        "Capture layers do not cover the complete decoder depth",
    )
    axis_path, _config_path = download_qwen_capping_assets(
        cache_dir=e2run.resolve_hub_cache(hf_home.resolve()), local_files_only=True
    )
    axis = load_axis(axis_path)
    require(tuple(sorted(caps)) == CAP_LAYERS, "Official p25 layers changed")
    require(axis.shape[1] == 5120, "Assistant Axis hidden size changed")
    require(axis.shape[0] > max(CAP_LAYERS), "Assistant Axis cap-layer coverage changed")
    official_axes = torch.stack(
        [
            axis[layer].float() / torch.linalg.vector_norm(axis[layer].float())
            for layer in CAP_LAYERS
        ]
    )
    save_file(
        {"official_axis_units": official_axes.contiguous()},
        HERE / "official_axis_units.safetensors",
    )
    directions, direction_metadata = build_random_directions(axis)
    repeated, repeated_metadata = build_random_directions(axis)
    require(torch.equal(directions, repeated), "Random direction rebuild is not deterministic")
    require(
        direction_metadata["tensor_sha256"] == repeated_metadata["tensor_sha256"],
        "Random direction hash changed on rebuild",
    )
    save_file({"directions": directions.contiguous()}, HERE / "random_directions.safetensors")
    write_json(HERE / "random_direction_metadata.json", direction_metadata)

    axis_mapping = {
        "layer45": "reporting_layer_without_authorized_axis_no_projection",
        "layers46_53": "official_layer_specific_assistant_axis",
        "layers54_63": "fixed_layer53_exit_axis",
        "random_layers46_53": "seed_specific_layer_direction",
        "random_layers54_63": "fixed_seed_specific_layer53_injection_direction",
        "downstream_caption_required": (
            "Downstream projections use the fixed layer-53 cap direction, not a "
            "layer-specific Assistant Axis."
        ),
    }
    write_json(HERE / "axis_mapping.json", axis_mapping)
    matrix_report = {
        "smoke_cells": len(smoke),
        "formal_cells": len(formal),
        "formal_official": formal_counts["official"],
        "formal_engineering_noop": formal_counts["engineering_noop"],
        "formal_random": formal_counts["random_reference"],
        "conditions": list(ALL_CONDITIONS),
        "capture_layers": list(CAPTURE_LAYERS),
        "reporting_layers": list(REPORTING_LAYERS),
        "trace_sources": list(TRACE_SOURCES),
        "smoke_prompt_ids": smoke_ids,
        "families": 12,
        "prompts": 96,
        "arms": dict(Counter(row["arm"] for row in traces)),
    }
    return {
        "cap_metadata": cap_metadata,
        "axis_shape": list(axis.shape),
        "axis_dtype": str(axis.dtype),
        "axis_mapping": axis_mapping,
        "random_directions": direction_metadata,
        "official_axis_units_sha256": tensor_sha256(official_axes),
        "matrix": matrix_report,
    }, matrix_report


def run_quality_gates(skip_tests: bool) -> dict[str, Any]:
    if skip_tests:
        return {"status": "skipped", "reason": "--skip-tests"}
    pytest_args = (
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_rg1p_stagea.py",
        "tests/test_capping.py",
        "tests/test_checkpoint.py",
        "tests/test_e2b_fixed_trace_cached.py",
    )
    lint_args = (
        sys.executable,
        "-m",
        "ruff",
        "check",
        "datasets/m2_2b_formal/rg1p_stagea",
        "scripts/run_rg1p_stagea.py",
        "scripts/audit_rg1p_stagea.py",
        "scripts/analyze_rg1p_stagea.py",
        "scripts/plot_rg1p_stagea.py",
        "tests/test_rg1p_stagea.py",
    )
    pytest = command(*pytest_args, check=False)
    lint = command(*lint_args, check=False)
    report = {
        "schema": "cedar.rg1p_stagea.cpu_test_report.v1",
        "gpu_used": False,
        "pytest": {
            "returncode": pytest.returncode,
            "stdout": pytest.stdout,
            "stderr": pytest.stderr,
        },
        "ruff": {
            "returncode": lint.returncode,
            "stdout": lint.stdout,
            "stderr": lint.stderr,
        },
        "status": "passed" if pytest.returncode == lint.returncode == 0 else "failed",
    }
    write_json(HERE / "CPU_TEST_REPORT.json", report)
    require(report["status"] == "passed", "CPU tests or lint failed")
    return report


def main() -> None:
    args = parse_args()
    require(Path(sys.prefix).name == "cedar", f"Expected cedar environment: {sys.prefix}")
    git = git_checks()
    sources = validate_sources()
    forward_equivalence = validate_forward_path_equivalence()
    release, matrix_report = build_release(args.hf_home)
    tests = run_quality_gates(args.skip_tests)
    require(tests.get("status") == "passed", "CPU quality gates must run and pass")

    release_paths = [
        PROTOCOL,
        HERE / "rg1p_common.py",
        HERE / "validate_rg1p_stagea.py",
        HERE / "smoke_matrix72.jsonl",
        HERE / "formal_matrix864.jsonl",
        HERE / "random_directions.safetensors",
        HERE / "official_axis_units.safetensors",
        HERE / "random_direction_metadata.json",
        HERE / "axis_mapping.json",
        HERE / "ANALYSIS_PLAN.json",
        HERE / "CPU_TEST_REPORT.json",
        ROOT / "scripts/run_rg1p_stagea.py",
        ROOT / "scripts/audit_rg1p_stagea.py",
        ROOT / "scripts/analyze_rg1p_stagea.py",
        ROOT / "scripts/plot_rg1p_stagea.py",
        ROOT / "tests/test_rg1p_stagea.py",
        ROOT / "docs/RG1P_STAGEA_ENGINEERING_CORRECTION_01.md",
    ]
    require(all(path.is_file() for path in release_paths), "RG1-P release file missing")
    release_files = {
        str(path.relative_to(ROOT)): (
            file_sha256(path) if path.suffix == ".safetensors" else canonical_text_sha256(path)
        )
        for path in release_paths
    }
    manifest = {
        "schema": "cedar.rg1p_stagea.manifest.v1",
        "status": "cpu_gate_ready",
        "gpu_used": False,
        "base_result_commit": BASE_RESULT_COMMIT,
        "model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "dtype": MODEL_DTYPE,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
        },
        "p25": {
            "experiment": CAP_EXPERIMENT,
            "layers": list(CAP_LAYERS),
            "axis_sha256": EXPECTED_AXIS_SHA256,
            "capping_config_sha256": EXPECTED_CAP_CONFIG_SHA256,
        },
        "replay_engine": "model.generate_kv_cache_forced_tokens",
        "terminal_probe": "e2b_capture_call_index",
        "matrix": matrix_report,
        "statistics": {
            "bootstrap_unit": "family",
            "families": 12,
            "draws": BOOTSTRAP_DRAWS,
            "seed": BOOTSTRAP_SEED,
            "layerwise_descriptive_draws": LAYER_BOOTSTRAP_DRAWS,
            "probe_family_label_flip_permutations": PROBE_PERMUTATIONS,
        },
        "capture": {
            "layers": list(CAPTURE_LAYERS),
            "num_hidden_layers": len(CAPTURE_LAYERS),
            "unique_complete_module_mapping": list(CAPTURE_LAYERS)
            == list(range(len(CAPTURE_LAYERS))),
            "reporting_layers": list(REPORTING_LAYERS),
            "storage_dtype": "bfloat16",
        },
        "logit_lens": {
            "path": "frozen_final_norm_then_full_checkpoint_lm_head",
            "execution_shape": "one_layer_per_call_[1,1,5120]_matching_terminal_decode",
            "engineering_correction": "RG1P_STAGEA_ENGINEERING_CORRECTION_01",
            "candidate_token_ids": {"A": 32, "B": 33, "C": 34, "D": 35},
            "final_layer_max_abs_tolerance": 0.001,
            "full_vocabulary_kl_appendix_only": True,
            "tuned_lens": False,
        },
        "engineering_correction": {
            "id": ENGINEERING_CORRECTION_ID,
            "failed_readout_cache_preserved": "lens_records/",
            "corrected_readout_cache": "lens_records_shape_faithful_v2/",
            "smoke_completed_forwards_preserved": 40,
            "gpu_forward_path_equivalence": forward_equivalence,
        },
        "engineering_noop": {
            "trace_source": PRIMARY_TRACE_SOURCE,
            "condition": NOOP_CONDITION,
            "smoke_rows": 8,
            "formal_rows": 96,
            "kl_opt_strict_upper_bound": 1e-6,
            "delta_h_l2_strict_upper_bound": 1e-4,
            "excluded_from_scientific_estimands": True,
        },
        "random_reference": {
            "seeds": list(RANDOM_SEEDS),
            "conditions": list(RANDOM_CONDITIONS),
            "orthogonal_tolerance": RANDOM_ORTHOGONAL_TOLERANCE,
            "unit_tolerance": RANDOM_UNIT_TOLERANCE,
            "schedule": "same_layer_same_token_same_official_actual_correction_norm",
        },
        "source_hashes": sources,
        "release_files": release_files,
        "release_object_sha256": object_sha256(release_files),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "git_at_build": git,
        "asset_validation": release,
    }
    write_json(HERE / "manifest.json", manifest)
    gate = {
        "schema": "cedar.rg1p_stagea.cpu_gate.v1",
        "gate": "RG1-P-Stage-A-fixed-trace-propagation",
        "status": "passed",
        "gpu_authorized": True,
        "authorized_scope": "RG1P_StageA_72_smoke_then_864_formal_only",
        "stage_b_authorized": False,
        "patching_authorized": False,
        "gpu_used": False,
        "manifest_canonical_lf_sha256": canonical_text_sha256(HERE / "manifest.json"),
        "cpu_test_report_canonical_lf_sha256": canonical_text_sha256(HERE / "CPU_TEST_REPORT.json"),
        "blockers": [],
        "gpu_preconditions": [
            "clean worktree",
            "HEAD equals upstream",
            "exactly four visible GPUs",
            "72-cell smoke audit passed before formal",
            "no-op all-layer and final-layer native-lens anchors passed",
        ],
    }
    write_json(HERE / "CPU_GATE.json", gate)
    print(json.dumps({"gate": gate, "matrix": matrix_report}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
