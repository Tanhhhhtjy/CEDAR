#!/usr/bin/env python3
"""Run the frozen 192-cell Evidence-only uncapped/p25 validation matrix."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA_ROOT = ROOT / "datasets" / "m2_2b_formal" / "prover_generator_evidence_validation_v1_1"
for path in (SRC, DATA_ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cedar_m2.assets.capping_config import (
    download_qwen_capping_assets,
    load_axis,
    load_capping_experiment,
    verify_axis_orientation,
)
from cedar_m2.checkpoint import RecordStore, atomic_write_json, file_sha256, initialize_run, object_sha256
from cedar_m2.interventions.capping import ActivationCapper
from cedar_m2.models.qwen import QwenModelSpec, load_qwen_model
from evidence_validation_common import (
    CAP_EXPERIMENT,
    CAP_LAYERS,
    CONDITIONS,
    EXPECTED_AXIS_SHA256,
    EXPECTED_CAP_CONFIG_SHA256,
    EXPECTED_RESULT_ROWS,
    MODEL_REPO,
    MODEL_REVISION,
    PARENT_RESULT_COMMIT,
    canonical_sha256,
    canonical_text_sha256,
    file_sha256 as data_file_sha256,
    ordered_run_cells,
    read_json,
    select_validation_rows,
    validate_module_selection,
)
from run_cpu_preflight import direction_sha256, resolve_hub_cache

GATE_PATH = DATA_ROOT / "EVIDENCE_VALIDATION_GATE.json"
MANIFEST_PATH = DATA_ROOT / "manifest.json"
FINGERPRINT_PATH = DATA_ROOT / "validation_input_fingerprints.json"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "m2" / "prover_generator_evidence_validation_v1_1"
MAX_NEW_TOKENS = 4096
SEED = 0
MODEL_DTYPE = "bfloat16"
ATTENTION_IMPLEMENTATION = "sdpa"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "generate", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--hf-home", type=Path, default=Path(os.environ.get("HF_HOME", "/data1/WuMingrui/HF_HOME")))
    parser.add_argument("--asset-cache-dir", type=Path)
    parser.add_argument("--gpu-memory-gib", type=int, default=18)
    return parser.parse_args()


def atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def git_provenance() -> dict[str, Any]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.rstrip()
    if status:
        raise RuntimeError("Evidence validation refuses to run from a dirty worktree")
    upstream = subprocess.run(["git", "rev-parse", "@{u}"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    if commit != upstream:
        raise RuntimeError("Evidence validation requires local HEAD to equal upstream")
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", PARENT_RESULT_COMMIT, commit], cwd=ROOT)
    if ancestor.returncode != 0:
        raise RuntimeError("Runner HEAD must descend from the frozen calibration result")
    return {
        "commit": commit,
        "branch": branch,
        "upstream_commit": upstream,
        "head_matches_upstream": True,
        "parent_result_commit": PARENT_RESULT_COMMIT,
        "parent_result_is_ancestor": True,
        "dirty": False,
    }


def validate_inputs() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_module_selection()
    rows = select_validation_rows()
    gate = read_json(GATE_PATH)
    manifest = read_json(MANIFEST_PATH)
    fingerprints = read_json(FINGERPRINT_PATH)
    if (
        gate.get("gate") != "E1-paired-evidence-validation"
        or gate.get("may_run_paired_validation") is not True
        or gate.get("p25_authorized_scope") != "evidence_only_internal_validation"
        or gate.get("p1_authorized") is not False
        or gate.get("social_formal_prover_authorized") is not False
    ):
        raise ValueError("Evidence-only Gate E1 is not frozen")
    if (
        manifest.get("status") != "ready_for_paired_evidence_validation"
        or manifest.get("formal_rows") != EXPECTED_RESULT_ROWS
        or manifest.get("conditions") != list(CONDITIONS)
        or manifest.get("p25_authorized_scope") != "evidence_only_internal_validation"
    ):
        raise ValueError("Evidence validation manifest is not ready")
    if canonical_text_sha256(FINGERPRINT_PATH) != manifest.get("input_fingerprints_canonical_sha256"):
        raise ValueError("Validation input fingerprints changed after Gate E1")
    for relative, expected in manifest.get("release_files", {}).items():
        path = ROOT / relative
        if not path.is_file() or canonical_text_sha256(path) != expected.get("canonical_text_sha256"):
            raise ValueError(f"Release source changed after Gate E1: {relative}")
    if fingerprints.get("status") != "passed" or fingerprints.get("selection", {}).get("rows") != 96:
        raise ValueError("Validation input fingerprints are absent or stale")
    if fingerprints.get("selection", {}).get("ids_sha256") != canonical_sha256([row["id"] for row in rows]):
        raise ValueError("Validation selected IDs changed")
    if fingerprints.get("selection", {}).get("messages_sha256") != canonical_sha256(
        [{"id": row["id"], "messages": row["messages"]} for row in rows]
    ):
        raise ValueError("Validation messages changed")
    return rows, fingerprints


def hook_counts(model: Any) -> dict[str, int]:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Pinned Qwen model does not expose model.layers")
    return {str(layer): len(model.model.layers[layer]._forward_hooks) for layer in CAP_LAYERS}


def stats_to_dict(stats: Mapping[int, Any]) -> dict[str, Any]:
    by_layer = {
        str(layer): {
            "total_positions": value.total_positions,
            "triggered_positions": value.triggered_positions,
            "trigger_rate": value.trigger_rate,
            "projection_correction_sum": value.projection_correction_sum,
            "mean_projection_correction": value.mean_projection_correction,
        }
        for layer, value in sorted(stats.items())
    }
    total = sum(value.total_positions for value in stats.values())
    triggered = sum(value.triggered_positions for value in stats.values())
    correction = sum(value.projection_correction_sum for value in stats.values())
    return {
        "definition": "batch-size-1 decoder token positions pooled over official layers 46-53",
        "by_layer": by_layer,
        "pooled": {
            "total_positions": total,
            "triggered_positions": triggered,
            "trigger_rate": triggered / total if total else 0.0,
            "projection_correction_sum": correction,
            "mean_projection_correction": correction / total if total else 0.0,
        },
    }


def load_p25_assets(cache_dir: Path) -> tuple[dict[int, Any], dict[str, Any]]:
    axis_path, config_path = download_qwen_capping_assets(cache_dir=resolve_hub_cache(cache_dir), local_files_only=True)
    axis_hash = data_file_sha256(axis_path)
    config_hash = data_file_sha256(config_path)
    if axis_hash != EXPECTED_AXIS_SHA256 or config_hash != EXPECTED_CAP_CONFIG_SHA256:
        raise ValueError("Frozen Assistant Axis or p25 config SHA-256 changed")
    axis = load_axis(axis_path)
    caps = load_capping_experiment(config_path, CAP_EXPERIMENT)
    if tuple(sorted(caps)) != CAP_LAYERS:
        raise ValueError("Official p25 layer range changed")
    orientation = verify_axis_orientation(caps, axis)
    metadata = {
        "axis_sha256": axis_hash,
        "capping_config_sha256": config_hash,
        "experiment": CAP_EXPERIMENT,
        "layers": {
            str(layer): {
                "threshold": caps[layer].threshold,
                "direction_sha256": direction_sha256(caps[layer]),
                "axis_cosine": orientation[layer],
            }
            for layer in CAP_LAYERS
        },
        "positions": "all",
    }
    return caps, metadata


def record_key(index: int, row: Mapping[str, Any], condition: str) -> dict[str, Any]:
    return {"index": index, "id": row["id"], "condition": condition, "seed": SEED}


def materialize(store: RecordStore, cells: Sequence[tuple[dict[str, Any], str]], results_path: Path, runtime_path: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    for index, (row, condition) in enumerate(cells):
        payload = store.read(record_key(index, row, condition))
        if payload is None:
            continue
        result = payload.get("result")
        if (
            payload.get("index") != index
            or not isinstance(result, dict)
            or result.get("id") != row["id"]
            or result.get("condition") != condition
        ):
            raise ValueError(f"Checkpoint mismatch at cell {index}")
        results.append(result)
        payloads.append(payload)
    atomic_write_jsonl(results_path, results)
    atomic_write_jsonl(runtime_path, payloads)
    return results


def update_runtime(path: Path, values: Mapping[str, Any]) -> None:
    runtime = read_json(path) if path.is_file() else {}
    runtime.update(values)
    atomic_write_json(path, runtime)


def configuration(rows: Sequence[dict[str, Any]], git: Mapping[str, Any], fingerprints: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": "CEDAR Evidence-only prespecified internal validation",
        "scope": "evidence_only_internal_validation",
        "development_split": "development_validation",
        "conditions": list(CONDITIONS),
        "paired_order": "each row uncapped then p25",
        "single_uninterrupted_matrix": True,
        "checkpoint_resume_allowed_without_selection": True,
        "seed": SEED,
        "dataset": {
            "parent_path": str((DATA_ROOT.parent / "prover_generator_development_v1_1" / "development_candidates336.jsonl").relative_to(ROOT)).replace("\\", "/"),
            "input_rows": len(rows),
            "result_rows": EXPECTED_RESULT_ROWS,
            "ids_sha256": canonical_sha256([row["id"] for row in rows]),
            "messages_sha256": fingerprints["selection"]["messages_sha256"],
            "families": sorted({row["family_id"] for row in rows}),
        },
        "model": {"repo_id": MODEL_REPO, "revision": MODEL_REVISION, "dtype": MODEL_DTYPE, "attention_implementation": ATTENTION_IMPLEMENTATION},
        "generation": {"batch_size": 1, "mode": "thinking", "do_sample": False, "num_beams": 1, "max_new_tokens": MAX_NEW_TOKENS, "content_retry": False, "answer_repair": False},
        "truncation_policy": {"classification": "invalid_or_abstain", "retry": False, "manual_credit": False, "itt_primary": True},
        "prohibited": ["AO", "S", "p1", "item_deletion", "LoFa", "J-Lens", "held_out_confirmation_families"],
        "git": dict(git),
    }


def main() -> None:
    args = parse_args()
    if args.gpu_memory_gib <= 0:
        raise ValueError("GPU memory must be positive")
    rows, fingerprints = validate_inputs()
    cells = ordered_run_cells(rows)
    if len(cells) != EXPECTED_RESULT_ROWS:
        raise ValueError("Paired validation matrix changed")
    git = git_provenance()
    run_dir = args.output_root / "formal-paired-uncapped-p25-seed0"
    manifest = initialize_run(run_dir, configuration(rows, git, fingerprints))
    store = RecordStore(run_dir / "records", run_fingerprint=str(manifest["fingerprint"]))
    results_path = run_dir / "results.jsonl"
    record_runtime_path = run_dir / "record_runtime.jsonl"
    runtime_path = run_dir / "runtime.json"
    results = materialize(store, cells, results_path, record_runtime_path)
    if args.stage in {"prepare", "all"}:
        print(
            json.dumps(
                {
                    "stage": "prepare",
                    "input_rows": len(rows),
                    "result_rows": len(cells),
                    "existing": len(results),
                    "conditions": list(CONDITIONS),
                    "run_dir": str(run_dir),
                    "runner_commit": git["commit"],
                    "gpu_used": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if args.stage == "prepare":
            return

    missing = [index for index, (row, condition) in enumerate(cells) if not store.contains(record_key(index, row, condition))]
    if missing:
        import torch
        import transformers
        import run_prover_generator_step5_uncapped as base

        if Path(sys.prefix).name != "cedar":
            raise RuntimeError(f"Generation requires cedar, found {sys.prefix}")
        if torch.cuda.device_count() != 4:
            raise RuntimeError("Evidence validation requires exactly four visible GPUs")
        os.environ["HF_HOME"] = str(args.hf_home.resolve())
        base.MAX_NEW_TOKENS = MAX_NEW_TOKENS
        base.SEED = SEED
        caps, cap_metadata = load_p25_assets(args.asset_cache_dir.resolve() if args.asset_cache_dir else args.hf_home.resolve())
        max_memory = {index: f"{args.gpu_memory_gib}GiB" for index in range(4)}
        spec = QwenModelSpec(repo_id=MODEL_REPO, revision=MODEL_REVISION, dtype=MODEL_DTYPE, attention_implementation=ATTENTION_IMPLEMENTATION)
        model, tokenizer = load_qwen_model(spec, max_memory=max_memory, local_files_only=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base.validate_label_tokens(tokenizer)
        rendered = [base.render_prompt(tokenizer, row, base.MODES["thinking"]) for row in rows]
        tokenized = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in rendered]
        rendered_hash = object_sha256([{"id": row["id"], "prompt": prompt} for row, prompt in zip(rows, rendered, strict=True)])
        input_ids_hash = object_sha256([{"id": row["id"], "input_ids": ids} for row, ids in zip(rows, tokenized, strict=True)])
        if rendered_hash != fingerprints["rendered_prompts_sha256"] or input_ids_hash != fingerprints["tokenized_input_ids_sha256"]:
            raise RuntimeError("Frozen validation rendered/tokenized input hash changed")
        update_runtime(
            runtime_path,
            {
                "generation_started_at": datetime.now().astimezone().isoformat(),
                "environment": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "visible_gpu_count": torch.cuda.device_count(), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "hf_home": str(args.hf_home.resolve())},
                "rendered_prompts_sha256": rendered_hash,
                "tokenized_input_ids_sha256": input_ids_hash,
                "official_p25": cap_metadata,
            },
        )
        for index in missing:
            row, condition = cells[index]
            before = hook_counts(model)
            capper = ActivationCapper(model, caps, positions="all", track_stats=True, defer_stats_sync=True) if condition == "p25" else None
            if capper is None:
                result = base.generate_one(model, tokenizer, transformers, row, scope="evidence-validation", mode_name="thinking", record_seed=index // 2)
                cap_stats = None
            else:
                with capper:
                    result = base.generate_one(model, tokenizer, transformers, row, scope="evidence-validation", mode_name="thinking", record_seed=index // 2)
                cap_stats = stats_to_dict(capper.stats)
            after = hook_counts(model)
            if after != before:
                raise RuntimeError(f"Forward-hook leak at cell {index}: before={before}, after={after}")
            result.update(
                {
                    "condition": condition,
                    "phase": "prespecified_internal_validation",
                    "development_split": row["development_split"],
                    "position_transition": row["position_transition"],
                    "pair_index": index // 2,
                    "paired_condition_order": list(CONDITIONS),
                    "cap_stats": cap_stats,
                    "hook_counts_before": before,
                    "hook_counts_after": after,
                }
            )
            store.write(record_key(index, row, condition), {"index": index, "result": result})
            results = materialize(store, cells, results_path, record_runtime_path)
            print(json.dumps({"stage": "generate", "complete": len(results), "expected": len(cells), "id": row["id"], "condition": condition}), flush=True)
        del model, tokenizer
        base.release_cuda_cache()

    results = materialize(store, cells, results_path, record_runtime_path)
    if len(results) == len(cells):
        p25_triggered = sum(
            int(row["cap_stats"]["pooled"]["triggered_positions"])
            for row in results
            if row["condition"] == "p25" and row.get("cap_stats")
        )
        update_runtime(
            runtime_path,
            {
                "generation_completed_at": datetime.now().astimezone().isoformat(),
                "complete_results": len(results),
                "results_sha256": file_sha256(results_path),
                "record_runtime_sha256": file_sha256(record_runtime_path),
                "content_retries": 0,
                "answer_repairs": 0,
                "parse_failures": [f"{row['id']}::{row['condition']}" for row in results if row["final_answer_parse_status"] != "passed"],
                "truncations": [f"{row['id']}::{row['condition']}" for row in results if row.get("truncated")],
                "missing_answer_logits": [f"{row['id']}::{row['condition']}" for row in results if row.get("answer_logits") is None],
                "p25_triggered_positions": p25_triggered,
                "single_matrix_completed_without_content_selection": True,
            },
        )
    print(json.dumps({"complete": len(results), "expected": len(cells), "scope": "evidence_only_internal_validation"}), flush=True)


if __name__ == "__main__":
    main()
