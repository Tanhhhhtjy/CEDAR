#!/usr/bin/env python3
"""Run the frozen E2B crossed fixed-trace teacher-forcing matrix."""

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

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA = ROOT / "datasets/m2_2b_formal/e2b_fixed_trace_cross"
E2A_DATA = ROOT / "datasets/m2_2b_formal/e2_phase_validity_choice"
for path in (SRC, DATA, E2A_DATA, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_e2_phase_validity_choice as e2run  # noqa: E402
from e2b_common import (  # noqa: E402
    ANCHOR_RESULT_COMMIT,
    ATTENTION_IMPLEMENTATION,
    CAP_EXPERIMENT,
    CAP_LAYERS,
    EXPECTED_AXIS_SHA256,
    EXPECTED_CAP_CONFIG_SHA256,
    FORWARD_CONDITIONS,
    LABELS,
    MODEL_DTYPE,
    MODEL_REPO,
    MODEL_REVISION,
    canonical_text_sha256,
    file_sha256,
    object_sha256,
    ordered_cells,
    read_json,
    read_jsonl,
    require,
)

from cedar_m2.checkpoint import RecordStore, atomic_write_json, initialize_run  # noqa: E402
from cedar_m2.interventions.capping import ActivationCapper  # noqa: E402
from cedar_m2.interventions.phase_capping import PhaseActivationCapper  # noqa: E402
from cedar_m2.models.qwen import QwenModelSpec, load_qwen_model, model_input_device  # noqa: E402

DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/m2/e2b_fixed_trace_cross"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("smoke", "formal"), required=True)
    parser.add_argument("--stage", choices=("prepare", "generate", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--hf-home", type=Path, default=Path(os.environ.get("HF_HOME", "<local-path>"))
    )
    parser.add_argument("--asset-cache-dir", type=Path)
    parser.add_argument("--gpu-memory-gib", type=int, default=18)
    return parser.parse_args()


def git_provenance(*, require_synced: bool) -> dict[str, Any]:
    def command(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", *args], cwd=ROOT, check=check, capture_output=True, text=True)

    commit = command("rev-parse", "HEAD").stdout.strip()
    branch = command("branch", "--show-current").stdout.strip()
    status = command("status", "--porcelain").stdout.rstrip()
    upstream_result = command("rev-parse", "@{u}", check=False)
    upstream = upstream_result.stdout.strip() if upstream_result.returncode == 0 else None
    ancestor = command("merge-base", "--is-ancestor", ANCHOR_RESULT_COMMIT, commit, check=False)
    require(ancestor.returncode == 0, "E2B runner HEAD must descend from the E2A result anchor")
    if require_synced:
        require(not status, "E2B GPU runner requires a clean worktree")
        require(
            upstream is not None and upstream == commit,
            "E2B GPU runner requires HEAD == upstream",
        )
    return {
        "commit": commit,
        "branch": branch,
        "upstream_commit": upstream,
        "head_matches_upstream": upstream == commit if upstream else False,
        "dirty": bool(status),
        "anchor_is_ancestor": True,
    }


def validate_release() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    traces = read_jsonl(DATA / "frozen_traces192.jsonl")
    manifest = read_json(DATA / "manifest.json")
    gate = read_json(DATA / "CPU_GATE.json")
    require(manifest.get("status") == "cpu_gate_ready", "E2B manifest is not ready")
    require(
        gate.get("gpu_authorized") is True and gate.get("authorized_scope") == "E2B_only",
        "E2B CPU Gate does not authorize this run",
    )
    for relative, expected in manifest["release_files"].items():
        path = ROOT / relative
        require(path.is_file(), f"Missing E2B release file: {relative}")
        require(canonical_text_sha256(path) == expected, f"E2B release hash changed: {relative}")
    require(
        canonical_text_sha256(DATA / "frozen_traces192.jsonl")
        == manifest["data_hashes"]["frozen_traces192.jsonl"],
        "Frozen traces changed",
    )
    require(
        canonical_text_sha256(DATA / "formal_matrix384.jsonl")
        == manifest["data_hashes"]["formal_matrix384.jsonl"],
        "Formal matrix changed",
    )
    return traces, manifest, gate


def hook_counts(model: Any) -> dict[str, int]:
    require(hasattr(model, "model") and hasattr(model.model, "layers"), "Qwen layers unavailable")
    return {str(layer): len(model.model.layers[layer]._forward_hooks) for layer in CAP_LAYERS}


def full_stats(stats: Mapping[int, Any]) -> dict[str, Any]:
    return {
        "by_layer": {
            str(layer): {
                "total_positions": value.total_positions,
                "triggered_positions": value.triggered_positions,
                "trigger_rate": value.trigger_rate,
                "projection_correction_sum": value.projection_correction_sum,
                "mean_projection_correction": value.mean_projection_correction,
            }
            for layer, value in sorted(stats.items())
        }
    }


def diagnostics(logits: Tensor, trace: Mapping[str, Any], tokenizer: Any) -> dict[str, Any]:
    result = e2run.option_diagnostics(logits, LABELS, trace["candidate_token_ids"])
    option_logits = result["logits"]
    initial = str(trace["initial_answer"])
    target = str(trace["target_answer"])
    result.update(
        {
            "initial_logit": option_logits[initial],
            "target_logit": option_logits[target],
            "target_minus_initial": option_logits[target] - option_logits[initial],
            "initial_rank": result["ranks"][initial],
            "target_rank": result["ranks"][target],
            "full_vocab_top_token_text": tokenizer.decode(
                [result["full_vocab_top_token_id"]], skip_special_tokens=False
            ),
            "top_token_is_option": result["full_vocab_top_token_id"]
            in set(trace["candidate_token_ids"].values()),
        }
    )
    return result


def run_one(
    model: Any,
    tokenizer: Any,
    caps: Mapping[int, Any],
    trace: Mapping[str, Any],
    condition: str,
) -> dict[str, Any]:
    prompt_row = next(
        row
        for row in read_jsonl(E2A_DATA / "answer_inputs96.jsonl")
        if row["id"] == trace["source_id"]
    )
    import run_prover_generator_step5_uncapped as base

    prompt = base.render_prompt(tokenizer, prompt_row, base.MODES["thinking"])
    prompt_ids = [int(value) for value in tokenizer.encode(prompt, add_special_tokens=False)]
    prefix_ids = [int(value) for value in trace["trace_prefix_token_ids"]]
    require(object_sha256(prompt) == trace["prompt_sha256"], "Prompt hash drift before forward")
    require(
        object_sha256(prompt_ids) == trace["prompt_token_ids_sha256"],
        "Prompt-token hash drift before forward",
    )
    require(
        object_sha256(prefix_ids) == trace["trace_prefix_token_ids_sha256"],
        "Trace-prefix token hash drift before forward",
    )
    input_ids_list = [*prompt_ids, *prefix_ids]
    require(
        object_sha256(input_ids_list) == trace["teacher_forced_input_token_ids_sha256"],
        "Teacher-forced input hash drift",
    )
    device = model_input_device(model)
    input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    before = hook_counts(model)
    monitor = PhaseActivationCapper(
        model,
        caps,
        enabled_phases=("prefill", "decode"),
        apply_changes=False,
        defer_stats_sync=True,
    )
    capper: ActivationCapper | None = None
    if condition == "full_p25":
        capper = ActivationCapper(
            model, caps, positions="all", track_stats=True, defer_stats_sync=True
        )
    elif condition != "uncapped":
        raise ValueError(f"Unknown E2B condition: {condition}")
    with torch.inference_mode():
        if capper is None:
            with monitor:
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
        else:
            with monitor:
                with capper:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=True,
                        return_dict=True,
                    )
    logits = outputs.logits[0, -1]
    observed = diagnostics(logits, trace, tokenizer)
    after = hook_counts(model)
    require(before == after, f"Hook leak: before={before}, after={after}")
    del outputs, input_ids, attention_mask
    return {
        "schema": "cedar.e2b_fixed_trace_cross.result.v1",
        "cell_id": f"{trace['trace_id']}::{condition}",
        "source_id": trace["source_id"],
        "trace_id": trace["trace_id"],
        "trace_source": trace["trace_source"],
        "source_condition": trace["source_condition"],
        "forward_condition": condition,
        "is_diagonal_replay": trace["source_condition"] == condition,
        "family_id": trace["family_id"],
        "base_id": trace["base_id"],
        "structure_class": trace["structure_class"],
        "variant": trace["variant"],
        "arm": trace["arm"],
        "initial_answer": trace["initial_answer"],
        "target_answer": trace["target_answer"],
        "gold_answer": trace["gold_answer"],
        "prompt_sha256": trace["prompt_sha256"],
        "prompt_token_ids_sha256": trace["prompt_token_ids_sha256"],
        "trace_prefix_text_sha256": trace["trace_prefix_text_sha256"],
        "trace_prefix_token_ids_sha256": trace["trace_prefix_token_ids_sha256"],
        "teacher_forced_input_token_ids_sha256": trace[
            "teacher_forced_input_token_ids_sha256"
        ],
        "prompt_tokens": len(prompt_ids),
        "trace_prefix_tokens": len(prefix_ids),
        "teacher_forced_input_tokens": len(input_ids_list),
        "candidate_token_ids": trace["candidate_token_ids"],
        "logit_diagnostics": observed,
        "anchor_answer_logits": trace["anchor_answer_logits"],
        "monitor_telemetry": monitor.telemetry(),
        "full_cap_stats": full_stats(capper.stats) if capper else None,
        "hook_counts_before": before,
        "hook_counts_after": after,
        "generated_text": False,
        "retry_count": 0,
        "answer_repair_performed": False,
    }


def atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
            newline="\n",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def record_key(index: int, trace: Mapping[str, Any], condition: str) -> dict[str, Any]:
    return {
        "index": index,
        "trace_id": trace["trace_id"],
        "forward_condition": condition,
    }


def materialize(
    store: RecordStore,
    matrix: Sequence[tuple[dict[str, Any], str]],
    results_path: Path,
) -> list[dict[str, Any]]:
    results = []
    for index, (trace, condition) in enumerate(matrix):
        payload = store.read(record_key(index, trace, condition))
        if payload is not None:
            require(payload.get("index") == index, f"Checkpoint index mismatch: {index}")
            results.append(payload["result"])
    atomic_write_jsonl(results_path, results)
    return results


def main() -> None:
    args = parse_args()
    require(args.gpu_memory_gib == 18, "E2B requires the frozen 18 GiB per-GPU cap")
    traces, manifest, _gate = validate_release()
    if args.scope == "formal" and args.stage in {"generate", "all"}:
        smoke_audit_path = args.output_root / "SMOKE_AUDIT.json"
        require(smoke_audit_path.is_file(), "Formal E2B is blocked until smoke audit exists")
        require(
            read_json(smoke_audit_path).get("status") == "passed",
            "Formal E2B is blocked until smoke audit passes",
        )
    matrix = ordered_cells(traces, scope=args.scope, smoke_ids=manifest["smoke_prompt_ids"])
    git = git_provenance(require_synced=args.stage in {"generate", "all"})
    run_dir = args.output_root / f"{args.scope}-fixed-trace-seed0"
    configuration = {
        "schema": "cedar.e2b_fixed_trace_cross.run.v1",
        "scope": args.scope,
        "cells": len(matrix),
        "trace_sources": list(manifest["trace_sources"]),
        "forward_conditions": list(FORWARD_CONDITIONS),
        "model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "dtype": MODEL_DTYPE,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
        },
        "intervention": {"experiment": CAP_EXPERIMENT, "layers": list(CAP_LAYERS)},
        "teacher_forcing": {"batch_size": 1, "generation": False},
        "git": git,
    }
    initialized = initialize_run(run_dir, configuration)
    store = RecordStore(run_dir / "records", run_fingerprint=str(initialized["fingerprint"]))
    results_path = run_dir / "results.jsonl"
    results = materialize(store, matrix, results_path)
    print(
        json.dumps(
            {
                "stage": "prepare",
                "scope": args.scope,
                "cells": len(matrix),
                "existing": len(results),
                "gpu_used": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if args.stage == "prepare":
        return

    import transformers

    require(Path(sys.prefix).name == "cedar", f"Generation requires cedar, found {sys.prefix}")
    require(torch.cuda.device_count() == 4, "E2B requires exactly four visible GPUs")
    os.environ["HF_HOME"] = str(args.hf_home.resolve())
    cache = args.asset_cache_dir.resolve() if args.asset_cache_dir else args.hf_home.resolve()
    caps, cap_metadata = e2run.load_p25_assets(cache)
    require(cap_metadata["axis_sha256"] == EXPECTED_AXIS_SHA256, "Assistant Axis hash drift")
    require(
        cap_metadata["capping_config_sha256"] == EXPECTED_CAP_CONFIG_SHA256,
        "Capping config hash drift",
    )
    spec = QwenModelSpec(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        dtype=MODEL_DTYPE,
        attention_implementation=ATTENTION_IMPLEMENTATION,
    )
    model, tokenizer = load_qwen_model(
        spec,
        max_memory={index: f"{args.gpu_memory_gib}GiB" for index in range(4)},
        local_files_only=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    import run_prover_generator_step5_uncapped as base

    base.validate_label_tokens(tokenizer)
    runtime_path = run_dir / "runtime.json"
    if not runtime_path.is_file():
        atomic_write_json(
            runtime_path,
            {
                "generation_started_at": datetime.now().astimezone().isoformat(),
                "environment": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "transformers": transformers.__version__,
                    "visible_gpu_count": torch.cuda.device_count(),
                    "visible_gpu_names": [
                        torch.cuda.get_device_name(index)
                        for index in range(torch.cuda.device_count())
                    ],
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                },
                "official_p25": cap_metadata,
                "resume_events": [],
            },
        )
    for index, (trace, condition) in enumerate(matrix):
        key = record_key(index, trace, condition)
        if store.contains(key):
            continue
        result = run_one(model, tokenizer, caps, trace, condition)
        store.write(key, {"index": index, "result": result})
        results = materialize(store, matrix, results_path)
        print(
            json.dumps(
                {
                    "stage": "generate",
                    "complete": len(results),
                    "expected": len(matrix),
                    "source_id": trace["source_id"],
                    "trace_source": trace["trace_source"],
                    "forward_condition": condition,
                }
            ),
            flush=True,
        )
    results = materialize(store, matrix, results_path)
    runtime = read_json(runtime_path)
    runtime.update(
        {
            "generation_completed_at": datetime.now().astimezone().isoformat(),
            "complete_results": len(results),
            "results_sha256": file_sha256(results_path),
            "nonfinite_logits": [
                row["cell_id"]
                for row in results
                if not all(
                    torch.isfinite(torch.tensor(value))
                    for value in row["logit_diagnostics"]["logits"].values()
                )
            ],
            "hook_leaks": [
                row["cell_id"]
                for row in results
                if row["hook_counts_before"] != row["hook_counts_after"]
            ],
        }
    )
    atomic_write_json(runtime_path, runtime)
    print(json.dumps({"complete": len(results), "expected": len(matrix)}), flush=True)


if __name__ == "__main__":
    main()
