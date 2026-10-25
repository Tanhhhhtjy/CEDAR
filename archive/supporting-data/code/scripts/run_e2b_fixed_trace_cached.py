#!/usr/bin/env python3
"""Run E2B v1.1 with cache-faithful forced incremental replay."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA = ROOT / "datasets/m2_2b_formal/e2b_fixed_trace_cached"
V1_DATA = ROOT / "datasets/m2_2b_formal/e2b_fixed_trace_cross"
E2A_DATA = ROOT / "datasets/m2_2b_formal/e2_phase_validity_choice"
for path in (SRC, DATA, V1_DATA, E2A_DATA, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_e2_phase_validity_choice as e2run  # noqa: E402
import run_e2b_fixed_trace_cross as v1run  # noqa: E402
from e2b_cached_common import (  # noqa: E402
    ATTENTION_IMPLEMENTATION,
    BASE_RESULT_COMMIT,
    BRANCH,
    CAP_EXPERIMENT,
    CAP_LAYERS,
    EXPECTED_AXIS_SHA256,
    EXPECTED_CAP_CONFIG_SHA256,
    FORWARD_CONDITIONS,
    MODEL_DTYPE,
    MODEL_REPO,
    MODEL_REVISION,
    SOURCE_MANIFEST,
    SOURCE_MATRIX,
    SOURCE_TRACES,
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

DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/m2/e2b_fixed_trace_cached_v1_1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("smoke", "formal"), required=True)
    parser.add_argument("--stage", choices=("prepare", "generate", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=Path(os.environ.get("HF_HOME", "<local-path>")),
    )
    parser.add_argument("--asset-cache-dir", type=Path)
    parser.add_argument("--gpu-memory-gib", type=int, default=18)
    return parser.parse_args()


def git_provenance(*, require_synced: bool) -> dict[str, Any]:
    def command(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=check, capture_output=True, text=True
        )

    commit = command("rev-parse", "HEAD").stdout.strip()
    branch = command("branch", "--show-current").stdout.strip()
    status = command("status", "--porcelain").stdout.rstrip()
    upstream_result = command("rev-parse", "@{u}", check=False)
    upstream = upstream_result.stdout.strip() if upstream_result.returncode == 0 else None
    ancestor = command("merge-base", "--is-ancestor", BASE_RESULT_COMMIT, commit, check=False)
    require(ancestor.returncode == 0, "Cached E2B HEAD must descend from the v1 failure result")
    require(branch == BRANCH, f"Cached E2B requires branch {BRANCH}, found {branch}")
    if require_synced:
        require(not status, "Cached E2B GPU runner requires a clean worktree")
        require(upstream == commit, "Cached E2B GPU runner requires HEAD == upstream")
    return {
        "commit": commit,
        "branch": branch,
        "upstream_commit": upstream,
        "head_matches_upstream": upstream == commit if upstream else False,
        "dirty": bool(status),
        "v1_failure_is_ancestor": True,
    }


def validate_release() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    traces = read_jsonl(SOURCE_TRACES)
    manifest = read_json(DATA / "manifest.json")
    gate = read_json(DATA / "CPU_GATE.json")
    require(manifest.get("status") == "cpu_gate_ready_cached_v1_1", "Manifest not ready")
    require(
        gate.get("gpu_authorized") is True
        and gate.get("authorized_scope") == "E2B_cached_v1_1_only",
        "Cached E2B CPU Gate does not authorize this run",
    )
    require(
        canonical_text_sha256(DATA / "manifest.json")
        == gate.get("manifest_canonical_lf_sha256"),
        "Cached E2B manifest hash changed",
    )
    for relative, expected in manifest["release_files"].items():
        path = ROOT / relative
        require(path.is_file(), f"Missing cached E2B release file: {relative}")
        require(
            canonical_text_sha256(path) == expected,
            f"Cached E2B release hash changed: {relative}",
        )
    source_hashes = manifest["source_hashes"]
    for path, key in (
        (SOURCE_TRACES, "frozen_traces192.jsonl"),
        (SOURCE_MATRIX, "formal_matrix384.jsonl"),
        (SOURCE_MANIFEST, "v1_manifest.json"),
    ):
        require(canonical_text_sha256(path) == source_hashes[key], f"Source hash changed: {key}")
    return traces, manifest, gate


class ProcessorOrderState:
    """Enforce capture-before-force and exact frozen-prefix state at each call."""

    def __init__(self, *, prompt_tokens: int, prefix_ids: Sequence[int]) -> None:
        self.prompt_tokens = prompt_tokens
        self.prefix_ids = tuple(int(value) for value in prefix_ids)
        self.capture_calls = 0
        self.force_calls = 0
        self.capture_call_index: int | None = None
        self.order_violations: list[str] = []

    def step(self, input_ids: Tensor) -> int:
        require(input_ids.ndim == 2 and input_ids.shape[0] == 1, "Replay requires batch size one")
        step = int(input_ids.shape[1]) - self.prompt_tokens
        require(0 <= step <= len(self.prefix_ids), f"Unexpected replay step: {step}")
        generated = tuple(int(value) for value in input_ids[0, self.prompt_tokens :].tolist())
        require(
            generated == self.prefix_ids[:step],
            f"Frozen replay prefix diverged at step {step}",
        )
        return step

    def before_capture(self, step: int) -> None:
        if self.capture_calls != self.force_calls or step != self.capture_calls:
            self.order_violations.append(
                f"capture:step={step}:capture={self.capture_calls}:force={self.force_calls}"
            )
            raise RuntimeError(self.order_violations[-1])
        self.capture_calls += 1

    def before_force(self, step: int) -> None:
        if self.capture_calls != self.force_calls + 1 or step != self.force_calls:
            self.order_violations.append(
                f"force:step={step}:capture={self.capture_calls}:force={self.force_calls}"
            )
            raise RuntimeError(self.order_violations[-1])
        self.force_calls += 1


class RawAnswerCapture:
    """Read raw answer-boundary scores before the forced-token processor."""

    def __init__(
        self,
        state: ProcessorOrderState,
        capture_fn: Callable[[Tensor], dict[str, Any]],
    ) -> None:
        self.state = state
        self.capture_fn = capture_fn
        self.observed: dict[str, Any] | None = None

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        step = self.state.step(input_ids)
        self.state.before_capture(step)
        if step == len(self.state.prefix_ids):
            require(self.observed is None, "Answer-boundary logits captured more than once")
            self.observed = self.capture_fn(scores[0])
            self.state.capture_call_index = step
        return scores


class FrozenTokenForcer:
    """Force the frozen prefix and one terminal probe label after raw capture."""

    def __init__(
        self,
        state: ProcessorOrderState,
        *,
        terminal_probe_token_id: int,
    ) -> None:
        self.state = state
        self.terminal_probe_token_id = int(terminal_probe_token_id)

    def __call__(self, input_ids: Tensor, scores: Tensor) -> Tensor:
        step = self.state.step(input_ids)
        self.state.before_force(step)
        token_id = (
            self.state.prefix_ids[step]
            if step < len(self.state.prefix_ids)
            else self.terminal_probe_token_id
        )
        require(0 <= token_id < scores.shape[-1], "Forced token ID is out of vocabulary")
        forced = torch.full_like(scores, -torch.inf)
        forced[:, token_id] = 0
        return forced


def run_one(
    model: Any,
    tokenizer: Any,
    transformers_module: Any,
    caps: Mapping[int, Any],
    prompt_rows: Mapping[str, Mapping[str, Any]],
    trace: Mapping[str, Any],
    condition: str,
) -> dict[str, Any]:
    import run_prover_generator_step5_uncapped as base

    prompt_row = prompt_rows[str(trace["source_id"])]
    prompt = base.render_prompt(tokenizer, prompt_row, base.MODES["thinking"])
    encoded = tokenizer([prompt], return_tensors="pt", padding=True, add_special_tokens=False)
    prompt_ids = [int(value) for value in encoded["input_ids"][0].tolist()]
    prefix_ids = [int(value) for value in trace["trace_prefix_token_ids"]]
    require(object_sha256(prompt) == trace["prompt_sha256"], "Prompt hash drift")
    require(object_sha256(prompt_ids) == trace["prompt_token_ids_sha256"], "Prompt-token drift")
    require(
        object_sha256(prefix_ids) == trace["trace_prefix_token_ids_sha256"],
        "Trace-prefix token drift",
    )
    require(
        object_sha256([*prompt_ids, *prefix_ids])
        == trace["teacher_forced_input_token_ids_sha256"],
        "Prompt-plus-prefix token drift",
    )
    encoded = {name: value.to(model_input_device(model)) for name, value in encoded.items()}
    state = ProcessorOrderState(prompt_tokens=len(prompt_ids), prefix_ids=prefix_ids)
    capture = RawAnswerCapture(
        state,
        lambda scores: v1run.diagnostics(scores, trace, tokenizer),
    )
    forcer = FrozenTokenForcer(
        state,
        terminal_probe_token_id=int(trace["generated_label_token_id"]),
    )
    before = v1run.hook_counts(model)
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
        raise ValueError(f"Unknown cached E2B condition: {condition}")

    generation_kwargs = {
        "do_sample": False,
        "num_beams": 1,
        "max_new_tokens": len(prefix_ids) + 1,
        "pad_token_id": tokenizer.pad_token_id,
        "use_cache": True,
        "return_dict_in_generate": True,
        "output_scores": False,
        "output_logits": False,
        "logits_processor": transformers_module.LogitsProcessorList([capture, forcer]),
    }
    transformers_module.set_seed(int(trace["anchor_record_seed"]))
    with torch.inference_mode():
        if capper is None:
            with monitor:
                generated = model.generate(**encoded, **generation_kwargs)
        else:
            with monitor:
                with capper:
                    generated = model.generate(**encoded, **generation_kwargs)

    replay_ids = [
        int(value) for value in generated.sequences[0, len(prompt_ids) :].tolist()
    ]
    require(replay_ids[:-1] == prefix_ids, "Forced replay emitted a changed trace prefix")
    require(
        replay_ids[-1] == int(trace["generated_label_token_id"]),
        "Terminal probe label changed",
    )
    require(capture.observed is not None, "Missing raw answer-boundary capture")
    require(state.capture_calls == len(prefix_ids) + 1, "Capture call count changed")
    require(state.force_calls == len(prefix_ids) + 1, "Force call count changed")
    require(state.capture_call_index == len(prefix_ids), "Capture boundary changed")
    require(not state.order_violations, "Capture-before-force ordering failed")
    after = v1run.hook_counts(model)
    require(before == after, f"Hook leak: before={before}, after={after}")
    result = {
        "schema": "cedar.e2b_fixed_trace_cached.result.v1_1",
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
        "teacher_forced_input_tokens": len(prompt_ids) + len(prefix_ids),
        "candidate_token_ids": trace["candidate_token_ids"],
        "forced_trace_token_ids_sha256": object_sha256(replay_ids[:-1]),
        "forced_trace_tokens": len(replay_ids) - 1,
        "terminal_probe_token_id": replay_ids[-1],
        "terminal_probe_label": trace["generated_label"],
        "processor_order": {
            "capture_calls": state.capture_calls,
            "force_calls": state.force_calls,
            "capture_call_index": state.capture_call_index,
            "capture_before_force_all": not state.order_violations,
            "violations": state.order_violations,
        },
        "replay_engine": "model.generate_kv_cache_forced_tokens",
        "model_generate_used": True,
        "natural_generation": False,
        "generated_text": False,
        "logit_diagnostics": capture.observed,
        "anchor_answer_logits": trace["anchor_answer_logits"],
        "monitor_telemetry": monitor.telemetry(),
        "full_cap_stats": v1run.full_stats(capper.stats) if capper else None,
        "hook_counts_before": before,
        "hook_counts_after": after,
        "retry_count": 0,
        "answer_repair_performed": False,
    }
    del generated, encoded
    return result


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
        "replay_engine": "cached_forced_v1_1",
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
    require(args.gpu_memory_gib == 18, "Cached E2B requires 18 GiB per visible GPU")
    if args.stage == "prepare" and not (DATA / "CPU_GATE.json").is_file():
        traces = read_jsonl(SOURCE_TRACES)
        manifest = read_json(SOURCE_MANIFEST)
    else:
        traces, manifest, _gate = validate_release()
    if args.scope == "formal" and args.stage in {"generate", "all"}:
        smoke_audit_path = args.output_root / "SMOKE_AUDIT.json"
        require(smoke_audit_path.is_file(), "Formal is blocked until cached smoke audit exists")
        require(
            read_json(smoke_audit_path).get("status") == "passed",
            "Formal is blocked until cached smoke audit passes",
        )
    matrix = ordered_cells(traces, scope=args.scope, smoke_ids=manifest["smoke_prompt_ids"])
    git = git_provenance(require_synced=args.stage in {"generate", "all"})
    run_dir = args.output_root / f"{args.scope}-fixed-trace-seed0"
    configuration = {
        "schema": "cedar.e2b_fixed_trace_cached.run.v1_1",
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
        "replay": {
            "engine": "model.generate_kv_cache_forced_tokens",
            "capture_before_force": True,
            "natural_generation": False,
            "terminal_probe_label_excluded_from_estimands": True,
            "batch_size": 1,
        },
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

    require(Path(sys.prefix).name == "cedar", f"GPU run requires cedar, found {sys.prefix}")
    require(torch.cuda.device_count() == 4, "Cached E2B requires exactly four visible GPUs")
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
    prompt_rows = {
        row["id"]: row for row in read_jsonl(E2A_DATA / "answer_inputs96.jsonl")
    }
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
                "replay_engine": "model.generate_kv_cache_forced_tokens",
                "resume_events": [],
            },
        )
    for index, (trace, condition) in enumerate(matrix):
        key = record_key(index, trace, condition)
        if store.contains(key):
            continue
        result = run_one(model, tokenizer, transformers, caps, prompt_rows, trace, condition)
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
            "processor_order_violations": [
                row["cell_id"]
                for row in results
                if not row["processor_order"]["capture_before_force_all"]
            ],
        }
    )
    atomic_write_json(runtime_path, runtime)
    print(json.dumps({"complete": len(results), "expected": len(matrix)}), flush=True)


if __name__ == "__main__":
    main()
