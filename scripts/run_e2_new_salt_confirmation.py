#!/usr/bin/env python3
# ruff: noqa: E402
"""Run the frozen E2 new-salt smoke or formal paired matrix."""

from __future__ import annotations

import argparse
import hashlib
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
DATA_ROOT = ROOT / "datasets" / "m2_2b_formal" / "e2_new_salt_confirmation"
for path in (SRC, DATA_ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from e2_new_salt_run_common import (
    ATTENTION_IMPLEMENTATION,
    CAP_EXPERIMENT,
    CAP_LAYERS,
    CONDITIONS,
    EXPECTED_AXIS_SHA256,
    EXPECTED_CAP_CONFIG_SHA256,
    FORMAL_RESULT_ROWS,
    MAX_NEW_TOKENS,
    MODEL_DTYPE,
    MODEL_REPO,
    MODEL_REVISION,
    SEED,
    SMOKE_RESULT_ROWS,
    ordered_run_cells,
    read_json,
    select_rows,
    selection_fingerprint,
    validate_cpu_gate,
)
from e2_new_salt_run_common import (
    file_sha256 as data_file_sha256,
)

from cedar_m2.assets.capping_config import (
    download_qwen_capping_assets,
    load_axis,
    load_capping_experiment,
    verify_axis_orientation,
)
from cedar_m2.checkpoint import (
    RecordStore,
    atomic_write_json,
    file_sha256,
    initialize_run,
    object_sha256,
)
from cedar_m2.interventions.capping import ActivationCapper
from cedar_m2.models.qwen import QwenModelSpec, load_qwen_model

BASE_COMMIT = "10120d04d00e3c9122bf2a2ba590bd403373fa45"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "m2" / "e2_new_salt_confirmation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("smoke", "formal"), required=True)
    parser.add_argument("--stage", choices=("prepare", "generate", "all"), default="all")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--hf-home",
        type=Path,
        default=Path(os.environ.get("HF_HOME", "/data1/WuMingrui/HF_HOME")),
    )
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
    def output(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
        ).stdout.strip()

    commit = output("rev-parse", "HEAD")
    upstream = output("rev-parse", "@{u}")
    branch = output("branch", "--show-current")
    status = output("status", "--porcelain")
    if status:
        raise RuntimeError("E2 new-salt runner refuses a dirty worktree")
    if commit != upstream:
        raise RuntimeError("E2 new-salt runner requires local HEAD to equal upstream")
    if branch != "agent/e2-new-salt-confirmation-v1":
        raise RuntimeError(f"Unexpected E2 new-salt branch: {branch}")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_COMMIT, commit], cwd=ROOT
    )
    if ancestor.returncode != 0:
        raise RuntimeError("Runner commit does not descend from the frozen base")
    return {
        "commit": commit,
        "upstream_commit": upstream,
        "branch": branch,
        "head_matches_upstream": True,
        "base_commit": BASE_COMMIT,
        "base_is_ancestor": True,
        "dirty": False,
    }


def hook_counts(model: Any) -> dict[str, int]:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Pinned Qwen model does not expose model.layers")
    return {
        str(layer): len(model.model.layers[layer]._forward_hooks) for layer in CAP_LAYERS
    }


def resolve_hub_cache(cache_dir: Path) -> Path:
    hub = cache_dir / "hub"
    return hub if hub.is_dir() else cache_dir


def direction_sha256(cap: Any) -> str:
    import torch

    tensor = cap.direction.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


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
        "definition": "batch-size-1 token positions pooled over official layers 46-53",
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
    axis_path, config_path = download_qwen_capping_assets(
        cache_dir=resolve_hub_cache(cache_dir), local_files_only=True
    )
    axis_hash = data_file_sha256(axis_path)
    config_hash = data_file_sha256(config_path)
    if axis_hash != EXPECTED_AXIS_SHA256 or config_hash != EXPECTED_CAP_CONFIG_SHA256:
        raise ValueError("Frozen Assistant Axis or p25 config changed")
    axis = load_axis(axis_path)
    caps = load_capping_experiment(config_path, CAP_EXPERIMENT)
    if tuple(sorted(caps)) != CAP_LAYERS:
        raise ValueError("Official p25 layer range changed")
    orientation = verify_axis_orientation(caps, axis)
    metadata = {
        "axis_sha256": axis_hash,
        "capping_config_sha256": config_hash,
        "experiment": CAP_EXPERIMENT,
        "positions": "all",
        "layers": {
            str(layer): {
                "threshold": caps[layer].threshold,
                "direction_sha256": direction_sha256(caps[layer]),
                "axis_cosine": orientation[layer],
            }
            for layer in CAP_LAYERS
        },
    }
    return caps, metadata


def record_key(index: int, row: Mapping[str, Any], condition: str, scope: str) -> dict[str, Any]:
    return {
        "index": index,
        "id": row["id"],
        "condition": condition,
        "scope": scope,
        "seed": SEED,
    }


def materialize(
    store: RecordStore,
    cells: Sequence[tuple[dict[str, Any], str]],
    results_path: Path,
    record_runtime_path: Path,
    scope: str,
) -> list[dict[str, Any]]:
    results = []
    payloads = []
    for index, (row, condition) in enumerate(cells):
        payload = store.read(record_key(index, row, condition, scope))
        if payload is None:
            continue
        result = payload.get("result")
        if (
            payload.get("index") != index
            or not isinstance(result, dict)
            or result.get("id") != row["id"]
            or result.get("condition") != condition
            or result.get("scope") != scope
        ):
            raise ValueError(f"Checkpoint mismatch at cell {index}")
        results.append(result)
        payloads.append(payload)
    atomic_write_jsonl(results_path, results)
    atomic_write_jsonl(record_runtime_path, payloads)
    return results


def update_runtime(path: Path, values: Mapping[str, Any]) -> None:
    runtime = read_json(path) if path.is_file() else {}
    runtime.update(values)
    atomic_write_json(path, runtime)


def passed_smoke_audit(output_root: Path) -> bool:
    path = output_root / "SMOKE_AUDIT.json"
    return path.is_file() and read_json(path).get("status") == "passed"


def configuration(
    rows: Sequence[dict[str, Any]], scope: str, git: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "kind": "CEDAR E2 new-salt unselected instance confirmation",
        "scope": scope,
        "claim_scope": "new_salt_unselected_instance_confirmation",
        "conditions": list(CONDITIONS),
        "paired_order": "each prompt uncapped then full_p25",
        "seed": SEED,
        "dataset": {
            "path": str((DATA_ROOT / "arms96.jsonl").relative_to(ROOT)),
            "input_rows": len(rows),
            "result_rows": len(rows) * 2,
            "selection": selection_fingerprint(scope),
            "families": sorted({row["family_id"] for row in rows}),
        },
        "model": {
            "repo_id": MODEL_REPO,
            "revision": MODEL_REVISION,
            "dtype": MODEL_DTYPE,
            "attention_implementation": ATTENTION_IMPLEMENTATION,
        },
        "generation": {
            "batch_size": 1,
            "mode": "thinking",
            "do_sample": False,
            "num_beams": 1,
            "max_new_tokens": MAX_NEW_TOKENS,
            "content_retry": False,
            "answer_repair": False,
        },
        "truncation_policy": {
            "classification": "invalid_or_abstain",
            "retry": False,
            "manual_credit": False,
            "itt": True,
        },
        "prohibited": [
            "p1",
            "additional_salt",
            "additional_seed",
            "item_deletion",
            "answer_repair",
            "patching",
            "LoFa",
            "J-Lens",
        ],
        "git": dict(git),
    }


def main() -> None:
    args = parse_args()
    if args.gpu_memory_gib != 18:
        raise ValueError("Frozen E2 new-salt run requires exactly 18 GiB per visible GPU")
    validate_cpu_gate()
    if args.scope == "formal" and not passed_smoke_audit(args.output_root.resolve()):
        raise ValueError("Formal run is blocked until SMOKE_AUDIT.json passes")
    rows = select_rows(args.scope)
    cells = ordered_run_cells(rows)
    expected = SMOKE_RESULT_ROWS if args.scope == "smoke" else FORMAL_RESULT_ROWS
    if len(cells) != expected:
        raise ValueError("Frozen paired matrix size changed")
    git = git_provenance()
    run_dir = args.output_root.resolve() / f"{args.scope}-paired-uncapped-full-p25-seed0"
    manifest = initialize_run(run_dir, configuration(rows, args.scope, git))
    store = RecordStore(run_dir / "records", run_fingerprint=str(manifest["fingerprint"]))
    results_path = run_dir / "results.jsonl"
    record_runtime_path = run_dir / "record_runtime.jsonl"
    runtime_path = run_dir / "runtime.json"
    results = materialize(store, cells, results_path, record_runtime_path, args.scope)
    if args.stage in {"prepare", "all"}:
        print(
            json.dumps(
                {
                    "stage": "prepare",
                    "scope": args.scope,
                    "input_rows": len(rows),
                    "result_rows": len(cells),
                    "existing": len(results),
                    "conditions": list(CONDITIONS),
                    "runner_commit": git["commit"],
                    "gpu_used": False,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if args.stage == "prepare":
            return

    missing = [
        index
        for index, (row, condition) in enumerate(cells)
        if not store.contains(record_key(index, row, condition, args.scope))
    ]
    if missing:
        import run_prover_generator_step5_uncapped as base
        import torch
        import transformers

        if Path(sys.prefix).name != "cedar":
            raise RuntimeError(f"Generation requires cedar, found {sys.prefix}")
        if torch.cuda.device_count() != 4:
            raise RuntimeError("E2 new-salt generation requires exactly four visible GPUs")
        os.environ["HF_HOME"] = str(args.hf_home.resolve())
        base.MAX_NEW_TOKENS = MAX_NEW_TOKENS
        base.SEED = SEED
        caps, cap_metadata = load_p25_assets(
            args.asset_cache_dir.resolve()
            if args.asset_cache_dir
            else args.hf_home.resolve()
        )
        max_memory = {index: f"{args.gpu_memory_gib}GiB" for index in range(4)}
        spec = QwenModelSpec(
            repo_id=MODEL_REPO,
            revision=MODEL_REVISION,
            dtype=MODEL_DTYPE,
            attention_implementation=ATTENTION_IMPLEMENTATION,
        )
        model, tokenizer = load_qwen_model(spec, max_memory=max_memory, local_files_only=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base.validate_label_tokens(tokenizer)
        rendered = [base.render_prompt(tokenizer, row, base.MODES["thinking"]) for row in rows]
        tokenized = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in rendered]
        rendered_hash = object_sha256(
            [
                {"id": row["id"], "prompt": prompt}
                for row, prompt in zip(rows, rendered, strict=True)
            ]
        )
        input_ids_hash = object_sha256(
            [
                {"id": row["id"], "input_ids": ids}
                for row, ids in zip(rows, tokenized, strict=True)
            ]
        )
        if args.scope == "formal":
            preflight = read_json(DATA_ROOT / "tokenizer_preflight.json")
            if (
                rendered_hash != preflight["inputs"]["rendered_prompts_sha256"]
                or input_ids_hash != preflight["inputs"]["tokenized_input_ids_sha256"]
            ):
                raise RuntimeError("Frozen formal rendered/tokenized input hash changed")
        update_runtime(
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
                    "hf_home": str(args.hf_home.resolve()),
                },
                "rendered_prompts_sha256": rendered_hash,
                "tokenized_input_ids_sha256": input_ids_hash,
                "official_p25": cap_metadata,
                "resume_events": [],
            },
        )
        for index in missing:
            row, condition = cells[index]
            before = hook_counts(model)
            capper = (
                ActivationCapper(
                    model,
                    caps,
                    positions="all",
                    track_stats=True,
                    defer_stats_sync=True,
                )
                if condition == "full_p25"
                else None
            )
            if capper is None:
                result = base.generate_one(
                    model,
                    tokenizer,
                    transformers,
                    row,
                    scope=args.scope,
                    mode_name="thinking",
                    record_seed=index // 2,
                )
                cap_stats = None
            else:
                with capper:
                    result = base.generate_one(
                        model,
                        tokenizer,
                        transformers,
                        row,
                        scope=args.scope,
                        mode_name="thinking",
                        record_seed=index // 2,
                    )
                cap_stats = stats_to_dict(capper.stats)
            after = hook_counts(model)
            if after != before:
                raise RuntimeError(
                    f"Forward-hook leak at cell {index}: before={before}, after={after}"
                )
            result.update(
                {
                    "condition": condition,
                    "experiment": "e2_new_salt_confirmation",
                    "pair_index": index // 2,
                    "paired_condition_order": list(CONDITIONS),
                    "cap_stats": cap_stats,
                    "hook_counts_before": before,
                    "hook_counts_after": after,
                }
            )
            store.write(
                record_key(index, row, condition, args.scope),
                {"index": index, "result": result},
            )
            results = materialize(
                store, cells, results_path, record_runtime_path, args.scope
            )
            print(
                json.dumps(
                    {
                        "stage": "generate",
                        "scope": args.scope,
                        "complete": len(results),
                        "expected": len(cells),
                        "id": row["id"],
                        "condition": condition,
                    }
                ),
                flush=True,
            )
        del model, tokenizer
        base.release_cuda_cache()

    results = materialize(store, cells, results_path, record_runtime_path, args.scope)
    if len(results) == len(cells):
        triggered = sum(
            int(row["cap_stats"]["pooled"]["triggered_positions"])
            for row in results
            if row["condition"] == "full_p25" and row.get("cap_stats")
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
                "parse_failures": [
                    f"{row['id']}::{row['condition']}"
                    for row in results
                    if row["final_answer_parse_status"] != "passed"
                ],
                "truncations": [
                    f"{row['id']}::{row['condition']}"
                    for row in results
                    if row.get("truncated")
                ],
                "missing_answer_logits": [
                    f"{row['id']}::{row['condition']}"
                    for row in results
                    if row.get("answer_logits") is None
                ],
                "full_p25_triggered_positions": triggered,
                "single_matrix_completed_without_content_selection": True,
            },
        )
    print(
        json.dumps(
            {
                "complete": len(results),
                "expected": len(cells),
                "scope": args.scope,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
