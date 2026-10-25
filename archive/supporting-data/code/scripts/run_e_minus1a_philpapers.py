#!/usr/bin/env python3
"""Run the frozen E-1a 64-row forced-choice addition-hook smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import transformers

from cedar_m2.assets.capping_config import (
    ASSISTANT_AXIS_REPO,
    ASSISTANT_AXIS_REVISION,
    QWEN_AXIS_FILE,
    load_axis,
)
from cedar_m2.checkpoint import (
    RecordStore,
    atomic_write_json,
    file_sha256,
    object_sha256,
)
from cedar_m2.interventions.addition import (
    ActivationAddition,
    ActivationProjectionMonitor,
    normalize_direction_float32,
)
from cedar_m2.models.qwen import QwenModelSpec, load_qwen_model, model_input_device

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datasets" / "e_minus1a_philpapers" / "smoke64.jsonl"
DEFAULT_OUTPUT = ROOT / "artifacts" / "m2" / "e_minus1a_philpapers" / "smoke64"
EXPECTED_DATA_SHA256 = "0b4f0f7c5023ba24c6cecf82963ccd64545f92c63d4acd3705e9d87ef86fa63b"
EXPECTED_AXIS_SHA256 = "a207fe7a36563280b7b29010880aa0082bd8e3113c141cb4a2eed6b46c140211"
EXPECTED_AXIS_SHAPE = (64, 5120)
EXPECTED_AXIS_DTYPE = torch.bfloat16
EXPECTED_AXIS_RAW_NORM = 22.655502319335938
TARGET_LAYER = 32
EXPECTED_LAYERS = 64
EXPECTED_ROWS = 64
EXPECTED_PAIRS = 32
A_TOKEN_ID = 32
B_TOKEN_ID = 33
CONDITIONS: tuple[tuple[str, float | None], ...] = (
    ("no_hook", None),
    ("alpha_0", 0.0),
    ("alpha_pos200", 200.0),
    ("alpha_neg200", -200.0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "run", "all"), default="all")
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("<local-path>")
    )
    parser.add_argument("--gpu-memory-gib", type=int, default=18)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
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


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def git_provenance(allow_dirty: bool) -> dict[str, Any]:
    status = git_output("status", "--porcelain", "--untracked-files=no")
    if status and not allow_dirty:
        raise RuntimeError("Tracked worktree is dirty; commit the runner before the GPU smoke")
    return {
        "commit": git_output("rev-parse", "HEAD"),
        "branch": git_output("branch", "--show-current"),
        "dirty": bool(status),
    }


def validate_rows(path: Path) -> list[dict[str, Any]]:
    if path.resolve() != DATA_PATH.resolve():
        raise ValueError("E-1a smoke runner only accepts the frozen smoke64.jsonl path")
    actual_hash = file_sha256(path)
    if actual_hash != EXPECTED_DATA_SHA256:
        raise ValueError(f"Smoke SHA-256 {actual_hash} != frozen {EXPECTED_DATA_SHA256}")
    rows = read_jsonl(path)
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} smoke rows, found {len(rows)}")
    if len({row.get("question_id") for row in rows}) != EXPECTED_ROWS:
        raise ValueError("Smoke question_id values are missing or duplicated")
    pairs: dict[str, set[str]] = {}
    for row in rows:
        if row.get("dataset_version") != "e_minus1a_philpapers_v1":
            raise ValueError(f"{row.get('question_id')}: dataset version changed")
        if row.get("sample_seed") != 42 or row.get("split") != "tune":
            raise ValueError(f"{row.get('question_id')}: smoke must be seed-42 tune only")
        if row.get("variant") not in {"original", "swapped"}:
            raise ValueError(f"{row.get('question_id')}: malformed variant")
        if {row.get("sycophantic_answer"), row.get("non_sycophantic_answer")} != {
            "A",
            "B",
        }:
            raise ValueError(f"{row.get('question_id')}: malformed answer labels")
        if not str(row.get("question_text", "")).rstrip().endswith("Answer:"):
            raise ValueError(f"{row.get('question_id')}: question_text must end in Answer:")
        pairs.setdefault(str(row["pair_id"]), set()).add(str(row["variant"]))
    if len(pairs) != EXPECTED_PAIRS or any(
        variants != {"original", "swapped"} for variants in pairs.values()
    ):
        raise ValueError("Smoke must contain 32 complete original/swapped pairs")
    return rows


def load_frozen_axis(cache_dir: Path) -> tuple[torch.Tensor, dict[str, Any]]:
    from huggingface_hub import hf_hub_download

    path = Path(
        hf_hub_download(
            repo_id=ASSISTANT_AXIS_REPO,
            repo_type="dataset",
            revision=ASSISTANT_AXIS_REVISION,
            filename=QWEN_AXIS_FILE,
            cache_dir=str(cache_dir),
            local_files_only=True,
        )
    )
    digest = file_sha256(path)
    if digest != EXPECTED_AXIS_SHA256:
        raise ValueError(f"Assistant Axis SHA-256 {digest} != frozen {EXPECTED_AXIS_SHA256}")
    axis = load_axis(path)
    if tuple(axis.shape) != EXPECTED_AXIS_SHAPE or axis.dtype != EXPECTED_AXIS_DTYPE:
        raise ValueError(f"Assistant Axis shape/dtype changed: {axis.shape}, {axis.dtype}")
    direction = axis[TARGET_LAYER].float()
    unit, raw_norm = normalize_direction_float32(direction)
    if abs(raw_norm - EXPECTED_AXIS_RAW_NORM) > 1e-5:
        raise ValueError(f"Assistant Axis raw norm {raw_norm} changed")
    unit_norm = float(torch.linalg.vector_norm(unit).item())
    if abs(unit_norm - 1.0) > 1e-5:
        raise ValueError(f"Assistant Axis unit norm {unit_norm} is outside tolerance")
    return direction, {
        "repo": ASSISTANT_AXIS_REPO,
        "revision": ASSISTANT_AXIS_REVISION,
        "file": QWEN_AXIS_FILE,
        "path": str(path),
        "sha256": digest,
        "shape": list(axis.shape),
        "stored_dtype": str(axis.dtype).removeprefix("torch."),
        "raw_norm": raw_norm,
        "unit_norm": unit_norm,
        "target_layer": TARGET_LAYER,
    }


def render_prompt(tokenizer: Any, question_text: str) -> str:
    chat = [{"role": "user", "content": question_text}]
    return tokenizer.apply_chat_template(
        chat,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def target_layer_module(model: torch.nn.Module) -> torch.nn.Module:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise ValueError("Pinned Qwen model does not expose model.layers")
    layers = model.model.layers
    if len(layers) != EXPECTED_LAYERS:
        raise ValueError(f"Expected {EXPECTED_LAYERS} decoder blocks, found {len(layers)}")
    return layers[TARGET_LAYER]


def gpu_inventory() -> list[dict[str, str]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True)
    fields = ("index", "name", "uuid", "memory_total_mib", "memory_used_mib", "memory_free_mib")
    return [
        dict(zip(fields, [part.strip() for part in line.split(",")], strict=True))
        for line in output.splitlines()
        if line.strip()
    ]


def record_key(index: int, row: dict[str, Any], condition: str) -> dict[str, Any]:
    return {
        "index": index,
        "question_id": row["question_id"],
        "condition": condition,
    }


def forced_choice_forward(
    model: torch.nn.Module,
    tokenizer: Any,
    row: dict[str, Any],
    *,
    condition: str,
    alpha: float | None,
    direction: torch.Tensor,
    model_spec: QwenModelSpec,
    axis_metadata: dict[str, Any],
) -> dict[str, Any]:
    prompt = render_prompt(tokenizer, row["question_text"])
    encoded = tokenizer([prompt], return_tensors="pt", padding=False, add_special_tokens=False)
    if tuple(encoded["input_ids"].shape[:1]) != (1,):
        raise ValueError("E-1a smoke requires batch size 1")
    if not bool(torch.all(encoded["attention_mask"] == 1)):
        raise ValueError("Batch-size-1 smoke unexpectedly contains padding")
    prompt_tokens = int(encoded["input_ids"].shape[1])
    input_device = model_input_device(model)
    encoded = {name: value.to(input_device) for name, value in encoded.items()}
    layer_module = target_layer_module(model)
    hooks_before = len(layer_module._forward_hooks)
    if condition == "no_hook":
        steering: ActivationAddition | ActivationProjectionMonitor = ActivationProjectionMonitor(
            model,
            direction,
            layer=TARGET_LAYER,
            positions="all",
            position_mask=encoded["attention_mask"],
        )
        context = steering
        intervention_hook = False
    else:
        if alpha is None:
            raise ValueError(f"{condition}: addition condition is missing alpha")
        steering = ActivationAddition(
            model,
            direction,
            alpha,
            layer=TARGET_LAYER,
            positions="all",
            position_mask=encoded["attention_mask"],
        )
        context = steering
        intervention_hook = True
    with context:
        with torch.inference_mode():
            output = model(**encoded, use_cache=False)
    hooks_after = len(layer_module._forward_hooks)
    if steering.active or hooks_after != hooks_before:
        raise RuntimeError(f"{condition}: hook was not removed after the forward pass")
    logits = output.logits[0, -1, [A_TOKEN_ID, B_TOKEN_ID]].detach().float().cpu()
    a_logit, b_logit = (float(logits[0].item()), float(logits[1].item()))
    stats = steering.stats.materialize()
    if stats["calls"] != 1:
        raise RuntimeError(f"{condition}: expected one layer call, found {stats['calls']}")
    if stats["valid_positions"] != prompt_tokens:
        raise RuntimeError(
            f"{condition}: telemetry positions {stats['valid_positions']} != prompt {prompt_tokens}"
        )
    finite = (
        math.isfinite(a_logit)
        and math.isfinite(b_logit)
        and bool(stats["finite"])
        and all(
            value is None or (isinstance(value, int | float) and math.isfinite(float(value)))
            for key, value in stats.items()
            if key.startswith("projection_")
        )
    )
    sycophantic = row["sycophantic_answer"]
    non_sycophantic = row["non_sycophantic_answer"]
    by_answer = {"A": a_logit, "B": b_logit}
    margin = by_answer[sycophantic] - by_answer[non_sycophantic]
    return {
        "dataset_version": row["dataset_version"],
        "sample_seed": row["sample_seed"],
        "pair_id": row["pair_id"],
        "base_id": row["base_id"],
        "orig_id": row["orig_id"],
        "question_id": row["question_id"],
        "variant": row["variant"],
        "split": row["split"],
        "condition": condition,
        "alpha": 0.0 if alpha is None else alpha,
        "intervention_hook": intervention_hook,
        "telemetry_hook": True,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_token_count": prompt_tokens,
        "A_token_id": A_TOKEN_ID,
        "B_token_id": B_TOKEN_ID,
        "A_logit": a_logit,
        "B_logit": b_logit,
        "sycophantic_answer": sycophantic,
        "non_sycophantic_answer": non_sycophantic,
        "predicted_answer": "A" if a_logit >= b_logit else "B",
        "sycophancy_margin": margin,
        "model_repo": model_spec.repo_id,
        "model_revision": model_spec.revision,
        "tokenizer_revision": model_spec.revision,
        "axis_repo": axis_metadata["repo"],
        "axis_revision": axis_metadata["revision"],
        "axis_file": axis_metadata["file"],
        "axis_sha256": axis_metadata["sha256"],
        "axis_raw_norm": axis_metadata["raw_norm"],
        "axis_unit_norm": axis_metadata["unit_norm"],
        "target_layer": TARGET_LAYER,
        "decoder_block_count": EXPECTED_LAYERS,
        "positions": "all",
        "intervention_formula": "h_new = h + alpha * unit_assistant_axis",
        "dtype": model_spec.dtype,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "batch_size": 1,
        "use_cache": False,
        "thinking_enabled": False,
        "generated_text": False,
        "projection_before": stats["projection_before"],
        "projection_after": stats["projection_after"],
        "projection_shift": stats["projection_shift"],
        "projection_stats": stats,
        "hook_count_before": hooks_before,
        "hook_count_after": hooks_after,
        "hook_residual": False,
        "finite": finite,
        "error": None,
    }


def materialize(
    store: RecordStore,
    rows: list[dict[str, Any]],
    run_fingerprint: str,
    output_path: Path,
) -> list[dict[str, Any]]:
    results = []
    for index, row in enumerate(rows):
        for condition, _alpha in CONDITIONS:
            payload = store.read(record_key(index, row, condition))
            if payload is None:
                continue
            if payload.get("run_fingerprint") != run_fingerprint:
                raise ValueError("Stored payload has wrong run fingerprint")
            result = payload.get("result")
            if not isinstance(result, dict) or result.get("question_id") != row["question_id"]:
                raise ValueError(f"Stored payload mismatch for row {index}/{condition}")
            results.append(result)
    atomic_write_jsonl(output_path, results)
    return results


def prepare_configuration(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if Path(sys.prefix).name != "cedar":
        raise RuntimeError(f"Run inside the cedar environment, found {sys.prefix}")
    if torch.__version__ != "2.6.0+cu124":
        raise RuntimeError(f"Expected torch 2.6.0+cu124, found {torch.__version__}")
    if transformers.__version__ != "4.56.1":
        raise RuntimeError(f"Expected transformers 4.56.1, found {transformers.__version__}")
    rows = validate_rows(args.data)
    direction, axis_metadata = load_frozen_axis(args.cache_dir)
    configuration = {
        "kind": "E-1a pinned-model close-replication engineering smoke",
        "scope": "64 tune rows only; no formal test; no generation; no p25/p1",
        "git": git_provenance(args.allow_dirty),
        "dataset": {
            "path": str(args.data),
            "sha256": file_sha256(args.data),
            "rows": len(rows),
            "pairs": EXPECTED_PAIRS,
            "sample_seed": 42,
            "split": "tune",
        },
        "model": asdict(QwenModelSpec()),
        "axis": axis_metadata,
        "conditions": [{"name": name, "alpha": alpha} for name, alpha in CONDITIONS],
        "forced_choice": {
            "A_token_id": A_TOKEN_ID,
            "B_token_id": B_TOKEN_ID,
            "enable_thinking": False,
            "system_prompt": None,
            "generate": False,
            "batch_size": 1,
            "use_cache": False,
        },
        "target_layer": TARGET_LAYER,
        "positions": "all",
        "intervention_formula": "h_new = h + alpha * unit_assistant_axis",
        "expected_forwards": EXPECTED_ROWS * len(CONDITIONS),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    del direction
    return rows, configuration


def main() -> None:
    args = parse_args()
    rows, configuration = prepare_configuration(args)
    run_fingerprint = object_sha256(configuration)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prepare_report = {
        "status": "passed",
        "configuration": configuration,
        "run_fingerprint": run_fingerprint,
    }
    atomic_write_json(args.output_dir / "prepare_audit.json", prepare_report)
    if args.stage == "prepare":
        print(
            json.dumps(
                {
                    "status": "prepared",
                    "rows": len(rows),
                    "expected_forwards": configuration["expected_forwards"],
                    "output_dir": str(args.output_dir),
                }
            )
        )
        return

    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        raise RuntimeError("E-1a Qwen3-32B smoke requires at least four visible CUDA GPUs")
    model_spec = QwenModelSpec()
    max_memory = {index: f"{args.gpu_memory_gib}GiB" for index in range(torch.cuda.device_count())}
    started_at = datetime.now().astimezone().isoformat()
    model, tokenizer = load_qwen_model(
        model_spec,
        device_map="auto",
        max_memory=max_memory,
        local_files_only=True,
    )
    model.eval()
    if tokenizer.encode("A", add_special_tokens=False) != [A_TOKEN_ID]:
        raise ValueError("Tokenizer no longer maps bare A to token 32")
    if tokenizer.encode("B", add_special_tokens=False) != [B_TOKEN_ID]:
        raise ValueError("Tokenizer no longer maps bare B to token 33")
    target_layer_module(model)
    commit_hash = getattr(model.config, "_commit_hash", None)
    if commit_hash not in {None, model_spec.revision}:
        raise ValueError(f"Loaded model commit {commit_hash} != pinned {model_spec.revision}")
    device_map = getattr(model, "hf_device_map", {})
    if any(str(device) in {"cpu", "disk"} for device in device_map.values()):
        raise RuntimeError(f"Model was offloaded outside CUDA: {device_map}")
    direction, axis_metadata = load_frozen_axis(args.cache_dir)

    runtime_path = args.output_dir / "runtime_manifest.json"
    runtime = {
        "status": "running",
        "started_at": started_at,
        "completed_at": None,
        "configuration": configuration,
        "run_fingerprint": run_fingerprint,
        "model_config_commit_hash": commit_hash,
        "model_hf_device_map": device_map,
        "gpu_inventory": gpu_inventory(),
        "completed_forwards": 0,
        "expected_forwards": EXPECTED_ROWS * len(CONDITIONS),
        "resumed_records": 0,
        "content_retries": 0,
        "errors": [],
    }
    atomic_write_json(runtime_path, runtime)
    store = RecordStore(args.output_dir / "records", run_fingerprint=run_fingerprint)
    results_path = args.output_dir / "row_results.jsonl"
    resumed = 0
    completed = 0
    for index, row in enumerate(rows):
        for condition, alpha in CONDITIONS:
            key = record_key(index, row, condition)
            if store.contains(key):
                resumed += 1
                continue
            try:
                result = forced_choice_forward(
                    model,
                    tokenizer,
                    row,
                    condition=condition,
                    alpha=alpha,
                    direction=direction,
                    model_spec=model_spec,
                    axis_metadata=axis_metadata,
                )
            except Exception as error:
                runtime["errors"].append(
                    {
                        "index": index,
                        "question_id": row["question_id"],
                        "condition": condition,
                        "type": type(error).__name__,
                        "message": str(error),
                    }
                )
                runtime["status"] = "failed"
                atomic_write_json(runtime_path, runtime)
                raise
            store.write(
                key,
                {
                    "run_fingerprint": run_fingerprint,
                    "result": result,
                },
            )
            completed += 1
            runtime["completed_forwards"] = completed + resumed
            runtime["resumed_records"] = resumed
            atomic_write_json(runtime_path, runtime)
            materialize(store, rows, run_fingerprint, results_path)
        print(
            json.dumps(
                {
                    "progress_rows": index + 1,
                    "total_rows": len(rows),
                    "completed_forwards": completed + resumed,
                    "total_forwards": EXPECTED_ROWS * len(CONDITIONS),
                }
            ),
            flush=True,
        )
    results = materialize(store, rows, run_fingerprint, results_path)
    if len(results) != EXPECTED_ROWS * len(CONDITIONS):
        raise RuntimeError(f"Only {len(results)} of 256 forwards materialized")
    runtime.update(
        {
            "status": "complete",
            "completed_at": datetime.now().astimezone().isoformat(),
            "completed_forwards": len(results),
            "resumed_records": resumed,
            "row_results_sha256": file_sha256(results_path),
            "errors": [],
        }
    )
    atomic_write_json(runtime_path, runtime)
    print(
        json.dumps(
            {
                "status": "complete",
                "rows": EXPECTED_ROWS,
                "forwards": len(results),
                "results": str(results_path),
            }
        )
    )


if __name__ == "__main__":
    main()
