#!/usr/bin/env python3
"""Generate uncapped greedy Qwen responses for the M2.2b v3 calibration set."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import transformers

from cedar_m2.checkpoint import (
    atomic_write_json,
    file_sha256,
    initialize_run,
    object_sha256,
)
from cedar_m2.models.qwen import QwenModelSpec, load_qwen_model, model_input_device

ARMS = ("N", "R", "V", "F", "S", "AO")
MESSAGE_FIELDS = ("system", "initial_user", "initial_assistant", "followup_user")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "generate", "all"), default="all")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "datasets/m2_2b_formal/v3_calibration/"
            "m2_2b_v3_calibration72_arms.jsonl"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/m2/m2_2b_v3_calibration"),
    )
    parser.add_argument("--run-name", default="qwen3_32b_greedy_seed0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--gpu-memory-gib", type=int, default=20)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def load_calibration_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 72:
        raise ValueError(f"Expected 72 calibration rows, found {len(rows)}")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(item_id, str) or not item_id for item_id in ids):
        raise ValueError("Every calibration row needs a non-empty string ID")
    if len(set(ids)) != len(ids):
        raise ValueError("Calibration row IDs must be unique")
    arm_counts = Counter(row.get("arm") for row in rows)
    if arm_counts != Counter({arm: 12 for arm in ARMS}):
        raise ValueError(f"Expected 12 rows per arm, found {dict(arm_counts)}")
    base_arms: dict[str, set[str]] = {}
    for row in rows:
        base_id = row.get("base_id")
        if not isinstance(base_id, str) or not base_id:
            raise ValueError(f"Row {row['id']} has a malformed base_id")
        base_arms.setdefault(base_id, set()).add(str(row["arm"]))
        for field in MESSAGE_FIELDS:
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Row {row['id']} has malformed {field}")
        try:
            initial = json.loads(row["initial_assistant"])
        except json.JSONDecodeError as error:
            raise ValueError(f"Row {row['id']} has invalid initial_assistant JSON") from error
        if initial.get("answer") != row.get("initial_answer_id"):
            raise ValueError(f"Row {row['id']} initial transcript answer does not match metadata")
    if len(base_arms) != 12 or any(arms != set(ARMS) for arms in base_arms.values()):
        raise ValueError("Expected 12 base units, each containing all six arms")
    return rows


def render_qwen_calibration_prompt(tokenizer: Any, row: dict[str, Any]) -> str:
    """Render the exact system/user/fixed-assistant/user transcript."""
    messages = [
        {"role": "system", "content": row["system"]},
        {"role": "user", "content": row["initial_user"]},
        {"role": "assistant", "content": row["initial_assistant"]},
        {"role": "user", "content": row["followup_user"]},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def load_existing_results(
    path: str | Path,
    *,
    valid_ids: set[str],
) -> dict[str, dict[str, Any]]:
    source = Path(path)
    if not source.exists():
        return {}
    results: dict[str, dict[str, Any]] = {}
    for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        item_id = row.get("id")
        if item_id not in valid_ids:
            raise ValueError(f"Unknown result ID on line {number}: {item_id!r}")
        if item_id in results:
            raise ValueError(f"Duplicate result ID on line {number}: {item_id!r}")
        if row.get("seed") != 0 or not isinstance(row.get("response"), str):
            raise ValueError(f"Malformed result on line {number}: {item_id!r}")
        results[item_id] = {
            "id": item_id,
            "seed": 0,
            "response": row["response"],
        }
    return results


def atomic_write_results(
    path: str | Path,
    dataset_rows: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in dataset_rows:
                if row["id"] in results:
                    handle.write(json.dumps(results[row["id"]], ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def git_provenance(*, allow_dirty: bool) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    branch = subprocess.run(
        ["git", "branch", "--show-current"], check=True, capture_output=True, text=True
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout.rstrip()
    if status and not allow_dirty:
        raise RuntimeError("Refusing an official v3 calibration run from a dirty worktree")
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], check=True, capture_output=True
    ).stdout
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
        "status": status.splitlines(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
    }


def build_configuration(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    visible_gpus = torch.cuda.device_count()
    return {
        "kind": "M2.2b v3 uncapped Qwen paradigm calibration",
        "scope": "development calibration only; capping prohibited",
        "dataset": {
            "path": str(args.data),
            "sha256": file_sha256(args.data),
            "dataset_version": rows[0]["dataset_version"],
            "row_ids_sha256": object_sha256([row["id"] for row in rows]),
            "rows": len(rows),
            "base_units": len({row["base_id"] for row in rows}),
            "arms": ARMS,
        },
        "model": asdict(QwenModelSpec()),
        "intervention": "uncapped",
        "messages": {
            "role_order": ("system", "user", "assistant", "user"),
            "content_fields": MESSAGE_FIELDS,
            "initial_assistant": "fixed transcript from dataset; never regenerated",
            "enable_thinking": False,
        },
        "generation": {
            "decoding": "greedy",
            "do_sample": False,
            "temperature": 0.0,
            "num_beams": 1,
            "seed": 0,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
            "output": "raw decoded continuation; no parsing, retry, or repair",
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": transformers.__version__,
        },
        "hardware": {
            "visible_gpu_count": visible_gpus,
            "visible_gpu_names": [torch.cuda.get_device_name(i) for i in range(visible_gpus)],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "qwen_max_memory_per_gpu_gib": args.gpu_memory_gib,
        },
        "environment": {
            "python_environment_name": Path(sys.prefix).name,
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "python_prefix": sys.prefix,
            "hf_home": os.environ.get("HF_HOME"),
            "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
            "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
        },
        "git": git_provenance(allow_dirty=args.allow_dirty),
    }


def update_runtime(run_dir: Path, values: dict[str, Any]) -> None:
    path = run_dir / "runtime.json"
    runtime = json.loads(path.read_text()) if path.exists() else {}
    runtime.update(values)
    atomic_write_json(path, runtime)


def generate_responses(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    output_path: Path,
    *,
    batch_size: int,
    max_new_tokens: int,
) -> int:
    missing = [row for row in rows if row["id"] not in results]
    if not missing:
        return 0
    device = model_input_device(model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    batches = (len(missing) + batch_size - 1) // batch_size
    generated_count = 0

    for batch_number, start in enumerate(range(0, len(missing), batch_size), start=1):
        batch = missing[start : start + batch_size]
        prompts = [render_qwen_calibration_prompt(tokenizer, row) for row in batch]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        encoded = {name: value.to(device) for name, value in encoded.items()}
        prompt_width = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            output_ids = model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        responses = tokenizer.batch_decode(
            output_ids[:, prompt_width:],
            skip_special_tokens=True,
        )
        for row, response in zip(batch, responses, strict=True):
            results[row["id"]] = {
                "id": row["id"],
                "seed": 0,
                "response": response.strip(),
            }
            generated_count += 1
        atomic_write_results(output_path, rows, results)
        print(
            json.dumps(
                {
                    "stage": "generate",
                    "batch": batch_number,
                    "batches": batches,
                    "generated": len(batch),
                    "complete": len(results),
                }
            ),
            flush=True,
        )
    return generated_count


def release_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if Path(sys.prefix).name != "cedar":
        raise RuntimeError(f"v3 calibration must run inside cedar, found {sys.prefix}")
    if args.batch_size <= 0 or args.max_new_tokens <= 0 or args.gpu_memory_gib <= 0:
        raise ValueError("Batch size, max new tokens, and GPU memory must be positive")

    rows = load_calibration_rows(args.data)
    run_dir = args.output_root / args.run_name
    configuration = build_configuration(args, rows)
    manifest = initialize_run(run_dir, configuration)
    output_path = run_dir / "qwen_results.jsonl"
    results = load_existing_results(output_path, valid_ids={row["id"] for row in rows})

    if args.stage == "prepare":
        print(
            json.dumps(
                {
                    "stage": "prepare",
                    "rows": len(rows),
                    "base_units": len({row["base_id"] for row in rows}),
                    "arms": ARMS,
                    "existing_results": len(results),
                    "run_fingerprint": manifest["fingerprint"],
                }
            ),
            flush=True,
        )

    if args.stage in {"generate", "all"}:
        if len(results) < len(rows):
            if torch.cuda.device_count() < 4:
                raise RuntimeError("Qwen-3-32B v3 calibration requires four visible GPUs")
            max_memory = {
                index: f"{args.gpu_memory_gib}GiB"
                for index in range(torch.cuda.device_count())
            }
            model, tokenizer = load_qwen_model(
                max_memory=max_memory,
                local_files_only=True,
            )
            update_runtime(run_dir, {"device_map": getattr(model, "hf_device_map", None)})
            newly_generated = generate_responses(
                model,
                tokenizer,
                rows,
                results,
                output_path,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
            )
            del model, tokenizer
            release_cuda_cache()
        else:
            newly_generated = 0
        print(
            json.dumps(
                {
                    "results": str(output_path),
                    "complete": len(results),
                    "newly_generated": newly_generated,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
