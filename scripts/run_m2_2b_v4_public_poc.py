#!/usr/bin/env python3
"""Run the uncapped Qwen3-32B thinking-mode v4 development diagnostic."""

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


ARMS = ("N", "R", "V", "F", "S")
MESSAGE_FIELDS = ("system", "initial_user", "initial_assistant", "followup_user")
EXPECTED_ROWS = 180
EXPECTED_BASES = 36


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "generate", "all"), default="all")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "datasets/m2_2b_formal/v4_public_poc/"
            "m2_2b_v4_public_poc180_arms.jsonl"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/m2/m2_2b_v4_public_poc"),
    )
    parser.add_argument(
        "--run-name",
        default="qwen3_32b_uncapped_thinking_greedy_seed0_development",
    )
    parser.add_argument("--results-filename", default="qwen_thinking_results.jsonl")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--gpu-memory-gib", type=int, default=18)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(rows)}")
    ids = [row.get("id") for row in rows]
    if any(not isinstance(item_id, str) or not item_id for item_id in ids):
        raise ValueError("Every row needs a non-empty string ID")
    if len(set(ids)) != len(ids):
        raise ValueError("Row IDs must be unique")
    if Counter(row.get("arm") for row in rows) != Counter({arm: 36 for arm in ARMS}):
        raise ValueError("Expected 36 records for each N/R/V/F/S arm")

    base_arms: dict[str, set[str]] = {}
    for row in rows:
        base_id = row.get("base_id")
        if not isinstance(base_id, str) or not base_id:
            raise ValueError(f"Row {row['id']} has a malformed base_id")
        base_arms.setdefault(base_id, set()).add(str(row.get("arm")))
        for field in MESSAGE_FIELDS:
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Row {row['id']} has malformed {field}")
        try:
            initial = json.loads(row["initial_assistant"])
        except json.JSONDecodeError as error:
            raise ValueError(f"Row {row['id']} has invalid initial_assistant JSON") from error
        if initial.get("answer") != row.get("initial_answer_id"):
            raise ValueError(
                f"Row {row['id']} initial transcript answer does not match metadata"
            )
        if row.get("recommended_enable_thinking") is not True:
            raise ValueError(f"Row {row['id']} does not require thinking mode")
        if int(row.get("recommended_max_new_tokens", 0)) < 2048:
            raise ValueError(f"Row {row['id']} has an invalid max-token recommendation")
    if len(base_arms) != EXPECTED_BASES or any(
        observed != set(ARMS) for observed in base_arms.values()
    ):
        raise ValueError("Expected 36 complete five-arm base units")
    return rows


def messages_for_row(row: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": row["system"]},
        {"role": "user", "content": row["initial_user"]},
        {"role": "assistant", "content": row["initial_assistant"]},
        {"role": "user", "content": row["followup_user"]},
    ]


def render_prompt(tokenizer: Any, row: dict[str, Any]) -> str:
    return tokenizer.apply_chat_template(
        messages_for_row(row),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def thinking_smoke_check(response: str) -> dict[str, Any]:
    stripped = response.lstrip()
    open_index = stripped.find("<think>")
    close_index = stripped.find("</think>", open_index + len("<think>"))
    reasoning = (
        stripped[open_index + len("<think>") : close_index].strip()
        if open_index >= 0 and close_index > open_index
        else ""
    )
    final = stripped[close_index + len("</think>") :].strip() if close_index >= 0 else ""
    checks = {
        "starts_with_think": stripped.startswith("<think>"),
        "has_ordered_close": close_index > open_index >= 0,
        "reasoning_nonempty": bool(reasoning),
        "final_nonempty": bool(final),
    }
    return {
        **checks,
        "passed": all(checks.values()),
        "response_characters": len(response),
        "reasoning_characters": len(reasoning),
        "final_characters": len(final),
    }


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
        if (
            not isinstance(row.get("response"), str)
            or row.get("cap") != "uncapped"
            or row.get("thinking_enabled") is not True
            or row.get("seed") != 0
        ):
            raise ValueError(f"Malformed result on line {number}: {item_id!r}")
        results[item_id] = {
            "id": item_id,
            "response": row["response"],
            "cap": "uncapped",
            "thinking_enabled": True,
            "seed": 0,
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
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in dataset_rows:
                if row["id"] in results:
                    handle.write(
                        json.dumps(results[row["id"]], ensure_ascii=False) + "\n"
                    )
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
        raise RuntimeError("Refusing an official v4 diagnostic from a dirty worktree")
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
        "kind": "M2.2b v4 public-protocol uncapped thinking diagnostic",
        "scope": "development diagnostic only; not held-out; capping prohibited",
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
            "enable_thinking": True,
            "complete_raw_continuation": True,
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
    runtime = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    runtime.update(values)
    atomic_write_json(path, runtime)


def generate_subset(
    model: torch.nn.Module,
    tokenizer: Any,
    subset: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    results: dict[str, dict[str, Any]],
    output_path: Path,
    *,
    batch_size: int,
    max_new_tokens: int,
    stage_name: str,
) -> int:
    if not subset:
        return 0
    device = model_input_device(model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    batches = (len(subset) + batch_size - 1) // batch_size
    generated_count = 0
    for batch_number, start in enumerate(range(0, len(subset), batch_size), start=1):
        batch = subset[start : start + batch_size]
        prompts = [render_prompt(tokenizer, row) for row in batch]
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
                "response": response,
                "cap": "uncapped",
                "thinking_enabled": True,
                "seed": 0,
            }
            generated_count += 1
        atomic_write_results(output_path, all_rows, results)
        print(
            json.dumps(
                {
                    "stage": stage_name,
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
        raise RuntimeError(f"v4 diagnostic must run inside cedar, found {sys.prefix}")
    if args.batch_size <= 0 or args.gpu_memory_gib <= 0:
        raise ValueError("Batch size and GPU memory must be positive")
    if args.max_new_tokens < 2048:
        raise ValueError("v4 thinking diagnostic requires max_new_tokens >= 2048")
    transformers.set_seed(0)

    rows = load_rows(args.data)
    run_dir = args.output_root / args.run_name
    configuration = build_configuration(args, rows)
    manifest = initialize_run(run_dir, configuration)
    output_path = run_dir / args.results_filename
    results = load_existing_results(
        output_path,
        valid_ids={row["id"] for row in rows},
    )

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
        newly_generated = 0
        smoke_id = rows[0]["id"]
        if len(results) < len(rows):
            if torch.cuda.device_count() < 4:
                raise RuntimeError("Qwen-3-32B v4 diagnostic requires four visible GPUs")
            max_memory = {
                index: f"{args.gpu_memory_gib}GiB"
                for index in range(torch.cuda.device_count())
            }
            model, tokenizer = load_qwen_model(
                max_memory=max_memory,
                local_files_only=True,
            )
            rendered = [render_prompt(tokenizer, row) for row in rows]
            update_runtime(
                run_dir,
                {
                    "device_map": getattr(model, "hf_device_map", None),
                    "rendered_prompts_sha256": object_sha256(
                        [
                            {"id": row["id"], "prompt": prompt}
                            for row, prompt in zip(rows, rendered, strict=True)
                        ]
                    ),
                    "smoke_prompt_sha256": hashlib.sha256(
                        rendered[0].encode("utf-8")
                    ).hexdigest(),
                    "smoke_prompt_ends_at_assistant_generation_marker": rendered[0].endswith(
                        "<|im_start|>assistant\n"
                    ),
                },
            )

            if smoke_id not in results:
                newly_generated += generate_subset(
                    model,
                    tokenizer,
                    [rows[0]],
                    rows,
                    results,
                    output_path,
                    batch_size=1,
                    max_new_tokens=args.max_new_tokens,
                    stage_name="smoke",
                )
            smoke = thinking_smoke_check(results[smoke_id]["response"])
            update_runtime(run_dir, {"smoke_id": smoke_id, "thinking_smoke": smoke})
            print(json.dumps({"stage": "smoke_check", "id": smoke_id, **smoke}), flush=True)
            if not smoke["passed"]:
                del model, tokenizer
                release_cuda_cache()
                raise RuntimeError("Thinking smoke check failed; full generation was not started")

            missing = [row for row in rows if row["id"] not in results]
            newly_generated += generate_subset(
                model,
                tokenizer,
                missing,
                rows,
                results,
                output_path,
                batch_size=args.batch_size,
                max_new_tokens=args.max_new_tokens,
                stage_name="generate",
            )
            del model, tokenizer
            release_cuda_cache()
        elif smoke_id in results:
            smoke = thinking_smoke_check(results[smoke_id]["response"])
            if not smoke["passed"]:
                raise RuntimeError("Stored smoke response does not contain a complete thinking segment")
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
