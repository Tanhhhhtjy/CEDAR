#!/usr/bin/env python3
"""Run the development-only uncapped Qwen objective-correction screen."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import transformers

from cedar_m2.checkpoint import (
    RecordStore,
    atomic_write_json,
    file_sha256,
    initialize_run,
    object_sha256,
)
from cedar_m2.data.correction_screen import VARIANTS, build_correction_cases, load_correction_screen
from cedar_m2.eval.correction import (
    CONTROL_RATE_THRESHOLD,
    MAIN_BROAD_CEILING,
    MAIN_NARROW_THRESHOLD,
    MAX_VERSION_A_EXCLUSION_RATE,
    MIN_CLEAN_ITEMS,
    correction_prompt_sha256,
    summarize_correction_screen,
)
from cedar_m2.models.qwen import QwenModelSpec, load_qwen_model
from cedar_m2.runners.correction import run_correction_scoring_stage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("prepare", "score", "summarize", "all"), default="all")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/development/m2_objective_correction_screen_v1.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/m2/objective_correction_screen"),
    )
    parser.add_argument("--run-name", default="qwen3_32b_uncapped_v4_five_arm")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gpu-memory-gib", type=int, default=20)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


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
        raise RuntimeError("Refusing an official correction-screen run from a dirty worktree")
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
    screen: Any,
    cases: tuple[Any, ...],
) -> dict[str, Any]:
    visible_gpus = torch.cuda.device_count()
    return {
        "kind": "M2.2 uncapped objective-correction diagnostic",
        "scope": "development screen; excluded from future confirmatory Pilot",
        "source": {
            "path": str(args.data),
            "sha256": file_sha256(args.data),
            "screen_id": screen.screen_id,
            "item_ids_sha256": object_sha256([item.item_id for item in screen.items]),
            "cases_sha256": object_sha256(
                [
                    {
                        "item_id": case.item_id,
                        "variant": case.variant,
                        "messages": list(case.messages),
                    }
                    for case in cases
                ]
            ),
        },
        "model": asdict(QwenModelSpec()),
        "intervention": "uncapped",
        "scoring": {
            "variants": VARIANTS,
            "prompt_sha256": correction_prompt_sha256(),
            "method": "multi-turn chat; single-token A/B/C logits; no sampling; thinking disabled",
            "batch_size": args.batch_size,
        },
        "decision_rule": {
            "main_narrow_minimum_rate": MAIN_NARROW_THRESHOLD,
            "main_broad_maximum_rate": MAIN_BROAD_CEILING,
            "control_minimum_rate": CONTROL_RATE_THRESHOLD,
            "maximum_version_a_exclusion_rate": MAX_VERSION_A_EXCLUSION_RATE,
            "minimum_clean_items": MIN_CLEAN_ITEMS,
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


def write_results_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    for record in records:
        row = {key: value for key, value in record.items() if key != "label_logits"}
        row.update({f"logit_{label}": record["label_logits"][label] for label in ("A", "B", "C")})
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["variant", "item_id"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def release_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if Path(sys.prefix).name != "cedar":
        raise RuntimeError(
            f"Correction screen must run inside the cedar environment, found {sys.prefix}"
        )
    if args.batch_size <= 0 or args.gpu_memory_gib <= 0:
        raise ValueError("Batch size and GPU memory must be positive")

    screen = load_correction_screen(args.data)
    cases = build_correction_cases(screen)
    expected = len(screen.items) * len(VARIANTS)
    if len(cases) != expected:
        raise ValueError(f"Expected {expected} expanded correction cases, found {len(cases)}")

    run_dir = args.output_root / args.run_name
    configuration = build_configuration(args, screen, cases)
    manifest = initialize_run(run_dir, configuration)
    fingerprint = str(manifest["fingerprint"])
    scores = RecordStore(run_dir / "scores", run_fingerprint=fingerprint)

    if args.stage == "prepare":
        print(
            json.dumps(
                {
                    "stage": "prepare",
                    "screen_id": screen.screen_id,
                    "items": len(screen.items),
                    "variants": VARIANTS,
                    "expected_scores": expected,
                    "prompt_sha256": correction_prompt_sha256(),
                    "run_fingerprint": fingerprint,
                }
            ),
            flush=True,
        )

    if args.stage in {"score", "all"}:
        existing = len(list(scores.payloads()))
        if existing > expected:
            raise ValueError(f"Score store has {existing} records, expected at most {expected}")
        if existing == expected:
            result = {"already_complete": existing, "newly_scored": 0}
        else:
            if torch.cuda.device_count() < 4:
                raise RuntimeError("Qwen-3-32B correction scoring requires four visible GPUs")
            max_memory = {
                index: f"{args.gpu_memory_gib}GiB" for index in range(torch.cuda.device_count())
            }
            model, tokenizer = load_qwen_model(max_memory=max_memory, local_files_only=True)
            update_runtime(run_dir, {"device_map": getattr(model, "hf_device_map", None)})
            result = run_correction_scoring_stage(
                model,
                tokenizer,
                cases,
                scores,
                batch_size=args.batch_size,
            )
            del model, tokenizer
            release_cuda_cache()
        print(json.dumps({"scoring": result}), flush=True)

    if args.stage in {"summarize", "all"}:
        records = list(scores.payloads())
        if len(records) != expected:
            raise ValueError(
                f"Expected {expected} score records before summary, found {len(records)}"
            )
        statistics = summarize_correction_screen(records)
        summary = {
            "kind": configuration["kind"],
            "screen_id": screen.screen_id,
            "run_fingerprint": fingerprint,
            "prompt_sha256": correction_prompt_sha256(),
            "statistics": statistics,
        }
        atomic_write_json(run_dir / "summary.json", summary)
        write_results_parquet(run_dir / "objective_correction_results.parquet", records)
        print(
            json.dumps(
                {
                    "items": statistics["items"],
                    "metrics": statistics["metrics"],
                    "conclusion": statistics["conclusion"],
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
