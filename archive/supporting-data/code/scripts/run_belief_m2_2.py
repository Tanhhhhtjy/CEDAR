#!/usr/bin/env python3
"""Run the sealed, resumable M2.2 Belief-R scoring pipeline."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import transformers

from cedar_m2.assets.capping_config import (
    ASSISTANT_AXIS_REPO,
    ASSISTANT_AXIS_REVISION,
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
from cedar_m2.data.belief_r import (
    BELIEF_R_REPO,
    BELIEF_R_REVISION,
    download_belief_r,
    load_protocol_splits,
    pair_belief_rows,
    read_belief_rows,
)
from cedar_m2.data.formal import read_formal_items, validate_formal_items
from cedar_m2.eval.belief import summarize_belief
from cedar_m2.eval.mcq import mcq_prompt_sha256
from cedar_m2.models.qwen import QwenModelSpec, load_qwen_model
from cedar_m2.runners.belief import (
    CONDITION_EXPERIMENTS,
    conditions_for_split,
    expected_score_count,
    pairs_for_split,
    run_belief_scoring_stage,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("prepare", "score", "summarize", "all"),
        default="all",
    )
    parser.add_argument(
        "--split",
        choices=("development", "pilot", "extension"),
        default="development",
    )
    parser.add_argument("--allow-pilot", action="store_true")
    parser.add_argument("--allow-extension", action="store_true")
    parser.add_argument("--prompt-sha256")
    parser.add_argument("--authorization-summary", type=Path)
    parser.add_argument(
        "--formal-data",
        type=Path,
        default=Path("datasets/formal_domain/formal_full.jsonl"),
    )
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("data/splits/m2_belief_r_ids.json"),
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("<local-path>"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/m2/belief_r"))
    parser.add_argument("--run-name")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gpu-memory-gib", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def git_provenance(*, allow_dirty: bool) -> dict[str, Any]:
    """Record the exact code state and reject dirty official runs."""
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
        raise RuntimeError("Refusing an official run from a dirty worktree; commit changes first")
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


def validate_authorization(args: argparse.Namespace, prompt_hash: str) -> None:
    """Keep Pilot and extension items sealed behind earlier-stage evidence."""
    if args.split == "development":
        if (
            args.allow_pilot
            or args.allow_extension
            or args.prompt_sha256
            or args.authorization_summary
        ):
            raise ValueError("Authorization flags must not be supplied for development")
        return

    expected_flag = args.allow_pilot if args.split == "pilot" else args.allow_extension
    irrelevant_flag = args.allow_extension if args.split == "pilot" else args.allow_pilot
    if irrelevant_flag:
        raise ValueError(f"Authorization flag does not apply to M2.2 {args.split}")
    if not expected_flag:
        raise PermissionError(f"M2.2 {args.split} remains sealed without its explicit allow flag")
    if args.prompt_sha256 != prompt_hash:
        raise PermissionError("--prompt-sha256 must match the frozen MCQ prompt fingerprint")
    if args.authorization_summary is None or not args.authorization_summary.is_file():
        raise PermissionError("An earlier-stage --authorization-summary is required")

    summary = json.loads(args.authorization_summary.read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping):
        raise PermissionError("Authorization summary must be a JSON object")
    if summary.get("prompt_sha256") != prompt_hash:
        raise PermissionError("Authorization summary uses a different MCQ prompt")
    statistics = summary.get("statistics", {})
    if not isinstance(statistics, Mapping):
        raise PermissionError("Authorization summary has malformed statistics")
    if args.split == "pilot":
        development_gate = statistics.get("development_gate")
        if (
            summary.get("split") != "development"
            or not isinstance(development_gate, Mapping)
            or not development_gate.get("prompt_valid")
        ):
            raise PermissionError("Pilot requires a passing development summary")
    else:
        pilot_gate = statistics.get("pilot_gate")
        if (
            summary.get("split") != "pilot"
            or not isinstance(pilot_gate, Mapping)
            or pilot_gate.get("next_step") != "expand_bu"
        ):
            raise PermissionError("Extension requires a Pilot summary requesting BU expansion")


def build_configuration(
    args: argparse.Namespace,
    *,
    belief_path: Path,
    axis_path: Path,
    capping_path: Path,
    sample_ids: list[str],
    conditions: tuple[str, ...],
    orientation: dict[str, dict[int, float]],
    prompt_hash: str,
) -> dict[str, Any]:
    """Build the immutable manifest configuration for one M2.2 phase."""
    visible_gpus = torch.cuda.device_count()
    return {
        "kind": f"M2.2 Belief-R {args.split}",
        "split": args.split,
        "sources": {
            "belief_r": {
                "repo_id": BELIEF_R_REPO,
                "revision": BELIEF_R_REVISION,
                "csv_sha256": file_sha256(belief_path),
            },
            "formal_data": {
                "path": str(args.formal_data),
                "sha256": file_sha256(args.formal_data),
            },
            "split_manifest": {
                "path": str(args.split_manifest),
                "sha256": file_sha256(args.split_manifest),
            },
            "sample_ids_sha256": object_sha256(sample_ids),
        },
        "model": asdict(QwenModelSpec()),
        "assistant_axis": {
            "repo_id": ASSISTANT_AXIS_REPO,
            "revision": ASSISTANT_AXIS_REVISION,
            "axis_sha256": file_sha256(axis_path),
            "capping_config_sha256": file_sha256(capping_path),
            "condition_experiments": {
                condition: CONDITION_EXPERIMENTS[condition] for condition in conditions
            },
            "orientation_cosines": {
                condition: {str(layer): value for layer, value in values.items()}
                for condition, values in orientation.items()
            },
        },
        "scoring": {
            "conditions": conditions,
            "steps": ("time_t", "time_t1"),
            "prompt_sha256": prompt_hash,
            "method": "single-token A/B/C logits; no sampling",
            "batch_size": args.batch_size,
        },
        "statistics": {
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": 20260711,
            "bootstrap_unit": "item",
            "confidence_interval": "one-sided 95% lower bound",
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
    """Write flattened score records atomically as the M2.2 tabular deliverable."""
    rows: list[dict[str, Any]] = []
    for record in records:
        row = {key: value for key, value in record.items() if key != "label_logits"}
        logits = record["label_logits"]
        row.update({f"logit_{label}": logits[label] for label in ("A", "B", "C")})
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["sample_id", "condition", "step"])
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
        raise RuntimeError(f"M2.2 must run inside the cedar environment, found {sys.prefix}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.gpu_memory_gib <= 0:
        raise ValueError("--gpu-memory-gib must be positive")
    if args.bootstrap_samples <= 0:
        raise ValueError("--bootstrap-samples must be positive")
    prompt_hash = mcq_prompt_sha256()
    validate_authorization(args, prompt_hash)

    belief_path = download_belief_r(cache_dir=args.cache_dir, local_files_only=True)
    source_pairs = pair_belief_rows(read_belief_rows(belief_path))
    frozen_splits = load_protocol_splits(args.split_manifest, source_pairs)
    all_frozen_pairs = (
        *frozen_splits.development,
        *frozen_splits.pilot_bu,
        *frozen_splits.pilot_bm,
        *frozen_splits.extension_bu,
    )
    formal_items = read_formal_items(args.formal_data)
    validate_formal_items(formal_items, all_frozen_pairs)
    pairs = pairs_for_split(frozen_splits, args.split)
    conditions = conditions_for_split(args.split)

    axis_path, capping_path = download_qwen_capping_assets(
        cache_dir=args.cache_dir,
        local_files_only=True,
    )
    axis = load_axis(axis_path)
    caps_by_condition = {
        condition: load_capping_experiment(capping_path, experiment)
        for condition in conditions
        if (experiment := CONDITION_EXPERIMENTS[condition]) is not None
    }
    orientation = {
        condition: verify_axis_orientation(caps, axis)
        for condition, caps in caps_by_condition.items()
    }

    run_name = args.run_name or f"m2_2_{args.split}_v1"
    run_dir = args.output_root / args.split / run_name
    configuration = build_configuration(
        args,
        belief_path=belief_path,
        axis_path=axis_path,
        capping_path=capping_path,
        sample_ids=[pair.sample_id for pair in pairs],
        conditions=conditions,
        orientation=orientation,
        prompt_hash=prompt_hash,
    )
    manifest = initialize_run(run_dir, configuration)
    fingerprint = str(manifest["fingerprint"])
    scores = RecordStore(run_dir / "scores", run_fingerprint=fingerprint)

    if args.stage == "prepare":
        print(
            json.dumps(
                {
                    "stage": "prepare",
                    "split": args.split,
                    "items": len(pairs),
                    "conditions": conditions,
                    "expected_scores": expected_score_count(len(pairs), conditions),
                    "prompt_sha256": prompt_hash,
                    "run_fingerprint": fingerprint,
                }
            ),
            flush=True,
        )

    if args.stage in {"score", "all"}:
        expected = expected_score_count(len(pairs), conditions)
        existing = len(list(scores.payloads()))
        if existing == expected:
            result = {"already_complete": existing, "newly_scored": 0}
        else:
            if existing > expected:
                raise ValueError(f"Score store has {existing} records, expected at most {expected}")
            if torch.cuda.device_count() < 4:
                raise RuntimeError("Qwen M2.2 scoring requires four visible GPUs")
            max_memory = {
                index: f"{args.gpu_memory_gib}GiB" for index in range(torch.cuda.device_count())
            }
            model, tokenizer = load_qwen_model(
                max_memory=max_memory,
                local_files_only=True,
            )
            update_runtime(run_dir, {"device_map": getattr(model, "hf_device_map", None)})
            result = run_belief_scoring_stage(
                model,
                tokenizer,
                pairs,
                conditions,
                caps_by_condition,
                scores,
                batch_size=args.batch_size,
            )
            del model, tokenizer
            release_cuda_cache()
        print(json.dumps({"scoring": result}), flush=True)

    if args.stage in {"summarize", "all"}:
        records = list(scores.payloads())
        expected = expected_score_count(len(pairs), conditions)
        if len(records) != expected:
            raise ValueError(
                f"Expected {expected} score records before summary, found {len(records)}"
            )
        statistics = summarize_belief(
            records,
            conditions,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=20260711,
        )
        summary = {
            "kind": configuration["kind"],
            "split": args.split,
            "run_fingerprint": fingerprint,
            "prompt_sha256": prompt_hash,
            "statistics": statistics,
        }
        atomic_write_json(run_dir / "summary.json", summary)
        write_results_parquet(run_dir / "m22_belief_results.parquet", records)
        print(
            json.dumps(
                {
                    "split": args.split,
                    "items": statistics["items"],
                    "development_gate": statistics.get("development_gate"),
                    "pilot_gate": statistics.get("pilot_gate"),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
