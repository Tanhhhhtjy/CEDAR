#!/usr/bin/env python3
"""Run the sealed, resumable M2.1 HarmBench generation/classification pipeline."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
from cedar_m2.data.harmbench import (
    HARMBENCH_REPO_URL,
    HARMBENCH_REVISION,
    HarmBenchBehavior,
    load_harmbench_split,
)
from cedar_m2.eval.harmbench import (
    HARMBENCH_CLASSIFIER_PROMPT,
    summarize_harmbench,
    summarize_harmbench_greedy,
)
from cedar_m2.models.harmbench import HarmBenchClassifierSpec, load_harmbench_classifier
from cedar_m2.models.qwen import QwenModelSpec, load_qwen_model
from cedar_m2.personas import PersonaConfig, load_persona_config
from cedar_m2.runners.harmbench import (
    CONDITIONS,
    SAMPLING_SEEDS,
    expected_generation_count,
    run_classification_stage,
    run_generation_stage,
)

PRIMARY_EXPERIMENT = "layers_46:54-p0.25"
TEST_EXPOSURE_NOTE = (
    "During schema inspection before template freeze, a repository search passively printed a "
    "small number of test input strings. No test generation, classifier output, or metric was "
    "observed; templates remain category-level and may only be tuned on validation."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=("prepare", "generate", "classify", "summarize", "all"),
        default="all",
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--test-template-sha256")
    parser.add_argument(
        "--harmbench-root",
        type=Path,
        default=Path("/data1/WuMingrui/HF_HOME/external/HarmBench"),
    )
    parser.add_argument(
        "--persona-config",
        type=Path,
        default=Path("configs/pilot/harmbench_personas.yaml"),
    )
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/m2/harmbench"))
    parser.add_argument("--run-name", default="m2_1_v1")
    parser.add_argument("--cache-dir", type=Path, default=Path("/data1/WuMingrui/HF_HOME/hub"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--classifier-batch-size", type=int, default=8)
    parser.add_argument("--gpu-memory-gib", type=int, default=20)
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--limit", type=int)
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


def select_limited_behaviors(
    behaviors: list[HarmBenchBehavior], limit: int | None, *, split: str
) -> list[HarmBenchBehavior]:
    if limit is None:
        return behaviors
    if split != "validation":
        raise ValueError("--limit is allowed only for validation diagnostics")
    if limit <= 0 or limit > len(behaviors):
        raise ValueError(f"--limit must be between 1 and {len(behaviors)}")
    by_category: dict[str, list[HarmBenchBehavior]] = defaultdict(list)
    for behavior in sorted(behaviors, key=lambda item: item.behavior_id):
        by_category[behavior.semantic_category].append(behavior)
    selected: list[HarmBenchBehavior] = []
    offset = 0
    while len(selected) < limit:
        added = False
        for category in sorted(by_category):
            if offset < len(by_category[category]) and len(selected) < limit:
                selected.append(by_category[category][offset])
                added = True
        if not added:
            break
        offset += 1
    return selected


def validate_test_authorization(args: argparse.Namespace, personas: PersonaConfig) -> None:
    if args.split != "test":
        if args.allow_test or args.test_template_sha256:
            raise ValueError("Test authorization flags must not be supplied for validation")
        return
    if not args.allow_test:
        raise PermissionError("HarmBench test remains sealed without --allow-test")
    personas.require_frozen()
    if args.test_template_sha256 != personas.sha256:
        raise PermissionError(
            "--test-template-sha256 must exactly match the frozen persona config fingerprint"
        )


def build_configuration(
    args: argparse.Namespace,
    personas: PersonaConfig,
    data_path: Path,
    behaviors: list[HarmBenchBehavior],
    axis_path: Path,
    capping_path: Path,
) -> dict[str, Any]:
    visible_gpus = torch.cuda.device_count()
    return {
        "kind": "M2.1 HarmBench validation diagnostic"
        if args.split == "validation"
        else "M2.1 HarmBench sealed test",
        "split": args.split,
        "limit": args.limit,
        "sources": {
            "harmbench": {
                "repo_url": HARMBENCH_REPO_URL,
                "revision": HARMBENCH_REVISION,
                "csv_sha256": file_sha256(data_path),
                "behavior_ids_sha256": object_sha256([item.behavior_id for item in behaviors]),
                "behavior_count": len(behaviors),
            },
            "assistant_axis": {
                "repo_id": ASSISTANT_AXIS_REPO,
                "revision": ASSISTANT_AXIS_REVISION,
                "axis_sha256": file_sha256(axis_path),
                "capping_config_sha256": file_sha256(capping_path),
                "experiment": PRIMARY_EXPERIMENT,
            },
        },
        "persona": {
            "path": str(personas.source_path),
            "version": personas.version,
            "status": personas.status,
            "sha256": personas.sha256,
        },
        "models": {
            "target": asdict(QwenModelSpec()),
            "classifier": asdict(HarmBenchClassifierSpec()),
            "classifier_prompt_sha256": hashlib.sha256(
                HARMBENCH_CLASSIFIER_PROMPT.encode()
            ).hexdigest(),
        },
        "generation": {
            "conditions": CONDITIONS,
            "sampled_seeds": SAMPLING_SEEDS,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 0,
            "max_new_tokens": 256,
            "greedy_robustness_check": True,
            "batch_size": args.batch_size,
            "batch_seed_derivation": (
                "sha256(base_seed:batch_index) first 31 bits; paired across conditions"
            ),
        },
        "classification": {
            "max_generation_tokens": 512,
            "max_new_tokens": 1,
            "batch_size": args.classifier_batch_size,
        },
        "statistics": {
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed": 20260711,
            "cluster_unit": "behavior",
            "bootstrap_stratification": "semantic_category",
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
        },
        "hardware": {
            "visible_gpu_count": visible_gpus,
            "visible_gpu_names": [torch.cuda.get_device_name(i) for i in range(visible_gpus)],
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
        "known_test_input_exposure": TEST_EXPOSURE_NOTE,
    }


def update_runtime(run_dir: Path, stage: str, values: dict[str, Any]) -> None:
    path = run_dir / "runtime.json"
    runtime = json.loads(path.read_text()) if path.exists() else {}
    runtime[stage] = values
    atomic_write_json(path, runtime)


def release_cuda_cache() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if Path(sys.prefix).name != "cedar":
        raise RuntimeError(f"M2.1 must run inside the cedar environment, found {sys.prefix}")
    personas = load_persona_config(args.persona_config)
    validate_test_authorization(args, personas)
    data_path, all_behaviors = load_harmbench_split(
        args.harmbench_root,
        args.split,
        allow_test=args.allow_test,
    )
    behaviors = select_limited_behaviors(all_behaviors, args.limit, split=args.split)

    axis_path, capping_path = download_qwen_capping_assets(
        cache_dir=args.cache_dir,
        local_files_only=True,
    )
    axis = load_axis(axis_path)
    caps = load_capping_experiment(capping_path, PRIMARY_EXPERIMENT)
    verify_axis_orientation(caps, axis)

    run_dir = args.output_root / args.split / args.run_name
    configuration = build_configuration(
        args,
        personas,
        data_path,
        behaviors,
        axis_path,
        capping_path,
    )
    manifest = initialize_run(run_dir, configuration)
    fingerprint = str(manifest["fingerprint"])
    generations = RecordStore(run_dir / "generations", run_fingerprint=fingerprint)
    classifications = RecordStore(run_dir / "classifications", run_fingerprint=fingerprint)

    if args.stage == "prepare":
        print(
            json.dumps(
                {
                    "stage": "prepare",
                    "split": args.split,
                    "behaviors": len(behaviors),
                    "run_fingerprint": fingerprint,
                    "persona_status": personas.status,
                }
            ),
            flush=True,
        )

    if args.stage in {"generate", "all"}:
        expected = expected_generation_count(len(behaviors))
        existing = len(list(generations.payloads()))
        if existing == expected:
            result = {"already_complete": existing, "newly_generated": 0}
        else:
            if existing > expected:
                raise ValueError(
                    f"Generation store has {existing} records, expected at most {expected}"
                )
            if torch.cuda.device_count() < 4:
                raise RuntimeError("Qwen M2.1 generation requires four visible GPUs")
            max_memory = {
                index: f"{args.gpu_memory_gib}GiB" for index in range(torch.cuda.device_count())
            }
            model, tokenizer = load_qwen_model(
                max_memory=max_memory,
                local_files_only=True,
            )
            update_runtime(
                run_dir,
                "generation_model",
                {"device_map": getattr(model, "hf_device_map", None)},
            )
            result = run_generation_stage(
                model,
                tokenizer,
                behaviors,
                personas,
                caps,
                generations,
                batch_size=args.batch_size,
            )
            del model, tokenizer
            release_cuda_cache()
        print(json.dumps({"generation": result}), flush=True)

    if args.stage in {"classify", "all"}:
        expected = expected_generation_count(len(behaviors))
        existing = len(list(classifications.payloads()))
        if existing == expected:
            result = {"already_complete": existing, "newly_classified": 0}
        else:
            if existing > expected:
                raise ValueError(
                    f"Classification store has {existing} records, expected at most {expected}"
                )
            if torch.cuda.device_count() < 1:
                raise RuntimeError("HarmBench classification requires one visible GPU")
            model, tokenizer = load_harmbench_classifier(
                device_map={"": 0},
                max_memory={0: "42GiB"},
                local_files_only=True,
            )
            update_runtime(
                run_dir,
                "classifier_model",
                {"device_map": getattr(model, "hf_device_map", None)},
            )
            result = run_classification_stage(
                model,
                tokenizer,
                behaviors,
                generations,
                classifications,
                batch_size=args.classifier_batch_size,
            )
            del model, tokenizer
            release_cuda_cache()
        print(json.dumps({"classification": result}), flush=True)

    if args.stage in {"summarize", "all"}:
        records = list(classifications.payloads())
        expected = expected_generation_count(len(behaviors))
        if len(records) != expected:
            raise ValueError(
                f"Expected {expected} classification records before summary, found {len(records)}"
            )
        summary = {
            "kind": configuration["kind"],
            "run_fingerprint": fingerprint,
            "persona_status": personas.status,
            "sampled_primary": summarize_harmbench(
                records,
                bootstrap_samples=args.bootstrap_samples,
                bootstrap_seed=20260711,
            ),
            "greedy_robustness": summarize_harmbench_greedy(records),
        }
        atomic_write_json(run_dir / "summary.json", summary)
        safe_summary = {
            "split": args.split,
            "sampled_asr": summary["sampled_primary"]["asr"],
            "sampled_gates": summary["sampled_primary"]["gates"],
            "greedy_asr": summary["greedy_robustness"]["asr"],
        }
        print(json.dumps(safe_summary), flush=True)


if __name__ == "__main__":
    main()
