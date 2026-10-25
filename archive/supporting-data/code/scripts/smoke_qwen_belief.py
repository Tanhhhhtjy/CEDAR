#!/usr/bin/env python3
"""Run an eight-item Qwen/Belief-R GPU smoke test without touching Pilot items."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import transformers

from cedar_m2.assets.capping_config import (
    download_qwen_capping_assets,
    load_axis,
    load_capping_experiment,
    verify_axis_orientation,
)
from cedar_m2.data.belief_r import download_belief_r, pair_belief_rows, read_belief_rows
from cedar_m2.eval.mcq import score_mcq_prompts
from cedar_m2.interventions.capping import ActivationCapper, CapStats
from cedar_m2.models.qwen import QwenModelSpec, load_qwen_model

EXPERIMENTS = (
    "layers_46:54-p0.25",
    "layers_46:54-p0.01",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-manifest", type=Path, default=Path("data/splits/m2_belief_r_ids.json")
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/m2/smoke_qwen_belief.json"))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--items-per-target", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gpu-memory-gib", type=int, default=20)
    return parser.parse_args()


def serialize_stats(stats: dict[int, CapStats]) -> dict[str, dict[str, float | int]]:
    return {
        str(layer): {
            **asdict(value),
            "trigger_rate": value.trigger_rate,
            "mean_projection_correction": value.mean_projection_correction,
        }
        for layer, value in stats.items()
    }


def main() -> None:
    args = parse_args()
    visible_gpus = torch.cuda.device_count()
    if visible_gpus < 2:
        raise RuntimeError(f"Qwen smoke test needs at least two visible GPUs, found {visible_gpus}")

    manifest = json.loads(args.split_manifest.read_text())
    development = manifest["development"]
    selected_records = []
    for target in ("BU", "BM"):
        selected_records.extend(
            [record for record in development if record["target"] == target][
                : args.items_per_target
            ]
        )

    belief_path = download_belief_r(cache_dir=args.cache_dir)
    pairs = pair_belief_rows(read_belief_rows(belief_path))
    pair_by_id = {pair.sample_id: pair for pair in pairs}
    selected_pairs = [pair_by_id[record["sample_id"]] for record in selected_records]
    questions = [pair.time_t1.question for pair in selected_pairs]
    labels = [pair.time_t1.ground_truth for pair in selected_pairs]

    axis_path, capping_path = download_qwen_capping_assets(cache_dir=args.cache_dir)
    axis = load_axis(axis_path)
    caps_by_experiment = {
        experiment: load_capping_experiment(capping_path, experiment) for experiment in EXPERIMENTS
    }
    orientation = {
        experiment: verify_axis_orientation(caps, axis)
        for experiment, caps in caps_by_experiment.items()
    }

    max_memory = {index: f"{args.gpu_memory_gib}GiB" for index in range(visible_gpus)}
    spec = QwenModelSpec()
    model, tokenizer = load_qwen_model(spec, max_memory=max_memory)

    conditions: dict[str, object] = {}
    baseline = score_mcq_prompts(
        model,
        tokenizer,
        questions,
        correct_labels=labels,
        batch_size=args.batch_size,
    )
    conditions["uncapped"] = {
        "scores": [score.to_dict() for score in baseline],
    }

    for experiment, caps in caps_by_experiment.items():
        with ActivationCapper(model, caps) as capper:
            scores = score_mcq_prompts(
                model,
                tokenizer,
                questions,
                correct_labels=labels,
                batch_size=args.batch_size,
            )
        conditions[experiment] = {
            "scores": [score.to_dict() for score in scores],
            "stats": serialize_stats(capper.stats),
        }

    output = {
        "kind": "development-only smoke test; not an M2 result",
        "created_at": datetime.now().astimezone().isoformat(),
        "system": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "visible_gpus": visible_gpus,
            "gpu_names": [torch.cuda.get_device_name(index) for index in range(visible_gpus)],
        },
        "model": asdict(spec),
        "device_map": getattr(model, "hf_device_map", None),
        "sample_ids": [pair.sample_id for pair in selected_pairs],
        "targets": [pair.target for pair in selected_pairs],
        "ground_truth": labels,
        "orientation_cosines": {
            experiment: {str(layer): value for layer, value in values.items()}
            for experiment, values in orientation.items()
        },
        "conditions": conditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")

    summary = {
        name: {
            "correct": sum(bool(score["correct"]) for score in result["scores"]),
            "items": len(result["scores"]),
            "triggered": sum(
                stats["triggered_positions"] for stats in result.get("stats", {}).values()
            ),
        }
        for name, result in conditions.items()
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
