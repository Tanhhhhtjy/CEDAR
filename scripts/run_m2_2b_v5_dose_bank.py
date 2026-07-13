#!/usr/bin/env python3
"""Run one frozen condition of the M2.2b v5 dose-response candidate scan."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import transformers
from run_m2_2b_v4_neutral_pair import (
    aggregate_cap_stats,
    batch_key,
    direction_sha256,
    fixed_batches,
    git_provenance,
    materialize_outputs,
    release_cuda_cache,
    render_prompt,
    stats_to_dict,
    thinking_smoke_check,
    update_runtime,
)

from cedar_m2.assets.capping_config import (
    ASSISTANT_AXIS_REPO,
    ASSISTANT_AXIS_REVISION,
    download_qwen_capping_assets,
    load_axis,
    load_capping_experiment,
    verify_axis_orientation,
)
from cedar_m2.checkpoint import RecordStore, file_sha256, initialize_run, object_sha256
from cedar_m2.interventions.capping import ActivationCapper, LayerCap
from cedar_m2.models.qwen import QwenModelSpec, load_qwen_model, model_input_device
from cedar_m2.runners.belief import CONDITION_EXPERIMENTS

CONDITIONS = ("uncapped", "p75", "p50", "p25", "p1")
EXPECTED_ROWS = 432
EXPECTED_BATCH_SIZE = 2
EXPECTED_MAX_NEW_TOKENS = 2048
EXPECTED_COUNTS = Counter({"V": 144, "F": 144, "S": 144})
DEFAULT_DATA = Path(
    "datasets/m2_2b_formal/v5_dose_response_bank/m2_2b_v5_dose_response_candidate432.jsonl"
)
DEFAULT_BANK_MANIFEST = Path("datasets/m2_2b_formal/v5_dose_response_bank/manifest.json")
RUN_NAMES = {
    condition: f"qwen3_32b_v5_dose_bank_{condition}_thinking_greedy_seed0_development"
    for condition in CONDITIONS
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--stage", choices=("prepare", "generate", "all"), default="all")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--bank-manifest", type=Path, default=DEFAULT_BANK_MANIFEST)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/m2/m2_2b_v5_dose_response_bank"),
    )
    parser.add_argument("--run-name")
    parser.add_argument("--cache-dir", type=Path, default=Path("/data1/WuMingrui/HF_HOME/hub"))
    parser.add_argument("--batch-size", type=int, default=EXPECTED_BATCH_SIZE)
    parser.add_argument("--max-new-tokens", type=int, default=EXPECTED_MAX_NEW_TOKENS)
    parser.add_argument("--gpu-memory-gib", type=int, default=18)
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} rows, found {len(rows)}")
    if len({row.get("id") for row in rows}) != EXPECTED_ROWS:
        raise ValueError("Candidate IDs must be unique")
    if Counter(row.get("arm") for row in rows) != EXPECTED_COUNTS:
        raise ValueError("Expected 144 V, 144 F, and 144 S candidates")
    if any(row.get("selection_status") != "unselected_candidate" for row in rows):
        raise ValueError("The five-dose scan accepts only the unselected candidate bank")
    if any(tuple(row.get("cap_scan_conditions", ())) != CONDITIONS for row in rows):
        raise ValueError("A candidate has a different cap order")
    for row in rows:
        for field in ("system", "initial_user", "initial_assistant", "followup_user"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"{row.get('id')}: malformed {field}")
        if row.get("recommended_enable_thinking") is not True:
            raise ValueError(f"{row['id']}: thinking is not frozen on")
        if row.get("recommended_do_sample") is not False:
            raise ValueError(f"{row['id']}: sampling must remain off")
        if int(row.get("recommended_max_new_tokens", 0)) != EXPECTED_MAX_NEW_TOKENS:
            raise ValueError(f"{row['id']}: max tokens changed")
    return rows


def validate_bank_manifest(path: Path, data: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "candidate_bank_ready_for_static_review":
        raise ValueError("Candidate bank manifest is not ready")
    if manifest.get("selection", {}).get("performed") is not False:
        raise ValueError("Candidate bank was already selected")
    expected = manifest.get("files", {}).get(data.name, {}).get("sha256")
    if expected != file_sha256(data):
        raise ValueError("Candidate bank hash does not match its manifest")
    leakage = manifest.get("observation_only_leakage", {})
    if leakage.get("status") != "passed":
        raise ValueError("Observation-only leakage gate did not pass")
    return manifest


def load_cap_contract(
    cache_dir: Path,
    condition: str,
) -> tuple[dict[int, LayerCap], dict[str, Any]]:
    axis_path, config_path = download_qwen_capping_assets(
        cache_dir=cache_dir,
        local_files_only=True,
    )
    metadata: dict[str, Any] = {
        "repo_id": ASSISTANT_AXIS_REPO,
        "revision": ASSISTANT_AXIS_REVISION,
        "axis_file": "qwen-3-32b/assistant_axis.pt",
        "axis_sha256": file_sha256(axis_path),
        "capping_config_file": "qwen-3-32b/capping_config.pt",
        "capping_config_sha256": file_sha256(config_path),
        "experiment_id": CONDITION_EXPERIMENTS[condition],
        "positions": "all decoder-block output positions",
        "layers": {},
    }
    if condition == "uncapped":
        return {}, metadata
    experiment = CONDITION_EXPERIMENTS[condition]
    if experiment is None:
        raise ValueError(f"Missing official experiment mapping for {condition}")
    axis = load_axis(axis_path)
    caps = load_capping_experiment(config_path, experiment)
    orientation = verify_axis_orientation(caps, axis)
    if sorted(caps) != list(range(46, 54)):
        raise ValueError(f"Official {condition} layers changed: {sorted(caps)}")
    metadata["layers"] = {
        str(layer): {
            "threshold": cap.threshold,
            "direction_sha256": direction_sha256(cap),
            "axis_cosine": orientation[layer],
        }
        for layer, cap in sorted(caps.items())
    }
    metadata["parsed_experiment_sha256"] = object_sha256(metadata["layers"])
    return caps, metadata


def build_configuration(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    bank_manifest: dict[str, Any],
    cap_metadata: dict[str, Any],
) -> dict[str, Any]:
    visible_gpus = torch.cuda.device_count()
    return {
        "kind": "M2.2b v5 five-dose candidate-bank scan",
        "scope": "response-enriched development calibration; not held-out or confirmatory",
        "condition": args.condition,
        "condition_order": CONDITIONS,
        "dataset": {
            "path": str(args.data),
            "sha256": file_sha256(args.data),
            "manifest_path": str(args.bank_manifest),
            "manifest_sha256": file_sha256(args.bank_manifest),
            "construction_contract_sha256": bank_manifest["construction_contract_sha256"],
            "rows": len(rows),
            "candidate_bases": len({row["candidate_base_id"] for row in rows}),
            "families": len({row["template_family"] for row in rows}),
            "row_ids_sha256": object_sha256([row["id"] for row in rows]),
        },
        "model": asdict(QwenModelSpec()),
        "intervention": {
            "condition": args.condition,
            "active": args.condition != "uncapped",
            "official_contract": cap_metadata,
            "implementation": "ActivationCapper on decoder block outputs",
            "stats_definition": (
                "hook-visible layer-token positions; fixed batch-2 padding positions "
                "are included in recorded exposure"
            ),
        },
        "messages": {
            "role_order": ("system", "user", "assistant", "user"),
            "enable_thinking": True,
            "initial_assistant": "fixed transcript from dataset; never regenerated",
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
            "output": "raw continuation; no parsing, content retry, or repair",
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


def generate_batch(
    model: torch.nn.Module,
    tokenizer: Any,
    batch: list[dict[str, Any]],
    condition: str,
    caps: dict[int, LayerCap],
    *,
    max_new_tokens: int,
) -> dict[str, Any]:
    prompts = [render_prompt(tokenizer, row) for row in batch]
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    prompt_nonpadding_tokens = [int(value) for value in encoded["attention_mask"].sum(dim=1)]
    prompt_width = int(encoded["input_ids"].shape[1])
    device = model_input_device(model)
    encoded = {name: value.to(device) for name, value in encoded.items()}

    if condition == "uncapped":
        from contextlib import nullcontext

        capper: ActivationCapper | None = None
        context = nullcontext()
    else:
        capper = ActivationCapper(
            model,
            caps,
            positions="all",
            track_stats=True,
            defer_stats_sync=True,
        )
        context = capper
    with context, torch.inference_mode():
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
    return {
        "ids": [row["id"] for row in batch],
        "prompt_padded_width": prompt_width,
        "prompt_nonpadding_tokens": prompt_nonpadding_tokens,
        "response_tokens": [
            len(tokenizer.encode(response, add_special_tokens=False)) for response in responses
        ],
        "cap_stats": stats_to_dict(capper.stats) if capper is not None else None,
        "results": [
            {
                "id": row["id"],
                "response": response,
                "cap": condition,
                "thinking_enabled": True,
                "seed": 0,
            }
            for row, response in zip(batch, responses, strict=True)
        ],
    }


def main() -> None:
    args = parse_args()
    if Path(sys.prefix).name != "cedar":
        raise RuntimeError(f"Dose scan must run inside cedar, found {sys.prefix}")
    if args.batch_size != EXPECTED_BATCH_SIZE:
        raise ValueError(f"Batch size is frozen at {EXPECTED_BATCH_SIZE}")
    if args.max_new_tokens != EXPECTED_MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens is frozen at {EXPECTED_MAX_NEW_TOKENS}")
    if args.gpu_memory_gib <= 0:
        raise ValueError("GPU memory must be positive")
    transformers.set_seed(0)

    rows = load_rows(args.data)
    bank_manifest = validate_bank_manifest(args.bank_manifest, args.data)
    caps, cap_metadata = load_cap_contract(args.cache_dir, args.condition)
    run_name = args.run_name or RUN_NAMES[args.condition]
    run_dir = args.output_root / run_name
    configuration = build_configuration(args, rows, bank_manifest, cap_metadata)
    manifest = initialize_run(run_dir, configuration)
    store = RecordStore(run_dir / "batches", run_fingerprint=str(manifest["fingerprint"]))
    batches = fixed_batches(rows, args.batch_size)
    results_path = run_dir / "qwen_thinking_results.jsonl"
    batch_runtime_path = run_dir / "batch_runtime.jsonl"
    results, payloads = materialize_outputs(
        store,
        batches,
        args.condition,
        results_path,
        batch_runtime_path,
    )

    if args.stage == "prepare":
        print(
            json.dumps(
                {
                    "stage": "prepare",
                    "condition": args.condition,
                    "rows": len(rows),
                    "batches": len(batches),
                    "existing_results": len(results),
                    "dataset_sha256": file_sha256(args.data),
                    "official_experiment": cap_metadata["experiment_id"],
                    "run_fingerprint": manifest["fingerprint"],
                }
            ),
            flush=True,
        )

    if args.stage in {"generate", "all"}:
        newly_generated = 0
        missing_indices = [
            index
            for index, batch in enumerate(batches)
            if not store.contains(batch_key(index, batch, args.condition))
        ]
        if missing_indices:
            if torch.cuda.device_count() < 4:
                raise RuntimeError("Qwen3-32B dose scan requires four visible GPUs")
            max_memory = {
                index: f"{args.gpu_memory_gib}GiB" for index in range(torch.cuda.device_count())
            }
            update_runtime(
                run_dir,
                {
                    "generation_started_at": datetime.now().astimezone().isoformat(),
                    "condition": args.condition,
                },
            )
            model, tokenizer = load_qwen_model(max_memory=max_memory, local_files_only=True)
            tokenizer.padding_side = "left"
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            rendered = [render_prompt(tokenizer, row) for row in rows]
            tokenized = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in rendered]
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
                    "tokenized_input_ids_sha256": object_sha256(
                        [
                            {"id": row["id"], "input_ids": input_ids}
                            for row, input_ids in zip(rows, tokenized, strict=True)
                        ]
                    ),
                },
            )
            for index in missing_indices:
                batch = batches[index]
                generated = generate_batch(
                    model,
                    tokenizer,
                    batch,
                    args.condition,
                    caps,
                    max_new_tokens=args.max_new_tokens,
                )
                payload = {
                    "batch_index": index,
                    "condition": args.condition,
                    **generated,
                }
                store.write(batch_key(index, batch, args.condition), payload)
                newly_generated += len(batch)
                results, payloads = materialize_outputs(
                    store,
                    batches,
                    args.condition,
                    results_path,
                    batch_runtime_path,
                )
                if index == 0:
                    smoke = thinking_smoke_check(results[0]["response"])
                    update_runtime(run_dir, {"thinking_smoke": {"id": results[0]["id"], **smoke}})
                    if not smoke["passed"]:
                        del model, tokenizer
                        release_cuda_cache()
                        raise RuntimeError("Thinking smoke check failed")
                print(
                    json.dumps(
                        {
                            "stage": "generate",
                            "condition": args.condition,
                            "batch": index + 1,
                            "batches": len(batches),
                            "complete": len(results),
                        }
                    ),
                    flush=True,
                )
            del model, tokenizer
            release_cuda_cache()

        results, payloads = materialize_outputs(
            store,
            batches,
            args.condition,
            results_path,
            batch_runtime_path,
        )
        if results and not thinking_smoke_check(results[0]["response"])["passed"]:
            raise RuntimeError("Stored smoke response lacks complete thinking output")
        if len(results) == EXPECTED_ROWS:
            values: dict[str, Any] = {
                "complete_results": len(results),
                "complete_batches": len(payloads),
                "results_sha256": file_sha256(results_path),
                "batch_runtime_sha256": file_sha256(batch_runtime_path),
                "cap_trigger_stats": aggregate_cap_stats(payloads),
            }
            if newly_generated:
                values["generation_completed_at"] = datetime.now().astimezone().isoformat()
            update_runtime(run_dir, values)
        print(
            json.dumps(
                {
                    "condition": args.condition,
                    "results": str(results_path),
                    "complete": len(results),
                    "newly_generated": newly_generated,
                    "expected": EXPECTED_ROWS,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
