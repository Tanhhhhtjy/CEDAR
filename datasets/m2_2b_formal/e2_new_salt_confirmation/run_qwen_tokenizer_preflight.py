#!/usr/bin/env python3
"""CPU-only tokenizer and official-p25 asset preflight for E2 new-salt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from e2_new_salt_common import (
    CAP_EXPERIMENT,
    CAP_LAYERS,
    EXPECTED_AXIS_SHA256,
    EXPECTED_CAP_CONFIG_SHA256,
    HERE,
    MODEL_REPO,
    MODEL_REVISION,
    OPTION_LETTERS,
    canonical_jsonl_sha256,
    file_sha256,
    object_sha256,
    read_json,
    require,
    write_json,
)

from cedar_m2.assets.capping_config import (
    download_qwen_capping_assets,
    load_axis,
    load_capping_experiment,
    verify_axis_orientation,
)

MAX_NEW_TOKENS = 4096


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def resolve_hub_cache(cache_dir: Path | None) -> Path | None:
    if cache_dir is None:
        return None
    hub = cache_dir / "hub"
    return hub if hub.is_dir() else cache_dir


def render_chat(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def contextual_labels(tokenizer: Any, prefix: str) -> dict[str, Any]:
    prefix_ids = list(tokenizer.encode(prefix, add_special_tokens=False))
    labels: dict[str, Any] = {}
    for label in OPTION_LETTERS:
        standalone = list(tokenizer.encode(label, add_special_tokens=False))
        combined = list(tokenizer.encode(prefix + label, add_special_tokens=False))
        boundary_preserved = combined[: len(prefix_ids)] == prefix_ids
        suffix = combined[len(prefix_ids) :] if boundary_preserved else []
        labels[label] = {
            "standalone_ids": standalone,
            "contextual_suffix_ids": suffix,
            "boundary_preserved": boundary_preserved,
            "single_contextual_token": boundary_preserved and len(suffix) == 1,
        }
    return {"prefix": prefix, "prefix_ids": prefix_ids, "labels": labels}


def run(cache_dir: Path | None) -> dict[str, Any]:
    from transformers import AutoTokenizer

    hub_cache = resolve_hub_cache(cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        cache_dir=str(hub_cache) if hub_cache else None,
        trust_remote_code=False,
        local_files_only=True,
    )
    arms_path = HERE / "arms96.jsonl"
    arms = read_jsonl(arms_path)
    rendered = [render_chat(tokenizer, row["messages"]) for row in arms]
    input_ids = [
        list(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in rendered
    ]
    labels = contextual_labels(tokenizer, '{"answer":"')
    label_tokens = [
        item["contextual_suffix_ids"][0]
        for item in labels["labels"].values()
        if item["single_contextual_token"]
    ]
    axis_path, config_path = download_qwen_capping_assets(
        cache_dir=hub_cache, local_files_only=True
    )
    axis_hash = file_sha256(axis_path)
    config_hash = file_sha256(config_path)
    axis = load_axis(axis_path)
    caps = load_capping_experiment(config_path, CAP_EXPERIMENT)
    orientation = verify_axis_orientation(caps, axis)
    model_limit = int(tokenizer.model_max_length)
    maximum = max(map(len, input_ids))
    checks = {
        "arms_96": len(arms) == 96,
        "unique_ids_96": len({row["id"] for row in arms}) == 96,
        "all_contextual_labels_single_token": all(
            item["single_contextual_token"] for item in labels["labels"].values()
        ),
        "contextual_label_tokens_distinct": len(set(label_tokens)) == 4,
        "prompt_plus_generation_within_context": maximum + MAX_NEW_TOKENS <= model_limit,
        "axis_hash_frozen": axis_hash == EXPECTED_AXIS_SHA256,
        "cap_config_hash_frozen": config_hash == EXPECTED_CAP_CONFIG_SHA256,
        "official_layers_46_53": tuple(sorted(caps)) == CAP_LAYERS,
        "construction_automatic_gate_passed": read_json(
            HERE / "automatic_validation_report.json"
        ).get("status")
        == "passed",
        "deterministic_rebuild_passed": read_json(
            HERE / "deterministic_rebuild_report.json"
        ).get("status")
        == "passed",
    }
    result = {
        "schema": "cedar.e2_new_salt_confirmation.tokenizer_preflight.v1",
        "status": "passed" if all(checks.values()) else "failed",
        "gpu_used": False,
        "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
        "tokenizer_class": type(tokenizer).__name__,
        "checks": checks,
        "answer_labels": labels,
        "inputs": {
            "rows": len(arms),
            "source_canonical_sha256": canonical_jsonl_sha256(arms_path),
            "rendered_prompts_sha256": object_sha256(
                [
                    {"id": row["id"], "prompt": prompt}
                    for row, prompt in zip(arms, rendered, strict=True)
                ]
            ),
            "tokenized_input_ids_sha256": object_sha256(
                [
                    {"id": row["id"], "input_ids": ids}
                    for row, ids in zip(arms, input_ids, strict=True)
                ]
            ),
            "token_counts": {
                "minimum": min(map(len, input_ids)),
                "maximum": maximum,
                "model_max_length": model_limit,
                "max_new_tokens": MAX_NEW_TOKENS,
            },
        },
        "official_p25": {
            "axis_path": str(axis_path),
            "config_path": str(config_path),
            "axis_sha256": axis_hash,
            "capping_config_sha256": config_hash,
            "experiment": CAP_EXPERIMENT,
            "layers": list(sorted(caps)),
            "orientation": {
                str(layer): orientation[layer] for layer in sorted(orientation)
            },
        },
    }
    write_json(HERE / "tokenizer_preflight.json", result)
    require(
        result["status"] == "passed",
        f"Tokenizer/asset preflight failed: {[k for k, v in checks.items() if not v]}",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    result = run(args.cache_dir.resolve() if args.cache_dir else None)
    print(
        json.dumps(
            {
                "status": result["status"],
                "rows": result["inputs"]["rows"],
                "minimum_tokens": result["inputs"]["token_counts"]["minimum"],
                "maximum_tokens": result["inputs"]["token_counts"]["maximum"],
                "gpu_used": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
