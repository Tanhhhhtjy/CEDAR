#!/usr/bin/env python3
"""CPU-only tokenizer and official-p25 asset preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = ROOT / "src"
for path in (HERE, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cedar_m2.assets.capping_config import (
    download_qwen_capping_assets,
    load_axis,
    load_capping_experiment,
    verify_axis_orientation,
)
from evidence_validation_common import (
    ASSET_REPO,
    ASSET_REVISION,
    CAP_EXPERIMENT,
    CAP_LAYERS,
    EXPECTED_AXIS_SHA256,
    EXPECTED_CAP_CONFIG_SHA256,
    MODEL_REPO,
    MODEL_REVISION,
    PARENT_CANDIDATES,
    canonical_sha256,
    file_sha256,
    select_validation_rows,
    validate_module_selection,
    write_json,
)


def resolve_hub_cache(cache_dir: Path | None) -> Path | None:
    if cache_dir is None:
        return None
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


def render_chat(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def run(cache_dir: Path | None) -> dict[str, Any]:
    from transformers import AutoTokenizer

    module_selection = validate_module_selection()
    rows = select_validation_rows()
    hub_cache = resolve_hub_cache(cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        cache_dir=str(hub_cache) if hub_cache else None,
        trust_remote_code=False,
        local_files_only=True,
    )
    rendered = [render_chat(tokenizer, row["messages"]) for row in rows]
    input_ids = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in rendered]
    labels = {label: list(tokenizer.encode(label, add_special_tokens=False)) for label in ("A", "B", "C", "D")}

    axis_path, config_path = download_qwen_capping_assets(
        cache_dir=hub_cache,
        local_files_only=True,
    )
    axis_hash = file_sha256(axis_path)
    config_hash = file_sha256(config_path)
    axis = load_axis(axis_path)
    caps = load_capping_experiment(config_path, CAP_EXPERIMENT)
    orientation = verify_axis_orientation(caps, axis)
    asset_checks = {
        "repo": ASSET_REPO,
        "revision": ASSET_REVISION,
        "axis_sha256": axis_hash,
        "capping_config_sha256": config_hash,
        "experiment": CAP_EXPERIMENT,
        "layers": list(sorted(caps)),
        "orientation": {str(layer): orientation[layer] for layer in sorted(orientation)},
        "directions": {str(layer): direction_sha256(caps[layer]) for layer in sorted(caps)},
    }
    passed = (
        len(rows) == 96
        and all(len(ids) == 1 for ids in labels.values())
        and axis_hash == EXPECTED_AXIS_SHA256
        and config_hash == EXPECTED_CAP_CONFIG_SHA256
        and tuple(sorted(caps)) == CAP_LAYERS
    )
    result = {
        "schema": "cedar.prover_generator.evidence_validation_v1_1.inputs.v1",
        "status": "passed" if passed else "failed",
        "gpu_used": False,
        "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
        "tokenizer_class": type(tokenizer).__name__,
        "answer_label_token_ids": labels,
        "selection": {
            "rows": len(rows),
            "families": len({row["family_id"] for row in rows}),
            "bases": len({row["base_id"] for row in rows}),
            "ids_sha256": canonical_sha256([row["id"] for row in rows]),
            "messages_sha256": canonical_sha256([{"id": row["id"], "messages": row["messages"]} for row in rows]),
            "parent_candidates_raw_sha256": file_sha256(PARENT_CANDIDATES),
        },
        "rendered_prompts_sha256": canonical_sha256(
            [{"id": row["id"], "prompt": prompt} for row, prompt in zip(rows, rendered, strict=True)]
        ),
        "tokenized_input_ids_sha256": canonical_sha256(
            [{"id": row["id"], "input_ids": ids} for row, ids in zip(rows, input_ids, strict=True)]
        ),
        "prompt_token_counts": {
            "minimum": min(map(len, input_ids)),
            "maximum": max(map(len, input_ids)),
            "by_id": {row["id"]: len(ids) for row, ids in zip(rows, input_ids, strict=True)},
        },
        "official_p25": asset_checks,
        "module_selection": module_selection,
    }
    write_json(HERE / "validation_input_fingerprints.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    result = run(args.cache_dir.resolve() if args.cache_dir else None)
    print(
        {
            "status": result["status"],
            "rows": result["selection"]["rows"],
            "families": result["selection"]["families"],
            "labels_single_token": all(len(ids) == 1 for ids in result["answer_label_token_ids"].values()),
            "p25_assets": result["official_p25"]["experiment"],
            "gpu_used": False,
        }
    )


if __name__ == "__main__":
    main()
