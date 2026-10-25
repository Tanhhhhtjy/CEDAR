#!/usr/bin/env python3
"""CPU-only tokenizer, prompt-boundary, and official-p25 asset preflight."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = ROOT / "src"
EVIDENCE_DATA = HERE.parent / "prover_generator_evidence_validation_v1_1"
for path in (HERE, SRC, EVIDENCE_DATA):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from e2_common import (  # noqa: E402
    ANSWER_LABELS,
    ASSET_REPO,
    ASSET_REVISION,
    CAP_EXPERIMENT,
    CAP_LAYERS,
    EXPECTED_AXIS_SHA256,
    EXPECTED_CAP_CONFIG_SHA256,
    MODEL_REPO,
    MODEL_REVISION,
    VALIDITY_LABELS,
    VALIDITY_PREFIX,
    canonical_text_sha256,
    direction_sha256,
    file_sha256,
    object_sha256,
    read_jsonl,
    require,
    resolve_hub_cache,
    validate_anchor,
    write_json,
)

from cedar_m2.assets.capping_config import (  # noqa: E402
    download_qwen_capping_assets,
    load_axis,
    load_capping_experiment,
    verify_axis_orientation,
)


def render_chat(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def contextual_labels(tokenizer: Any, prefix: str, labels: tuple[str, ...]) -> dict[str, Any]:
    prefix_ids = list(tokenizer.encode(prefix, add_special_tokens=False))
    result: dict[str, Any] = {"prefix_ids": prefix_ids, "labels": {}}
    for label in labels:
        standalone = list(tokenizer.encode(label, add_special_tokens=False))
        combined = list(tokenizer.encode(prefix + label, add_special_tokens=False))
        boundary_preserved = combined[: len(prefix_ids)] == prefix_ids
        suffix = combined[len(prefix_ids) :] if boundary_preserved else []
        result["labels"][label] = {
            "standalone_ids": standalone,
            "contextual_suffix_ids": suffix,
            "boundary_preserved": boundary_preserved,
            "single_contextual_token": boundary_preserved and len(suffix) == 1,
        }
    return result


def run(cache_dir: Path | None, *, allow_download: bool) -> dict[str, Any]:
    from transformers import AutoTokenizer

    anchor = validate_anchor()
    answers = read_jsonl(HERE / "answer_inputs96.jsonl")
    validity = read_jsonl(HERE / "validity_inputs48.jsonl")
    hub_cache = resolve_hub_cache(cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        cache_dir=str(hub_cache) if hub_cache else None,
        trust_remote_code=False,
        local_files_only=not allow_download,
    )
    answer_rendered = [render_chat(tokenizer, row["messages"]) for row in answers]
    validity_rendered = [render_chat(tokenizer, row["messages"]) for row in validity]
    answer_ids = [tokenizer.encode(text, add_special_tokens=False) for text in answer_rendered]
    validity_ids = [tokenizer.encode(text, add_special_tokens=False) for text in validity_rendered]
    answer_labels = contextual_labels(tokenizer, '{"answer":"', ANSWER_LABELS)
    validity_labels = contextual_labels(tokenizer, VALIDITY_PREFIX, VALIDITY_LABELS)
    validity_label_variants = {
        prefix: contextual_labels(tokenizer, prefix, VALIDITY_LABELS)
        for prefix in (
            VALIDITY_PREFIX,
            '{"judgment": "',
            '{ "judgment":"',
            '{ "judgment": "',
        )
    }

    axis_path, config_path = download_qwen_capping_assets(
        cache_dir=hub_cache,
        local_files_only=not allow_download,
    )
    axis_hash = file_sha256(axis_path)
    config_hash = file_sha256(config_path)
    axis = load_axis(axis_path)
    caps = load_capping_experiment(config_path, CAP_EXPERIMENT)
    orientation = verify_axis_orientation(caps, axis)
    official_p25 = {
        "repo": ASSET_REPO,
        "revision": ASSET_REVISION,
        "axis_sha256": axis_hash,
        "capping_config_sha256": config_hash,
        "experiment": CAP_EXPERIMENT,
        "layers": list(sorted(caps)),
        "orientation": {str(layer): orientation[layer] for layer in sorted(orientation)},
        "directions": {str(layer): direction_sha256(caps[layer]) for layer in sorted(caps)},
    }
    labels_pass = all(
        item["single_contextual_token"]
        for group in (answer_labels, validity_labels)
        for item in group["labels"].values()
    ) and all(
        item["single_contextual_token"]
        and item["contextual_suffix_ids"]
        == validity_labels["labels"][label]["contextual_suffix_ids"]
        for group in validity_label_variants.values()
        for label, item in group["labels"].items()
    )
    passed = (
        len(answers) == 96
        and len(validity) == 48
        and labels_pass
        and axis_hash == EXPECTED_AXIS_SHA256
        and config_hash == EXPECTED_CAP_CONFIG_SHA256
        and tuple(sorted(caps)) == CAP_LAYERS
    )
    result = {
        "schema": "cedar.e2_phase_validity_choice.preflight.v1",
        "status": "passed" if passed else "failed",
        "gpu_used": False,
        "download_allowed_for_preflight": allow_download,
        "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
        "tokenizer_class": type(tokenizer).__name__,
        "answer_labels": answer_labels,
        "validity_labels": validity_labels,
        "validity_label_variants": validity_label_variants,
        "answer": {
            "rows": len(answers),
            "source_canonical_text_sha256": canonical_text_sha256(HERE / "answer_inputs96.jsonl"),
            "rendered_prompts_sha256": object_sha256(
                [
                    {"id": row["id"], "prompt": prompt}
                    for row, prompt in zip(answers, answer_rendered, strict=True)
                ]
            ),
            "tokenized_input_ids_sha256": object_sha256(
                [
                    {"id": row["id"], "input_ids": ids}
                    for row, ids in zip(answers, answer_ids, strict=True)
                ]
            ),
            "token_counts": {
                "minimum": min(map(len, answer_ids)),
                "maximum": max(map(len, answer_ids)),
            },
        },
        "validity": {
            "rows": len(validity),
            "source_canonical_text_sha256": canonical_text_sha256(HERE / "validity_inputs48.jsonl"),
            "rendered_prompts_sha256": object_sha256(
                [
                    {"id": row["id"], "prompt": prompt}
                    for row, prompt in zip(validity, validity_rendered, strict=True)
                ]
            ),
            "tokenized_input_ids_sha256": object_sha256(
                [
                    {"id": row["id"], "input_ids": ids}
                    for row, ids in zip(validity, validity_ids, strict=True)
                ]
            ),
            "token_counts": {
                "minimum": min(map(len, validity_ids)),
                "maximum": max(map(len, validity_ids)),
            },
        },
        "official_p25": official_p25,
        "anchor": anchor,
    }
    require(
        passed, f"E2 CPU tokenizer/asset preflight failed: {json.dumps(result, sort_keys=True)}"
    )
    write_json(HERE / "tokenizer_preflight.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    result = run(
        args.cache_dir.resolve() if args.cache_dir else None, allow_download=args.allow_download
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "gpu_used": result["gpu_used"],
                "answer_rows": result["answer"]["rows"],
                "validity_rows": result["validity"]["rows"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
