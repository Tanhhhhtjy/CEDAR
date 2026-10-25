#!/usr/bin/env python3
"""CPU-only fixed-Qwen tokenizer preflight for development v1.1."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from development_v1_1_common import (
    CALIBRATION_FAMILIES,
    EVIDENCE_ARMS,
    HERE,
    SOCIAL_CANDIDATE_ARMS,
    TOKENIZER_BINDING,
    canonical_sha256,
    file_sha256,
    read_jsonl,
    write_json,
)

SMOKE_BASE_IDS = {
    "CEDAR-DEV-001",
    "CEDAR-DEV-009",
    "CEDAR-DEV-018",
    "CEDAR-DEV-026",
}


def resolve_transformers_cache_dir(cache_dir: Path | None) -> Path | None:
    """Accept either a Hugging Face home directory or its hub cache."""
    if cache_dir is None:
        return None
    hub = cache_dir / "hub"
    return hub if hub.is_dir() else cache_dir


def _encode_chat(tokenizer: Any, messages: list[dict[str, str]], *, thinking: bool) -> list[int]:
    kwargs = {"tokenize": True, "add_generation_prompt": True}
    try:
        return list(tokenizer.apply_chat_template(messages, enable_thinking=thinking, **kwargs))
    except TypeError:
        return list(tokenizer.apply_chat_template(messages, **kwargs))


def _render_chat(tokenizer: Any, messages: list[dict[str, str]], *, thinking: bool) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _runner_fingerprints(tokenizer: Any, arms: list[dict[str, Any]], candidate_path: Path) -> dict[str, Any]:
    cells = {}
    tracks = {
        "evidence": {"arms": set(EVIDENCE_ARMS), "thinking": True},
        "social-candidates": {"arms": set(SOCIAL_CANDIDATE_ARMS), "thinking": False},
    }
    for scope in ("smoke", "formal"):
        for track, spec in tracks.items():
            selected = [
                row
                for row in arms
                if row["family_id"] in set(CALIBRATION_FAMILIES)
                and row["development_split"] == "template_calibration"
                and row["arm"] in spec["arms"]
                and (scope == "formal" or row["base_id"] in SMOKE_BASE_IDS)
            ]
            rendered = [
                _render_chat(tokenizer, row["messages"], thinking=spec["thinking"])
                for row in selected
            ]
            tokenized = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in rendered]
            key = f"template-calibration/{scope}/{track}"
            cells[key] = {
                "rows": len(selected),
                "selected_ids_sha256": canonical_sha256([row["id"] for row in selected]),
                "rendered_prompts_sha256": canonical_sha256(
                    [
                        {"id": row["id"], "prompt": prompt}
                        for row, prompt in zip(selected, rendered, strict=True)
                    ]
                ),
                "tokenized_input_ids_sha256": canonical_sha256(
                    [
                        {"id": row["id"], "input_ids": ids}
                        for row, ids in zip(selected, tokenized, strict=True)
                    ]
                ),
            }
    return {
        "schema": "cedar.prover_generator_development.v1_1.runner_inputs.v1",
        "status": "passed",
        "binding": TOKENIZER_BINDING,
        "candidate_raw_sha256": file_sha256(candidate_path),
        "cells": cells,
        "gpu_used": False,
    }


def run(root: Path, cache_dir: Path | None) -> dict[str, Any]:
    from transformers import AutoTokenizer

    transformers_cache = resolve_transformers_cache_dir(cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_BINDING["repo"],
        revision=TOKENIZER_BINDING["revision"],
        cache_dir=str(transformers_cache) if transformers_cache else None,
        trust_remote_code=False,
        local_files_only=True,
    )
    candidate_path = root / "development_candidates336.jsonl"
    arms = read_jsonl(candidate_path)
    by_base: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in arms:
        by_base[row["base_id"]][row["arm"]] = row
    rows = []
    for base_id in sorted(by_base):
        group = by_base[base_id]
        social = [group[arm] for arm in ("AO", "S1", "S2")]
        followup_ids = {
            row["arm"]: list(tokenizer.encode(row["messages"][-1]["content"], add_special_tokens=False))
            for row in social
        }
        full_ids = {
            row["arm"]: _encode_chat(tokenizer, row["messages"], thinking=False)
            for row in social
        }
        vf_followup = {
            arm: list(tokenizer.encode(group[arm]["messages"][-1]["content"], add_special_tokens=False))
            for arm in ("V", "F")
        }
        vf_full = {
            arm: _encode_chat(tokenizer, group[arm]["messages"], thinking=True)
            for arm in ("V", "F")
        }
        checks = {
            "social_followup_token_count_equal": len({len(ids) for ids in followup_ids.values()}) == 1,
            "social_full_chat_token_count_equal": len({len(ids) for ids in full_ids.values()}) == 1,
            "social_shared_prefix_before_followup": len({canonical_sha256(row["messages"][:-1]) for row in social}) == 1,
            "social_same_target": len({row["target_answer"] for row in social}) == 1,
            "social_same_gold": len({row["gold_answer"] for row in social}) == 1,
            "vf_followup_token_count_equal": len(vf_followup["V"]) == len(vf_followup["F"]),
            "vf_full_chat_token_count_equal": len(vf_full["V"]) == len(vf_full["F"]),
            "vf_shared_prefix_before_followup": group["V"]["messages"][:-1] == group["F"]["messages"][:-1],
            "vf_same_target": group["V"]["target_answer"] == group["F"]["target_answer"],
        }
        rows.append(
            {
                "base_id": base_id,
                "target_answer": social[0]["target_answer"],
                "social_followup_tokens": {arm: len(ids) for arm, ids in followup_ids.items()},
                "social_full_chat_tokens": {arm: len(ids) for arm, ids in full_ids.items()},
                "social_followup_ids_sha256": {arm: canonical_sha256(ids) for arm, ids in followup_ids.items()},
                "vf_followup_tokens": {arm: len(ids) for arm, ids in vf_followup.items()},
                "vf_full_chat_tokens": {arm: len(ids) for arm, ids in vf_full.items()},
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    labels = {}
    for label in ("A", "B", "C", "D"):
        ids = list(tokenizer.encode(label, add_special_tokens=False))
        labels[label] = {"ids": ids, "single_token": len(ids) == 1}
    all_social = len(rows) == 48 and all(
        row["checks"]["social_followup_token_count_equal"]
        and row["checks"]["social_full_chat_token_count_equal"]
        for row in rows
    )
    all_vf = len(rows) == 48 and all(
        row["checks"]["vf_followup_token_count_equal"]
        and row["checks"]["vf_full_chat_token_count_equal"]
        for row in rows
    )
    result = {
        "schema": "cedar.prover_generator_development.v1_1.tokenizer_preflight.v1",
        "status": "passed" if all_social and all_vf and all(item["single_token"] for item in labels.values()) else "failed",
        "binding": TOKENIZER_BINDING,
        "tokenizer_class": type(tokenizer).__name__,
        "base_count": len(rows),
        "all_exact_social_lengths": all_social,
        "all_exact_vf_lengths": all_vf,
        "answer_label_tokens": labels,
        "rows": rows,
        "gpu_used": False,
    }
    write_json(root / "tokenizer_preflight.json", result)
    write_json(root / "runner_input_fingerprints.json", _runner_fingerprints(tokenizer, arms, candidate_path))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    result = run(args.root.resolve(), args.cache_dir.resolve() if args.cache_dir else None)
    print({"status": result["status"], "bases": result["base_count"], "social_equal": result["all_exact_social_lengths"], "vf_equal": result["all_exact_vf_lengths"], "gpu_used": False})


if __name__ == "__main__":
    main()
