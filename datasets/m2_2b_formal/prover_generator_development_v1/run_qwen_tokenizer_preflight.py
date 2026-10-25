"""CPU-only exact AO/S length preflight for the frozen Qwen tokenizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from development_common import TOKENIZER_BINDING
from hardening_common import canonical_sha256, write_json  # type: ignore

HERE = Path(__file__).resolve().parent


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _encode_chat(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": True,
    }
    try:
        return list(tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs))
    except TypeError:
        return list(tokenizer.apply_chat_template(messages, **kwargs))


def run(root: Path, cache_dir: Path | None) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_BINDING["repo"],
        revision=TOKENIZER_BINDING["revision"],
        cache_dir=str(cache_dir) if cache_dir else None,
        trust_remote_code=False,
    )
    arms = read_jsonl(root / "development_arms288.jsonl")
    by_base: dict[str, dict[str, dict[str, Any]]] = {}
    for row in arms:
        by_base.setdefault(row["base_id"], {})[row["arm"]] = row
    rows = []
    for base_id in sorted(by_base):
        ao = by_base[base_id]["AO"]
        social = by_base[base_id]["S"]
        ao_followup_ids = list(tokenizer.encode(ao["messages"][-1]["content"], add_special_tokens=False))
        s_followup_ids = list(
            tokenizer.encode(social["messages"][-1]["content"], add_special_tokens=False)
        )
        ao_full_ids = _encode_chat(tokenizer, ao["messages"])
        s_full_ids = _encode_chat(tokenizer, social["messages"])
        checks = {
            "followup_token_count_equal": len(ao_followup_ids) == len(s_followup_ids),
            "full_chat_token_count_equal": len(ao_full_ids) == len(s_full_ids),
            "shared_prefix_before_followup": ao["messages"][:-1] == social["messages"][:-1],
            "same_target": ao["target_answer"] == social["target_answer"],
            "same_gold": ao["gold_answer"] == social["gold_answer"],
        }
        rows.append(
            {
                "base_id": base_id,
                "target_answer": ao["target_answer"],
                "ao_followup_tokens": len(ao_followup_ids),
                "s_followup_tokens": len(s_followup_ids),
                "ao_full_chat_tokens": len(ao_full_ids),
                "s_full_chat_tokens": len(s_full_ids),
                "ao_followup_ids_sha256": canonical_sha256(ao_followup_ids),
                "s_followup_ids_sha256": canonical_sha256(s_followup_ids),
                "checks": checks,
                "passed": all(checks.values()),
            }
        )
    special = {}
    for label in ("A", "B", "C", "D"):
        ids = tokenizer.encode(label, add_special_tokens=False)
        special[label] = {"ids": list(ids), "single_token": len(ids) == 1}
    result = {
        "schema": "cedar.prover_generator_development.tokenizer_preflight.v1",
        "status": "passed" if len(rows) == 48 and all(row["passed"] for row in rows) else "failed",
        "binding": TOKENIZER_BINDING,
        "tokenizer_class": type(tokenizer).__name__,
        "base_count": len(rows),
        "all_exact_ao_s_length": len(rows) == 48 and all(row["passed"] for row in rows),
        "answer_label_tokens": special,
        "rows": rows,
        "gpu_used": False,
    }
    write_json(root / "tokenizer_preflight.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=HERE)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    result = run(args.root.resolve(), args.cache_dir.resolve() if args.cache_dir else None)
    print(
        {
            "status": result["status"],
            "bases": result["base_count"],
            "all_exact_ao_s_length": result["all_exact_ao_s_length"],
            "gpu_used": False,
        }
    )


if __name__ == "__main__":
    main()
