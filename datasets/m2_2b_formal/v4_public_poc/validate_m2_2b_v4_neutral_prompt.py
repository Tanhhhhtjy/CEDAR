#!/usr/bin/env python3
"""Validate the neutral prompt derivative against the frozen source rows."""

from __future__ import annotations

import json

from build_m2_2b_v4_neutral_prompt import AUDIT, OUTPUT, SOURCE, load_jsonl

from cedar_m2.checkpoint import file_sha256
from cedar_m2.data.m2_2b_v4_neutral import neutralize_rows


def main() -> None:
    source = load_jsonl(SOURCE)
    observed = load_jsonl(OUTPUT)
    expected = neutralize_rows(source)
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    checks = {
        "source_rows_180": len(source) == 180,
        "neutral_rows_180": len(observed) == 180,
        "exact_deterministic_rebuild": observed == expected,
        "audit_passed": audit.get("status") == "passed",
        "audit_source_hash_matches": audit.get("source_sha256") == file_sha256(SOURCE),
        "audit_neutral_hash_matches": audit.get("neutral_sha256") == file_sha256(OUTPUT),
        "ids_and_order_unchanged": [row["id"] for row in source]
        == [row["id"] for row in observed],
    }
    if not all(checks.values()):
        raise SystemExit(json.dumps(checks, ensure_ascii=False, indent=2))
    print(
        json.dumps(
            {
                "status": "passed",
                "checks": checks,
                "neutral_sha256": file_sha256(OUTPUT),
                "prompt_contract_sha256": audit["prompt_contract_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
