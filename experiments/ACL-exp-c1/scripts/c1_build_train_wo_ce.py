#!/usr/bin/env python3
"""Derive the 4-class leave-out training file (excluding `cross_entity_transfer`) for Step 2B Plan A.

Leaves frozen `data/c1_train_examples_nothink.jsonl` 100% untouched. Writes:
  - data/c1_train_examples_wo_ce_nothink.jsonl (400 bases = 2,400 rule examples, exact 50/50 tone symmetry)
  - data/c1_train_wo_ce_manifest.json

Usage:
  python experiments/ACL-exp-c1/scripts/c1_build_train_wo_ce.py
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
C1 = HERE.parents[0]


def main() -> None:
    src = C1 / "data" / "c1_train_examples_nothink.jsonl"
    src_sha = hashlib.sha256(src.read_bytes()).hexdigest()
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    kept = [r for r in rows if r["fallacy_class"] != "cross_entity_transfer"]

    assert len(kept) == 2400, f"Expected 2400 examples, got {len(kept)}"
    assert len({r["base_id"] for r in kept}) == 400
    assert all(r["fallacy_class"] != "cross_entity_transfer" for r in kept)

    cnt = Counter(r["arm"] for r in kept)
    assert cnt["V"] == cnt["F"] == 400
    assert cnt["V_hes"] == cnt["F_hes"] == 200
    assert cnt["V_auth"] == cnt["F_auth"] == 200

    out = C1 / "data" / "c1_train_examples_wo_ce_nothink.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in kept) + "\n", encoding="utf-8", newline="\n")
    out_sha = hashlib.sha256(out.read_bytes()).hexdigest()

    manifest = {
        "schema": "cedar.acl_exp_c1.train_wo_ce_manifest.v1",
        "source_file": "experiments/ACL-exp-c1/data/c1_train_examples_nothink.jsonl",
        "source_sha256": src_sha,
        "held_out_fallacy_class": "cross_entity_transfer",
        "n_bases": 400,
        "n_examples": len(kept),
        "by_fallacy_class": dict(Counter(r["fallacy_class"] for r in kept)),
        "by_arm": dict(cnt),
        "sha256": {"c1_train_examples_wo_ce_nothink.jsonl": out_sha},
    }
    (C1 / "data" / "c1_train_wo_ce_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"[OK] wrote {out} ({len(kept)} examples / 400 bases, sha256={out_sha[:16]}...)")


if __name__ == "__main__":
    main()
