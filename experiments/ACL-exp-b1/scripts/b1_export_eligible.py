#!/usr/bin/env python3
"""Export frozen A1 v2.2 no_think Turn-1 eligible base_ids for ACL-exp-b1 eligible-subset verification.

Eligibility definition matches experiments/ACL-exp-a1/scripts/analyze_v22_2x2.py::eligible_items:
  Turn-1 natural K=4 (sample_idx in 0..3): itt_outcome == 'initial' on >= 3/4 samples for BOTH
  turn1_V and turn1_F.

Reads directly from git ref `origin/ACL-exp-a1` (commit 069f830 or given --git-ref) so it works
from the ACL-exp-b1 branch without switching branches.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import io
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
B1 = HERE.parents[0]
REPO = B1.parents[1]
DATASETS = ("symbolic_pilot", "natural_pilot")
MODELS = ("qwen3_8b", "qwen3_14b")


def git_rev_parse(ref: str) -> str:
    return subprocess.check_output(["git", "rev-parse", ref], cwd=REPO).decode("utf-8").strip()


def extract_eligible(ref: str, alias: str) -> dict:
    rel = f"experiments/ACL-exp-a1/results/v2_2/raw_records/{alias}__no_think.jsonl.gz"
    raw = subprocess.check_output(["git", "show", f"{ref}:{rel}"], cwd=REPO)
    sha = hashlib.sha256(raw).hexdigest()
    ok = collections.Counter()
    by_ds: dict[str, set[str]] = collections.defaultdict(set)
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
        for line in gz:
            r = json.loads(line.decode("utf-8"))
            ds = r["dataset"]
            if ds not in DATASETS:
                continue
            by_ds[ds].add(r["base_id"])
            if r["arm"] in ("turn1_V", "turn1_F") and 0 <= r["sample_idx"] <= 3:
                if r["itt_outcome"] == "initial":
                    ok[(r["base_id"], r["arm"])] += 1
    eligible_by_ds = {
        ds: sorted(b for b in by_ds[ds] if ok[(b, "turn1_V")] >= 3 and ok[(b, "turn1_F")] >= 3)
        for ds in DATASETS
    }
    all_ids = sorted({b for ids in eligible_by_ds.values() for b in ids})
    return {
        "source_file": rel,
        "source_sha256": sha,
        "counts": {ds: len(eligible_by_ds[ds]) for ds in DATASETS},
        "total": len(all_ids),
        "by_dataset": eligible_by_ds,
        "eligible_base_ids": all_ids,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--git-ref", default="origin/ACL-exp-a1")
    ap.add_argument("--out", default=str(B1 / "data" / "eligible_ids_v2_2_no_think.json"))
    args = ap.parse_args()
    commit = git_rev_parse(args.git_ref)
    payload = {
        "source_git_ref": args.git_ref,
        "source_commit": commit,
        "criterion": "turn1_V >= 3/4 AND turn1_F >= 3/4 (sample_idx 0..3, itt_outcome == 'initial', no_think)",
        "models": {alias: extract_eligible(commit, alias) for alias in MODELS},
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    for alias, info in payload["models"].items():
        print(f"[OK] {alias}: total={info['total']} ({info['counts']})")
    print(f"[OK] wrote {out}")


if __name__ == "__main__":
    main()
