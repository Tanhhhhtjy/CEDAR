#!/usr/bin/env python3
"""Prepare and freeze external replay & reference-KL corpora for ACL-exp-c1.

Outputs (written to experiments/ACL-exp-c1/data/):
  1. `c1_replay_norobots_529.jsonl`:
     529 stratified general-instruction examples from `HuggingFaceH4/no_robots` (`train` split, seed=20270104),
     giving exact 529 / (3000 + 529) = 14.99% replay ratio. Each row stores:
     `example_id`, `pair_group_id`, `is_replay=True`, `margin_sign=0`, `category`, `messages` (prefix turns),
     and `assistant_target` (final assistant reply).
  2. `c1_kl_wikitext2_200.jsonl`:
     200 clean prose passages (>= 200 chars, headings `= ... =` stripped) from
     `Salesforce/wikitext` (`wikitext-2-raw-v1`, `test` split, seed=20270104) for exact next-token KL evaluation.
  3. `c1_external_manifest.json`: counts, category distribution, and SHA-256 checksums.

Usage (on GPU host with HF cache / network):
  python experiments/ACL-exp-c1/scripts/c1_prepare_external.py

Usage (offline CPU self-test):
  python experiments/ACL-exp-c1/scripts/c1_prepare_external.py --selftest-offline
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
C1 = HERE.parents[0]
DATA_DIR = C1 / "data"
SEED = 20270104


def format_norobots_rows(raw_rows: list[dict[str, Any]], n_target: int = 529, seed: int = SEED) -> list[dict[str, Any]]:
    """Stratified-sample n_target valid chat rows from HuggingFaceH4/no_robots."""
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, row in enumerate(raw_rows):
        msgs = row.get("messages") or []
        if len(msgs) < 2 or msgs[-1].get("role") != "assistant":
            continue
        target = (msgs[-1].get("content") or "").strip()
        prefix = [
            {"role": m["role"], "content": (m.get("content") or "").strip()}
            for m in msgs[:-1]
            if m.get("role") in ("system", "user", "assistant") and (m.get("content") or "").strip()
        ]
        if not prefix or not target or prefix[-1]["role"] != "user":
            continue
        cat = str(row.get("category") or "General")
        by_cat[cat].append({"src_idx": idx, "category": cat, "messages": prefix, "assistant_target": target})

    rng = random.Random(seed)
    total_avail = sum(len(v) for v in by_cat.values())
    assert total_avail >= n_target, f"Not enough valid no_robots rows: {total_avail} < {n_target}"

    # Proportional stratified allocation across categories
    cats = sorted(by_cat.keys())
    alloc: dict[str, int] = {}
    rem_pool: list[tuple[float, str]] = []
    for c in cats:
        exact = n_target * len(by_cat[c]) / total_avail
        base = min(len(by_cat[c]), int(exact))
        alloc[c] = base
        rem_pool.append((exact - base, c))
    rem_pool.sort(reverse=True)
    need = n_target - sum(alloc.values())
    for _, c in rem_pool:
        if need <= 0:
            break
        if alloc[c] < len(by_cat[c]):
            alloc[c] += 1
            need -= 1

    out: list[dict[str, Any]] = []
    k = 0
    for c in cats:
        pool = list(by_cat[c])
        rng.shuffle(pool)
        for item in pool[: alloc[c]]:
            k += 1
            eid = f"C1-REPLAY-NR-{k:04d}"
            out.append(
                {
                    "example_id": eid,
                    "pair_group_id": eid,
                    "is_replay": True,
                    "margin_sign": 0,
                    "source": "HuggingFaceH4/no_robots:train",
                    "category": item["category"],
                    "messages": item["messages"],
                    "assistant_target": item["assistant_target"],
                }
            )
    rng.shuffle(out)
    assert len(out) == n_target
    return out


def format_wikitext_passages(raw_lines: list[str], n_target: int = 200, min_chars: int = 250, seed: int = SEED) -> list[dict[str, Any]]:
    """Pack non-heading wikitext-2 test paragraphs into clean prose windows for next-token KL."""
    clean: list[str] = []
    buf: list[str] = []
    for line in raw_lines:
        s = line.strip()
        if not s or (s.startswith("=") and s.endswith("=")):
            if buf:
                joined = " ".join(buf)
                if len(joined) >= min_chars:
                    clean.append(joined)
                buf = []
            continue
        buf.append(s)
        if sum(len(x) for x in buf) >= 1200:
            clean.append(" ".join(buf))
            buf = []
    if buf:
        joined = " ".join(buf)
        if len(joined) >= min_chars:
            clean.append(joined)

    rng = random.Random(seed)
    assert len(clean) >= n_target, f"Not enough clean wikitext passages: {len(clean)} < {n_target}"
    rng.shuffle(clean)
    return [
        {
            "passage_id": f"C1-KL-WT2-{i + 1:03d}",
            "source": "Salesforce/wikitext:wikitext-2-raw-v1:test",
            "text": txt,
        }
        for i, txt in enumerate(clean[:n_target])
    ]


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_selftest_offline() -> None:
    fake_nr = [
        {
            "category": ("Summarize" if i % 3 == 0 else ("Rewrite" if i % 3 == 1 else "Brainstorm")),
            "messages": [
                {"role": "user", "content": f"Synthetic prompt {i}"},
                {"role": "assistant", "content": f"Synthetic answer {i}"},
            ],
        }
        for i in range(60)
    ]
    nr_out = format_norobots_rows(fake_nr, n_target=30, seed=SEED)
    assert len(nr_out) == 30
    assert Counter(r["category"] for r in nr_out) == {"Summarize": 10, "Rewrite": 10, "Brainstorm": 10}
    assert all(r["is_replay"] and r["margin_sign"] == 0 for r in nr_out)

    fake_wt = []
    for i in range(25):
        fake_wt.append(f" = Heading {i} = ")
        fake_wt.append(f"Clean prose paragraph number {i}. " * 15)
        fake_wt.append("")
    wt_out = format_wikitext_passages(fake_wt, n_target=20, min_chars=200, seed=SEED)
    assert len(wt_out) == 20
    assert all(not r["text"].startswith("=") for r in wt_out)
    print("[OK] c1_prepare_external offline self-test passed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--n-replay", type=int, default=529)
    ap.add_argument("--n-kl", type=int, default=200)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--selftest-offline", action="store_true")
    args = ap.parse_args()

    if args.selftest_offline:
        run_selftest_offline()
        return

    from datasets import load_dataset

    print("[c1_ext] Loading HuggingFaceH4/no_robots (train)...")
    ds_nr = load_dataset("HuggingFaceH4/no_robots", split="train")
    nr_rows = format_norobots_rows(list(ds_nr), n_target=args.n_replay, seed=args.seed)
    nr_path = args.out_dir / f"c1_replay_norobots_{args.n_replay}.jsonl"
    nr_sha = _write_jsonl(nr_rows, nr_path)

    print("[c1_ext] Loading Salesforce/wikitext (wikitext-2-raw-v1, test)...")
    ds_wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    wt_rows = format_wikitext_passages([r["text"] for r in ds_wt], n_target=args.n_kl, seed=args.seed)
    wt_path = args.out_dir / f"c1_kl_wikitext2_{args.n_kl}.jsonl"
    wt_sha = _write_jsonl(wt_rows, wt_path)

    manifest = {
        "schema": "cedar.acl_exp_c1.external_corpora.v1",
        "seed": args.seed,
        "replay": {
            "path": nr_path.name,
            "n": len(nr_rows),
            "by_category": dict(Counter(r["category"] for r in nr_rows)),
            "sha256": nr_sha,
        },
        "kl_corpus": {
            "path": wt_path.name,
            "n": len(wt_rows),
            "sha256": wt_sha,
        },
    }
    man_path = args.out_dir / "c1_external_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("[OK] Frozen external corpora written:")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
