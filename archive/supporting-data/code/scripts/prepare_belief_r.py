#!/usr/bin/env python3
"""Download pinned Belief-R data and freeze the preregistered M2 sample IDs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cedar_m2.data.belief_r import (
    build_protocol_splits,
    download_belief_r,
    pair_belief_rows,
    read_belief_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("data/splits/m2_belief_r_ids.json"))
    parser.add_argument("--seed", type=int, default=20260711)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = download_belief_r(cache_dir=args.cache_dir)
    rows = read_belief_rows(csv_path)
    pairs = pair_belief_rows(rows)
    splits = build_protocol_splits(pairs, seed=args.seed)

    manifest = splits.manifest()
    manifest["seed"] = args.seed
    manifest["counts"] = {
        "raw_rows": len(rows),
        "paired_items": len(pairs),
        "development": len(splits.development),
        "pilot_bu": len(splits.pilot_bu),
        "pilot_bm": len(splits.pilot_bm),
        "extension_bu": len(splits.extension_bu),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest["counts"], ensure_ascii=False))


if __name__ == "__main__":
    main()
