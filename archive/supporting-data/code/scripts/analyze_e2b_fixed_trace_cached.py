#!/usr/bin/env python3
"""Run the frozen E2B analysis on cache-faithful replay results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets/m2_2b_formal/e2b_fixed_trace_cached"
for path in (DATA, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import analyze_e2b_fixed_trace_cross as frozen_analysis  # noqa: E402
from e2b_cached_common import write_json  # noqa: E402

DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/m2/e2b_fixed_trace_cached_v1_1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    analysis = frozen_analysis.analyze(output_root)
    analysis.update(
        {
            "schema": "cedar.e2b_fixed_trace_cached.analysis.v1_1",
            "replay_engine": "model.generate_kv_cache_forced_tokens",
            "engineering_amendment_only": True,
            "v1_whole_prefix_failure_retained": True,
        }
    )
    write_json(output_root / "E2B_ANALYSIS.json", analysis)
    markdown_path = output_root / "E2B_ANALYSIS.md"
    existing = markdown_path.read_text(encoding="utf-8")
    note = (
        "Cache-faithful forced replay v1.1 was used. The scientific estimands and "
        "frozen family bootstrap are unchanged from E2B v1.\n\n"
    )
    markdown_path.write_text(
        existing.replace("\n\n", f"\n\n{note}", 1),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(analysis, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
