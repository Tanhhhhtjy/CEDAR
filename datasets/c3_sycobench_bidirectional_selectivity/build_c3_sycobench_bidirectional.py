#!/usr/bin/env python3
"""Dataset-local entry point for the C3 builder."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_c3_sycobench_bidirectional import main  # noqa: E402, I001


if __name__ == "__main__":
    main()
