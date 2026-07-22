#!/usr/bin/env python3
"""Dataset-local entry point for the frozen C2/V8 builder."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_c2_v8_natural_confirmation import main  # noqa: E402

if __name__ == "__main__":
    main()
