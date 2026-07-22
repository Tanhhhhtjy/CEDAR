#!/usr/bin/env python3
"""Dataset-local entry point for C2/V8 source, construction, and review gates."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from validate_c2_v8 import main  # noqa: E402

if __name__ == "__main__":
    main()
