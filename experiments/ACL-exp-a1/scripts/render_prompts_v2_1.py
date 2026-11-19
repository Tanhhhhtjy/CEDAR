#!/usr/bin/env python3
"""Compatibility shim: protocol v2.2 moved the renderer to render_prompts.py.

Kept so that runner code written for v2.1 (`from render_prompts_v2_1 import ...`) keeps working and
automatically renders v2.2 prompts from configs/exp_a1_config.yaml. New code: import render_prompts.
The exact v2.1 renderer is preserved in git history (commit f2a0739).
"""

from render_prompts import *  # noqa: F401,F403
from render_prompts import main

if __name__ == "__main__":
    main()
