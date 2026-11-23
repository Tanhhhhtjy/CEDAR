#!/usr/bin/env python3
"""ACL-exp-c3.5 vLLM Native LoRA Parity Verifier wrapping `c3_check_vllm_lora_parity.py` with `c35_modules` & `c35_export_peft_lora`.

CPU Self-Test:
  python experiments/ACL-exp-c3.5/scripts/c35_check_vllm_lora_parity.py --selftest-tiny
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
C35 = HERE.parents[0]
REPO = C35.parents[1]
C3 = REPO / "experiments" / "ACL-exp-c3"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(C3 / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-a1" / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-b1" / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-c1" / "scripts"))

import c35_export_peft_lora as EXP35  # noqa: E402
import c35_modules as M35  # noqa: E402
import c3_check_vllm_lora_parity as PAR3  # noqa: E402
import c3_eval as C3EV  # noqa: E402

C3EV.M3 = M35
PAR3.M3 = M35
PAR3.EXP3 = EXP35


def main() -> None:
    if "--components-file" not in sys.argv and "--selftest-tiny" not in sys.argv:
        sys.argv.extend(["--components-file", str(C35 / "data" / "c35_components_frozen.json")])
    PAR3.main()


if __name__ == "__main__":
    main()
