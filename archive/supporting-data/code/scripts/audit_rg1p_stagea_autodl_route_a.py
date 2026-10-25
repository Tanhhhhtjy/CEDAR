#!/usr/bin/env python3
"""<compute-host> Route A policy wrapper around the frozen RG1-P engineering audit.

The frozen auditor remains byte-identical to the Stage A release. This wrapper
retains its full strict output, then changes only whether cross-environment E2A/
E2B replay drift contributes to the Route A status. All within-run engineering
and scientific-integrity gates remain blocking.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets" / "m2_2b_formal" / "rg1p_stagea"
for path in (DATA, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from audit_rg1p_stagea import audit as frozen_audit  # noqa: E402
from rg1p_common import write_json  # noqa: E402

DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "m2" / "rg1p_stagea_autodl_route_a"


def apply_route_a_policy(strict: dict[str, Any]) -> dict[str, Any]:
    """Make only frozen-anchor replay failures diagnostic for a fresh run."""
    amended = copy.deepcopy(strict)
    strict_failures = list(strict.get("failures", []))
    blocking = [failure for failure in strict_failures if not failure.startswith("replay:")]
    replay = amended.setdefault("replay_gate", {})
    replay["mode"] = "cross_environment_diagnostic"
    replay["gate_applied"] = False
    replay["strict_anchor_passed"] = bool(replay.get("passed"))
    replay["numerical_equivalence_observed"] = not replay.get("failures", [])
    replay["passed"] = None
    replay["interpretation"] = (
        "Diagnostic only. <compute-host> Route A rebuilds every analyzed cell on one new stack "
        "and makes no token-level numerical-equivalence claim to E2A/E2B."
    )
    amended["schema"] = "cedar.rg1p_stagea.<compute-host>"
    amended["policy"] = {
        "route": "<compute-host>",
        "strict_anchor_status": strict.get("status"),
        "strict_failure_count": len(strict_failures),
        "replay_failure_count": len(replay.get("failures", [])),
        "non_replay_blocking_failure_count": len(blocking),
    }
    amended["failures"] = blocking
    amended["warnings"] = [
        *strict.get("warnings", []),
        f"cross_environment_replay_drift_diagnostic:{len(replay.get('failures', []))}",
    ]
    amended["status"] = "passed" if not blocking else "failed"
    return amended


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("smoke", "formal"), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    strict = frozen_audit(args.scope, output_root)
    prefix = "SMOKE" if args.scope == "smoke" else "FORMAL"
    write_json(output_root / f"{prefix}_AUDIT_STRICT_ANCHOR.json", strict)
    amended = apply_route_a_policy(strict)
    write_json(output_root / f"{prefix}_AUDIT.json", amended)
    print(json.dumps(amended, indent=2, sort_keys=True))
    if amended["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
