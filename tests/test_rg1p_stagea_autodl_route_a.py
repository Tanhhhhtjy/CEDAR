from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from audit_rg1p_stagea_autodl_route_a import apply_route_a_policy  # noqa: E402


def strict_record(*failures: str) -> dict[str, object]:
    return {
        "schema": "cedar.rg1p_stagea.audit.v1",
        "status": "failed" if failures else "passed",
        "replay_gate": {"failures": [{"cell_id": "x"}], "passed": False},
        "failures": list(failures),
        "warnings": [],
    }


def test_replay_drift_alone_is_diagnostic() -> None:
    result = apply_route_a_policy(strict_record("replay:x"))
    assert result["status"] == "passed"
    assert result["failures"] == []
    assert result["replay_gate"]["gate_applied"] is False
    assert result["replay_gate"]["strict_anchor_passed"] is False


def test_non_replay_failure_remains_blocking() -> None:
    result = apply_route_a_policy(strict_record("replay:x", "tensor:x"))
    assert result["status"] == "failed"
    assert result["failures"] == ["tensor:x"]
