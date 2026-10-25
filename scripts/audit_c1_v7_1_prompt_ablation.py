#!/usr/bin/env python3
"""Audit frozen C1/V7.1 smoke and formal runs."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from scripts.audit_c1_v7_heldout import (
        LAYERS,
        by_id,
        comparable_configuration,
        hook_clean,
        load_phase,
        max_logit_difference,
    )
except ModuleNotFoundError:
    from audit_c1_v7_heldout import (
        LAYERS,
        by_id,
        comparable_configuration,
        hook_clean,
        load_phase,
        max_logit_difference,
    )

from cedar_m2.checkpoint import atomic_write_json

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts" / "m2" / "c1_v7_1_prompt_cue_ablation"


def p25_cap_stats_complete(row: Mapping[str, Any]) -> bool:
    stats = row.get("cap_stats")
    if not isinstance(stats, Mapping):
        return False
    by_layer = stats.get("by_layer")
    if not isinstance(by_layer, Mapping):
        return False
    layer_keys = {str(layer) for layer in LAYERS}
    return (
        set(by_layer) == layer_keys
        and all(
            isinstance(by_layer[str(layer)], Mapping)
            and int(by_layer[str(layer)].get("total_positions", 0)) > 0
            for layer in LAYERS
        )
        and all(
            float(by_layer[str(layer)].get("projection_correction_sum", -1.0)) >= 0.0
            for layer in LAYERS
        )
    )


def audit_smoke(root: Path) -> dict[str, Any]:
    nohook = load_phase(root, "smoke-nohook", 3)
    monitor = load_phase(root, "smoke-monitor", 3)
    capped = load_phase(root, "smoke-p25", 1)
    left = by_id(nohook["results"])
    right = by_id(monitor["results"])
    ids = sorted(left)
    max_difference = max_logit_difference(nohook["results"], monitor["results"])
    checks = {
        "same_ids": set(left) == set(right),
        "same_responses": all(
            left[item_id].get("response") == right[item_id].get("response")
            for item_id in ids
        ),
        "same_prompt_hashes": all(
            left[item_id].get("prompt_sha256") == right[item_id].get("prompt_sha256")
            and left[item_id].get("prompt_token_ids_sha256")
            == right[item_id].get("prompt_token_ids_sha256")
            for item_id in ids
        ),
        "logit_difference_le_1e_4": max_difference <= 1e-4,
        "nohook_has_no_telemetry": all(
            row.get("axis_telemetry") is None and row.get("cap_stats") is None
            for row in nohook["results"]
        ),
        "monitor_has_telemetry": all(
            isinstance(row.get("axis_telemetry"), Mapping) for row in monitor["results"]
        ),
        "all_hooks_clean": all(
            hook_clean(row) for row in [*nohook["results"], *monitor["results"], *capped["results"]]
        ),
        "thinking_integrity": all(
            row.get("thinking_integrity", {}).get("passed") is True
            for row in [*nohook["results"], *monitor["results"], *capped["results"]]
        ),
    }
    checks["p25_stats_complete_layers_46_53"] = all(
        p25_cap_stats_complete(row) for row in capped["results"]
    )
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "status": "passed" if not failures else "failed",
        "checks": checks,
        "failures": failures,
        "max_abc_logit_difference": max_difference,
        "behavior_not_used_as_gate": True,
    }


def audit_formal(root: Path) -> dict[str, Any]:
    uncapped = load_phase(root, "formal-uncapped", 144)
    capped = load_phase(root, "formal-p25", 144)
    left = by_id(uncapped["results"])
    right = by_id(capped["results"])
    ids = sorted(left)
    left_config = uncapped["manifest"]["configuration"]
    right_config = capped["manifest"]["configuration"]
    checks = {
        "same_ids": set(left) == set(right),
        "same_order": [row["id"] for row in uncapped["results"]]
        == [row["id"] for row in capped["results"]],
        "same_prompts": all(
            left[item_id].get("prompt_sha256") == right[item_id].get("prompt_sha256")
            and left[item_id].get("prompt_token_ids_sha256")
            == right[item_id].get("prompt_token_ids_sha256")
            for item_id in ids
        ),
        "same_row_metadata": all(
            all(
                left[item_id].get(field) == right[item_id].get(field)
                for field in (
                    "base_id",
                    "family_id",
                    "arm",
                    "initial_answer",
                    "target_answer",
                    "distractor_answer",
                    "gold_answer",
                )
            )
            for item_id in ids
        ),
        "conditions_exact": left_config.get("condition") == "uncapped"
        and right_config.get("condition") == "p25",
        "configuration_equal_except_condition_and_phase": comparable_configuration(left_config)
        == comparable_configuration(right_config),
        "all_hooks_clean": all(
            hook_clean(row) for row in [*uncapped["results"], *capped["results"]]
        ),
        "all_telemetry_present": all(
            isinstance(row.get("axis_telemetry"), Mapping)
            for row in [*uncapped["results"], *capped["results"]]
        ),
        "uncapped_has_no_cap_stats": all(
            row.get("cap_stats") is None for row in uncapped["results"]
        ),
        "p25_has_cap_stats": all(
            isinstance(row.get("cap_stats"), Mapping) for row in capped["results"]
        ),
        "prompt_ablation_equal": left_config.get("prompt_ablation")
        == right_config.get("prompt_ablation"),
    }
    failures = [name for name, passed in checks.items() if not passed]
    return {
        "status": "passed" if not failures else "failed",
        "checks": checks,
        "failures": failures,
        "invalid_or_truncated_retained": True,
        "content_retries": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "formal", "all"), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    args = parser.parse_args()
    reports: dict[str, Any] = {}
    if args.stage in {"smoke", "all"}:
        reports["smoke"] = audit_smoke(args.output_root)
        atomic_write_json(args.output_root / "SMOKE_AUDIT.json", reports["smoke"])
    if args.stage in {"formal", "all"}:
        reports["formal"] = audit_formal(args.output_root)
        atomic_write_json(args.output_root / "CROSS_CONDITION_AUDIT.json", reports["formal"])
    print(json.dumps(reports, ensure_ascii=False, indent=2, sort_keys=True))
    if any(report["status"] != "passed" for report in reports.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
