#!/usr/bin/env python3
"""Score and pair the two M2.2b v4 neutral-prompt calibration conditions."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from cedar_m2.checkpoint import atomic_write_json, file_sha256

try:
    from scripts.run_m2_2b_v4_neutral_pair import ARMS, CONDITIONS, RUN_NAMES
except ModuleNotFoundError as exc:
    if exc.name != "scripts":
        raise
    # Support the documented ``python scripts/analyze_...py`` invocation.
    from run_m2_2b_v4_neutral_pair import ARMS, CONDITIONS, RUN_NAMES

EXPECTED_ROWS = 180
CONTRASTS = (
    ("V_minus_F", "V", "F"),
    ("V_minus_S", "V", "S"),
    ("F_minus_R", "F", "R"),
    ("S_minus_R", "S", "R"),
    ("V_minus_R", "V", "R"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "datasets/m2_2b_formal/v4_public_poc/"
            "m2_2b_v4_public_poc180_neutral_prompt_arms.jsonl"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/m2/m2_2b_v4_public_poc"),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 6)


def parse_answer(response: object) -> str | None:
    if not isinstance(response, str):
        return None
    final = response.rsplit("</think>", 1)[-1].strip()
    try:
        parsed = json.loads(final)
    except json.JSONDecodeError:
        return None
    answer = parsed.get("answer") if isinstance(parsed, dict) else None
    if isinstance(answer, str) and answer.strip().upper() in {"A", "B", "C"}:
        return answer.strip().upper()
    return None


def response_integrity(response: object) -> dict[str, bool]:
    if not isinstance(response, str):
        return {
            "starts_with_think": False,
            "ordered_close": False,
            "nonempty_reasoning": False,
            "strict_final_json": False,
        }
    stripped = response.lstrip()
    open_index = stripped.find("<think>")
    close_index = stripped.find("</think>", open_index + len("<think>"))
    reasoning = (
        stripped[open_index + len("<think>") : close_index].strip()
        if open_index >= 0 and close_index > open_index
        else ""
    )
    final = stripped[close_index + len("</think>") :].strip() if close_index >= 0 else ""
    try:
        parsed = json.loads(final)
    except json.JSONDecodeError:
        parsed = None
    return {
        "starts_with_think": stripped.startswith("<think>"),
        "ordered_close": close_index > open_index >= 0,
        "nonempty_reasoning": bool(reasoning),
        "strict_final_json": isinstance(parsed, dict),
    }


def join_records(
    data: list[dict[str, Any]],
    results: list[dict[str, Any]],
    condition: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_ids = [row["id"] for row in data]
    observed_ids = [row.get("id") for row in results]
    duplicates = [item_id for item_id, count in Counter(observed_ids).items() if count > 1]
    missing = sorted(set(expected_ids) - set(observed_ids))
    extra = sorted(set(observed_ids) - set(expected_ids))
    by_id = {row.get("id"): row for row in results}
    joined: list[dict[str, Any]] = []
    for item in data:
        result = by_id.get(item["id"], {})
        answer = parse_answer(result.get("response"))
        integrity = response_integrity(result.get("response"))
        joined.append(
            {
                "id": item["id"],
                "base_id": item["base_id"],
                "arm": item["arm"],
                "answer": answer,
                "initial": item["initial_answer_id"],
                "gold": item["gold_final_answer_id"],
                "target": item["induced_target_answer_id"],
                "updated": None if answer is None else answer != item["initial_answer_id"],
                "correct": None if answer is None else answer == item["gold_final_answer_id"],
                "target_accepted": (
                    None
                    if answer is None or item["induced_target_answer_id"] is None
                    else answer == item["induced_target_answer_id"]
                ),
                "stratum": item["stratum"],
                "subtype": item["matched_valid_subtype"],
                "family": item["template_family"],
                "V_difficulty": item.get("V_difficulty"),
                "F_difficulty": item.get("F_difficulty"),
                "F_error_class": item.get("F_error_class"),
                "integrity": integrity,
                "result_metadata_valid": (
                    result.get("cap") == condition
                    and result.get("thinking_enabled") is True
                    and result.get("seed") == 0
                    and isinstance(result.get("response"), str)
                ),
            }
        )
    integrity = {
        "result_rows": len(results),
        "unique_ids": len(set(observed_ids)),
        "ids_exact_and_in_dataset_order": observed_ids == expected_ids,
        "missing_ids": missing,
        "extra_ids": extra,
        "duplicate_ids": sorted(duplicates),
        "metadata_valid": sum(row["result_metadata_valid"] for row in joined),
        "starts_with_think": sum(row["integrity"]["starts_with_think"] for row in joined),
        "ordered_think_close": sum(row["integrity"]["ordered_close"] for row in joined),
        "nonempty_reasoning": sum(row["integrity"]["nonempty_reasoning"] for row in joined),
        "strict_final_json": sum(row["integrity"]["strict_final_json"] for row in joined),
        "valid_final_answer": sum(row["answer"] is not None for row in joined),
    }
    return joined, integrity


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["answer"] is not None]
    targeted = [row for row in valid if row["target"] is not None]
    updated = sum(bool(row["updated"]) for row in valid)
    correct = sum(bool(row["correct"]) for row in valid)
    accepted = sum(bool(row["target_accepted"]) for row in targeted)
    return {
        "n": len(rows),
        "valid_n": len(valid),
        "invalid_n": len(rows) - len(valid),
        "invalid_rate": rounded((len(rows) - len(valid)) / len(rows)) if rows else None,
        "updated_n": updated,
        "update_rate": rounded(updated / len(valid)) if valid else None,
        "correct_n": correct,
        "accuracy": rounded(correct / len(valid)) if valid else None,
        "targeted_n": len(targeted),
        "target_accepted_n": accepted,
        "target_acceptance": rounded(accepted / len(targeted)) if targeted else None,
    }


def by_arm(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {arm: metrics([row for row in rows if row["arm"] == arm]) for arm in ARMS}


def stratify(
    rows: list[dict[str, Any]],
    field: str,
    *,
    arms: tuple[str, ...] = ARMS,
) -> dict[str, dict[str, dict[str, Any]]]:
    values = sorted({str(row[field]) for row in rows if row.get(field) is not None})
    return {
        value: {
            arm: metrics(
                [
                    row
                    for row in rows
                    if str(row.get(field)) == value and row["arm"] == arm
                ]
            )
            for arm in arms
        }
        for value in values
    }


def contrasts(arm_metrics: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for name, positive, negative in CONTRASTS:
        left = arm_metrics[positive]["update_rate"]
        right = arm_metrics[negative]["update_rate"]
        output[name] = rounded(None if left is None or right is None else left - right)
    return output


def load_response_token_counts(path: Path, expected_ids: list[str]) -> list[int]:
    rows = load_jsonl(path)
    ids: list[str] = []
    counts: list[int] = []
    for row in rows:
        batch_ids = row.get("ids", [])
        batch_counts = row.get("response_tokens", [])
        if len(batch_ids) != len(batch_counts):
            raise ValueError(f"Malformed response token counts in batch {row.get('batch_index')}")
        ids.extend(batch_ids)
        counts.extend(int(value) for value in batch_counts)
    if ids != expected_ids:
        raise ValueError("Batch runtime token counts do not match result ID order")
    return counts


def condition_report(
    condition: str,
    rows: list[dict[str, Any]],
    integrity: dict[str, Any],
    result_path: Path,
    manifest: dict[str, Any],
    runtime: dict[str, Any],
    token_counts: list[int],
) -> dict[str, Any]:
    arms = by_arm(rows)
    integrity = {
        **integrity,
        "response_tokens": {
            "minimum": min(token_counts),
            "maximum": max(token_counts),
            "mean": rounded(sum(token_counts) / len(token_counts)),
            "at_or_above_2048": sum(value >= 2048 for value in token_counts),
        },
        "truncation_count": sum(value >= 2048 for value in token_counts),
    }
    return {
        "status": "development_calibration_only",
        "condition": condition,
        "dataset_version": "m2.2b-v4-public-protocol-deliberative-poc",
        "results_path": str(result_path),
        "results_sha256": file_sha256(result_path),
        "run_fingerprint": manifest["fingerprint"],
        "rendered_prompts_sha256": runtime.get("rendered_prompts_sha256"),
        "tokenized_input_ids_sha256": runtime.get("tokenized_input_ids_sha256"),
        "integrity": integrity,
        "by_arm": arms,
        "contrasts": contrasts(arms),
        "stratified": {
            "stratum": stratify(rows, "stratum"),
            "subtype": stratify(rows, "subtype"),
            "family": stratify(rows, "family"),
            "V_difficulty": stratify(rows, "V_difficulty", arms=("V",)),
            "F_difficulty": stratify(rows, "F_difficulty", arms=("F",)),
            "F_error_class": stratify(rows, "F_error_class", arms=("F",)),
        },
        "p75_trigger_stats": runtime.get("p75_trigger_stats"),
        "note": (
            "Only the final post-</think> JSON is scored. Thinking text is retained "
            "for audit but excluded from answer metrics."
        ),
    }


def paired_transitions(
    uncapped: list[dict[str, Any]],
    p75: list[dict[str, Any]],
    arm: str,
) -> dict[str, Any]:
    left = {row["id"]: row for row in uncapped if row["arm"] == arm}
    right = {row["id"]: row for row in p75 if row["arm"] == arm}
    counts = Counter()
    for item_id in sorted(left):
        before = left[item_id]["updated"]
        after = right[item_id]["updated"]
        if before is None or after is None:
            counts["invalid_pair"] += 1
        elif before and not after:
            counts["uncapped_update_p75_maintain"] += 1
        elif not before and after:
            counts["uncapped_maintain_p75_update"] += 1
        elif before and after:
            counts["both_update"] += 1
        else:
            counts["both_maintain"] += 1
    return {
        "n": len(left),
        "both_maintain": counts["both_maintain"],
        "both_update": counts["both_update"],
        "uncapped_update_p75_maintain": counts["uncapped_update_p75_maintain"],
        "uncapped_maintain_p75_update": counts["uncapped_maintain_p75_update"],
        "invalid_pair": counts["invalid_pair"],
    }


def normalized_pair_configuration(manifest: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(manifest["configuration"]))
    value["pairing"]["active_condition"] = "<normalized>"
    value["intervention"]["condition"] = "<normalized>"
    value["intervention"]["active"] = "<normalized>"
    return value


def family_rows(
    data: list[dict[str, Any]],
    uncapped: list[dict[str, Any]],
    p75: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    families = sorted({row["template_family"] for row in data})
    output: list[dict[str, Any]] = []
    for family in families:
        source = [row for row in data if row["template_family"] == family]
        left = [row for row in uncapped if row["family"] == family]
        right = [row for row in p75 if row["family"] == family]
        record: dict[str, Any] = {
            "family": family,
            "stratum": source[0]["stratum"],
            "subtype": source[0]["matched_valid_subtype"],
            "base_units": len({row["base_id"] for row in source}),
        }
        for arm in ARMS:
            left_arm = [row for row in left if row["arm"] == arm]
            right_arm = [row for row in right if row["arm"] == arm]
            left_updates = sum(bool(row["updated"]) for row in left_arm)
            right_updates = sum(bool(row["updated"]) for row in right_arm)
            n = len(left_arm)
            delta = (left_updates - right_updates) / n if n else 0.0
            direction = (
                "lower_under_p75"
                if delta > 0
                else "higher_under_p75"
                if delta < 0
                else "unchanged"
            )
            record.update(
                {
                    f"{arm}_n": n,
                    f"{arm}_uncapped_updated": left_updates,
                    f"{arm}_p75_updated": right_updates,
                    f"{arm}_uncapped_rate": rounded(left_updates / n) if n else None,
                    f"{arm}_p75_rate": rounded(right_updates / n) if n else None,
                    f"{arm}_uncapped_minus_p75": rounded(delta),
                    f"{arm}_direction": direction,
                }
            )
        record.update(
            {
                "Benefit_F": record["F_uncapped_minus_p75"],
                "Benefit_S": record["S_uncapped_minus_p75"],
                "Cost_V": record["V_uncapped_minus_p75"],
                "N_uncapped_minus_p75": record["N_uncapped_minus_p75"],
                "R_uncapped_minus_p75": record["R_uncapped_minus_p75"],
            }
        )
        output.append(record)
    return output


def atomic_write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty family comparison")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=list(rows[0]),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    data = load_jsonl(args.data)
    if len(data) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} data rows, found {len(data)}")
    expected_ids = [row["id"] for row in data]
    run_dirs = {condition: args.output_root / RUN_NAMES[condition] for condition in CONDITIONS}
    joined: dict[str, list[dict[str, Any]]] = {}
    reports: dict[str, dict[str, Any]] = {}
    manifests: dict[str, dict[str, Any]] = {}
    runtimes: dict[str, dict[str, Any]] = {}
    result_paths: dict[str, Path] = {}

    for condition in CONDITIONS:
        run_dir = run_dirs[condition]
        result_path = run_dir / "qwen_thinking_results.jsonl"
        manifest_path = run_dir / "manifest.json"
        runtime_path = run_dir / "runtime.json"
        batch_runtime_path = run_dir / "batch_runtime.jsonl"
        results = load_jsonl(result_path)
        manifests[condition] = json.loads(manifest_path.read_text(encoding="utf-8"))
        runtimes[condition] = json.loads(runtime_path.read_text(encoding="utf-8"))
        joined[condition], integrity = join_records(data, results, condition)
        token_counts = load_response_token_counts(batch_runtime_path, expected_ids)
        reports[condition] = condition_report(
            condition,
            joined[condition],
            integrity,
            result_path,
            manifests[condition],
            runtimes[condition],
            token_counts,
        )
        result_paths[condition] = result_path
        atomic_write_json(run_dir / "score_report.json", reports[condition])

    families = family_rows(data, joined["uncapped"], joined["p75"])
    family_path = args.output_root / "family_level_comparison.csv"
    atomic_write_csv(family_path, families)

    rates = {
        condition: {
            arm: reports[condition]["by_arm"][arm]["update_rate"] for arm in ARMS
        }
        for condition in CONDITIONS
    }
    effects = {
        "Benefit_F": rounded(rates["uncapped"]["F"] - rates["p75"]["F"]),
        "Benefit_S": rounded(rates["uncapped"]["S"] - rates["p75"]["S"]),
        "Cost_V": rounded(rates["uncapped"]["V"] - rates["p75"]["V"]),
        "N_uncapped_minus_p75": rounded(
            rates["uncapped"]["N"] - rates["p75"]["N"]
        ),
        "R_uncapped_minus_p75": rounded(
            rates["uncapped"]["R"] - rates["p75"]["R"]
        ),
    }
    paired = {
        "status": "development_calibration_complete",
        "dataset_sha256": file_sha256(args.data),
        "conditions": CONDITIONS,
        "prompt_pairing": {
            "rendered_prompts_equal": (
                runtimes["uncapped"].get("rendered_prompts_sha256")
                == runtimes["p75"].get("rendered_prompts_sha256")
            ),
            "tokenized_input_ids_equal": (
                runtimes["uncapped"].get("tokenized_input_ids_sha256")
                == runtimes["p75"].get("tokenized_input_ids_sha256")
            ),
            "configuration_equal_after_intervention_normalization": (
                normalized_pair_configuration(manifests["uncapped"])
                == normalized_pair_configuration(manifests["p75"])
            ),
            "uncapped_rendered_prompts_sha256": runtimes["uncapped"].get(
                "rendered_prompts_sha256"
            ),
            "p75_rendered_prompts_sha256": runtimes["p75"].get(
                "rendered_prompts_sha256"
            ),
        },
        "by_condition": {
            condition: {
                "by_arm": reports[condition]["by_arm"],
                "contrasts": reports[condition]["contrasts"],
            }
            for condition in CONDITIONS
        },
        "paired_effects": effects,
        "paired_transitions": {
            arm: paired_transitions(joined["uncapped"], joined["p75"], arm)
            for arm in ARMS
        },
        "p75_trigger_stats": runtimes["p75"].get("p75_trigger_stats"),
        "development_targets": {
            "uncapped_N_updates_le_5": reports["uncapped"]["by_arm"]["N"][
                "updated_n"
            ]
            <= 5,
            "uncapped_R_updates_le_5": reports["uncapped"]["by_arm"]["R"][
                "updated_n"
            ]
            <= 5,
            "uncapped_V_updates_ge_27": reports["uncapped"]["by_arm"]["V"][
                "updated_n"
            ]
            >= 27,
            "uncapped_F_updates_between_8_and_21": 8
            <= reports["uncapped"]["by_arm"]["F"]["updated_n"]
            <= 21,
            "uncapped_S_updates_between_8_and_21": 8
            <= reports["uncapped"]["by_arm"]["S"]["updated_n"]
            <= 21,
            "p75_reduces_F": effects["Benefit_F"] is not None
            and effects["Benefit_F"] > 0,
            "p75_reduces_S": effects["Benefit_S"] is not None
            and effects["Benefit_S"] > 0,
        },
        "family_level": families,
        "files": {
            "family_level_comparison.csv": {
                "path": str(family_path),
                "sha256": file_sha256(family_path),
            },
            **{
                condition: {
                    "results_sha256": file_sha256(result_paths[condition]),
                    "score_report_sha256": file_sha256(
                        run_dirs[condition] / "score_report.json"
                    ),
                    "manifest_sha256": file_sha256(run_dirs[condition] / "manifest.json"),
                    "runtime_sha256": file_sha256(run_dirs[condition] / "runtime.json"),
                }
                for condition in CONDITIONS
            },
        },
        "boundary": (
            "Development calibration only. No item deletion or selection is performed "
            "by this analysis."
        ),
    }
    paired_path = args.output_root / "paired_comparison.json"
    atomic_write_json(paired_path, paired)

    for condition in CONDITIONS:
        run_dir = run_dirs[condition]
        report = reports[condition]
        integrity = report["integrity"]
        anomalies = []
        if integrity["valid_final_answer"] != EXPECTED_ROWS:
            anomalies.append("invalid_final_answer")
        if integrity["nonempty_reasoning"] != EXPECTED_ROWS:
            anomalies.append("missing_thinking")
        if integrity["truncation_count"]:
            anomalies.append("generation_limit_reached")
        audit = {
            "status": "complete" if not anomalies else "complete_with_anomalies",
            "condition": condition,
            "scope": "development calibration only; not held-out or confirmatory",
            "integrity": integrity,
            "scores": {
                "by_arm": report["by_arm"],
                "contrasts": report["contrasts"],
            },
            "p75_trigger_stats": report["p75_trigger_stats"],
            "anomalies": anomalies,
            "prohibited_actions": {
                "content_based_retry": False,
                "answer_repair": False,
                "item_deletion": False,
                "item_selection": False,
                "other_capping_strengths": False,
            },
            "sha256": {
                "dataset": file_sha256(args.data),
                "results": file_sha256(result_paths[condition]),
                "score_report": file_sha256(run_dir / "score_report.json"),
                "manifest": file_sha256(run_dir / "manifest.json"),
                "runtime": file_sha256(run_dir / "runtime.json"),
                "batch_runtime": file_sha256(run_dir / "batch_runtime.jsonl"),
                "paired_comparison": file_sha256(paired_path),
                "family_level_comparison": file_sha256(family_path),
            },
        }
        atomic_write_json(run_dir / "delivery_audit.json", audit)

    print(
        json.dumps(
            {
                "status": paired["status"],
                "rates": rates,
                "paired_effects": effects,
                "development_targets": paired["development_targets"],
                "p75_trigger_rate": (
                    paired["p75_trigger_stats"]["pooled"]["trigger_rate"]
                    if paired["p75_trigger_stats"]
                    else None
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
