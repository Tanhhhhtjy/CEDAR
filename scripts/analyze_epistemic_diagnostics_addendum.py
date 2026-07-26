#!/usr/bin/env python3
"""Post-hoc zero-GPU epistemic diagnostics for E2A and E2C."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import tempfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as st
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.cov_struct import Exchangeable
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.families.links import Probit

ROOT = Path(__file__).resolve().parents[1]
E2A_PATH = (
    ROOT
    / "artifacts"
    / "m2"
    / "e2_phase_validity_choice"
    / "formal-all-seed0"
    / "results.jsonl"
)
E2C_PATH = (
    ROOT
    / "artifacts"
    / "m2"
    / "e2_new_salt_confirmation"
    / "formal-paired-uncapped-full-p25-seed0"
    / "results.jsonl"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "m2" / "epistemic_diagnostics_addendum_v1"

EXPECTED_HASHES = {
    "E2A": "5ba0116b51f677f72498fd5df4a46cae4e53084715cb9905150866ef6bb28358",
    "E2C": "e9fc1c502b45d76e55432e5491d3a285f58a36ff373a65bb44f2164890383c8f",
}
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260726
EPS = 1e-15
OPTION_LABELS = ("A", "B", "C", "D")
CAP_CONDITIONS = {
    "E2A": ("full_p25", "prefill_only", "decode_only"),
    "E2C": ("full_p25",),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_offset(*parts: str, modulo: int = 10_000) -> int:
    payload = "::".join(parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % modulo


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    return float(np.quantile(np.asarray(values, dtype=float), probability))


def descriptive(values: Sequence[float]) -> dict[str, Any]:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return {
            "n": 0,
            "mean": None,
            "sd": None,
            "median": None,
            "minimum": None,
            "maximum": None,
        }
    return {
        "n": len(clean),
        "mean": statistics.fmean(clean),
        "sd": statistics.stdev(clean) if len(clean) > 1 else None,
        "median": statistics.median(clean),
        "minimum": min(clean),
        "maximum": max(clean),
    }


def family_means(records: Sequence[tuple[str, float]]) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for family, value in records:
        grouped[str(family)].append(float(value))
    return {
        family: statistics.fmean(values)
        for family, values in sorted(grouped.items())
    }


def family_bootstrap(
    records: Sequence[tuple[str, float]],
    *,
    seed_offset: int = 0,
) -> dict[str, Any]:
    scores = family_means(records)
    families = sorted(scores)
    if not families:
        return {
            "estimate": None,
            "family_cluster_bootstrap_95ci": [None, None],
            "families": 0,
            "records": 0,
            "probability_positive": None,
            "probability_negative": None,
            "family_scores": {},
        }
    estimate = statistics.fmean(scores.values())
    rng = random.Random(BOOTSTRAP_SEED + seed_offset)
    draws = [
        statistics.fmean(scores[rng.choice(families)] for _ in families)
        for _ in range(BOOTSTRAP_DRAWS)
    ]
    return {
        "estimate": estimate,
        "family_cluster_bootstrap_95ci": [
            percentile(draws, 0.025),
            percentile(draws, 0.975),
        ],
        "families": len(families),
        "records": len(records),
        "probability_positive": sum(value > 0 for value in draws) / len(draws),
        "probability_negative": sum(value < 0 for value in draws) / len(draws),
        "family_scores": scores,
    }


def conditional_probabilities(row: Mapping[str, Any]) -> dict[str, float] | None:
    payload = row.get("answer_logits")
    if not isinstance(payload, Mapping):
        return None
    values = payload.get("conditional_probabilities")
    if not isinstance(values, Mapping):
        return None
    labels = set(values)
    if not labels.issubset(set(OPTION_LABELS)) or len(labels) not in {2, 4}:
        return None
    return {str(label): float(value) for label, value in values.items()}


def option_logits(row: Mapping[str, Any]) -> dict[str, float] | None:
    payload = row.get("answer_logits")
    if not isinstance(payload, Mapping):
        return None
    values = payload.get("logits")
    if not isinstance(values, Mapping):
        return None
    return {str(label): float(value) for label, value in values.items()}


def normalized_entropy(probabilities: Mapping[str, float]) -> float:
    values = np.asarray([max(float(value), EPS) for value in probabilities.values()])
    values = values / values.sum()
    return float(-(values * np.log(values)).sum() / math.log(len(values)))


def answer_geometry(row: Mapping[str, Any]) -> dict[str, float] | None:
    logits = option_logits(row)
    if logits is None or set(logits) != set(OPTION_LABELS):
        return None
    initial = str(row["initial_answer"])
    target = str(row["target_answer"])
    background = [label for label in OPTION_LABELS if label not in {initial, target}]
    m_target = logits[target] - logits[initial]
    m_background = statistics.fmean(logits[label] - logits[initial] for label in background)
    return {
        "m_target": m_target,
        "m_background": m_background,
        "m_general": statistics.fmean(
            logits[label] - logits[initial] for label in OPTION_LABELS if label != initial
        ),
        "m_specific": logits[target] - statistics.fmean(logits[label] for label in background),
        "top_gap": sorted(logits.values(), reverse=True)[0]
        - sorted(logits.values(), reverse=True)[1],
    }


def target_follow(row: Mapping[str, Any]) -> int:
    return int(row.get("outcome_class") == "induced_target")


def validate_inputs(
    e2a_rows: Sequence[dict[str, Any]], e2c_rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    failures: list[str] = []
    actual_hashes = {"E2A": sha256(E2A_PATH), "E2C": sha256(E2C_PATH)}
    for dataset, expected in EXPECTED_HASHES.items():
        if actual_hashes[dataset] != expected:
            failures.append(
                f"{dataset} SHA mismatch: {actual_hashes[dataset]} != {expected}"
            )
    if len(e2a_rows) != 576:
        failures.append(f"E2A expected 576 rows, found {len(e2a_rows)}")
    if len(e2c_rows) != 192:
        failures.append(f"E2C expected 192 rows, found {len(e2c_rows)}")

    e2a_counts = Counter(
        (row.get("assay"), row.get("condition"), row.get("arm")) for row in e2a_rows
    )
    for condition in ("uncapped", "full_p25", "prefill_only", "decode_only"):
        for arm in ("N", "R", "V", "F"):
            if e2a_counts[("answer", condition, arm)] != 24:
                failures.append(f"E2A answer {condition}/{arm} is not 24")
        if e2a_counts[("validity", condition, None)] != 48:
            failures.append(f"E2A validity {condition} is not 48")

    e2c_counts = Counter((row.get("condition"), row.get("arm")) for row in e2c_rows)
    for condition in ("uncapped", "full_p25"):
        for arm in ("N", "R", "V", "F"):
            if e2c_counts[(condition, arm)] != 24:
                failures.append(f"E2C {condition}/{arm} is not 24")

    e2a_families = sorted({str(row["family_id"]) for row in e2a_rows})
    e2c_families = sorted({str(row["family_id"]) for row in e2c_rows})
    if len(e2a_families) != 12:
        failures.append(f"E2A expected 12 families, found {len(e2a_families)}")
    if len(e2c_families) != 12:
        failures.append(f"E2C expected 12 families, found {len(e2c_families)}")

    e2a_missing = [
        row["id"]
        for row in e2a_rows
        if row["assay"] == "answer" and conditional_probabilities(row) is None
    ]
    e2c_missing = [
        row["id"] for row in e2c_rows if conditional_probabilities(row) is None
    ]
    if e2a_missing:
        failures.append(f"E2A missing answer logits: {e2a_missing}")
    if len(e2c_missing) != 1:
        failures.append(f"E2C expected one missing-logit row, found {e2c_missing}")

    answer_index = {
        (str(row["id"]), str(row["condition"])): row
        for row in e2a_rows
        if row["assay"] == "answer"
    }
    validity_pairs = 0
    for row in e2a_rows:
        if row["assay"] != "validity":
            continue
        key = (str(row["source_answer_id"]), str(row["condition"]))
        if key not in answer_index:
            failures.append(f"Missing answer pair for validity row {row['id']}")
        else:
            validity_pairs += 1
    if validity_pairs != 192:
        failures.append(f"Expected 192 validity-answer pairs, found {validity_pairs}")

    return {
        "status": "passed" if not failures else "failed",
        "failures": failures,
        "actual_hashes": actual_hashes,
        "expected_hashes": EXPECTED_HASHES,
        "row_counts": {"E2A": len(e2a_rows), "E2C": len(e2c_rows)},
        "family_counts": {"E2A": len(e2a_families), "E2C": len(e2c_families)},
        "validity_answer_pairs": validity_pairs,
        "missing_answer_logits": {"E2A": e2a_missing, "E2C": e2c_missing},
    }


def analyze_validity_choice(
    e2a_rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    answer_index = {
        (str(row["id"]), str(row["condition"])): row
        for row in e2a_rows
        if row["assay"] == "answer"
    }
    pairs: list[dict[str, Any]] = []
    for validity in e2a_rows:
        if validity["assay"] != "validity":
            continue
        answer = answer_index[(str(validity["source_answer_id"]), str(validity["condition"]))]
        arm = str(validity["source_arm"])
        validity_correct = int(bool(validity["gold_correct"]))
        followed = target_follow(answer)
        stayed_initial = int(answer["outcome_class"] == "initial")
        pair = {
            "base_id": answer["base_id"],
            "family_id": answer["family_id"],
            "condition": answer["condition"],
            "arm": arm,
            "validity_correct": validity_correct,
            "target_follow": followed,
            "stayed_initial": stayed_initial,
            "answer_outcome": answer["outcome_class"],
            "recognized_and_yielded": int(
                arm == "F" and validity_correct == 1 and followed == 1
            ),
            "misjudged_and_yielded": int(
                arm == "F" and validity_correct == 0 and followed == 1
            ),
            "recognized_and_updated": int(
                arm == "V" and validity_correct == 1 and followed == 1
            ),
            "recognized_but_refused": int(
                arm == "V" and validity_correct == 1 and stayed_initial == 1
            ),
        }
        pairs.append(pair)

    metrics_by_arm = {
        "V": (
            "validity_correct",
            "target_follow",
            "recognized_and_updated",
            "recognized_but_refused",
        ),
        "F": (
            "validity_correct",
            "target_follow",
            "recognized_and_yielded",
            "misjudged_and_yielded",
        ),
    }
    raw: dict[str, Any] = {}
    contrasts: dict[str, Any] = {}
    for condition in ("uncapped", "full_p25", "prefill_only", "decode_only"):
        raw[condition] = {}
        for arm, metrics in metrics_by_arm.items():
            selected = [
                row for row in pairs if row["condition"] == condition and row["arm"] == arm
            ]
            raw[condition][arm] = {
                metric: {
                    "count": sum(int(row[metric]) for row in selected),
                    "n": len(selected),
                    "rate": statistics.fmean(float(row[metric]) for row in selected),
                }
                for metric in metrics
            }
            correct = [row for row in selected if row["validity_correct"]]
            if arm == "F":
                numerator = sum(row["recognized_and_yielded"] for row in correct)
                label = "yield_rate_given_correct_invalidity_judgment"
            else:
                numerator = sum(row["recognized_and_updated"] for row in correct)
                label = "update_rate_given_correct_validity_judgment"
            raw[condition][arm][label] = {
                "count": numerator,
                "n": len(correct),
                "rate": numerator / len(correct) if correct else None,
            }

    by_key = {
        (row["base_id"], row["condition"], row["arm"]): row for row in pairs
    }
    for condition in ("full_p25", "prefill_only", "decode_only"):
        contrasts[condition] = {}
        for arm, metrics in metrics_by_arm.items():
            contrasts[condition][arm] = {}
            bases = sorted(
                {
                    row["base_id"]
                    for row in pairs
                    if row["condition"] == "uncapped" and row["arm"] == arm
                }
            )
            for metric in metrics:
                records = []
                for base in bases:
                    left = by_key[(base, "uncapped", arm)]
                    right = by_key[(base, condition, arm)]
                    records.append(
                        (str(left["family_id"]), float(right[metric]) - float(left[metric]))
                    )
                contrasts[condition][arm][metric] = family_bootstrap(
                    records, seed_offset=stable_offset(condition, arm, metric)
                )

    return {
        "scope": "E2A matched-item, separate-call judgment/action diagnostic",
        "pairs": len(pairs),
        "families": len({row["family_id"] for row in pairs}),
        "raw": raw,
        "paired_contrasts_vs_uncapped": contrasts,
    }, pairs


def hautus_rate(successes: int, total: int) -> float:
    return (successes + 0.5) / (total + 1.0)


def sdt_from_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    v = [row for row in rows if row["arm"] == "V"]
    f = [row for row in rows if row["arm"] == "F"]
    hits = sum(target_follow(row) for row in v)
    false_alarms = sum(target_follow(row) for row in f)
    hit_rate = hautus_rate(hits, len(v))
    false_alarm_rate = hautus_rate(false_alarms, len(f))
    z_hit = float(st.norm.ppf(hit_rate))
    z_fa = float(st.norm.ppf(false_alarm_rate))
    return {
        "v_target_count": hits,
        "v_n": len(v),
        "f_target_count": false_alarms,
        "f_n": len(f),
        "hit_rate_corrected": hit_rate,
        "false_alarm_rate_corrected": false_alarm_rate,
        "d_prime": z_hit - z_fa,
        "criterion": -0.5 * (z_hit + z_fa),
    }


def bootstrap_sdt_change(
    rows: Sequence[dict[str, Any]], condition: str, *, seed_offset: int
) -> dict[str, Any]:
    selected = [
        row
        for row in rows
        if row["arm"] in {"V", "F"} and row["condition"] in {"uncapped", condition}
    ]
    families = sorted({str(row["family_id"]) for row in selected})
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected:
        by_family[str(row["family_id"])].append(row)

    point_uncapped = sdt_from_rows(
        [row for row in selected if row["condition"] == "uncapped"]
    )
    point_cap = sdt_from_rows([row for row in selected if row["condition"] == condition])
    point_delta = {
        "d_prime": point_cap["d_prime"] - point_uncapped["d_prime"],
        "criterion": point_cap["criterion"] - point_uncapped["criterion"],
    }
    rng = random.Random(BOOTSTRAP_SEED + seed_offset)
    d_draws: list[float] = []
    c_draws: list[float] = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = [rng.choice(families) for _ in families]
        draw_rows: list[dict[str, Any]] = []
        for duplicate_index, family in enumerate(sampled):
            for row in by_family[family]:
                copied = dict(row)
                copied["family_id"] = f"{family}::boot{duplicate_index}"
                draw_rows.append(copied)
        uncapped = sdt_from_rows(
            [row for row in draw_rows if row["condition"] == "uncapped"]
        )
        cap = sdt_from_rows([row for row in draw_rows if row["condition"] == condition])
        d_draws.append(cap["d_prime"] - uncapped["d_prime"])
        c_draws.append(cap["criterion"] - uncapped["criterion"])
    return {
        "uncapped": point_uncapped,
        "cap": point_cap,
        "change_cap_minus_uncapped": {
            "d_prime": {
                "estimate": point_delta["d_prime"],
                "family_cluster_bootstrap_95ci": [
                    percentile(d_draws, 0.025),
                    percentile(d_draws, 0.975),
                ],
                "probability_positive": sum(value > 0 for value in d_draws) / len(d_draws),
                "probability_negative": sum(value < 0 for value in d_draws) / len(d_draws),
            },
            "criterion": {
                "estimate": point_delta["criterion"],
                "family_cluster_bootstrap_95ci": [
                    percentile(c_draws, 0.025),
                    percentile(c_draws, 0.975),
                ],
                "probability_positive": sum(value > 0 for value in c_draws) / len(c_draws),
                "probability_negative": sum(value < 0 for value in c_draws) / len(c_draws),
                "direction_note": "negative means a more liberal target-update criterion",
            },
        },
    }


def fit_sdt_gee(rows: Sequence[dict[str, Any]], condition: str) -> dict[str, Any]:
    records = []
    for row in rows:
        if row["arm"] not in {"V", "F"} or row["condition"] not in {
            "uncapped",
            condition,
        }:
            continue
        records.append(
            {
                "target_follow": target_follow(row),
                "signal": int(row["arm"] == "V"),
                "cap": int(row["condition"] == condition),
                "family_id": str(row["family_id"]),
            }
        )
    frame = pd.DataFrame(records)
    try:
        model = sm.GEE.from_formula(
            "target_follow ~ signal * cap",
            groups="family_id",
            data=frame,
            cov_struct=Exchangeable(),
            family=Binomial(link=Probit()),
        )
        fitted = model.fit()
        coefficients = {}
        for name in fitted.params.index:
            coefficients[str(name)] = {
                "estimate": float(fitted.params[name]),
                "robust_se": float(fitted.bse[name]),
                "z": float(fitted.tvalues[name]),
                "p_value": float(fitted.pvalues[name]),
                "95ci": [
                    float(fitted.conf_int().loc[name, 0]),
                    float(fitted.conf_int().loc[name, 1]),
                ],
            }
        return {
            "status": "complete",
            "model": "population-averaged probit GEE with exchangeable family clustering",
            "rows": len(frame),
            "clusters": int(frame["family_id"].nunique()),
            "coefficients": coefficients,
            "interpretation": {
                "signal": "uncapped V/F sensitivity contrast on probit scale",
                "cap": "cap shift in false-alarm propensity",
                "signal:cap": "change in V/F sensitivity under cap",
            },
            "caveat": "Only 12 structural-family clusters; cluster bootstrap is primary.",
        }
    except Exception as exc:  # pragma: no cover - defensive diagnostic
        return {"status": "failed", "error": repr(exc), "rows": len(frame)}


def analyze_sdt(
    datasets: Mapping[str, Sequence[dict[str, Any]]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output: dict[str, Any] = {}
    table: list[dict[str, Any]] = []
    for dataset, rows in datasets.items():
        output[dataset] = {}
        for index, condition in enumerate(CAP_CONDITIONS[dataset]):
            summary = bootstrap_sdt_change(
                rows, condition, seed_offset=1_000 * (index + 1) + len(dataset)
            )
            summary["gee"] = fit_sdt_gee(rows, condition)
            output[dataset][condition] = summary
            for state, values in (
                ("uncapped", summary["uncapped"]),
                (condition, summary["cap"]),
            ):
                table.append(
                    {
                        "dataset": dataset,
                        "comparison": condition,
                        "condition": state,
                        **values,
                    }
                )
            table.append(
                {
                    "dataset": dataset,
                    "comparison": condition,
                    "condition": "change_cap_minus_uncapped",
                    "d_prime": summary["change_cap_minus_uncapped"]["d_prime"]["estimate"],
                    "criterion": summary["change_cap_minus_uncapped"]["criterion"]["estimate"],
                    "d_prime_ci_low": summary["change_cap_minus_uncapped"]["d_prime"][
                        "family_cluster_bootstrap_95ci"
                    ][0],
                    "d_prime_ci_high": summary["change_cap_minus_uncapped"]["d_prime"][
                        "family_cluster_bootstrap_95ci"
                    ][1],
                    "criterion_ci_low": summary["change_cap_minus_uncapped"]["criterion"][
                        "family_cluster_bootstrap_95ci"
                    ][0],
                    "criterion_ci_high": summary["change_cap_minus_uncapped"]["criterion"][
                        "family_cluster_bootstrap_95ci"
                    ][1],
                }
            )
    return {
        "definition": {
            "signal": "V",
            "noise": "F",
            "yes_response": "induced target",
            "correction": "Hautus log-linear",
            "primary_uncertainty": "paired family-cluster bootstrap",
        },
        "datasets": output,
    }, table


def role_values(row: Mapping[str, Any]) -> dict[str, float] | None:
    probabilities = conditional_probabilities(row)
    logits = option_logits(row)
    if (
        probabilities is None
        or logits is None
        or set(probabilities) != set(OPTION_LABELS)
        or set(logits) != set(OPTION_LABELS)
    ):
        return None
    initial = str(row["initial_answer"])
    target = str(row["target_answer"])
    background = [label for label in OPTION_LABELS if label not in {initial, target}]
    return {
        "p_initial": probabilities[initial],
        "p_target": probabilities[target],
        "p_background": sum(probabilities[label] for label in background),
        "logit_target_initial": logits[target] - logits[initial],
        "logit_background_initial": statistics.fmean(
            logits[label] - logits[initial] for label in background
        ),
    }


def analyze_probability_flow(
    datasets: Mapping[str, Sequence[dict[str, Any]]]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    base_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    output: dict[str, Any] = {}
    for dataset, rows in datasets.items():
        index = {
            (str(row["base_id"]), str(row["condition"]), str(row["arm"])): row
            for row in rows
        }
        output[dataset] = {"comparisons": {}, "missing_pairs": []}
        bases = sorted({str(row["base_id"]) for row in rows})
        for condition in ("uncapped", *CAP_CONDITIONS[dataset]):
            for base in bases:
                r_row = index[(base, condition, "R")]
                r_values = role_values(r_row)
                if r_values is None:
                    output[dataset]["missing_pairs"].append(
                        {"base_id": base, "condition": condition, "arm": "R"}
                    )
                    continue
                for arm in ("V", "F"):
                    arm_row = index[(base, condition, arm)]
                    arm_values = role_values(arm_row)
                    if arm_values is None:
                        output[dataset]["missing_pairs"].append(
                            {"base_id": base, "condition": condition, "arm": arm}
                        )
                        continue
                    base_rows.append(
                        {
                            "dataset": dataset,
                            "base_id": base,
                            "family_id": arm_row["family_id"],
                            "condition": condition,
                            "arm": arm,
                            **{f"arm_{key}": value for key, value in arm_values.items()},
                            **{f"r_{key}": value for key, value in r_values.items()},
                            **{
                                f"flow_{key}": arm_values[key] - r_values[key]
                                for key in arm_values
                            },
                        }
                    )

        by_key = {
            (row["base_id"], row["condition"], row["arm"]): row
            for row in base_rows
            if row["dataset"] == dataset
        }
        for cap_index, condition in enumerate(CAP_CONDITIONS[dataset]):
            output[dataset]["comparisons"][condition] = {}
            for arm in ("V", "F"):
                output[dataset]["comparisons"][condition][arm] = {}
                metrics = (
                    "p_initial",
                    "p_target",
                    "p_background",
                    "logit_target_initial",
                    "logit_background_initial",
                )
                for metric_index, metric in enumerate(metrics):
                    records: list[tuple[str, float]] = []
                    missing: list[str] = []
                    for base in bases:
                        left = by_key.get((base, "uncapped", arm))
                        right = by_key.get((base, condition, arm))
                        if left is None or right is None:
                            missing.append(base)
                            continue
                        records.append(
                            (
                                str(left["family_id"]),
                                float(right[f"flow_{metric}"])
                                - float(left[f"flow_{metric}"]),
                            )
                        )
                    summary = family_bootstrap(
                        records,
                        seed_offset=20_000
                        + cap_index * 100
                        + (0 if arm == "V" else 50)
                        + metric_index,
                    )
                    summary["missing_base_ids"] = missing
                    summary["definition"] = (
                        "(arm minus matched R) under cap minus "
                        "(arm minus matched R) under uncapped"
                    )
                    output[dataset]["comparisons"][condition][arm][metric] = summary
                    summary_rows.append(
                        {
                            "dataset": dataset,
                            "condition": condition,
                            "arm": arm,
                            "metric": metric,
                            "estimate": summary["estimate"],
                            "ci_low": summary["family_cluster_bootstrap_95ci"][0],
                            "ci_high": summary["family_cluster_bootstrap_95ci"][1],
                            "families": summary["families"],
                            "records": summary["records"],
                            "missing_bases": ";".join(missing),
                        }
                    )
    return {
        "definition": (
            "R-adjusted evidence assimilation difference-in-differences; "
            "conditional option probabilities are separate from full-vocabulary option mass."
        ),
        "datasets": output,
    }, base_rows, summary_rows


CALIBRATION_METRICS = (
    "brier",
    "nll",
    "conditional_entropy",
    "confidence",
    "signed_confidence_gap",
    "option_mass",
    "full_vocab_entropy",
    "correct",
)


def row_calibration(row: Mapping[str, Any]) -> dict[str, float] | None:
    probabilities = conditional_probabilities(row)
    if probabilities is None or set(probabilities) != set(OPTION_LABELS):
        return None
    gold = str(row["gold_answer"])
    normalized = np.asarray([probabilities[label] for label in OPTION_LABELS], dtype=float)
    normalized = normalized / normalized.sum()
    gold_index = OPTION_LABELS.index(gold)
    truth = np.zeros(4, dtype=float)
    truth[gold_index] = 1.0
    confidence = float(normalized.max())
    predicted = OPTION_LABELS[int(normalized.argmax())]
    correct = float(predicted == gold)
    payload = row["answer_logits"]
    return {
        "brier": float(np.square(normalized - truth).sum()),
        "nll": float(-math.log(max(float(normalized[gold_index]), EPS))),
        "conditional_entropy": normalized_entropy(
            {label: normalized[index] for index, label in enumerate(OPTION_LABELS)}
        ),
        "confidence": confidence,
        "signed_confidence_gap": confidence - correct,
        "option_mass": float(payload["option_mass"]),
        "full_vocab_entropy": float(payload["full_vocab_entropy"]),
        "correct": correct,
    }


def aurc(rows: Sequence[Mapping[str, Any]]) -> float | None:
    scored = []
    for row in rows:
        metrics = row_calibration(row)
        if metrics is None:
            continue
        scored.append((metrics["confidence"], 1.0 - metrics["correct"]))
    if not scored:
        return None
    scored.sort(reverse=True)
    cumulative_errors = 0.0
    risks = []
    for index, (_, error) in enumerate(scored, start=1):
        cumulative_errors += error
        risks.append(cumulative_errors / index)
    return statistics.fmean(risks)


def analyze_calibration(
    datasets: Mapping[str, Sequence[dict[str, Any]]]
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    per_row: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    output: dict[str, Any] = {}
    for dataset, rows in datasets.items():
        output[dataset] = {"raw": {}, "contrasts": {}, "missing_pairs": []}
        for row in rows:
            metrics = row_calibration(row)
            if metrics is None:
                continue
            per_row.append(
                {
                    "dataset": dataset,
                    "id": row["id"],
                    "base_id": row["base_id"],
                    "family_id": row["family_id"],
                    "condition": row["condition"],
                    "arm": row["arm"],
                    **metrics,
                }
            )
        dataset_rows = [row for row in per_row if row["dataset"] == dataset]
        for condition in ("uncapped", *CAP_CONDITIONS[dataset]):
            output[dataset]["raw"][condition] = {}
            for arm in ("N", "R", "V", "F"):
                original = [
                    row
                    for row in rows
                    if row["condition"] == condition and row["arm"] == arm
                ]
                selected = [
                    row
                    for row in dataset_rows
                    if row["condition"] == condition and row["arm"] == arm
                ]
                output[dataset]["raw"][condition][arm] = {
                    metric: descriptive([float(row[metric]) for row in selected])
                    for metric in CALIBRATION_METRICS
                }
                output[dataset]["raw"][condition][arm]["aurc"] = aurc(original)
                output[dataset]["raw"][condition][arm]["complete_rows"] = len(selected)
                output[dataset]["raw"][condition][arm]["all_rows"] = len(original)

        index = {
            (row["id"], row["condition"]): row
            for row in dataset_rows
        }
        original_uncapped = [
            row for row in rows if row["condition"] == "uncapped"
        ]
        for cap_index, condition in enumerate(CAP_CONDITIONS[dataset]):
            output[dataset]["contrasts"][condition] = {}
            for arm in ("N", "R", "V", "F"):
                output[dataset]["contrasts"][condition][arm] = {}
                uncapped_ids = [
                    str(row["id"]) for row in original_uncapped if row["arm"] == arm
                ]
                for metric_index, metric in enumerate(CALIBRATION_METRICS):
                    records: list[tuple[str, float]] = []
                    missing: list[str] = []
                    for item_id in uncapped_ids:
                        left = index.get((item_id, "uncapped"))
                        right = index.get((item_id, condition))
                        if left is None or right is None:
                            missing.append(item_id)
                            continue
                        records.append(
                            (
                                str(left["family_id"]),
                                float(right[metric]) - float(left[metric]),
                            )
                        )
                    summary = family_bootstrap(
                        records,
                        seed_offset=30_000
                        + cap_index * 500
                        + ("NRVF".index(arm) * 50)
                        + metric_index,
                    )
                    summary["missing_pair_ids"] = missing
                    output[dataset]["contrasts"][condition][arm][metric] = summary
                    summary_rows.append(
                        {
                            "dataset": dataset,
                            "condition": condition,
                            "arm": arm,
                            "metric": metric,
                            "estimate": summary["estimate"],
                            "ci_low": summary["family_cluster_bootstrap_95ci"][0],
                            "ci_high": summary["family_cluster_bootstrap_95ci"][1],
                            "families": summary["families"],
                            "records": summary["records"],
                            "missing_pairs": ";".join(missing),
                        }
                    )
    return {
        "definitions": {
            "brier": "sum over A/B/C/D of squared probability error",
            "nll": "negative log option-conditional probability of gold",
            "conditional_entropy": "A/B/C/D entropy divided by log(4)",
            "signed_confidence_gap": "max option probability minus correctness",
            "option_mass": "full-vocabulary mass on A/B/C/D",
            "aurc": "mean selective risk over descending confidence coverages; descriptive",
            "ece": "not reported because 24 rows per arm do not support stable binning",
        },
        "datasets": output,
    }, per_row, summary_rows


def bootstrap_spearman(
    records: Sequence[dict[str, Any]],
    x_key: str,
    y_key: str,
    *,
    seed_offset: int,
) -> dict[str, Any]:
    clean = [
        row
        for row in records
        if math.isfinite(float(row[x_key])) and math.isfinite(float(row[y_key]))
    ]
    if len(clean) < 4 or len({float(row[x_key]) for row in clean}) < 3:
        return {
            "status": "not_estimable",
            "reason": "insufficient non-degenerate predictor variation",
            "records": len(clean),
        }
    point = st.spearmanr(
        [float(row[x_key]) for row in clean],
        [float(row[y_key]) for row in clean],
    )
    families = sorted({str(row["family_id"]) for row in clean})
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clean:
        by_family[str(row["family_id"])].append(row)
    rng = random.Random(BOOTSTRAP_SEED + seed_offset)
    draws: list[float] = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = [rng.choice(families) for _ in families]
        draw = [row for family in sampled for row in by_family[family]]
        result = st.spearmanr(
            [float(row[x_key]) for row in draw],
            [float(row[y_key]) for row in draw],
        )
        if math.isfinite(float(result.statistic)):
            draws.append(float(result.statistic))
    return {
        "status": "complete",
        "spearman_rho": float(point.statistic),
        "nominal_p_value": float(point.pvalue),
        "family_cluster_bootstrap_95ci": [
            percentile(draws, 0.025),
            percentile(draws, 0.975),
        ],
        "records": len(clean),
        "families": len(families),
        "bootstrap_valid_draws": len(draws),
    }


def model_result(fitted: Any, *, clusters: int) -> dict[str, Any]:
    coefficients = {}
    conf = fitted.conf_int()
    for name in fitted.params.index:
        coefficients[str(name)] = {
            "estimate": float(fitted.params[name]),
            "robust_se": float(fitted.bse[name]),
            "statistic": float(fitted.tvalues[name]),
            "p_value": float(fitted.pvalues[name]),
            "95ci": [float(conf.loc[name, 0]), float(conf.loc[name, 1])],
        }
    return {
        "status": "complete",
        "rows": int(fitted.nobs),
        "clusters": clusters,
        "coefficients": coefficients,
    }


def analyze_muse_proxy(
    datasets: Mapping[str, Sequence[dict[str, Any]]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output: dict[str, Any] = {
        "scope": (
            "MUSE-style moderator using exact forced-choice R-baseline uncertainty; "
            "not a sample-entropy reproduction."
        ),
        "datasets": {},
    }
    table: list[dict[str, Any]] = []
    for dataset, rows in datasets.items():
        index = {
            (str(row["base_id"]), str(row["condition"]), str(row["arm"])): row
            for row in rows
        }
        bases = sorted({str(row["base_id"]) for row in rows})
        output["datasets"][dataset] = {}
        for cap_index, condition in enumerate(CAP_CONDITIONS[dataset]):
            records: list[dict[str, Any]] = []
            long_records: list[dict[str, Any]] = []
            missing: list[str] = []
            for base in bases:
                r_uncapped = index[(base, "uncapped", "R")]
                r_probabilities = conditional_probabilities(r_uncapped)
                r_geometry = answer_geometry(r_uncapped)
                if r_probabilities is None or r_geometry is None:
                    missing.append(f"{base}:R")
                    continue
                h0 = normalized_entropy(r_probabilities)
                gap0 = r_geometry["top_gap"]
                for arm in ("V", "F"):
                    left = index[(base, "uncapped", arm)]
                    right = index[(base, condition, arm)]
                    left_geometry = answer_geometry(left)
                    right_geometry = answer_geometry(right)
                    if left_geometry is None or right_geometry is None:
                        missing.append(f"{base}:{arm}")
                        continue
                    record = {
                        "dataset": dataset,
                        "condition": condition,
                        "base_id": base,
                        "family_id": left["family_id"],
                        "arm": arm,
                        "signal": int(arm == "V"),
                        "h0_entropy": h0,
                        "h0_top_gap": gap0,
                        "delta_target_margin": right_geometry["m_target"]
                        - left_geometry["m_target"],
                        "delta_target_follow": target_follow(right) - target_follow(left),
                    }
                    records.append(record)
                    for cap_value, row in ((0, left), (1, right)):
                        long_records.append(
                            {
                                "target_follow": target_follow(row),
                                "signal": int(arm == "V"),
                                "cap": cap_value,
                                "h0_entropy": h0,
                                "h0_top_gap": gap0,
                                "family_id": str(left["family_id"]),
                            }
                        )
            frame = pd.DataFrame(records)
            long_frame = pd.DataFrame(long_records)
            h0_summary = descriptive(frame["h0_entropy"].tolist())
            gap_summary = descriptive(frame["h0_top_gap"].tolist())
            entropy_degenerate = (
                h0_summary["sd"] is None or float(h0_summary["sd"]) < 1e-4
            )
            result: dict[str, Any] = {
                "records": len(records),
                "families": int(frame["family_id"].nunique()) if not frame.empty else 0,
                "missing": missing,
                "baseline_entropy": h0_summary,
                "baseline_top_gap": gap_summary,
                "entropy_degenerate": entropy_degenerate,
                "spearman": {
                    "entropy_vs_delta_margin": bootstrap_spearman(
                        records,
                        "h0_entropy",
                        "delta_target_margin",
                        seed_offset=40_000 + cap_index,
                    ),
                    "top_gap_vs_delta_margin": bootstrap_spearman(
                        records,
                        "h0_top_gap",
                        "delta_target_margin",
                        seed_offset=41_000 + cap_index,
                    ),
                },
            }
            if not frame.empty:
                frame["h0_entropy_z"] = (
                    frame["h0_entropy"] - frame["h0_entropy"].mean()
                ) / max(float(frame["h0_entropy"].std(ddof=0)), EPS)
                frame["h0_top_gap_z"] = (
                    frame["h0_top_gap"] - frame["h0_top_gap"].mean()
                ) / max(float(frame["h0_top_gap"].std(ddof=0)), EPS)
                try:
                    fitted_margin = smf.ols(
                        "delta_target_margin ~ signal + h0_top_gap_z + "
                        "signal:h0_top_gap_z",
                        data=frame,
                    ).fit(cov_type="cluster", cov_kwds={"groups": frame["family_id"]})
                    result["margin_moderator_cluster_ols"] = model_result(
                        fitted_margin, clusters=int(frame["family_id"].nunique())
                    )
                except Exception as exc:  # pragma: no cover
                    result["margin_moderator_cluster_ols"] = {
                        "status": "failed",
                        "error": repr(exc),
                    }
            if not long_frame.empty:
                long_frame["h0_top_gap_z"] = (
                    long_frame["h0_top_gap"] - long_frame["h0_top_gap"].mean()
                ) / max(float(long_frame["h0_top_gap"].std(ddof=0)), EPS)
                try:
                    fitted_choice = sm.GEE.from_formula(
                        "target_follow ~ signal * cap + h0_top_gap_z * cap",
                        groups="family_id",
                        data=long_frame,
                        cov_struct=Exchangeable(),
                        family=Binomial(link=Probit()),
                    ).fit()
                    result["choice_moderator_gee"] = model_result(
                        fitted_choice,
                        clusters=int(long_frame["family_id"].nunique()),
                    )
                except Exception as exc:  # pragma: no cover
                    result["choice_moderator_gee"] = {
                        "status": "failed",
                        "error": repr(exc),
                    }
            output["datasets"][dataset][condition] = result
            for record in records:
                table.append(record)
    return output, table


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def apply_holm_corrections(sdt: dict[str, Any], muse: dict[str, Any]) -> None:
    sdt_targets: list[dict[str, Any]] = []
    for dataset in sdt["datasets"].values():
        for comparison in dataset.values():
            coefficients = comparison.get("gee", {}).get("coefficients", {})
            for name in ("cap", "signal:cap"):
                if name in coefficients:
                    sdt_targets.append(coefficients[name])
    if sdt_targets:
        adjusted = holm_adjust([entry["p_value"] for entry in sdt_targets])
        for entry, value in zip(sdt_targets, adjusted, strict=True):
            entry["holm_p_value_sdt_family"] = value

    muse_targets: list[dict[str, Any]] = []
    for dataset in muse["datasets"].values():
        for comparison in dataset.values():
            choice = comparison.get("choice_moderator_gee", {}).get("coefficients", {})
            if "h0_top_gap_z:cap" in choice:
                muse_targets.append(choice["h0_top_gap_z:cap"])
            margin = comparison.get("margin_moderator_cluster_ols", {}).get(
                "coefficients", {}
            )
            for name in ("h0_top_gap_z", "signal:h0_top_gap_z"):
                if name in margin:
                    muse_targets.append(margin[name])
    if muse_targets:
        adjusted = holm_adjust([entry["p_value"] for entry in muse_targets])
        for entry, value in zip(muse_targets, adjusted, strict=True):
            entry["holm_p_value_moderator_family"] = value


def export_figure(fig: Any, base_path: Path) -> None:
    os.environ.setdefault("SOURCE_DATE_EPOCH", "0")
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        base_path.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"Creator": "CEDAR epistemic diagnostics", "CreationDate": None},
    )
    fig.savefig(
        base_path.with_suffix(".png"),
        bbox_inches="tight",
        dpi=600,
        metadata={"Software": "CEDAR epistemic diagnostics"},
    )


def make_figures(
    output: Path,
    validity_choice: Mapping[str, Any],
    validity_pairs: Sequence[Mapping[str, Any]],
    sdt: Mapping[str, Any],
    probability_flow_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    muse_rows: Sequence[Mapping[str, Any]],
) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cedar-mpl-cache")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure_dir = output / "figures"
    colors = {
        "uncapped": "#777777",
        "full_p25": "#D55E00",
        "prefill_only": "#0072B2",
        "decode_only": "#009E73",
    }

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.15), sharey=True)
    condition_order = ("uncapped", "prefill_only", "decode_only", "full_p25")
    panels = (
        ("F", "recognized_and_yielded", "Recognized F invalid,\nbut followed target"),
        ("V", "recognized_but_refused", "Recognized V valid,\nbut retained initial"),
    )
    for axis, (arm, metric, title) in zip(axes, panels, strict=True):
        values = []
        lower = []
        upper = []
        for condition in condition_order:
            records = [
                (str(row["family_id"]), float(row[metric]))
                for row in validity_pairs
                if row["condition"] == condition and row["arm"] == arm
            ]
            summary = family_bootstrap(
                records, seed_offset=50_000 + condition_order.index(condition)
            )
            values.append(summary["estimate"])
            lower.append(
                summary["estimate"] - summary["family_cluster_bootstrap_95ci"][0]
            )
            upper.append(
                summary["family_cluster_bootstrap_95ci"][1] - summary["estimate"]
            )
        x = np.arange(len(condition_order))
        axis.bar(
            x,
            values,
            color=[colors[condition] for condition in condition_order],
            width=0.7,
        )
        axis.errorbar(
            x,
            values,
            yerr=np.asarray([lower, upper]),
            fmt="none",
            color="black",
            capsize=3,
            linewidth=1,
        )
        axis.set_xticks(x, ("Uncap", "Prefill", "Decode", "Full"))
        axis.set_title(title, fontweight="bold")
        axis.set_ylim(0, 1.0)
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Joint rate")
    fig.tight_layout()
    export_figure(fig, figure_dir / "figure-01-validity-choice-gap")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2))
    entries = []
    for dataset_name, dataset in sdt["datasets"].items():
        for condition, summary in dataset.items():
            entries.append((dataset_name, condition, summary))
    y = np.arange(len(entries))[::-1]
    labels = [
        f"{dataset} / {condition.replace('_', ' ')}"
        for dataset, condition, _ in entries
    ]
    for axis, metric, label, neutral in (
        (axes[0], "d_prime", "Change in sensitivity (Δd′)", 0.0),
        (axes[1], "criterion", "Change in update criterion (Δc)", 0.0),
    ):
        for y_value, (_dataset, condition, summary) in zip(y, entries, strict=True):
            value = summary["change_cap_minus_uncapped"][metric]
            estimate = value["estimate"]
            low, high = value["family_cluster_bootstrap_95ci"]
            color = "#D55E00" if condition == "full_p25" else "#0072B2"
            axis.errorbar(
                estimate,
                y_value,
                xerr=[[estimate - low], [high - estimate]],
                fmt="o",
                color=color,
                capsize=3,
            )
        axis.axvline(neutral, color="#777777", linestyle="--", linewidth=1)
        axis.set_xlabel(label)
        axis.grid(axis="x", alpha=0.2)
    axes[0].set_yticks(y, labels)
    axes[1].set_yticks(y, [])
    fig.tight_layout()
    export_figure(fig, figure_dir / "figure-02-sdt-sensitivity-criterion")
    plt.close(fig)

    selected_flow = [
        row
        for row in probability_flow_rows
        if row["condition"] == "full_p25"
        and row["metric"] in {"p_initial", "p_target", "p_background"}
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.2), sharey=True)
    role_order = ("p_initial", "p_target", "p_background")
    role_labels = ("Initial", "Target", "Background")
    for axis, dataset in zip(axes, ("E2A", "E2C"), strict=True):
        subset = [row for row in selected_flow if row["dataset"] == dataset]
        x = np.arange(len(role_order))
        width = 0.34
        for arm_index, (arm, color) in enumerate(
            (("V", "#0072B2"), ("F", "#D55E00"))
        ):
            arm_rows = {row["metric"]: row for row in subset if row["arm"] == arm}
            values = [arm_rows[role]["estimate"] for role in role_order]
            lower = [
                arm_rows[role]["estimate"] - arm_rows[role]["ci_low"]
                for role in role_order
            ]
            upper = [
                arm_rows[role]["ci_high"] - arm_rows[role]["estimate"]
                for role in role_order
            ]
            positions = x + (arm_index - 0.5) * width
            axis.bar(positions, values, width=width, color=color, label=arm)
            axis.errorbar(
                positions,
                values,
                yerr=np.asarray([lower, upper]),
                fmt="none",
                color="black",
                capsize=2,
                linewidth=0.8,
            )
        axis.axhline(0, color="#777777", linestyle="--", linewidth=1)
        axis.set_xticks(x, role_labels)
        axis.set_title(dataset, fontweight="bold")
        axis.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("p25 change in R-adjusted probability")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    export_figure(fig, figure_dir / "figure-03-r-adjusted-probability-flow")
    plt.close(fig)

    selected_calibration = [
        row
        for row in calibration_rows
        if row["condition"] == "full_p25"
        and row["metric"] in {"brier", "conditional_entropy", "option_mass"}
    ]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.9))
    metric_titles = {
        "brier": "Δ Brier\n(lower is better)",
        "conditional_entropy": "Δ option entropy",
        "option_mass": "Δ option mass",
    }
    for axis, metric in zip(
        axes, ("brier", "conditional_entropy", "option_mass"), strict=True
    ):
        subset = [row for row in selected_calibration if row["metric"] == metric]
        lookup = {(row["dataset"], row["arm"]): row for row in subset}
        x = np.arange(4)
        width = 0.34
        for dataset_index, (dataset, color) in enumerate(
            (("E2A", "#0072B2"), ("E2C", "#E69F00"))
        ):
            rows_by_arm = [lookup[(dataset, arm)] for arm in ("N", "R", "V", "F")]
            values = [row["estimate"] for row in rows_by_arm]
            lower = [row["estimate"] - row["ci_low"] for row in rows_by_arm]
            upper = [row["ci_high"] - row["estimate"] for row in rows_by_arm]
            positions = x + (dataset_index - 0.5) * width
            axis.bar(
                positions,
                values,
                color=color,
                width=width,
                label=dataset,
            )
            axis.errorbar(
                positions,
                values,
                yerr=np.asarray([lower, upper]),
                fmt="none",
                color="black",
                capsize=2,
                linewidth=0.7,
            )
        axis.axhline(0, color="#777777", linestyle="--", linewidth=1)
        axis.set_xticks(x, ("N", "R", "V", "F"))
        axis.set_title(metric_titles[metric], fontweight="bold")
        axis.grid(axis="y", alpha=0.2)
    axes[2].legend(frameon=False, loc="best")
    fig.tight_layout()
    export_figure(fig, figure_dir / "figure-04-calibration-uncertainty")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), sharey=True)
    for axis, dataset in zip(axes, ("E2A", "E2C"), strict=True):
        selected = [
            row
            for row in muse_rows
            if row["dataset"] == dataset and row["condition"] == "full_p25"
        ]
        for arm, color, marker in (
            ("V", "#0072B2", "o"),
            ("F", "#D55E00", "s"),
        ):
            arm_rows = [row for row in selected if row["arm"] == arm]
            axis.scatter(
                [row["h0_top_gap"] for row in arm_rows],
                [row["delta_target_margin"] for row in arm_rows],
                color=color,
                marker=marker,
                alpha=0.65,
                s=22,
                label=arm,
            )
        axis.axhline(0, color="#777777", linestyle="--", linewidth=1)
        axis.set_title(dataset, fontweight="bold")
        axis.set_xlabel("Uncapped R top-1/top-2 logit gap")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("p25 − uncapped target margin")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    export_figure(fig, figure_dir / "figure-05-uncertainty-moderator")
    plt.close(fig)


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    numeric = float(value)
    if numeric != 0 and abs(numeric) < 10 ** (-digits):
        return f"{numeric:.2e}"
    return f"{numeric:.{digits}f}"


def fmt_ci(summary: Mapping[str, Any], digits: int = 3) -> str:
    estimate = summary.get("estimate")
    low, high = summary.get("family_cluster_bootstrap_95ci", [None, None])
    return f"{fmt(estimate, digits)} [{fmt(low, digits)}, {fmt(high, digits)}]"


def write_reports(output: Path, analysis: Mapping[str, Any]) -> None:
    validity = analysis["validity_choice_gap"]
    sdt = analysis["sdt"]
    flow = analysis["probability_flow"]
    calibration = analysis["calibration"]
    muse = analysis["muse_style_moderator"]

    e2a_full_f = validity["paired_contrasts_vs_uncapped"]["full_p25"]["F"]
    e2a_full_v = validity["paired_contrasts_vs_uncapped"]["full_p25"]["V"]
    e2a_sdt = sdt["datasets"]["E2A"]["full_p25"][
        "change_cap_minus_uncapped"
    ]
    e2c_sdt = sdt["datasets"]["E2C"]["full_p25"][
        "change_cap_minus_uncapped"
    ]
    e2a_flow = flow["datasets"]["E2A"]["comparisons"]["full_p25"]
    e2c_flow = flow["datasets"]["E2C"]["comparisons"]["full_p25"]

    report = [
        "# E2 Epistemic Diagnostics Addendum v1",
        "",
        "> Post-hoc zero-GPU analysis of frozen E2A and E2C outputs. It does not identify",
        "> a neural circuit and is not a preregistered confirmation.",
        "",
        "## Analysis questions",
        "",
        "1. Does explicit validity judgment dissociate from final answer policy?",
        "2. Does p25 change V/F sensitivity, the update criterion, or both?",
        "3. Does lost initial-answer probability move to the induced target or background options?",
        "4. Does capping change calibration/uncertainty, and does baseline uncertainty",
        "   moderate it?",
        "",
        "## Executive summary",
        "",
        "- In E2A, p25 primarily moved the **update criterion** toward more target",
        "  acceptance (Δcriterion below zero), while the V/F sensitivity interval included",
        "  zero.",
        "- In the new-salt E2C assay, the pattern changed: V/F sensitivity decreased while",
        "  the criterion interval included zero. The two datasets therefore should not be",
        "  averaged into a single 'universal de-anchoring' estimate.",
        "- R-adjusted probability flow and calibration agree on the boundary condition:",
        "  E2A moved probability from the initial answer toward a warranted V target, whereas",
        "  E2C reduced V-target probability and moved some mass toward the initial/background",
        "  alternatives. Option-token mass remained practically stable.",
        "- The available deterministic confidence proxy does not support a MUSE-style",
        "  moderation claim; R-baseline entropy is nearly degenerate, so this is a support",
        "  limitation rather than a faithful test of stochastic semantic entropy.",
        "",
        "## Data and method",
        "",
        "- Frozen inputs: 576 E2A records and 192 E2C records.",
        "- Primary uncertainty unit: structural family (12 clusters per dataset).",
        f"- Primary intervals: {BOOTSTRAP_DRAWS:,}-draw percentile family bootstrap.",
        "- E2C contains one disclosed missing continuous F row; discrete ITT retains it as",
        "  invalid/abstain, while continuous paired analyses use available complete pairs.",
        "- All analyses in this addendum are post-hoc and zero-GPU.",
        "",
        "## Main findings",
        "",
        "### 1. Matched validity judgment and final choice",
        "",
        (
            "- E2A full p25 change in **recognized-invalid F followed anyway**: "
            f"{fmt_ci(e2a_full_f['recognized_and_yielded'])}."
        ),
        (
            "- E2A full p25 change in **recognized-valid V but initial retained**: "
            f"{fmt_ci(e2a_full_v['recognized_but_refused'])}."
        ),
        (
            "- Change in explicit F validity accuracy: "
            f"{fmt_ci(e2a_full_f['validity_correct'])}; change in explicit V validity "
            f"accuracy: {fmt_ci(e2a_full_v['validity_correct'])}."
        ),
        "",
        "These are matched-item, separate-call results. They distinguish explicit judgment",
        "performance from answer policy at the assay level, but do not prove that a particular",
        "generation internally represented the correct verdict before answering. In particular,",
        "the increase in F target-follow is not cleanly attributable to the",
        "recognized-invalid-but-followed cell.",
        "",
        "### 2. SDT decomposition",
        "",
        (
            "- E2A full p25: "
            f"Δd′ = {fmt_ci(e2a_sdt['d_prime'])}; "
            f"Δcriterion = {fmt_ci(e2a_sdt['criterion'])}. "
            "A negative criterion change is a more liberal target-update policy."
        ),
        (
            "- E2C full p25: "
            f"Δd′ = {fmt_ci(e2c_sdt['d_prime'])}; "
            f"Δcriterion = {fmt_ci(e2c_sdt['criterion'])}."
        ),
        "",
        "The family-cluster bootstrap is primary. Probit GEE estimates are secondary because",
        "each assay contains only 12 structural-family clusters.",
        "",
        "### 3. R-adjusted probability flow",
        "",
    ]
    for dataset, dataset_flow in (("E2A", e2a_flow), ("E2C", e2c_flow)):
        for arm in ("V", "F"):
            report.append(
                f"- {dataset} {arm}: initial {fmt_ci(dataset_flow[arm]['p_initial'])}; "
                f"target {fmt_ci(dataset_flow[arm]['p_target'])}; "
                f"background {fmt_ci(dataset_flow[arm]['p_background'])}."
            )
    report.extend(
        [
            "",
            "Each number is the p25 change in `(arm − matched R)` option probability. This",
            "removes the same-condition recheck baseline before comparing interventions.",
            "",
            "### 4. Calibration and uncertainty",
            "",
        ]
    )
    for dataset in ("E2A", "E2C"):
        report.append(f"- **{dataset}, full p25 versus uncapped:**")
        for arm in ("N", "R", "V", "F"):
            values = calibration["datasets"][dataset]["contrasts"]["full_p25"][arm]
            report.append(
                f"  - {arm}: ΔBrier {fmt_ci(values['brier'])}; "
                f"Δoption entropy {fmt_ci(values['conditional_entropy'])}; "
                f"Δoption mass {fmt_ci(values['option_mass'])}."
            )
    report.extend(
        [
            "",
            "The MUSE-style proxy uses uncapped R uncertainty rather than stochastic",
            "sample entropy. Its baseline entropy support is:",
        ]
    )
    for dataset in ("E2A", "E2C"):
        values = muse["datasets"][dataset]["full_p25"]
        report.append(
            f"- {dataset}: normalized R entropy mean {fmt(values['baseline_entropy']['mean'], 6)}, "
            f"SD {fmt(values['baseline_entropy']['sd'], 6)}; "
            f"entropy_degenerate = `{str(values['entropy_degenerate']).lower()}`."
        )
        moderator = values["spearman"]["top_gap_vs_delta_margin"]
        moderator_summary = {
            "estimate": moderator.get("spearman_rho"),
            "family_cluster_bootstrap_95ci": moderator.get(
                "family_cluster_bootstrap_95ci", [None, None]
            ),
        }
        report.append(
            "  Top-gap vs Δtarget-margin Spearman: "
            f"{fmt_ci(moderator_summary)}."
        )
    report.extend(
        [
            "",
            "## Claim candidates",
            "",
            "- Claim:",
            "  - Source evidence: E2A matched validity/choice cells and family-cluster contrasts.",
            "  - Allowed wording: aggregate explicit-validity accuracy and final-choice rates",
            "    need not move together under p25 on matched formal items.",
            "  - Forbidden stronger wording: the model recognized F as invalid and followed it",
            "    anyway; the same generation knew the argument was invalid.",
            "  - Decision: weaken; retain as a measurement distinction, not a mechanism claim.",
            "",
            "- Claim:",
            "  - Source evidence: SDT Δd′ and Δcriterion in E2A/E2C.",
            "  - Allowed wording: p25 changes the operating point for target updating, with the",
            "    sensitivity effect reported separately.",
            "  - Forbidden stronger wording: p25 is a pure criterion manipulation.",
            "  - Decision: report both datasets; do not average them into one pseudo-replication.",
            "",
            "- Claim:",
            "  - Source evidence: R-adjusted option probability and logit flows.",
            "  - Allowed wording: p25 redistribution is target-specific or background-directed",
            "    only to the degree supported by the corresponding intervals.",
            "  - Forbidden stronger wording: all lost initial probability is transferred",
            "    to target.",
            "  - Decision: keep the decomposition; state E2A and E2C directions separately.",
            "",
            "- Claim:",
            "  - Source evidence: option entropy, Brier/NLL, and R-baseline moderator.",
            "  - Allowed wording: current forced-choice uncertainty is saturated or informative",
            "    according to the observed support.",
            "  - Forbidden stronger wording: this is a faithful MUSE replication.",
            "  - Decision: keep calibration; report the moderator as inconclusive.",
            "",
            "## Boundaries",
            "",
            "- E2A is a development mechanism assay; E2C has one missing continuous F pair.",
            "- There is one model, one primary decoding configuration, and 12 families per",
            "  dataset.",
            "- All four diagnostic families are post-hoc.",
            "- C2 and E−1b are not pooled because they lack the same R/V/F/Validity structure.",
            "",
            "## Recommended use",
            "",
            "- Use the SDT and R-adjusted probability-flow panels as the main manuscript",
            "  diagnostics; they give distinct, directly interpretable estimands.",
            "- Use the matched validity-choice analysis to motivate separate judgment and",
            "  action readouts, not to claim latent knowledge or a specific circuit.",
            "- Treat the MUSE-style result as a negative feasibility check. A faithful MUSE",
            "  analysis would require stochastic samples or another non-degenerate uncertainty",
            "  source and therefore new model inference.",
        ]
    )
    (output / "analysis-report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8", newline="\n"
    )

    appendix = [
        "# Statistical appendix",
        "",
        "## Units and uncertainty",
        "",
        "- Unit of pairing: base × arm × condition.",
        "- Unit of resampling/clustering: structural family.",
        f"- Bootstrap draws: {BOOTSTRAP_DRAWS:,}; seed: {BOOTSTRAP_SEED}.",
        "- Intervals: percentile family-cluster bootstrap 95% intervals.",
        "- SDT extreme rates: Hautus log-linear correction.",
        "- GEE: probit link with exchangeable within-family correlation.",
        "- ECE omitted because 24 items per arm do not support stable binning.",
        "",
        "## Missingness",
        "",
        "- E2A missing answer-logit rows: "
        f"{analysis['input_audit']['missing_answer_logits']['E2A']}.",
        "- E2C missing answer-logit rows: "
        f"{analysis['input_audit']['missing_answer_logits']['E2C']}.",
        "- Discrete ITT retains invalid/abstain rows; continuous metrics use complete pairs only.",
        "",
        "## Multiple comparisons",
        "",
        "- Bootstrap intervals and exact effect sizes are primary.",
        "- Nominal GEE/OLS p-values are secondary.",
        "- Holm adjustments are stored in the machine-readable analysis for SDT and moderator",
        "  coefficient families.",
        "",
        "## Interpretation rules",
        "",
        "- Δd′ below zero: reduced V/F discrimination.",
        "- Δcriterion below zero: more liberal target updating.",
        "- Positive ΔBrier or ΔNLL: worse probabilistic accuracy.",
        "- Positive Δconditional entropy: less concentrated A/B/C/D decision distribution.",
        "- Negative Δoption mass: probability leaves the valid option-token space.",
        "- Probability-flow contrasts are difference-in-differences against matched R.",
        "",
        "## Model caveats",
        "",
        "- Twelve family clusters are too few for asymptotic cluster-robust p-values to be",
        "  definitive; the family bootstrap is emphasized.",
        "- The SDT mapping is a measurement framework, not a literal cognitive model.",
        "- The MUSE-style proxy is limited by nearly saturated R-baseline option distributions.",
    ]
    (output / "stats-appendix.md").write_text(
        "\n".join(appendix) + "\n", encoding="utf-8", newline="\n"
    )

    catalog = [
        "# Figure catalog",
        "",
        "## Figure 1 — Validity–choice gap",
        "",
        "- File: `figures/figure-01-validity-choice-gap.pdf`",
        "- Purpose: test whether explicit validity judgment and final choice dissociate.",
        "- Data: 192 E2A matched validity/answer pairs; 12-family bootstrap intervals.",
        "- Notice: compare recognized-invalid F target following with recognized-valid V refusal.",
        "- Implication: separates matched-item judgment performance from answer policy.",
        "- Caveat: separate calls; not token-level evidence of latent knowledge.",
        "",
        "## Figure 2 — SDT sensitivity and criterion",
        "",
        "- File: `figures/figure-02-sdt-sensitivity-criterion.pdf`",
        "- Purpose: distinguish V/F sensitivity changes from target-update criterion shifts.",
        "- Data: E2A and E2C V/F choices; family-cluster bootstrap 95% intervals.",
        "- Notice: Δd′ and Δcriterion are separate axes of change.",
        "- Implication: determines whether 'more updating' reflects discrimination or policy.",
        "",
        "## Figure 3 — R-adjusted probability flow",
        "",
        "- File: `figures/figure-03-r-adjusted-probability-flow.pdf`",
        "- Purpose: locate probability leaving the initial answer.",
        "- Data: complete option-logit pairs; each arm first differenced against matched R.",
        "- Notice: target and background movement need not have the same sign.",
        "- Implication: distinguishes target assimilation from general output diffusion.",
        "",
        "## Figure 4 — Calibration and uncertainty",
        "",
        "- File: `figures/figure-04-calibration-uncertainty.pdf`",
        "- Purpose: test whether behavioral shifts coincide with probabilistic degradation.",
        "- Data: Brier, normalized A/B/C/D entropy, and full-vocabulary option mass.",
        "- Notice: accuracy/calibration and format-space mass are separate.",
        "- Implication: identifies whether p25 changes epistemic quality or output geometry.",
        "",
        "## Figure 5 — Baseline confidence moderator",
        "",
        "- File: `figures/figure-05-uncertainty-moderator.pdf`",
        "- Purpose: test the MUSE-style prediction that baseline uncertainty moderates",
        "  cap effects.",
        "- Data: uncapped R top-1/top-2 logit gap and paired p25 target-margin change.",
        "- Notice: R entropy is nearly saturated, so the logit gap is the usable proxy.",
        "- Implication: a null result limits rather than confirms the uncertainty account.",
    ]
    (output / "figure-catalog.md").write_text(
        "\n".join(catalog) + "\n", encoding="utf-8", newline="\n"
    )


def delivery_manifest(output: Path) -> dict[str, Any]:
    files = sorted(path for path in output.rglob("*") if path.is_file())
    return {
        "status": "complete",
        "files": {
            path.relative_to(output).as_posix(): sha256(path)
            for path in files
            if path.name != "delivery_manifest.json"
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    e2a_all = read_jsonl(E2A_PATH)
    e2c_all = read_jsonl(E2C_PATH)
    input_audit = validate_inputs(e2a_all, e2c_all)
    write_json(output / "input_audit.json", input_audit)
    if input_audit["status"] != "passed":
        raise SystemExit(f"Input audit failed: {input_audit['failures']}")

    e2a_answer = [row for row in e2a_all if row["assay"] == "answer"]
    datasets: dict[str, Sequence[dict[str, Any]]] = {
        "E2A": e2a_answer,
        "E2C": e2c_all,
    }
    validity_choice, validity_pairs = analyze_validity_choice(e2a_all)
    sdt, sdt_rows = analyze_sdt(datasets)
    probability_flow, probability_flow_base, probability_flow_rows = (
        analyze_probability_flow(datasets)
    )
    calibration, calibration_per_row, calibration_rows = analyze_calibration(datasets)
    muse, muse_rows = analyze_muse_proxy(datasets)
    apply_holm_corrections(sdt, muse)

    analysis = {
        "status": "complete",
        "scope": "post_hoc_zero_gpu_epistemic_diagnostics",
        "input_audit": input_audit,
        "validity_choice_gap": validity_choice,
        "sdt": sdt,
        "probability_flow": probability_flow,
        "calibration": calibration,
        "muse_style_moderator": muse,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED},
    }
    write_json(output / "EPISTEMIC_DIAGNOSTICS_ANALYSIS.json", analysis)
    write_csv(output / "validity_choice_pairs.csv", validity_pairs)
    write_csv(output / "sdt_summary.csv", sdt_rows)
    write_csv(output / "probability_flow_base.csv", probability_flow_base)
    write_csv(output / "probability_flow_summary.csv", probability_flow_rows)
    write_csv(output / "calibration_per_row.csv", calibration_per_row)
    write_csv(output / "calibration_summary.csv", calibration_rows)
    write_csv(output / "muse_style_moderator_rows.csv", muse_rows)
    if not args.skip_figures:
        make_figures(
            output,
            validity_choice,
            validity_pairs,
            sdt,
            probability_flow_rows,
            calibration_rows,
            muse_rows,
        )
    write_reports(output, analysis)
    write_json(output / "delivery_manifest.json", delivery_manifest(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
