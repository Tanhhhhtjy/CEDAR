#!/usr/bin/env python3
"""Observation-only family-grouped lexical leakage audit for the v5 bank."""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from build_m2_2b_v5_dose_response_bank import MANIFEST, OUTPUT, atomic_write_json, load_jsonl

from cedar_m2.checkpoint import file_sha256
from cedar_m2.data.m2_2b_v4_neutral import split_vf_followup

HERE = Path(__file__).resolve().parent
REPORT = HERE / "leakage_report.json"
SEED = 20260714
THRESHOLD = 0.65
TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def train(rows: list[dict[str, Any]]) -> dict[str, Any]:
    token_counts = {label: Counter() for label in ("V", "F")}
    totals: Counter[str] = Counter()
    docs: Counter[str] = Counter()
    vocabulary: set[str] = set()
    for row in rows:
        label = row["arm"]
        tokens = tokenize(row["observation"])
        token_counts[label].update(tokens)
        totals[label] += len(tokens)
        docs[label] += 1
        vocabulary.update(tokens)
    return {"token_counts": token_counts, "totals": totals, "docs": docs, "vocabulary": vocabulary}


def predict(model: dict[str, Any], text: str) -> str:
    tokens = tokenize(text)
    total_docs = sum(model["docs"].values())
    vocab_size = max(1, len(model["vocabulary"]))
    scores: dict[str, float] = {}
    for label in ("V", "F"):
        value = math.log((model["docs"][label] + 1) / (total_docs + 2))
        denominator = model["totals"][label] + vocab_size
        for token in tokens:
            value += math.log((model["token_counts"][label][token] + 1) / denominator)
        scores[label] = value
    return max(("V", "F"), key=lambda label: (scores[label], label))


def accuracy(training: list[dict[str, Any]], test: list[dict[str, Any]]) -> float:
    model = train(training)
    return sum(predict(model, row["observation"]) == row["arm"] for row in test) / len(test)


def grouped_cv(rows: list[dict[str, Any]], folds: int = 6) -> tuple[float, list[dict[str, Any]]]:
    rng = random.Random(SEED)
    families = sorted({row["template_family"] for row in rows})
    rng.shuffle(families)
    buckets = [families[index::folds] for index in range(folds)]
    reports: list[dict[str, Any]] = []
    correct = 0.0
    total = 0
    for fold, heldout in enumerate(buckets):
        test = [row for row in rows if row["template_family"] in heldout]
        training = [row for row in rows if row["template_family"] not in heldout]
        score = accuracy(training, test)
        reports.append(
            {
                "fold": fold,
                "families": sorted(heldout),
                "items": len(test),
                "accuracy": round(score, 6),
            }
        )
        correct += score * len(test)
        total += len(test)
    return correct / total, reports


def random_split(rows: list[dict[str, Any]]) -> float:
    rng = random.Random(SEED + 1)
    training: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for label in ("V", "F"):
        chunk = [row for row in rows if row["arm"] == label]
        rng.shuffle(chunk)
        cut = round(len(chunk) * 0.25)
        test.extend(chunk[:cut])
        training.extend(chunk[cut:])
    return accuracy(training, test)


def leave_one_family_out(rows: list[dict[str, Any]]) -> tuple[float, list[dict[str, Any]]]:
    reports: list[dict[str, Any]] = []
    correct = 0.0
    total = 0
    for family in sorted({row["template_family"] for row in rows}):
        test = [row for row in rows if row["template_family"] == family]
        training = [row for row in rows if row["template_family"] != family]
        score = accuracy(training, test)
        reports.append({"family": family, "items": len(test), "accuracy": round(score, 6)})
        correct += score * len(test)
        total += len(test)
    return correct / total, reports


def predictive_tokens(rows: list[dict[str, Any]], limit: int = 24) -> list[dict[str, Any]]:
    model = train(rows)
    vocab_size = max(1, len(model["vocabulary"]))
    output: list[dict[str, Any]] = []
    for token in model["vocabulary"]:
        pv = (model["token_counts"]["V"][token] + 1) / (model["totals"]["V"] + vocab_size)
        pf = (model["token_counts"]["F"][token] + 1) / (model["totals"]["F"] + vocab_size)
        output.append(
            {
                "token": token,
                "V_count": model["token_counts"]["V"][token],
                "F_count": model["token_counts"]["F"][token],
                "log_odds_V_minus_F": round(math.log(pv / pf), 6),
            }
        )
    return sorted(output, key=lambda row: abs(row["log_odds_V_minus_F"]), reverse=True)[:limit]


def main() -> None:
    candidates = load_jsonl(OUTPUT)
    rows = []
    for row in candidates:
        if row["arm"] not in {"V", "F"}:
            continue
        observation, _ = split_vf_followup(row["followup_user"])
        rows.append({**row, "observation": observation})
    if len(rows) != 288 or Counter(row["arm"] for row in rows) != Counter({"V": 144, "F": 144}):
        raise ValueError("Expected 144 V and 144 F observations")

    grouped, folds = grouped_cv(rows)
    lofo, lofo_rows = leave_one_family_out(rows)
    cue_tokens = (
        "active",
        "remains",
        "only",
        "complete",
        "independent",
        "explicit",
        "limited",
        "verified",
    )
    cue_counts = {
        token: {
            label: sum(
                tokenize(row["observation"]).count(token) for row in rows if row["arm"] == label
            )
            for label in ("V", "F")
        }
        for token in cue_tokens
    }
    report = {
        "dataset_version": "m2.2b-v5-dose-response-enriched-development-candidate-bank",
        "classifier": (
            "multinomial Naive Bayes over Observation-only unigrams with Laplace smoothing"
        ),
        "items": len(rows),
        "families": len({row["template_family"] for row in rows}),
        "family_grouped_cross_validation": {"accuracy": round(grouped, 6), "folds": folds},
        "random_split_accuracy": round(random_split(rows), 6),
        "random_split_interpretation": (
            "Diagnostic only: a row-random split places alternate levels of the "
            "same base on both sides and therefore leaks level siblings."
        ),
        "leave_one_family_out": {"accuracy": round(lofo, 6), "families": lofo_rows},
        "top_predictive_tokens": predictive_tokens(rows),
        "source_and_operation_cue_counts": cue_counts,
        "hard_threshold": THRESHOLD,
        "status": "passed" if grouped <= THRESHOLD else "failed_rewrite_required",
        "note": "Candidate-bank diagnostic only; all prompt levels remain family-grouped.",
        "candidate_sha256": file_sha256(OUTPUT),
    }
    atomic_write_json(REPORT, report)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["observation_only_leakage"] = {
        "status": report["status"],
        "family_grouped_accuracy": report["family_grouped_cross_validation"]["accuracy"],
        "random_split_accuracy": report["random_split_accuracy"],
        "leave_one_family_out_accuracy": report["leave_one_family_out"]["accuracy"],
        "hard_threshold": THRESHOLD,
    }
    manifest.setdefault("files", {})[REPORT.name] = {
        "sha256": file_sha256(REPORT),
        "bytes": REPORT.stat().st_size,
    }
    atomic_write_json(MANIFEST, manifest)
    print(
        json.dumps(
            {
                "status": report["status"],
                "family_grouped_accuracy": report["family_grouped_cross_validation"]["accuracy"],
                "random_split_accuracy": report["random_split_accuracy"],
                "leave_one_family_out_accuracy": report["leave_one_family_out"]["accuracy"],
            },
            indent=2,
        )
    )
    if grouped > THRESHOLD:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
