"""Observation-only lexical leakage diagnostics for the v4 development PoC."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
SEED = 20260713
THRESHOLD = 0.70
TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def observation(row: dict) -> str:
    text = row["followup_user"]
    match = re.search(r"Observation:\s*(.*?)\s*Candidate reasoning:", text, flags=re.S)
    if not match:
        raise AssertionError(f"missing Observation wrapper: {row['id']}")
    return match.group(1).strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def train(rows: list[dict]) -> dict:
    labels = ("V", "F")
    token_counts = {label: Counter() for label in labels}
    totals = Counter()
    docs = Counter()
    vocabulary: set[str] = set()
    for row in rows:
        label = row["arm"]
        tokens = tokenize(row["observation"])
        token_counts[label].update(tokens)
        totals[label] += len(tokens)
        docs[label] += 1
        vocabulary.update(tokens)
    return {
        "token_counts": token_counts,
        "totals": totals,
        "docs": docs,
        "vocabulary": vocabulary,
    }


def predict(model: dict, text: str) -> str:
    labels = ("V", "F")
    total_docs = sum(model["docs"].values())
    vocab_size = max(1, len(model["vocabulary"]))
    tokens = tokenize(text)
    scores = {}
    for label in labels:
        scores[label] = math.log((model["docs"][label] + 1) / (total_docs + len(labels)))
        denominator = model["totals"][label] + vocab_size
        for token in tokens:
            scores[label] += math.log((model["token_counts"][label][token] + 1) / denominator)
    return max(labels, key=lambda label: (scores[label], label))


def accuracy(train_rows: list[dict], test_rows: list[dict]) -> float:
    model = train(train_rows)
    return sum(predict(model, row["observation"]) == row["arm"] for row in test_rows) / len(test_rows)


def grouped_cv(rows: list[dict], folds: int = 6) -> tuple[float, list[dict]]:
    rng = random.Random(SEED)
    families = sorted({row["template_family"] for row in rows})
    rng.shuffle(families)
    buckets = [families[i::folds] for i in range(folds)]
    reports = []
    weighted_correct = 0.0
    total = 0
    for index, heldout in enumerate(buckets):
        test = [row for row in rows if row["template_family"] in heldout]
        training = [row for row in rows if row["template_family"] not in heldout]
        score = accuracy(training, test)
        weighted_correct += score * len(test)
        total += len(test)
        reports.append({"fold": index, "families": sorted(heldout), "items": len(test), "accuracy": round(score, 6)})
    return weighted_correct / total, reports


def random_split(rows: list[dict]) -> float:
    rng = random.Random(SEED + 1)
    test: list[dict] = []
    training: list[dict] = []
    for label in ("V", "F"):
        chunk = [row for row in rows if row["arm"] == label]
        rng.shuffle(chunk)
        cut = max(1, round(len(chunk) * 0.25))
        test.extend(chunk[:cut])
        training.extend(chunk[cut:])
    return accuracy(training, test)


def leave_one_family_out(rows: list[dict]) -> tuple[float, list[dict]]:
    reports = []
    correct = 0.0
    total = 0
    for family in sorted({row["template_family"] for row in rows}):
        test = [row for row in rows if row["template_family"] == family]
        training = [row for row in rows if row["template_family"] != family]
        score = accuracy(training, test)
        correct += score * len(test)
        total += len(test)
        reports.append({"family": family, "items": len(test), "accuracy": round(score, 6)})
    return correct / total, reports


def predictive_tokens(rows: list[dict], limit: int = 20) -> list[dict]:
    model = train(rows)
    vocab_size = max(1, len(model["vocabulary"]))
    output = []
    for token in model["vocabulary"]:
        pv = (model["token_counts"]["V"][token] + 1) / (model["totals"]["V"] + vocab_size)
        pf = (model["token_counts"]["F"][token] + 1) / (model["totals"]["F"] + vocab_size)
        output.append({
            "token": token,
            "V_count": model["token_counts"]["V"][token],
            "F_count": model["token_counts"]["F"][token],
            "log_odds_V_minus_F": round(math.log(pv / pf), 6),
        })
    return sorted(output, key=lambda row: abs(row["log_odds_V_minus_F"]), reverse=True)[:limit]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    flats = load_jsonl(HERE / "m2_2b_v4_public_poc180_arms.jsonl")
    rows = [
        {
            "id": row["id"],
            "arm": row["arm"],
            "template_family": row["template_family"],
            "observation": observation(row),
        }
        for row in flats if row["arm"] in {"V", "F"}
    ]
    assert len(rows) == 72
    assert Counter(row["arm"] for row in rows) == {"V": 36, "F": 36}
    grouped, folds = grouped_cv(rows)
    random_accuracy = random_split(rows)
    lofo, lofo_rows = leave_one_family_out(rows)
    cues = ["audit", "correction", "retraction", "status", "report", "remains", "active", "invalid", "failed"]
    cue_counts = {
        cue: {
            label: sum(tokenize(row["observation"]).count(cue) for row in rows if row["arm"] == label)
            for label in ("V", "F")
        }
        for cue in cues
    }
    report = {
        "dataset_version": "m2.2b-v4-public-protocol-deliberative-poc",
        "classifier": "multinomial Naive Bayes over Observation-only unigrams with Laplace smoothing",
        "items": len(rows),
        "families": 18,
        "family_grouped_cross_validation": {"accuracy": round(grouped, 6), "folds": folds},
        "random_split_accuracy": round(random_accuracy, 6),
        "leave_one_family_out": {"accuracy": round(lofo, 6), "families": lofo_rows},
        "top_predictive_tokens": predictive_tokens(rows),
        "source_and_operation_cue_counts": cue_counts,
        "development_diagnostic_threshold": THRESHOLD,
        "status": "passed" if grouped <= THRESHOLD else "failed_rewrite_required",
        "note": "This development threshold diagnoses surface asymmetry; it is not a confirmatory statistical result.",
    }
    report_path = HERE / "leakage_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")

    manifest_path = HERE / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["observation_only_leakage"] = {
        "status": report["status"],
        "family_grouped_accuracy": round(grouped, 6),
        "random_split_accuracy": round(random_accuracy, 6),
        "leave_one_family_out_accuracy": round(lofo, 6),
        "development_diagnostic_threshold": THRESHOLD,
    }
    manifest["sha256"]["leakage_report.json"] = sha256(report_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "status": report["status"],
        "family_grouped_accuracy": round(grouped, 6),
        "random_split_accuracy": round(random_accuracy, 6),
        "leave_one_family_out_accuracy": round(lofo, 6),
    }, indent=2))
    if grouped > THRESHOLD:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
