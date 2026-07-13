"""Observation-only bag-of-words leakage diagnostics for M2.2b v3.

No model under evaluation is called.  This script trains a small deterministic
multinomial Naive Bayes classifier only on V/F Observation text.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent
BASE = HERE / "m2_2b_v3_heldout60_base.jsonl"
REPORT = HERE / "leakage_report.json"
MANIFEST = HERE / "manifest.json"
SEED = 20260713
THRESHOLD = 0.65
TOKEN_RE = re.compile(r"[a-z]+(?:-[a-z]+)?|\d+(?:\.\d+)?")
SOURCE_AUTHORITY_CUES = (
    "record", "register", "audit", "report", "source", "certificate", "diagnostic",
    "validated", "explicit", "signed", "archive", "metadata", "log", "manual",
    "target", "prior", "preceding", "companion", "another", "different", "other",
    "neighboring", "adjacent", "authority", "authoritative",
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def fit_nb(samples: list[dict]) -> dict:
    docs = Counter()
    token_counts = {"V": Counter(), "F": Counter()}
    token_totals = Counter()
    vocab: set[str] = set()
    for sample in samples:
        label = sample["label"]
        counts = Counter(tokens(sample["observation"]))
        docs[label] += 1
        token_counts[label].update(counts)
        token_totals[label] += sum(counts.values())
        vocab.update(counts)
    return {
        "docs": docs,
        "token_counts": token_counts,
        "token_totals": token_totals,
        "vocab": vocab,
        "n_docs": len(samples),
    }


def predict(model: dict, observation: str) -> str:
    counts = Counter(tokens(observation))
    scores = {}
    vocab_size = max(1, len(model["vocab"]))
    for label in ("V", "F"):
        prior = (model["docs"][label] + 1) / (model["n_docs"] + 2)
        denominator = model["token_totals"][label] + vocab_size
        score = math.log(prior)
        for token, count in counts.items():
            if token in model["vocab"]:
                score += count * math.log((model["token_counts"][label][token] + 1) / denominator)
        scores[label] = score
    return max(("V", "F"), key=lambda label: (scores[label], label))


def accuracy(train: list[dict], test: list[dict]) -> tuple[int, int]:
    model = fit_nb(train)
    correct = sum(predict(model, sample["observation"]) == sample["label"] for sample in test)
    return correct, len(test)


def build_samples() -> list[dict]:
    samples: list[dict] = []
    for base in load_jsonl(BASE):
        for label in ("V", "F"):
            samples.append({
                "base_id": base["base_id"],
                "family": base["template_family"],
                "variant": base["template_variant"],
                "label": label,
                "observation": base["arms"][label]["observation"],
            })
    return samples


def grouped_cv(samples: list[dict], folds: int = 5) -> dict:
    families = sorted({x["family"] for x in samples})
    random.Random(SEED).shuffle(families)
    fold_reports = []
    total_correct = total_test = 0
    for fold in range(folds):
        held = set(families[fold::folds])
        train = [x for x in samples if x["family"] not in held]
        test = [x for x in samples if x["family"] in held]
        correct, count = accuracy(train, test)
        total_correct += correct
        total_test += count
        fold_reports.append({
            "fold": fold + 1,
            "heldout_families": sorted(held),
            "train_observations": len(train),
            "test_observations": count,
            "correct": correct,
            "accuracy": round(correct / count, 6),
        })
    return {
        "method": "5-fold family-grouped multinomial bag-of-words Naive Bayes",
        "correct": total_correct,
        "observations": total_test,
        "accuracy": round(total_correct / total_test, 6),
        "folds": fold_reports,
    }


def random_split(samples: list[dict]) -> dict:
    by_label = {label: [x for x in samples if x["label"] == label] for label in ("V", "F")}
    rng = random.Random(SEED + 1)
    train: list[dict] = []
    test: list[dict] = []
    for label in ("V", "F"):
        rng.shuffle(by_label[label])
        cut = int(0.8 * len(by_label[label]))
        train.extend(by_label[label][:cut])
        test.extend(by_label[label][cut:])
    correct, count = accuracy(train, test)
    leaked_families = len({x["family"] for x in train} & {x["family"] for x in test})
    return {
        "method": "stratified random 80/20 observation split",
        "train_observations": len(train),
        "test_observations": count,
        "families_present_in_both_splits": leaked_families,
        "correct": correct,
        "accuracy": round(correct / count, 6),
        "warning": "Diagnostic only; this split may place variants from one family on both sides and is not the hard gate.",
    }


def leave_one_family_out(samples: list[dict]) -> dict:
    rows = []
    total_correct = total_count = 0
    for family in sorted({x["family"] for x in samples}):
        train = [x for x in samples if x["family"] != family]
        test = [x for x in samples if x["family"] == family]
        correct, count = accuracy(train, test)
        total_correct += correct
        total_count += count
        rows.append({"family": family, "correct": correct, "observations": count, "accuracy": round(correct / count, 6)})
    return {
        "method": "leave-one-family-out",
        "correct": total_correct,
        "observations": total_count,
        "accuracy": round(total_correct / total_count, 6),
        "families": rows,
    }


def top_predictive_tokens(samples: list[dict], n: int = 20) -> dict:
    model = fit_nb(samples)
    vocab_size = len(model["vocab"])
    denominators = {label: model["token_totals"][label] + vocab_size for label in ("V", "F")}
    scored = []
    for token in model["vocab"]:
        log_v = math.log((model["token_counts"]["V"][token] + 1) / denominators["V"])
        log_f = math.log((model["token_counts"]["F"][token] + 1) / denominators["F"])
        scored.append((log_v - log_f, token, model["token_counts"]["V"][token], model["token_counts"]["F"][token]))
    scored.sort()
    encode = lambda row: {"token": row[1], "log_odds_V_minus_F": round(row[0], 6), "V_count": row[2], "F_count": row[3]}
    return {
        "F_predictive": [encode(row) for row in scored[:n]],
        "V_predictive": [encode(row) for row in reversed(scored[-n:])],
    }


def cue_counts(samples: list[dict]) -> dict:
    result = {}
    for cue in SOURCE_AUTHORITY_CUES:
        entry = {}
        pattern = re.compile(rf"\b{re.escape(cue)}\w*\b", re.IGNORECASE)
        for label in ("V", "F"):
            observations = [x["observation"] for x in samples if x["label"] == label]
            entry[label] = {
                "total_occurrences": sum(len(pattern.findall(text)) for text in observations),
                "observations_with_cue": sum(bool(pattern.search(text)) for text in observations),
            }
        result[cue] = entry
    return result


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    samples = build_samples()
    grouped = grouped_cv(samples)
    random_report = random_split(samples)
    lofo = leave_one_family_out(samples)
    passed = grouped["accuracy"] <= THRESHOLD
    report = {
        "status": "passed" if passed else "failed_rewrite_required",
        "scope": "Observation text only; Candidate reasoning and every other field excluded",
        "classifier": "standard-library multinomial Naive Bayes over unigrams with Laplace smoothing",
        "random_seed": SEED,
        "sample_count": len(samples),
        "family_count": len({x["family"] for x in samples}),
        "class_counts": dict(Counter(x["label"] for x in samples)),
        "hard_gate": {"metric": "family_grouped_accuracy", "maximum": THRESHOLD, "observed": grouped["accuracy"], "passed": passed},
        "family_grouped_cross_validation": grouped,
        "random_split": random_report,
        "leave_one_family_out": lofo,
        "top_predictive_tokens": top_predictive_tokens(samples),
        "source_authority_cue_counts": cue_counts(samples),
        "anti_padding_audit": "No shared filler was added by this script; it reads the authored Observation field verbatim.",
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest["observation_only_leakage"] = {
        "status": report["status"],
        "family_grouped_accuracy": grouped["accuracy"],
        "random_split_accuracy": random_report["accuracy"],
        "leave_one_family_out_accuracy": lofo["accuracy"],
        "hard_threshold": THRESHOLD,
        "classifier": report["classifier"],
    }
    manifest.setdefault("sha256", {})[REPORT.name] = file_sha256(REPORT)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)
    print("OBSERVATION-ONLY LEAKAGE GATE PASSED")


if __name__ == "__main__":
    main()
