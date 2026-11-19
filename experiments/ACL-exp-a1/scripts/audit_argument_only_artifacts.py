#!/usr/bin/env python3
"""Argument-Only Artifact Audit for ACL-exp-a1 (Opus P0-2 Defense).

Tests whether a classifier seeing ONLY the Turn-2 user argument text (without the Turn-1
problem context/rules) can predict V vs. F above chance (0.50).

- On `context_flip` twins: Turn-2 text is byte-for-byte identical between V and F, so any
  argument-only classifier MUST score strictly at chance (~0.50).
- On `argument_flip` pairs: Measures residual lexical/structural leakage in Turn-2 text
  alone, broken down overall and per fallacy class, using 5-fold GroupKFold by base_id.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run_cv_audit(rows: list[dict[str, Any]], label_tag: str) -> dict[str, Any]:
    texts: list[str] = []
    labels: list[int] = []
    groups: list[str] = []

    for r in rows:
        texts.append(r["arms"]["V"])
        labels.append(1)
        groups.append(r["base_id"])

        texts.append(r["arms"]["F"])
        labels.append(0)
        groups.append(r["base_id"])

    y = np.array(labels)
    g = np.array(groups)
    n_unique_groups = len(set(groups))
    n_splits = min(5, n_unique_groups)
    gkf = GroupKFold(n_splits=n_splits)

    word_clf = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 3), lowercase=True)),
            ("lr", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ]
    )
    combo_clf = Pipeline(
        [
            (
                "feats",
                FeatureUnion(
                    [
                        ("word", TfidfVectorizer(ngram_range=(1, 3), lowercase=True)),
                        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)),
                    ]
                ),
            ),
            ("lr", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
        ]
    )

    out: dict[str, Any] = {
        "subset": label_tag,
        "n_bases": len(rows),
        "n_examples": len(texts),
        "exact_turn2_byte_match_rate": float(
            np.mean([r["arms"]["V"] == r["arms"]["F"] for r in rows])
        ),
        "mean_char_len_v": float(np.mean([len(r["arms"]["V"]) for r in rows])),
        "mean_char_len_f": float(np.mean([len(r["arms"]["F"]) for r in rows])),
    }

    for name, pipe in (("word_ngram_lr", word_clf), ("word_plus_char_ngram_lr", combo_clf)):
        oof_probs = np.zeros(len(texts), dtype=float)
        for tr_idx, te_idx in gkf.split(texts, y, g):
            x_tr = [texts[i] for i in tr_idx]
            x_te = [texts[i] for i in te_idx]
            pipe.fit(x_tr, y[tr_idx])
            oof_probs[te_idx] = pipe.predict_proba(x_te)[:, 1]
        auc = float(roc_auc_score(y, oof_probs))
        acc = float(accuracy_score(y, (oof_probs >= 0.5).astype(int)))
        out[name] = {
            "oof_roc_auc": round(auc, 4),
            "oof_accuracy": round(acc, 4),
            "chance_level_passed_auc_le_0_55": auc <= 0.55,
        }
    return out


def main() -> None:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    in_path = data_dir / "symbolic_pilot_120_dev_only.jsonl"
    rows = load_jsonl(in_path)

    cf_rows = [r for r in rows if r["pair_type"] == "context_flip"]
    af_rows = [r for r in rows if r["pair_type"] == "argument_flip"]

    report: dict[str, Any] = {
        "input_file": str(in_path.name),
        "overall": run_cv_audit(rows, "all_120"),
        "context_flip": run_cv_audit(cf_rows, "context_flip_60"),
        "argument_flip": run_cv_audit(af_rows, "argument_flip_60"),
        "argument_flip_by_fallacy_class": {},
    }

    for fc in sorted({r["fallacy_class"] for r in af_rows}):
        sub = [r for r in af_rows if r["fallacy_class"] == fc]
        report["argument_flip_by_fallacy_class"][fc] = run_cv_audit(sub, f"argument_flip_{fc}")

    out_path = data_dir / "artifact_audit_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=== Argument-Only Artifact Audit Summary ===")
    for key in ("context_flip", "argument_flip", "overall"):
        sec = report[key]
        print(
            f"[{key:14s}] n_bases={sec['n_bases']:3d} | byte_match={sec['exact_turn2_byte_match_rate']:.2f} | "
            f"word_LR AUC={sec['word_ngram_lr']['oof_roc_auc']:.4f} | "
            f"word+char_LR AUC={sec['word_plus_char_ngram_lr']['oof_roc_auc']:.4f}"
        )
    print("\n[argument_flip per fallacy class (word+char LR AUC)]:")
    for fc, sec in report["argument_flip_by_fallacy_class"].items():
        print(
            f"  - {fc:22s}: AUC={sec['word_plus_char_ngram_lr']['oof_roc_auc']:.4f}, "
            f"Acc={sec['word_plus_char_ngram_lr']['oof_accuracy']:.4f}, "
            f"len(V)={sec['mean_char_len_v']:.1f} vs len(F)={sec['mean_char_len_f']:.1f}"
        )
    print(f"\n[OK] Saved audit report -> {out_path}")


if __name__ == "__main__":
    main()
