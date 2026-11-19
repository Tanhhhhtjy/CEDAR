#!/usr/bin/env python3
"""Build ACL-exp-b1 contrast pairs from the frozen A1 v2.2 datasets (CPU only).

Every pair = (clean_messages, corrupt_messages) rendered with experiments/ACL-exp-a1/scripts/render_prompts.py in
the no_think regime, so prompts are byte-identical to the A1 v2.2 behavioural run.

Contrasts (clean vs corrupt; metric = logit[pos] - logit[neg] at the token after the readout key):
  C_soc        S            vs AO           dialog, context F      pos=a_t neg=a0   (Pinpoint-style social pathway)
  C_form       shell_only   vs AO           dialog, context F      pos=a_t neg=a0   (argument-form compliance)
  C_Fshell     F            vs shell_only   dialog, context F      pos=a_t neg=a0   (plausible content beyond form)
  C_content_AF V            vs F            dialog, argument_flip  pos=a_t neg=a0   (explicit fact-rule conflict)
  C_content_N  standalone V vs standalone F argument_flip          pos=VALID neg=INVALID, key "ArgumentCheck:"
  C_content_CF V(ctxV)      vs F(ctxF)      dialog, context_flip   pos=a_t neg=a0   [DEFERRED: gated by A1 2x2]
  C_ctx_shell  shell_ctxV   vs shell_only   dialog, context_flip   pos=a_t neg=a0   [DEFERRED: context main effect]
Interaction attribution for context_flip = AtP(C_content_CF) - AtP(C_ctx_shell)  (only if the A1 v2.2 2x2 gate passes).

Usage: python experiments/ACL-exp-b1/scripts/b1_build_pairs.py [--datasets symbolic_pilot natural_pilot]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
B1 = HERE.parents[0]
REPO = B1.parents[1]
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-a1" / "scripts"))
from render_prompts import build_messages, dataset_paths, load_config  # noqa: E402

CONTRASTS = {
    #  name           clean arm               corrupt arm             pair_type filter   readout           pos/neg
    "C_soc":        ("S",                    "AO",                   None,              "Final:",         "target"),
    "C_form":       ("shell_only",           "AO",                   None,              "Final:",         "target"),
    "C_Fshell":     ("F",                    "shell_only",           None,              "Final:",         "target"),
    "C_content_AF": ("V",                    "F",                    "argument_flip",   "Final:",         "target"),
    "C_content_N":  ("standalone_neutral_V", "standalone_neutral_F", "argument_flip",   "ArgumentCheck:", "verdict"),
    "C_content_CF": ("V",                    "F",                    "context_flip",    "Final:",         "target"),
    "C_ctx_shell":  ("shell_only_ctxV",      "shell_only",           "context_flip",    "Final:",         "target"),
}
DEFERRED = {"C_content_CF", "C_ctx_shell"}


def half_of(family_id: str, contrast: str) -> str:
    h = int(hashlib.sha256(f"{contrast}|{family_id}".encode()).hexdigest(), 16)
    return "A" if h % 2 == 0 else "B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["symbolic_pilot", "natural_pilot"])
    ap.add_argument("--out", default=str(B1 / "data" / "b1_pairs_v2_2.jsonl"))
    args = ap.parse_args()
    cfg = load_config()
    assert cfg["protocol_version"] == "2.2", "B1 must use the A1 v2.2 prompts"
    paths = dataset_paths(cfg)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = {}
    with out.open("w", encoding="utf-8", newline="\n") as f:
        for ds in args.datasets:
            for line in paths[ds].read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                fam = row.get("family_id", row["base_id"])
                for cname, (ca, ka, pt, key, kind) in CONTRASTS.items():
                    if pt and row["pair_type"] != pt:
                        continue
                    pos, neg = (row["target_answer"], row["initial_answer"]) if kind == "target" else ("VALID", "INVALID")
                    rec = {
                        "pair_id": f"{cname}|{row['base_id']}",
                        "contrast": cname,
                        "deferred": cname in DEFERRED,
                        "dataset": ds,
                        "base_id": row["base_id"],
                        "family_id": fam,
                        "pair_type": row["pair_type"],
                        "fallacy_class": row["fallacy_class"],
                        "difficulty_depth": row["difficulty_depth"],
                        "split_half": half_of(fam, cname),
                        "readout_key": key,
                        "pos_word": pos,
                        "neg_word": neg,
                        "clean_arm": ca,
                        "corrupt_arm": ka,
                        "clean_messages": build_messages(cfg, row, ca, "no_think"),
                        "corrupt_messages": build_messages(cfg, row, ka, "no_think"),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n[cname] = n.get(cname, 0) + 1
    print(f"[OK] wrote {out}  pairs per contrast: {n}")


if __name__ == "__main__":
    main()
