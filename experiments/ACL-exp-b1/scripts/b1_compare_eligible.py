#!/usr/bin/env python3
"""Compare full-set vs Turn-1-eligible-only ACL-exp-b1 results (CPU only).

Usage:
  python experiments/ACL-exp-b1/scripts/b1_compare_eligible.py --alias qwen3_14b
  python experiments/ACL-exp-b1/scripts/b1_compare_eligible.py --alias qwen3_8b
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
B1 = HERE.parents[0]
sys.path.insert(0, str(HERE))
import b1_lib as L  # noqa: E402


def weighted_mass_retained(top_elig: list[tuple[int, int]], S_full: torch.Tensor, top_full: list[tuple[int, int]]) -> float:
    shared = set(top_elig) & set(top_full)
    denom = sum(max(0.0, float(S_full[li, h])) for li, h in top_full)
    if denom <= 1e-12:
        return 0.0
    return sum(max(0.0, float(S_full[li, h])) for li, h in shared) / denom


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def compare_pair(full_dir: Path, elig_dir: Path, ks=(10, 20, 50)) -> dict:
    """Decision rule (agreed 2026-11-20): a contrast's localisation is 'stable on the eligible subset' if
      (a) weighted_mass_of_full@20 >= 0.70 (primary: the heavy heads survive), AND
      (b) jaccard@20 >= the full run's OWN split-half Jaccard@20 (noise ceiling; a fixed 0.70 bar is stricter
          than noise for small contrasts, e.g. 14B C_content_CF split-half J@20 = 0.538).
    Note: eligible pairs are a subset of the full-run pairs, so eligible-vs-full overlap is biased upward;
    `shared_pair_fraction` is reported so readers can discount it."""
    out: dict = {"full_dir": str(full_dir), "eligible_dir": str(elig_dir), "decision_rule": compare_pair.__doc__,
                 "atp_heads": {}, "atp_mlp": {}, "dla_heads": {}}
    full_rel = _load_json(full_dir / "stage_reliability.json").get("within_split_half", {})
    full_atp = _load_json(full_dir / "stage_atp.json")
    elig_atp = _load_json(elig_dir / "stage_atp.json")
    for pt in sorted(elig_dir.glob("atp_*.pt")):
        c = pt.stem.removeprefix("atp_")
        full_pt = full_dir / pt.name
        if not full_pt.exists():
            continue
        bf, be = torch.load(full_pt), torch.load(pt)
        if "head_all" in bf and "head_all" in be:
            Sf, Se = bf["head_all"], be["head_all"]
            row = {}
            for k in ks:
                tf, te = L.top_k(Sf, k), L.top_k(Se, k)
                row[f"jaccard@{k}"] = round(L.jaccard(tf, te), 4)
                row[f"weighted_mass_of_full@{k}"] = round(weighted_mass_retained(te, Sf, tf), 4)
                row[f"weighted_mass_of_elig@{k}"] = round(weighted_mass_retained(tf, Se, te), 4)
            # Also check B-half stability (the selection half used in verify/dissociation)
            if "head_B" in bf and "head_B" in be:
                row["half_B_jaccard@20"] = round(L.jaccard(L.top_k(bf["head_B"], 20), L.top_k(be["head_B"], 20)), 4)
            row["top20_full"] = L.top_k(Sf, 20)
            row["top20_elig"] = L.top_k(Se, 20)
            row["shared_top20"] = sorted(set(row["top20_full"]) & set(row["top20_elig"]))
            ceil = full_rel.get(c, {})
            ceil = ceil.get("20", ceil.get(20))
            row["noise_ceiling_full_split_half_J@20"] = ceil
            row["passes_weighted_mass_0.70"] = bool(row["weighted_mass_of_full@20"] >= 0.70)
            row["passes_noise_ceiling"] = None if ceil is None else bool(row["jaccard@20"] >= ceil)
            row["stable_on_eligible"] = None if ceil is None else bool(row["passes_weighted_mass_0.70"] and row["passes_noise_ceiling"])
            row["passes_0.70_at_20"] = bool(row["jaccard@20"] >= 0.70)  # legacy fixed bar, reported only
            ea, fa = elig_atp.get(c, {}), full_atp.get(c, {})
            row["n_used_eligible"] = {"all": ea.get("n_used"), "A": ea.get("n_A"), "B": ea.get("n_B")}
            row["n_used_full"] = {"all": fa.get("n_used"), "A": fa.get("n_A"), "B": fa.get("n_B")}
            if ea.get("n_used") and fa.get("n_used"):
                row["shared_pair_fraction"] = round(ea["n_used"] / fa["n_used"], 3)
            out["atp_heads"][c] = row
        if "mlp_all" in bf and "mlp_all" in be:
            mf = bf["mlp_all"].view(-1, 1)
            me = be["mlp_all"].view(-1, 1)
            tf5 = [li for li, _ in L.top_k(mf, 5)]
            te5 = [li for li, _ in L.top_k(me, 5)]
            out["atp_mlp"][c] = {
                "top5_full": tf5,
                "top5_elig": te5,
                "jaccard@5": round(L.jaccard(tf5, te5), 4),
            }
    for pt in sorted(elig_dir.glob("dla_*.pt")):
        c = pt.stem.removeprefix("dla_")
        full_pt = full_dir / pt.name
        if not full_pt.exists():
            continue
        bf, be = torch.load(full_pt), torch.load(pt)
        Sf = bf.get("dla_all", bf.get(f"dla_{c}"))
        Se = be.get("dla_all", be.get(f"dla_{c}"))
        if Sf is not None and Se is not None:
            row = {}
            for k in ks:
                row[f"jaccard@{k}"] = round(L.jaccard(L.top_k(Sf, k), L.top_k(Se, k)), 4)
            out["dla_heads"][c] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alias", required=True, help="base alias, e.g. qwen3_14b or qwen3_8b")
    ap.add_argument("--eligible-alias", help="defaults to <alias>_eligible")
    args = ap.parse_args()
    full_dir = B1 / "results" / args.alias
    elig_dir = B1 / "results" / (args.eligible_alias or f"{args.alias}_eligible")
    report = compare_pair(full_dir, elig_dir)
    out = elig_dir / "eligible_vs_full_comparison.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"[OK] wrote {out}")
    for c, v in report["atp_heads"].items():
        print(f"  AtP {c:16s}: J@20={v['jaccard@20']:.3f} (ceiling {v['noise_ceiling_full_split_half_J@20']}) "
              f"mass_full@20={v['weighted_mass_of_full@20']:.3f} n_elig(A/B)={v['n_used_eligible']['A']}/{v['n_used_eligible']['B']} "
              f"stable={v['stable_on_eligible']}")
    for c, v in report["atp_mlp"].items():
        print(f"  MLP {c:16s}: J@5={v['jaccard@5']:.3f} full={v['top5_full']} elig={v['top5_elig']}")
    for c, v in report["dla_heads"].items():
        print(f"  DLA {c:16s}: J@10={v['jaccard@10']:.3f} J@20={v['jaccard@20']:.3f}")


if __name__ == "__main__":
    main()
