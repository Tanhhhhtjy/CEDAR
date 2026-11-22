#!/usr/bin/env python3
"""Build a 100% domain-disjoint, solver-certified, shortcut-audited Proxy-OOD dataset for Step 2 (CPU only).

Design guarantees (strictly preserving sealed c1_test_id & c1_test_ood):
1. ZERO PEEK at sealed test sets:
   - Never opens or reads c1_test_id_bases.jsonl or c1_test_ood_bases.jsonl.
   - Mathematical disjointness from A1 AND ALL C1 splits (train/dev/test_id/test_ood) is guaranteed by
     6 brand-new domain pools (airspace_corridor, cleanroom_fab, freight_customs, orbital_relay,
     coldchain_depot, mining_hoist) with 100% disjoint domain names, entity prefixes, entity names,
     zones, guards, metrics, predicates, and outcome labels.
2. Strictly avoids Test-OOD holdout axes:
   - Uses ONLY the 5 seen fallacy classes (scope_swap, cross_entity_transfer, missing_conjunct,
     negation_drop, threshold_violation) and depths 2 & 3 (120 depth-2 std + 120 depth-3 hard w/ lures).
   - STRICTLY EXCLUDES `converse_error` and `depth-4` so sealed `c1_test_ood` remains 100% untouched.
3. Statistical power for Step 2 paired arm comparison (240 bases):
   - 2 pair_types (context_flip, argument_flip) x 2 depths (2 std, 3 hard) x 5 classes x 12 reps = 240 bases.
   - Expected ~130-150 M0 dual-sided Turn-1 eligible bases in no_think and ~240 in think.
4. Built-in shortcut & G0 artifact audits (same hard thresholds as c1_build_datasets.py):
   - Zero overlap of domains, entities, base_ids, V/F contexts, or message hashes with A1, c1_train, c1_dev.
   - Balanced transitions: (domain, a0) -> at accuracy <= 0.45.
   - Counterbalanced cross_entity_transfer/argument_flip entity match in [0.45, 0.55].
   - Widened threshold_violation/argument_flip obs_val ROC AUC in [0.40, 0.60].
   - G0 argument-only TF-IDF GroupKFold AUC in [0.38, 0.62] overall, [0.35, 0.65] AF, exact 1.000 CF byte match.

Usage:
  python experiments/ACL-exp-c1/scripts/c1_build_proxy_ood.py
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
C1 = HERE.parents[0]
REPO = C1.parents[1]
A1_SCRIPTS = REPO / "experiments" / "ACL-exp-a1" / "scripts"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(A1_SCRIPTS))

import audit_argument_only_artifacts as A1_AUDIT  # noqa: E402
import build_symbolic_pilot_v2 as A1_SYM  # noqa: E402
import c1_build_datasets as C1B  # noqa: E402
from a1_arm_wrappers import ARM_ORDER, build_arms  # noqa: E402

PROXY_OOD_SEED = 20270219
SEEN_CLASSES = C1B.SEEN_CLASSES

# 6 brand-new domains with 100% zero overlap against BOTH A1_SYM.DOMAINS and C1B.C1_DOMAINS
PROXY_OOD_DOMAINS: tuple[dict[str, Any], ...] = (
    {
        "name": "airspace_corridor",
        "intro": "A closed airspace sector controller enforces exact routing rules. Use only the listed logs and rules.",
        "entity_prefix": "Flight",
        "entities": ("Jet-14", "Jet-28", "Jet-37", "Jet-49"),
        "zones": ("Sector East", "Sector West"),
        "guard_name": "weather_hold",
        "guard_active_text": "the weather hold is active",
        "guard_inactive_text": "the weather hold is inactive",
        "metric_name": "altitude index",
        "props": (
            ("p_obs", "broadcast a transponder ping"),
            ("p_mid1", "enters corridor screening"),
            ("p_mid2", "unlocks flight override"),
            ("p_mid3", "clears terminal priority"),
            ("p_mid4", "receives final vector release"),
            ("p_aux", "carries an IFR endorsement"),
            ("p_alt_mid", "holds a waypoint token"),
            ("p_noise1", "passed radar calibration"),
            ("p_noise2", "logged squawk verification"),
        ),
        "outcomes": ("Route-Alpha", "Route-Beta", "Route-Gamma", "Route-Delta"),
    },
    {
        "name": "cleanroom_fab",
        "intro": "A closed semiconductor cleanroom scheduler applies exact lot rules. Use only the listed records and rules.",
        "entity_prefix": "Wafer",
        "entities": ("Lot-105", "Lot-214", "Lot-329", "Lot-442"),
        "zones": ("Bay Quartz", "Bay Silicon"),
        "guard_name": "purge_lock",
        "guard_active_text": "the purge lock is active",
        "guard_inactive_text": "the purge lock is inactive",
        "metric_name": "particle count",
        "props": (
            ("p_obs", "recorded an etch drift"),
            ("p_mid1", "enters metrology audit"),
            ("p_mid2", "triggers litho override"),
            ("p_mid3", "grants reticle priority"),
            ("p_mid4", "receives final stepper dispatch"),
            ("p_aux", "holds a mask clearance"),
            ("p_alt_mid", "carries a vacuum stamp"),
            ("p_noise1", "passed optical inspection"),
            ("p_noise2", "logged cassette sync"),
        ),
        "outcomes": ("Tool-M1", "Tool-M2", "Tool-M3", "Tool-M4"),
    },
    {
        "name": "freight_customs",
        "intro": "A closed port customs manifest board follows exact inspection rules. Use only the listed entries and rules.",
        "entity_prefix": "Container",
        "entities": ("Manifest-11", "Manifest-24", "Manifest-36", "Manifest-47"),
        "zones": ("Dock Amber", "Dock Cobalt"),
        "guard_name": "tariff_freeze",
        "guard_active_text": "the tariff freeze is active",
        "guard_inactive_text": "the tariff freeze is inactive",
        "metric_name": "duty score",
        "props": (
            ("p_obs", "triggered an xray flag"),
            ("p_mid1", "enters broker review"),
            ("p_mid2", "authorizes customs override"),
            ("p_mid3", "opens bonded clearance"),
            ("p_mid4", "receives final gate release"),
            ("p_aux", "carries a bill endorsement"),
            ("p_alt_mid", "holds a seal certificate"),
            ("p_noise1", "passed weighbridge check"),
            ("p_noise2", "logged pallet manifest"),
        ),
        "outcomes": ("Lane-10", "Lane-20", "Lane-30", "Lane-40"),
    },
    {
        "name": "orbital_relay",
        "intro": "A closed deep-space ground station applies exact downlink rules. Use only the listed logs and rules.",
        "entity_prefix": "Orbiter",
        "entities": ("Sat-61", "Sat-73", "Sat-85", "Sat-94"),
        "zones": ("Band Zenith", "Band Nadir"),
        "guard_name": "solar_shield",
        "guard_active_text": "the solar shield is active",
        "guard_inactive_text": "the solar shield is inactive",
        "metric_name": "flux metric",
        "props": (
            ("p_obs", "emitted a carrier spike"),
            ("p_mid1", "enters ephemeris check"),
            ("p_mid2", "unlocks dish override"),
            ("p_mid3", "enables high-gain priority"),
            ("p_mid4", "receives final beacon lock"),
            ("p_aux", "holds a spectrum lease"),
            ("p_alt_mid", "carries an orbit key"),
            ("p_noise1", "passed doppler sweep"),
            ("p_noise2", "logged clock alignment"),
        ),
        "outcomes": ("Link-K1", "Link-K2", "Link-K3", "Link-K4"),
    },
    {
        "name": "coldchain_depot",
        "intro": "A closed refrigerated logistics depot enforces exact storage rules. Use only the listed sheets and rules.",
        "entity_prefix": "Pallet",
        "entities": ("Unit-33", "Unit-44", "Unit-57", "Unit-68"),
        "zones": ("Chiller Prime", "Chiller Reserve"),
        "guard_name": "frost_seal",
        "guard_active_text": "the frost seal is active",
        "guard_inactive_text": "the frost seal is inactive",
        "metric_name": "thermal index",
        "props": (
            ("p_obs", "logged a sensor spike"),
            ("p_mid1", "enters cryo inspection"),
            ("p_mid2", "activates vault override"),
            ("p_mid3", "grants express cold transfer"),
            ("p_mid4", "receives final loading pass"),
            ("p_aux", "holds a cold-pack voucher"),
            ("p_alt_mid", "carries a thermal tag"),
            ("p_noise1", "passed door-seal test"),
            ("p_noise2", "verified batch label"),
        ),
        "outcomes": ("Chamber-Red", "Chamber-Blue", "Chamber-Green", "Chamber-White"),
    },
    {
        "name": "mining_hoist",
        "intro": "A closed shaft hoist dispatcher applies exact cage rules. Use only the listed records and rules.",
        "entity_prefix": "Cage",
        "entities": ("Shaft-19", "Shaft-27", "Shaft-38", "Shaft-51"),
        "zones": ("Level Upper", "Level Lower"),
        "guard_name": "brake_interlock",
        "guard_active_text": "the brake interlock is active",
        "guard_inactive_text": "the brake interlock is inactive",
        "metric_name": "load tension",
        "props": (
            ("p_obs", "tripped a cable gauge"),
            ("p_mid1", "enters hoist verification"),
            ("p_mid2", "grants winch override"),
            ("p_mid3", "opens priority ascent"),
            ("p_mid4", "receives final drum clearance"),
            ("p_aux", "carries a safety permit"),
            ("p_alt_mid", "holds a shaft token"),
            ("p_noise1", "passed rope tension test"),
            ("p_noise2", "logged winch telemetry"),
        ),
        "outcomes": ("Deck-N1", "Deck-N2", "Deck-N3", "Deck-N4"),
    },
)
PROXY_DOMAIN_BY_NAME = {d["name"]: d for d in PROXY_OOD_DOMAINS}


def _with_proxy_domains(fn, *args, **kwargs):
    old_a1_dom, old_a1_seed = A1_SYM.DOMAINS, A1_SYM.MASTER_SEED
    old_c1_dom, old_c1_map, old_c1_seed = C1B.C1_DOMAINS, C1B.DOMAIN_BY_NAME, C1B.C1_MASTER_SEED
    A1_SYM.DOMAINS = PROXY_OOD_DOMAINS
    A1_SYM.MASTER_SEED = PROXY_OOD_SEED
    C1B.C1_DOMAINS = PROXY_OOD_DOMAINS
    C1B.DOMAIN_BY_NAME = PROXY_DOMAIN_BY_NAME
    C1B.C1_MASTER_SEED = PROXY_OOD_SEED
    try:
        return fn(*args, **kwargs)
    finally:
        A1_SYM.DOMAINS = old_a1_dom
        A1_SYM.MASTER_SEED = old_a1_seed
        C1B.C1_DOMAINS = old_c1_dom
        C1B.DOMAIN_BY_NAME = old_c1_map
        C1B.C1_MASTER_SEED = old_c1_seed


def verify_domain_disjointness() -> None:
    """Assert zero lexical overlap between PROXY_OOD_DOMAINS and both A1_SYM.DOMAINS & C1B.C1_DOMAINS."""
    existing = list(A1_SYM.DOMAINS) + list(C1B.C1_DOMAINS)
    for field in ("name", "entity_prefix", "guard_name", "metric_name"):
        ex_set = {d[field] for d in existing}
        pr_set = {d[field] for d in PROXY_OOD_DOMAINS}
        assert len(pr_set) == len(PROXY_OOD_DOMAINS), f"Duplicate proxy {field}"
        assert not (ex_set & pr_set), f"Overlap on {field}: {ex_set & pr_set}"
    for seq_field in ("entities", "zones", "outcomes"):
        ex_set = {x for d in existing for x in d[seq_field]}
        pr_set = {x for d in PROXY_OOD_DOMAINS for x in d[seq_field]}
        assert len(pr_set) == sum(len(d[seq_field]) for d in PROXY_OOD_DOMAINS), f"Duplicate proxy {seq_field}"
        assert not (ex_set & pr_set), f"Overlap on {seq_field}: {ex_set & pr_set}"
    ex_props = {txt for d in existing for _, txt in d["props"]}
    pr_props = {txt for d in PROXY_OOD_DOMAINS for _, txt in d["props"]}
    assert not (ex_props & pr_props), f"Predicate overlap: {ex_props & pr_props}"


def run_proxy_shortcut_audits(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Run all 4 shortcut & G0 artifact audits on c1_proxy_ood_bases.jsonl."""
    report: dict[str, Any] = {}

    # 1. (domain, a0) -> at internal majority-class upper bound (chance = 0.333)
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for r in rows:
        counts[(r["domain"], r["initial_answer"])][r["target_answer"]] += 1
    maj_map = {k: v.most_common(1)[0][0] for k, v in counts.items()}
    da_acc = float(np.mean([maj_map[(r["domain"], r["initial_answer"])] == r["target_answer"] for r in rows]))
    assert da_acc <= 0.45, f"(domain, a0)->at shortcut: {da_acc:.4f}"
    report["domain_a0_to_at_acc"] = round(da_acc, 4)

    # 2. cross_entity_transfer/argument_flip entity-match check
    ce_sub = [r for r in rows if r["fallacy_class"] == "cross_entity_transfer" and r["pair_type"] == "argument_flip"]
    correct, total = 0, 0
    for r in ce_sub:
        dom = PROXY_DOMAIN_BY_NAME[r["domain"]]
        ep = dom["entity_prefix"]
        pat = re.compile(rf"\b{re.escape(ep)} [A-Za-z]+-\d+\b")
        for arm, label in (("V", 1), ("F", 0)):
            txt = r["arms"][arm]
            obs_part, concl_part = txt.split("Conclusion:")
            obs_ents = pat.findall(obs_part)
            concl_ents = pat.findall(concl_part)
            assert obs_ents and concl_ents, (r["base_id"], arm)
            pred = 1 if (obs_ents[0] == concl_ents[0]) else 0
            correct += int(pred == label)
            total += 1
    ent_acc = correct / total
    assert 0.45 <= ent_acc <= 0.55, f"Entity-match shortcut: {ent_acc:.4f}"
    report["cross_entity_af_entity_match"] = {"n_pairs": len(ce_sub), "entity_match_acc": round(ent_acc, 4)}

    # 3. threshold_violation/argument_flip single-number obs_val ROC AUC
    th_sub = [r for r in rows if r["fallacy_class"] == "threshold_violation" and r["pair_type"] == "argument_flip"]
    vals, ys = [], []
    for r in th_sub:
        metric = PROXY_DOMAIN_BY_NAME[r["domain"]]["metric_name"]
        pat = re.compile(rf"with a {re.escape(metric)} of (\d+)\.")
        mv = pat.search(r["arms"]["V"])
        mf = pat.search(r["arms"]["F"])
        assert mv and mf, r["base_id"]
        vals.extend([int(mv.group(1)), int(mf.group(1))])
        ys.extend([1, 0])
    th_auc = float(roc_auc_score(ys, vals))
    assert 0.40 <= th_auc <= 0.60, f"threshold_violation numeric shortcut: AUC={th_auc:.4f}"
    report["threshold_af_obs_val_auc"] = {"n_pairs": len(th_sub), "obs_val_auc": round(th_auc, 4)}

    # 4. G0 argument-only TF-IDF GroupKFold audit (two-sided)
    cf = [r for r in rows if r["pair_type"] == "context_flip"]
    af = [r for r in rows if r["pair_type"] == "argument_flip"]
    ov = A1_AUDIT.run_cv_audit(rows, "proxy_ood_all")
    cf_res = A1_AUDIT.run_cv_audit(cf, "proxy_ood_cf")
    af_res = A1_AUDIT.run_cv_audit(af, "proxy_ood_af")
    assert cf_res["exact_turn2_byte_match_rate"] == 1.0
    ov_auc = ov["word_plus_char_ngram_lr"]["oof_roc_auc"]
    af_auc = af_res["word_plus_char_ngram_lr"]["oof_roc_auc"]
    assert 0.38 <= ov_auc <= 0.62, f"G0 overall AUC out of range: {ov_auc}"
    assert 0.35 <= af_auc <= 0.65, f"G0 argument_flip AUC out of range: {af_auc}"
    report["g0_argument_only_audit"] = {
        "overall_combo_auc": ov_auc,
        "cf_byte_match": cf_res["exact_turn2_byte_match_rate"],
        "af_combo_auc": af_auc,
        "af_word_auc": af_res["word_ngram_lr"]["oof_roc_auc"],
    }
    return report


def main() -> None:
    verify_domain_disjointness()
    data_dir = C1 / "data"
    dedup = C1B.SplitDeduplicator()
    C1B.load_a1_rows_into_dedup(dedup)

    # Register existing non-sealed splits ONLY (NEVER read c1_test_id or c1_test_ood!)
    for fn in ("c1_train_bases.jsonl", "c1_dev_bases.jsonl"):
        p = data_dir / fn
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    dedup.register_external(json.loads(line))

    proxy_rows: list[dict[str, Any]] = []
    gidx = 70000
    k = 0
    d2_ce_af_k = 0
    # 2 pair_types x 2 depths x 12 reps x 5 seen classes = 240 bases
    for pair_type in ("context_flip", "argument_flip"):
        for depth in (2, 3):
            for rep in range(12):
                before = ((rep % 2 == 0) ^ (depth == 3) ^ (pair_type == "argument_flip"))
                pos = "before_rules" if before else "after_rules"
                add_lures = (depth == 3)  # 120 std depth-2 + 120 hard depth-3
                for fc in SEEN_CLASSES:
                    k += 1
                    bid = f"C1-POOD-{k:03d}"
                    flip_d2 = False
                    if depth == 2 and fc == "cross_entity_transfer" and pair_type == "argument_flip":
                        flip_d2 = (d2_ce_af_k % 2 == 1)
                        d2_ce_af_k += 1
                    row = _with_proxy_domains(
                        C1B._make_standard_or_hard,
                        gidx,
                        pair_type,
                        depth,
                        fc,
                        pos,
                        add_lures,
                        bid,
                        "proxy_ood",
                        dedup,
                        19,
                        flip_d2,
                    )
                    assert tuple(row["arms"]) == ARM_ORDER
                    proxy_rows.append(row)
                    gidx += 1

    assert len(proxy_rows) == 240
    assert all(r["fallacy_class"] in SEEN_CLASSES and r["difficulty_depth"] in (2, 3) for r in proxy_rows)
    assert all(r["domain"] in PROXY_DOMAIN_BY_NAME for r in proxy_rows)

    audit_report = run_proxy_shortcut_audits(proxy_rows)

    out_jsonl = data_dir / "c1_proxy_ood_bases.jsonl"
    sha = C1B.write_jsonl(proxy_rows, out_jsonl)

    manifest = {
        "schema": "cedar.acl_exp_c1.proxy_ood_manifest.v1",
        "purpose": "Step 2 independent unseen-domain Proxy-OOD evaluation set (strictly disjoint from A1, train, dev, and sealed test_id/test_ood)",
        "master_seed": PROXY_OOD_SEED,
        "sealed_test_sets_touched": False,
        "domain_disjointness_verified": True,
        "domains": [d["name"] for d in PROXY_OOD_DOMAINS],
        "counts": {
            "total_bases": len(proxy_rows),
            "by_pair_type": dict(Counter(r["pair_type"] for r in proxy_rows)),
            "by_depth": dict(Counter(str(r["difficulty_depth"]) for r in proxy_rows)),
            "by_fallacy_class": dict(Counter(r["fallacy_class"] for r in proxy_rows)),
            "by_domain": dict(Counter(r["domain"] for r in proxy_rows)),
        },
        "shortcut_audits": audit_report,
        "sha256": {
            "c1_proxy_ood_bases.jsonl": sha,
        },
    }
    out_manifest = data_dir / "c1_proxy_ood_manifest.json"
    out_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"[OK] Built & audited c1_proxy_ood_bases.jsonl ({len(proxy_rows)} bases, sha256={sha[:16]}...)")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
