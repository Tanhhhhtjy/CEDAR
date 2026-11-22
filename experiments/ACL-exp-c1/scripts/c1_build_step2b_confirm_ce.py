#!/usr/bin/env python3
"""Build the single-chain 2x2 crossed, information-theoretically zero-shortcut Step 2B primary confirmation set (1,728 bases).

Combines BOTH guarantees simultaneously (informed by Astra's audit + Step 2B-Preflight M0 calibration):
  1. Single-chain 4-rule (depth 2) / 8-rule (depth 3) Turn-1 context matching `c1_proxy_ood` distribution,
     so M0 `no_think` Turn-1 accuracy stays at ~70%+ (>= 1,100 / 1,728 eligible bases, well above 800 floor!).
  2. Exact 2x2 crossed counterfactual symmetry per `cell_id` (864 cells x 2 bases = 1,728 bases):
      * C_1: Rule 1 requires `{e1} {p_obs} -> {e1} {p_step1}` (same-entity trigger).
      * C_2: Rule 1 requires `{e2} {p_obs} -> {e1} {p_step1}` (legitimate cross-entity trigger e2 -> e1).
      * Arg_e1: observes `{e1} {p_obs}` and chains Rule 1..d -> `{e1}` override.
      * Arg_e2: observes `{e2} {p_obs}` and chains Rule 1..d -> `{e1}` override.
      * Row A (ctx_V = ctx_F = C_1): V = Arg_e1, F = Arg_e2.
      * Row B (ctx_V = ctx_F = C_2): V = Arg_e2, F = Arg_e1.
  3. Information-theoretic zero-shortcut guarantees (all asserted before saving!):
      - Turn-1 context-only Bayes ceiling == EXACT 0.5000 (`ctx_V == ctx_F` on 1,728/1,728; Astra Rule-1/2
        checker scores 0/1,728 double-correct; Turn-1 TF-IDF GroupKFold AUC == 0.5000).
      - Turn-2 argument-only Bayes ceiling == EXACT 0.5000 (`Counter(V) == Counter(F)`; entity-switch
        counter == 0.5000; Turn-2 TF-IDF GroupKFold AUC == 0.5000).
      - Solver & hop-by-hop proof verified via `verify_step_by_step_proof` + `A1_SYM.evaluate_world`.
      - Zero peek at sealed test sets; zero Turn-2 V/F string overlap with ALL open splits.

Usage:
  python experiments/ACL-exp-c1/scripts/c1_build_step2b_confirm_ce.py
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
C1 = HERE.parents[0]
REPO = C1.parents[1]
A1_SCRIPTS = REPO / "experiments" / "ACL-exp-a1" / "scripts"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(A1_SCRIPTS))

import build_symbolic_pilot_v2 as A1_SYM  # noqa: E402
import c1_build_datasets as C1B  # noqa: E402
import c1_build_proxy_ood as P1  # noqa: E402
from a1_arm_wrappers import ARM_ORDER, build_arms  # noqa: E402

CONFIRM_CE_SEED = 20270423
OPTION_LETTERS = ("A", "B", "C", "D")
ALL_TRANSITIONS = tuple((a, b) for a in OPTION_LETTERS for b in OPTION_LETTERS if a != b)
assert len(ALL_TRANSITIONS) == 12


def _build_single_chain_2x2_cell(
    *,
    cell_idx: int,
    dom_idx: int,
    depth: int,
    pos: str,
    trans: tuple[str, str],
    cell_id: str,
    bid_a: str,
    bid_b: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Construct one single-chain 2x2 crossed cell (Row A on C_1 same-entity, Row B on C_2 cross-entity)."""
    assert depth in (2, 3)
    rng = random.Random(CONFIRM_CE_SEED + cell_idx * 7919)
    dom = P1.PROXY_OOD_DOMAINS[dom_idx]
    p = dict(dom["props"])
    a0, at = trans

    outcomes = list(dom["outcomes"])
    rng.shuffle(outcomes)
    letter_to_outcome = dict(zip(OPTION_LETTERS, outcomes))
    a0_name = letter_to_outcome[a0]
    at_name = letter_to_outcome[at]

    ep = dom["entity_prefix"]
    ents = list(dom["entities"])
    rng.shuffle(ents)
    e1 = f"{ep} {ents[0]}"
    e2 = f"{ep} {ents[1]}"
    e3 = f"{ep} {ents[2]}"
    e4 = f"{ep} {ents[3]}"

    zones = list(dom["zones"])
    rng.shuffle(zones)
    z1, z2 = zones[0], zones[1]

    # Use clean event predicates that have ZERO Turn-2 string overlap with c1_proxy_ood!
    p_obs = p["p_obs"]
    p_trig = p["p_mid2"]
    p_s1 = p["p_mid3"] if depth == 2 else p["p_mid1"]
    p_s2 = p["p_mid4"]  # used in depth 3

    # Exact same 5-fact baseline structure as c1_proxy_ood (preserves ~70%+ M0 no_think Turn-1!)
    fact_lines = [
        f"{e1} is registered under active default status.",
        f"{e1} belongs to {z1}.",
        f"For {e1}, {dom['guard_inactive_text']}.",
        f"{e3} {p['p_noise1']}.",
        f"{e4} {p['p_noise2']}.",
    ]
    base_facts = {
        f"default_active({e1})",
        f"in_zone({e1},{z1})",
        f"guard_inactive({e1})",
        f"noise1({e3})",
        f"noise2({e4})",
    }
    default_policy_text = (
        f"Default policy: {e1} is under active default status, so {e1} is assigned to {a0_name} "
        f"unless the rules establish that {e1} {p_trig}."
    )
    override_trigger = f"trig({e1})"

    def _make_context(trigger_entity: str) -> tuple[str, list[dict[str, Any]], int]:
        if depth == 2:
            rule_lines = [
                f"If {trigger_entity} {p_obs}, then {e1} {p_s1}.",
                f"If {e1} {p_s1}, then {e1} {p_trig}.",
                f"If {e1} {p_trig}, override the default classification and assign {e1} to {at_name}.",
                f"If {e3} {p['p_noise1']} and {e4} {p['p_noise2']}, record a routine audit note for {e3}.",
            ]
            rules = [
                {"id": "Rule 1", "conditions": [f"obs({trigger_entity})"], "conclusion": f"s1({e1})"},
                {"id": "Rule 2", "conditions": [f"s1({e1})"], "conclusion": f"trig({e1})"},
                {"id": "Rule 4", "conditions": [f"noise1({e3})", f"noise2({e4})"], "conclusion": f"audit({e3})"},
            ]
            ov_num = 3
        else:
            rule_lines = [
                f"If {trigger_entity} {p_obs}, then {e1} {p_s1}.",
                f"If {e1} {p_s1}, then {e1} {p_s2}.",
                f"If {e1} {p_s2}, then {e1} {p_trig}.",
                f"If {e1} {p_trig}, override the default classification and assign {e1} to {at_name}.",
                f"If {e3} {p['p_noise1']} and {e4} {p['p_noise2']}, record a routine audit note for {e3}.",
                f"If {e2} {p_obs}, then {e2} is placed on a watch list.",
                f"If {e1} {p_s1}, then {e1} receives a periodic log review.",
                f"For entries in {z2}, if {e1} {p_obs}, then {e1} is queued for archival.",
            ]
            rules = [
                {"id": "Rule 1", "conditions": [f"obs({trigger_entity})"], "conclusion": f"s1({e1})"},
                {"id": "Rule 2", "conditions": [f"s1({e1})"], "conclusion": f"s2({e1})"},
                {"id": "Rule 3", "conditions": [f"s2({e1})"], "conclusion": f"trig({e1})"},
                {"id": "Rule 5", "conditions": [f"noise1({e3})", f"noise2({e4})"], "conclusion": f"audit({e3})"},
                {"id": "Rule 6", "conditions": [f"obs({e2})"], "conclusion": f"watch({e2})"},
                {"id": "Rule 7", "conditions": [f"s1({e1})"], "conclusion": f"review({e1})"},
                {"id": "Rule 8", "conditions": [f"in_zone({e1},{z2})", f"obs({e1})"], "conclusion": f"archive({e1})"},
            ]
            ov_num = 4

        ctx = A1_SYM.render_context_prompt(
            dom, fact_lines, rule_lines, e1, letter_to_outcome, default_policy_text, pos
        )
        return ctx, rules, ov_num

    ctx_1, rules_1, ov_num = _make_context(e1)
    ctx_2, rules_2, _ = _make_context(e2)
    assert ctx_1 != ctx_2

    def _make_arg(obs_entity: str) -> str:
        if depth == 2:
            return (
                f"New verified observation: {obs_entity} {p_obs}. "
                f"Step 1: Since {obs_entity} {p_obs}, Rule 1 implies that {e1} {p_s1}. "
                f"Step 2: Applying Rule 2 to that intermediate status, {e1} {p_trig}. "
                f"Conclusion: Under the override rule (Rule {ov_num}), {e1} is reassigned to {at_name} (Option {at})."
            )
        return (
            f"New verified observation: {obs_entity} {p_obs}. "
            f"Step 1: Since {obs_entity} {p_obs}, Rule 1 implies that {e1} {p_s1}. "
            f"Step 2: Applying Rule 2 to that intermediate status, {e1} {p_s2}. "
            f"Step 3: Applying Rule 3 to that intermediate status, {e1} {p_trig}. "
            f"Conclusion: Under the override rule (Rule {ov_num}), {e1} is reassigned to {at_name} (Option {at})."
        )

    arg_1 = _make_arg(e1)
    arg_2 = _make_arg(e2)
    obs_1, obs_2 = {f"obs({e1})"}, {f"obs({e2})"}
    claimed = [f"s1({e1})", override_trigger] if depth == 2 else [f"s1({e1})", f"s2({e1})", override_trigger]

    # Solver & hop-by-hop verification for ALL 4 combinations of the 2x2 cell!
    _, t1_a = A1_SYM.evaluate_world(base_facts, {}, rules_1, a0_name, override_trigger, at_name)
    _, t1_b = A1_SYM.evaluate_world(base_facts, {}, rules_2, a0_name, override_trigger, at_name)
    assert t1_a == a0_name and t1_b == a0_name

    # C_1 + Arg_1 -> VALID; C_1 + Arg_2 -> INVALID (breaks at Step 1!)
    _, out_11 = A1_SYM.evaluate_world(base_facts | obs_1, {}, rules_1, a0_name, override_trigger, at_name)
    _, out_12 = A1_SYM.evaluate_world(base_facts | obs_2, {}, rules_1, a0_name, override_trigger, at_name)
    assert out_11 == at_name and out_12 == a0_name
    C1B.verify_step_by_step_proof(base_facts, obs_1, rules_1, claimed, expected_broken_step=None)
    C1B.verify_step_by_step_proof(base_facts, obs_2, rules_1, claimed, expected_broken_step=1)

    # C_2 + Arg_2 -> VALID; C_2 + Arg_1 -> INVALID (breaks at Step 1!)
    _, out_22 = A1_SYM.evaluate_world(base_facts | obs_2, {}, rules_2, a0_name, override_trigger, at_name)
    _, out_21 = A1_SYM.evaluate_world(base_facts | obs_1, {}, rules_2, a0_name, override_trigger, at_name)
    assert out_22 == at_name and out_21 == a0_name
    C1B.verify_step_by_step_proof(base_facts, obs_2, rules_2, claimed, expected_broken_step=None)
    C1B.verify_step_by_step_proof(base_facts, obs_1, rules_2, claimed, expected_broken_step=1)

    shell_only = (
        f"New administrative log: routine archive headers were checked. "
        f"Step 1: Since those clerical timestamps appear in the log, the standard display checklist applies. "
        f"Step 2: Applying display formatting conventions to archived entries adds the revised presentation tag. "
        f"Conclusion: Accordingly, {e1} should be listed under {at_name} (Option {at})."
    )
    i0_text = (
        f"Verification note: {e1} remains governed by the default policy with no valid override triggered, "
        f"so Option {a0} remains valid. Please confirm your final answer."
    )

    def _pack_row(bid: str, member: str, ctx: str, core_v: str, core_f: str) -> dict[str, Any]:
        arms = build_arms(
            core_v=core_v,
            core_f=core_f,
            shell_only=shell_only,
            a0=a0,
            at=at,
            i0_text=i0_text,
            final_noun="answer",
        )
        assert tuple(arms) == ARM_ORDER
        for an, atxt in arms.items():
            if an.startswith(("F", "V", "shell_only")):
                assert len(re.findall(rf"\bOption {at}\b", atxt)) == 1, (bid, an)
        return {
            "schema": "cedar.acl_exp_c1.step2b_confirm_ce_2x2.v3",
            "split": "step2b_confirm_ce",
            "base_id": bid,
            "cell_id": cell_id,
            "cell_member": member,
            "family_id": cell_id,
            "domain": dom["name"],
            "pair_type": "crossed_2x2",
            "difficulty_depth": depth,
            "fallacy_class": "cross_entity_transfer",
            "default_policy_position": pos,
            "initial_answer": a0,
            "target_answer": at,
            "letter_to_outcome": letter_to_outcome,
            "context_prompt_v": ctx,
            "context_prompt_f": ctx,
            "context_prompt_f_placebo": ctx,
            "context_prompt_shared_for_non_v": ctx,
            "arms": arms,
        }

    row_a = _pack_row(bid_a, "A", ctx_1, arg_1, arg_2)
    row_b = _pack_row(bid_b, "B", ctx_2, arg_2, arg_1)
    return row_a, row_b


def _run_group_tfidf_auc(texts: list[str], labels: list[int], groups: list[str], ngram_max: int = 2) -> float:
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(texts), dtype=np.float64)
    y = np.array(labels, dtype=np.int64)
    g = np.array(groups)
    for tr, va in gkf.split(texts, y, g):
        vec = TfidfVectorizer(ngram_range=(1, ngram_max), min_df=2, max_features=12000)
        X_tr = vec.fit_transform([texts[i] for i in tr])
        X_va = vec.transform([texts[i] for i in va])
        clf = LogisticRegression(max_iter=1000, C=1.0)
        clf.fit(X_tr, y[tr])
        oof[va] = clf.predict_proba(X_va)[:, 1]
    return float(roc_auc_score(y, oof))


def run_adversarial_audits(rows: list[dict[str, Any]], external_turn2_vf: set[str]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    assert len(rows) == 1728, len(rows)

    by_domain = Counter(r["domain"] for r in rows)
    by_depth = Counter(r["difficulty_depth"] for r in rows)
    by_pos = Counter(r["default_policy_position"] for r in rows)
    by_trans = Counter(f"{r['initial_answer']}->{r['target_answer']}" for r in rows)
    by_cell = Counter(r["cell_id"] for r in rows)
    assert len(by_domain) == 6 and set(by_domain.values()) == {288}, by_domain
    assert by_depth == {2: 864, 3: 864}, by_depth
    assert by_pos == {"before_rules": 864, "after_rules": 864}, by_pos
    assert len(by_trans) == 12 and set(by_trans.values()) == {144}, by_trans
    assert len(by_cell) == 864 and set(by_cell.values()) == {2}

    # (domain, a0) -> at check
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for r in rows:
        counts[(r["domain"], r["initial_answer"])][r["target_answer"]] += 1
    maj_map = {k: v.most_common(1)[0][0] for k, v in counts.items()}
    da_acc = float(np.mean([maj_map[(r["domain"], r["initial_answer"])] == r["target_answer"] for r in rows]))
    assert da_acc <= 0.36
    report["domain_a0_to_at_acc"] = round(da_acc, 4)

    # TURN-1 CONTEXT-ONLY ADVERSARIAL AUDIT
    ctx_eq_rate = float(np.mean([r["context_prompt_v"] == r["context_prompt_f"] for r in rows]))
    assert ctx_eq_rate == 1.0
    ctx_only_dc = 0
    for r in rows:
        def _astra_rule_check(ctx: str) -> str:
            m1 = re.search(r"Rule 1\.\s*If ([^,]+?), then ([^\.]+)\.", ctx)
            assert m1
            e_ante = m1.group(1).split()[:2]
            e_cons = m1.group(2).split()[:2]
            return r["target_answer"] if e_ante == e_cons else r["initial_answer"]

        pv = _astra_rule_check(r["context_prompt_v"])
        pf = _astra_rule_check(r["context_prompt_f"])
        ctx_only_dc += int(pv == r["target_answer"] and pf == r["initial_answer"])
    assert ctx_only_dc == 0

    ctx_texts, ctx_labels, ctx_groups = [], [], []
    for r in rows:
        ctx_texts.extend([r["context_prompt_v"], r["context_prompt_f"]])
        ctx_labels.extend([1, 0])
        ctx_groups.extend([r["cell_id"], r["cell_id"]])
    ctx_tfidf_auc = _run_group_tfidf_auc(ctx_texts, ctx_labels, ctx_groups, ngram_max=2)
    assert 0.45 <= ctx_tfidf_auc <= 0.55
    report["turn1_context_only_audit"] = {
        "exact_turn1_ctx_vf_byte_match": ctx_eq_rate,
        "astra_rule_checker_double_correct": ctx_only_dc,
        "tfidf_groupkfold_auc": round(ctx_tfidf_auc, 4),
    }

    # TURN-2 ARGUMENT-ONLY ADVERSARIAL AUDIT
    assert Counter(r["arms"]["V"] for r in rows) == Counter(r["arms"]["F"] for r in rows)
    by_cid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cid[r["cell_id"]].append(r)
    for cid, pair in by_cid.items():
        assert len(pair) == 2 and pair[0]["arms"]["V"] == pair[1]["arms"]["F"] and pair[0]["arms"]["F"] == pair[1]["arms"]["V"]

    sw_correct, sw_total = 0, 0
    for r in rows:
        dom = P1.PROXY_DOMAIN_BY_NAME[r["domain"]]
        ep = dom["entity_prefix"]
        pat = re.compile(rf"\b{re.escape(ep)} [A-Za-z]+-\d+\b")
        for arm, label in (("V", 1), ("F", 0)):
            ents = pat.findall(r["arms"][arm])
            n_switches = sum(1 for a, b in zip(ents, ents[1:]) if a != b)
            sw_correct += int((n_switches == 0) == bool(label))
            sw_total += 1
    sw_acc = sw_correct / sw_total
    assert abs(sw_acc - 0.5) < 1e-9

    arg_texts, arg_labels, arg_groups = [], [], []
    for r in rows:
        arg_texts.extend([r["arms"]["V"], r["arms"]["F"]])
        arg_labels.extend([1, 0])
        arg_groups.extend([r["cell_id"], r["cell_id"]])
    arg_tfidf_auc = _run_group_tfidf_auc(arg_texts, arg_labels, arg_groups, ngram_max=2)
    assert 0.45 <= arg_tfidf_auc <= 0.55
    report["turn2_argument_only_audit"] = {
        "exact_turn2_vf_multiset_equal": True,
        "entity_switch_counter_acc": round(sw_acc, 4),
        "tfidf_groupkfold_auc": round(arg_tfidf_auc, 4),
    }

    cur_t2 = {r["arms"]["V"] for r in rows} | {r["arms"]["F"] for r in rows}
    t2_overlap = len(cur_t2 & external_turn2_vf)
    assert t2_overlap == 0, f"Turn-2 V/F overlap with open splits: {t2_overlap}"
    report["external_turn2_vf_overlap"] = t2_overlap
    report["counts"] = {
        "total_bases": len(rows),
        "total_2x2_cells": len(by_cell),
        "by_domain": dict(sorted(by_domain.items())),
        "by_depth": {str(k): v for k, v in sorted(by_depth.items())},
        "by_default_policy_position": dict(sorted(by_pos.items())),
        "by_transition": dict(sorted(by_trans.items())),
    }
    return report


def main() -> None:
    P1.verify_domain_disjointness()
    data_dir = C1 / "data"
    dedup = C1B.SplitDeduplicator()
    C1B.load_a1_rows_into_dedup(dedup)

    external_turn2_vf: set[str] = set()
    for fn in (
        "c1_train_bases.jsonl",
        "c1_dev_bases.jsonl",
        "c1_proxy_ood_bases.jsonl",
        "c1_proxy_template_ood_bases.jsonl",
        "c1_proxy_template_v2_bases.jsonl",
    ):
        p = data_dir / fn
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    dedup.register_external(r)
                    external_turn2_vf.add(r["arms"]["V"])
                    external_turn2_vf.add(r["arms"]["F"])

    rows: list[dict[str, Any]] = []
    cell_k = 0
    base_k = 0
    for depth in (2, 3):
        for pos in ("before_rules", "after_rules"):
            for trans in ALL_TRANSITIONS:
                for rep in range(3):
                    for dom_idx in range(len(P1.PROXY_OOD_DOMAINS)):
                        cell_k += 1
                        base_k += 2
                        cid = f"C1-S2BCE-CELL-{cell_k:04d}"
                        bid_a = f"C1-S2BCE-{base_k - 1:04d}"
                        bid_b = f"C1-S2BCE-{base_k:04d}"
                        attempt = 0
                        while True:
                            row_a, row_b = _build_single_chain_2x2_cell(
                                cell_idx=500000 + cell_k * 101 + attempt * 10007,
                                dom_idx=dom_idx,
                                depth=depth,
                                pos=pos,
                                trans=trans,
                                cell_id=cid,
                                bid_a=bid_a,
                                bid_b=bid_b,
                            )
                            if (
                                not dedup.has_collision(row_a)
                                and not dedup.has_collision(row_b)
                                and row_a["arms"]["V"] not in external_turn2_vf
                                and row_a["arms"]["F"] not in external_turn2_vf
                            ):
                                dedup.commit(row_a)
                                dedup.commit(row_b)
                                break
                            attempt += 1
                            assert attempt < 80, f"Too many dedup retries for {cid}"
                        rows.extend([row_a, row_b])

    audit = run_adversarial_audits(rows, external_turn2_vf)

    out_jsonl = data_dir / "c1_step2b_confirm_ce_bases.jsonl"
    out_manifest = data_dir / "c1_step2b_confirm_ce_manifest.json"
    out_jsonl.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    sha = hashlib.sha256(out_jsonl.read_bytes()).hexdigest()
    manifest = {
        "schema": "cedar.acl_exp_c1.step2b_confirm_ce_manifest.v3",
        "design": "single_chain_2x2_crossed_counterfactual_zero_shortcut",
        "seed": CONFIRM_CE_SEED,
        "sealed_test_sets_touched": False,
        "excluded_ood_axes": ["converse_error", "depth_4"],
        "file": str(out_jsonl.relative_to(REPO)).replace("\\", "/"),
        "sha256": sha,
        "audits": audit,
    }
    out_manifest.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"[OK] wrote {out_jsonl} ({len(rows)} bases / 864 2x2 cells, sha256={sha[:12]}...)")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print("\n--- Sample Cell 1 Row A context_prompt_v ---\n" + rows[0]["context_prompt_v"])
    print("\n--- Sample Cell 1 Row A arm V ---\n" + rows[0]["arms"]["V"])
    print("\n--- Sample Cell 1 Row A arm F ---\n" + rows[0]["arms"]["F"])


if __name__ == "__main__":
    main()
