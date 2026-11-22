#!/usr/bin/env python3
"""Build the 2x2 crossed, information-theoretically zero-shortcut Step 2B primary confirmation set (1,728 bases, CPU only).

Fixes Astra's Turn-1 context-only shortcut finding with a mathematically symmetric 2x2 crossed design:
  - Each atomic 2x2 cell (`cell_id`, 864 cells x 2 bases = 1,728 bases) defines:
      * Two contexts (C_A, C_B) with identical facts, entities (e1 target, e2 companion), options, and
        rule numbering, each containing BOTH one legitimate cross-entity handoff rule (e2 -> e1) AND
        one entity-local non-transfer rule (e2 -> e2).
        - C_A: Branch A transfers e2 -> e1 (valid), Branch B stays e2 -> e2 (invalid).
        - C_B: Branch A stays e2 -> e2 (invalid), Branch B transfers e2 -> e1 (valid).
      * Two cross-entity Turn-2 arguments (Arg_A, Arg_B), BOTH starting on e2 and concluding on e1:
        - Row 1 (context_prompt_v = context_prompt_f = C_A): V = Arg_A, F = Arg_B.
        - Row 2 (context_prompt_v = context_prompt_f = C_B): V = Arg_B, F = Arg_A.
  - Information-theoretic zero-shortcut guarantees (asserted before saving!):
      1. Turn-1 context-only Bayes ceiling == EXACT 0.5000:
         Within EVERY base, `context_prompt_v == context_prompt_f` (1,728 / 1,728 byte-identical!).
         Any Turn-1-only rule (including Astra's Rule-1/2 entity-consistency checker) scores 0 / 1,728 double-correct!
      2. Turn-2 argument-only Bayes ceiling == EXACT 0.5000:
         Across the dataset, `Counter(V) == Counter(F)` (every single Turn-2 argument string appears
         EXACTLY once as V and once as F!). Entity-switch counts, rule numbers, lengths, and TF-IDF
         (GroupKFold by `cell_id`) all sit strictly at chance (~0.500).
      3. Solver & hop-by-hop proof verified via `verify_step_by_step_proof` + `A1_SYM.evaluate_world`.
      4. Zero peek at sealed test sets (`c1_test_id` / `c1_test_ood` NEVER opened); zero overlap of
         contexts OR Turn-2 V/F strings with A1, `c1_train`, `c1_dev`, `c1_proxy_ood`, `c1_proxy_template_ood`.
      5. Exact structural balance: 6 proxy domains (288 each) x 2 depths (864 each) x 2 policy positions
         (864 each) x 12 ordered (a0, at) transitions (144 each, exact `(domain, a0)->at = 0.3333`) x 3 cells x 2 rows = 1,728 bases.

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

CONFIRM_CE_SEED = 20270419
OPTION_LETTERS = ("A", "B", "C", "D")
ALL_TRANSITIONS = tuple((a, b) for a in OPTION_LETTERS for b in OPTION_LETTERS if a != b)
assert len(ALL_TRANSITIONS) == 12


def _build_2x2_cell(
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
    """Construct one exact 2x2 crossed counterfactual cell (Row A on C_A, Row B on C_B)."""
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
    z1 = zones[0]

    # Two completely distinct predicate branches on companion entity e2, converging to shared p_trig = p["p_mid2"]
    p_trig = p["p_mid2"]
    if depth == 2:
        # Branch A: p_obs -> p_mid1 -> p_trig
        # Branch B: p_aux -> p_alt_mid -> p_trig
        br_a = [p["p_obs"], p["p_mid1"]]
        br_b = [p["p_aux"], p["p_alt_mid"]]
    else:
        # Depth 3: 3-hop chains on both branches
        # Branch A: p_obs -> p_mid1 -> p_mid3 -> p_trig
        # Branch B: p_aux -> p_alt_mid -> p_mid4 -> p_trig
        br_a = [p["p_obs"], p["p_mid1"], p["p_mid3"]]
        br_b = [p["p_aux"], p["p_alt_mid"], p["p_mid4"]]

    # Randomize which branch uses the earlier vs later predicate pair so neither branch name correlates with position
    if rng.random() < 0.5:
        br_a, br_b = br_b, br_a

    # Facts (identical across C_A and C_B!)
    fact_lines = [
        f"{e1} is registered under active default status.",
        f"{e1} belongs to {z1}.",
        f"{e2} belongs to {z1}.",
        f"For {e1}, {dom['guard_inactive_text']}.",
    ]
    base_facts = {
        f"default_active({e1})",
        f"in_zone({e1},{z1})",
        f"in_zone({e2},{z1})",
        f"guard_inactive({e1})",
    }
    if depth == 3:
        fact_lines.extend([
            f"{e3} {p['p_noise1']}.",
            f"{e4} {p['p_noise2']}.",
        ])
        base_facts.update({f"noise1({e3})", f"noise2({e4})"})

    default_policy_text = (
        f"Default policy: {e1} is under active default status, so {e1} is assigned to {a0_name} "
        f"unless the rules establish that {e1} {p_trig}."
    )
    override_trigger = f"trig({e1})"

    def _make_rules_and_context(branch_a_transfers: bool) -> tuple[str, list[dict[str, Any]], int]:
        """Build exact solver rules + rendered context prompt.
        branch_a_transfers=True  -> C_A (Branch A ends on e1, Branch B stays on e2)
        branch_a_transfers=False -> C_B (Branch A stays on e2, Branch B ends on e1)
        """
        end_a = e1 if branch_a_transfers else e2
        end_b = e2 if branch_a_transfers else e1
        rule_lines: list[str] = []
        rules: list[dict[str, Any]] = []

        if depth == 2:
            # Rule 1..2: Branch A; Rule 3..4: Branch B; Rule 5: Override
            rule_lines = [
                f"If {e2} {br_a[0]}, then {e2} {br_a[1]}.",
                f"If {e2} {br_a[1]}, then {end_a} {p_trig}.",
                f"If {e2} {br_b[0]}, then {e2} {br_b[1]}.",
                f"If {e2} {br_b[1]}, then {end_b} {p_trig}.",
                f"If {e1} {p_trig}, override the default classification and assign {e1} to {at_name}.",
            ]
            rules = [
                {"id": "Rule 1", "conditions": [f"obs_a({e2})"], "conclusion": f"s1_a({e2})"},
                {"id": "Rule 2", "conditions": [f"s1_a({e2})"], "conclusion": f"trig({end_a})"},
                {"id": "Rule 3", "conditions": [f"obs_b({e2})"], "conclusion": f"s1_b({e2})"},
                {"id": "Rule 4", "conditions": [f"s1_b({e2})"], "conclusion": f"trig({end_b})"},
            ]
            ov_rule_num = 5
        else:
            # Rule 1..3: Branch A; Rule 4..6: Branch B; Rule 7: Override; Rule 8: Distractor lure
            rule_lines = [
                f"If {e2} {br_a[0]}, then {e2} {br_a[1]}.",
                f"If {e2} {br_a[1]}, then {e2} {br_a[2]}.",
                f"If {e2} {br_a[2]}, then {end_a} {p_trig}.",
                f"If {e2} {br_b[0]}, then {e2} {br_b[1]}.",
                f"If {e2} {br_b[1]}, then {e2} {br_b[2]}.",
                f"If {e2} {br_b[2]}, then {end_b} {p_trig}.",
                f"If {e1} {p_trig}, override the default classification and assign {e1} to {at_name}.",
                f"If {e3} {p['p_noise1']} and {e4} {p['p_noise2']}, record a routine audit note for {e3}.",
            ]
            rules = [
                {"id": "Rule 1", "conditions": [f"obs_a({e2})"], "conclusion": f"s1_a({e2})"},
                {"id": "Rule 2", "conditions": [f"s1_a({e2})"], "conclusion": f"s2_a({e2})"},
                {"id": "Rule 3", "conditions": [f"s2_a({e2})"], "conclusion": f"trig({end_a})"},
                {"id": "Rule 4", "conditions": [f"obs_b({e2})"], "conclusion": f"s1_b({e2})"},
                {"id": "Rule 5", "conditions": [f"s1_b({e2})"], "conclusion": f"s2_b({e2})"},
                {"id": "Rule 6", "conditions": [f"s2_b({e2})"], "conclusion": f"trig({end_b})"},
                {"id": "Rule 8", "conditions": [f"noise1({e3})", f"noise2({e4})"], "conclusion": f"audit({e3})"},
            ]
            ov_rule_num = 7

        ctx = A1_SYM.render_context_prompt(
            dom, fact_lines, rule_lines, e1, letter_to_outcome, default_policy_text, pos
        )
        return ctx, rules, ov_rule_num

    ctx_a, rules_a, ov_num = _make_rules_and_context(branch_a_transfers=True)
    ctx_b, rules_b, _ = _make_rules_and_context(branch_a_transfers=False)
    assert ctx_a != ctx_b

    # Build the TWO completely symmetric cross-entity Turn-2 arguments (Arg_A, Arg_B)
    if depth == 2:
        arg_a = (
            f"New verified observation: {e2} {br_a[0]}. "
            f"Step 1: Since {e2} {br_a[0]}, Rule 1 implies that {e2} {br_a[1]}. "
            f"Step 2: Applying Rule 2 to that intermediate status, {e1} {p_trig}. "
            f"Conclusion: Under the override rule (Rule {ov_num}), {e1} is reassigned to {at_name} (Option {at})."
        )
        arg_b = (
            f"New verified observation: {e2} {br_b[0]}. "
            f"Step 1: Since {e2} {br_b[0]}, Rule 3 implies that {e2} {br_b[1]}. "
            f"Step 2: Applying Rule 4 to that intermediate status, {e1} {p_trig}. "
            f"Conclusion: Under the override rule (Rule {ov_num}), {e1} is reassigned to {at_name} (Option {at})."
        )
        obs_a, claimed_a = {f"obs_a({e2})"}, [f"s1_a({e2})", override_trigger]
        obs_b, claimed_b = {f"obs_b({e2})"}, [f"s1_b({e2})", override_trigger]
    else:
        arg_a = (
            f"New verified observation: {e2} {br_a[0]}. "
            f"Step 1: Since {e2} {br_a[0]}, Rule 1 implies that {e2} {br_a[1]}. "
            f"Step 2: Applying Rule 2 to that intermediate status, {e2} {br_a[2]}. "
            f"Step 3: Applying Rule 3 to that intermediate status, {e1} {p_trig}. "
            f"Conclusion: Under the override rule (Rule {ov_num}), {e1} is reassigned to {at_name} (Option {at})."
        )
        arg_b = (
            f"New verified observation: {e2} {br_b[0]}. "
            f"Step 1: Since {e2} {br_b[0]}, Rule 4 implies that {e2} {br_b[1]}. "
            f"Step 2: Applying Rule 5 to that intermediate status, {e2} {br_b[2]}. "
            f"Step 3: Applying Rule 6 to that intermediate status, {e1} {p_trig}. "
            f"Conclusion: Under the override rule (Rule {ov_num}), {e1} is reassigned to {at_name} (Option {at})."
        )
        obs_a, claimed_a = {f"obs_a({e2})"}, [f"s1_a({e2})", f"s2_a({e2})", override_trigger]
        obs_b, claimed_b = {f"obs_b({e2})"}, [f"s1_b({e2})", f"s2_b({e2})", override_trigger]

    # Solver & step-by-step verification for ALL 4 combinations of the 2x2 cell!
    # 1. Turn-1 (no Turn-2 obs): BOTH C_A and C_B derive default a0_name
    _, out_t1_a = A1_SYM.evaluate_world(base_facts, {}, rules_a, a0_name, override_trigger, at_name)
    _, out_t1_b = A1_SYM.evaluate_world(base_facts, {}, rules_b, a0_name, override_trigger, at_name)
    assert out_t1_a == a0_name and out_t1_b == a0_name

    # 2. C_A + Arg_A -> VALID (at_name); C_A + Arg_B -> INVALID (a0_name, broken at final step = depth)
    _, out_aa = A1_SYM.evaluate_world(base_facts | obs_a, {}, rules_a, a0_name, override_trigger, at_name)
    _, out_ab = A1_SYM.evaluate_world(base_facts | obs_b, {}, rules_a, a0_name, override_trigger, at_name)
    assert out_aa == at_name and out_ab == a0_name
    C1B.verify_step_by_step_proof(base_facts, obs_a, rules_a, claimed_a, expected_broken_step=None)
    C1B.verify_step_by_step_proof(base_facts, obs_b, rules_a, claimed_b, expected_broken_step=depth)

    # 3. C_B + Arg_B -> VALID (at_name); C_B + Arg_A -> INVALID (a0_name, broken at final step = depth)
    _, out_bb = A1_SYM.evaluate_world(base_facts | obs_b, {}, rules_b, a0_name, override_trigger, at_name)
    _, out_ba = A1_SYM.evaluate_world(base_facts | obs_a, {}, rules_b, a0_name, override_trigger, at_name)
    assert out_bb == at_name and out_ba == a0_name
    C1B.verify_step_by_step_proof(base_facts, obs_b, rules_b, claimed_b, expected_broken_step=None)
    C1B.verify_step_by_step_proof(base_facts, obs_a, rules_b, claimed_a, expected_broken_step=depth)

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
            "schema": "cedar.acl_exp_c1.step2b_confirm_ce_2x2.v2",
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

    row_a = _pack_row(bid_a, "A", ctx_a, arg_a, arg_b)
    row_b = _pack_row(bid_b, "B", ctx_b, arg_b, arg_a)
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


def run_adversarial_audits(
    rows: list[dict[str, Any]],
    external_turn2_vf: set[str],
) -> dict[str, Any]:
    """Run BOTH Turn-1 context-only AND Turn-2 argument-only adversarial shortcut audits + balance checks."""
    report: dict[str, Any] = {}
    assert len(rows) == 1728, len(rows)

    # 1. Exact structural balance
    by_domain = Counter(r["domain"] for r in rows)
    by_depth = Counter(r["difficulty_depth"] for r in rows)
    by_pos = Counter(r["default_policy_position"] for r in rows)
    by_trans = Counter(f"{r['initial_answer']}->{r['target_answer']}" for r in rows)
    by_cell = Counter(r["cell_id"] for r in rows)
    assert len(by_domain) == 6 and set(by_domain.values()) == {288}, by_domain
    assert by_depth == {2: 864, 3: 864}, by_depth
    assert by_pos == {"before_rules": 864, "after_rules": 864}, by_pos
    assert len(by_trans) == 12 and set(by_trans.values()) == {144}, by_trans
    assert len(by_cell) == 864 and set(by_cell.values()) == {2}, "Each 2x2 cell must have exactly 2 paired rows!"

    # 2. (domain, a0) -> at check (exact chance = 1/3)
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for r in rows:
        counts[(r["domain"], r["initial_answer"])][r["target_answer"]] += 1
    maj_map = {k: v.most_common(1)[0][0] for k, v in counts.items()}
    da_acc = float(np.mean([maj_map[(r["domain"], r["initial_answer"])] == r["target_answer"] for r in rows]))
    assert da_acc <= 0.36, f"(domain, a0)->at shortcut: {da_acc:.4f}"
    report["domain_a0_to_at_acc"] = round(da_acc, 4)

    # 3. TURN-1 CONTEXT-ONLY ADVERSARIAL AUDIT (Astra's exact exploit!)
    # 3a. Exact byte identity of Turn-1 context between V and F within EVERY base
    ctx_eq_rate = float(np.mean([r["context_prompt_v"] == r["context_prompt_f"] for r in rows]))
    assert ctx_eq_rate == 1.0, f"Turn-1 context differs between V and F: {ctx_eq_rate}"

    # 3b. Explicitly run Astra's Rule-1/2 entity-consistency checker on Turn-1 context ONLY:
    # Any deterministic function f(context) outputs the same decision for V and F -> 0/1728 double-correct!
    ctx_only_dc = 0
    for r in rows:
        def _astra_rule_check(ctx: str) -> str:
            # Predicts target_answer if Rule 2 stays same-entity, else initial_answer
            m1 = re.search(r"Rule 1\.\s*If ([^,]+?), then ([^\.]+)\.", ctx)
            m2 = re.search(r"Rule 2\.\s*If ([^,]+?), then ([^\.]+)\.", ctx)
            assert m1 and m2
            e_ante = m1.group(1).split()[:2]
            e_cons = m2.group(2).split()[:2]
            return r["target_answer"] if e_ante == e_cons else r["initial_answer"]

        pv = _astra_rule_check(r["context_prompt_v"])
        pf = _astra_rule_check(r["context_prompt_f"])
        ctx_only_dc += int(pv == r["target_answer"] and pf == r["initial_answer"])
    assert ctx_only_dc == 0, f"Context-only rule achieved {ctx_only_dc} double-correct!"

    # 3c. Turn-1 context-only TF-IDF GroupKFold AUC (grouped by cell_id)
    ctx_texts, ctx_labels, ctx_groups = [], [], []
    for r in rows:
        ctx_texts.extend([r["context_prompt_v"], r["context_prompt_f"]])
        ctx_labels.extend([1, 0])
        ctx_groups.extend([r["cell_id"], r["cell_id"]])
    ctx_tfidf_auc = _run_group_tfidf_auc(ctx_texts, ctx_labels, ctx_groups, ngram_max=2)
    assert 0.45 <= ctx_tfidf_auc <= 0.55, f"Turn-1 context-only TF-IDF AUC out of chance band: {ctx_tfidf_auc}"
    report["turn1_context_only_audit"] = {
        "exact_turn1_ctx_vf_byte_match": ctx_eq_rate,
        "astra_rule_checker_double_correct": ctx_only_dc,
        "tfidf_groupkfold_auc": round(ctx_tfidf_auc, 4),
    }

    # 4. TURN-2 ARGUMENT-ONLY ADVERSARIAL AUDIT
    # 4a. Exact multiset equality between ALL V strings and ALL F strings (Bayes ceiling == EXACT 0.5000!)
    v_counter = Counter(r["arms"]["V"] for r in rows)
    f_counter = Counter(r["arms"]["F"] for r in rows)
    assert v_counter == f_counter, "Turn-2 V and F multisets are NOT identical!"

    # 4b. Verify within EVERY 2x2 cell: rowA.V == rowB.F and rowA.F == rowB.V
    by_cid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_cid[r["cell_id"]].append(r)
    for cid, pair in by_cid.items():
        assert len(pair) == 2
        ra, rb = pair[0], pair[1]
        assert ra["arms"]["V"] == rb["arms"]["F"] and ra["arms"]["F"] == rb["arms"]["V"], cid

    # 4c. Turn-2 entity-switch counter audit (Astra's depth-3 AF exploit)
    sw_correct, sw_total = 0, 0
    for r in rows:
        dom = P1.PROXY_DOMAIN_BY_NAME[r["domain"]]
        ep = dom["entity_prefix"]
        pat = re.compile(rf"\b{re.escape(ep)} [A-Za-z]+-\d+\b")
        for arm, label in (("V", 1), ("F", 0)):
            ents = pat.findall(r["arms"][arm])
            n_switches = sum(1 for a, b in zip(ents, ents[1:]) if a != b)
            # Predict valid iff n_switches == 1 (or any deterministic threshold)
            sw_correct += int((n_switches == 1) == bool(label))
            sw_total += 1
    sw_acc = sw_correct / sw_total
    assert abs(sw_acc - 0.5) < 1e-9, f"Entity-switch shortcut detected: {sw_acc}"

    # 4d. Turn-2 argument-only TF-IDF GroupKFold AUC (grouped by cell_id)
    arg_texts, arg_labels, arg_groups = [], [], []
    for r in rows:
        arg_texts.extend([r["arms"]["V"], r["arms"]["F"]])
        arg_labels.extend([1, 0])
        arg_groups.extend([r["cell_id"], r["cell_id"]])
    arg_tfidf_auc = _run_group_tfidf_auc(arg_texts, arg_labels, arg_groups, ngram_max=2)
    assert 0.45 <= arg_tfidf_auc <= 0.55, f"Turn-2 argument-only TF-IDF AUC out of chance band: {arg_tfidf_auc}"
    report["turn2_argument_only_audit"] = {
        "exact_turn2_vf_multiset_equal": True,
        "entity_switch_counter_acc": round(sw_acc, 4),
        "tfidf_groupkfold_auc": round(arg_tfidf_auc, 4),
    }

    # 5. Zero Turn-2 V/F string overlap with ALL existing open splits
    cur_t2 = {r["arms"]["V"] for r in rows} | {r["arms"]["F"] for r in rows}
    t2_overlap = len(cur_t2 & external_turn2_vf)
    assert t2_overlap == 0, f"Turn-2 V/F string overlap with existing open splits: {t2_overlap}"
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

    # Register existing non-sealed splits ONLY (NEVER open c1_test_id or c1_test_ood!)
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
    # 2 depths x 2 positions x 12 transitions x 3 cell reps x 6 domains = 864 cells x 2 rows = 1,728 bases!
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
                            row_a, row_b = _build_2x2_cell(
                                cell_idx=400000 + cell_k * 101 + attempt * 10007,
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
        "schema": "cedar.acl_exp_c1.step2b_confirm_ce_manifest.v2",
        "design": "2x2_crossed_counterfactual_zero_shortcut",
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
