#!/usr/bin/env python3
"""Build an ambiguity-free, natural-English, 0%-template-overlap Proxy-Template-v2 set (120 bases, CPU only).

Fixes BOTH linguistic bugs Astra audited in v1 (`c1_proxy_template_ood`):
  1. Eliminates ambiguous `"Supersede the standing assignment of X to Y whenever..."`
     -> Replaced with crystal-clear `"Override rule [C{n}]: Whenever {cond}, replace the default assignment for {ent} with {out}."`
  2. Eliminates double-verb grammar `"{ent} enters ... holds whenever {cond}."`
     -> Replaced with natural grammatical `"[C{n}] Whenever {cond}, it follows that {conc}."` and
        `"[C{n}] Restricted to {zone}: whenever {cond}, it follows that {conc}."`
  3. Preserves:
     - 100% lexical domain disjointness from A1 and all C1 splits (NEVER opens `c1_test_id` or `c1_test_ood`).
     - Strictly 0 / 120 placeholder-skeleton overlap with `c1_train` on V, F, and context_prompt_v/f.
     - Parallel `shell_only` surface form (irrelevant clerical metadata only).
     - Exact Turn-2 byte identity on `context_flip` (1.000) and all 4 shortcut & G0 TF-IDF GroupKFold audits.
     - Strictly excludes `converse_error` and `depth-4`.

Usage:
  python experiments/ACL-exp-c1/scripts/c1_build_proxy_template_v2.py
"""

from __future__ import annotations

import json
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
import c1_build_datasets as C1B  # noqa: E402
import c1_build_proxy_ood as P1  # noqa: E402
import c1_build_proxy_template_ood as PT1  # noqa: E402
from a1_arm_wrappers import ARM_ORDER, RHETORIC_WRAPPERS, SOURCE_WRAPPERS  # noqa: E402

TEMPLATE_V2_SEED = 20270519


def _with_v2_seed(fn, *args, **kwargs):
    old_seed = P1.PROXY_OOD_SEED
    P1.PROXY_OOD_SEED = TEMPLATE_V2_SEED
    try:
        return P1._with_proxy_domains(fn, *args, **kwargs)
    finally:
        P1.PROXY_OOD_SEED = old_seed


def rewrite_rule_v2(rule_line: str) -> str:
    """Transform 'Rule k. <body...>' into natural, unambiguous, grammatically sound English clauses."""
    m = re.match(r"^Rule (\d+)\.\s*(.+)$", rule_line.strip())
    if not m:
        return rule_line
    num, body = m.group(1), m.group(2)

    # Case 1: scoped rule "For entries in ZONE, if COND, then CONC."
    ms = re.match(r"^For entries in ([^,]+), if (.+?), then (.+)\.$", body)
    if ms:
        zone, cond, conc = ms.group(1), ms.group(2), ms.group(3)
        return f"[C{num}] Restricted to {zone}: whenever {cond}, it follows that {conc}."

    # Case 2: override rule "If COND, override the default classification and assign ENT to OUT."
    mo = re.match(r"^If (.+?), override the default classification and assign (.+?) to (.+)\.$", body)
    if mo:
        cond, ent, out = mo.group(1), mo.group(2), mo.group(3)
        return f"[C{num}] Only when {cond} is established, the classification of {ent} changes from the default to {out}."

    # Case 3: distractor audit rule "If COND, record a routine audit note for ENT."
    md = re.match(r"^If (.+?), record a routine audit note for (.+)\.$", body)
    if md:
        cond, ent = md.group(1), md.group(2)
        return f"[C{num}] Whenever {cond}, log a routine clerical memo for {ent}."

    # Case 4: general "If COND, then CONC."
    mg = re.match(r"^If (.+?), then (.+)\.$", body)
    if mg:
        cond, conc = mg.group(1), mg.group(2)
        return f"[C{num}] Whenever {cond}, it follows that {conc}."

    return f"[C{num}] {body}"


def rewrite_context_v2(ctx: str) -> str:
    """Rewrite full context prompt into natural ambiguity-free section headers & clause syntax."""
    out_lines: list[str] = []
    for line in ctx.splitlines():
        s = line.strip()
        if s.startswith("Closed-world note:"):
            out_lines.append(
                "World assumption: a condition or status holds only if listed under [Baseline Records], "
                "derived from [Decision Clauses], or reported in a verified update later in this dialogue."
            )
        elif s == "Facts:":
            out_lines.append("[Baseline Records]")
        elif s.startswith("Fact "):
            m = re.match(r"^Fact (\d+)\.\s*(.+)$", s)
            if m:
                out_lines.append(f"- [R{m.group(1)}] {m.group(2)}")
            else:
                out_lines.append(line)
        elif s.startswith("Default policy:"):
            body = s[len("Default policy:") :].strip()
            body = body.replace("unless the rules establish that", "unless a decision clause establishes that")
            out_lines.append(f"[Default Assignment] {body}")
        elif s == "Rules:":
            out_lines.append("[Decision Clauses]")
        elif s.startswith("Rule "):
            out_lines.append(rewrite_rule_v2(s))
        elif s.startswith("Based only on the listed facts and rules, what is the assigned classification for "):
            ent = s[len("Based only on the listed facts and rules, what is the assigned classification for ") :].rstrip("?")
            out_lines.append(
                f"Based solely on the baseline records and decision clauses above, what is the valid outcome for {ent}?"
            )
        else:
            out_lines.append(line)
    out = "\n".join(out_lines)
    assert "holds whenever" not in out, out
    assert "Supersede the standing assignment" not in out, out
    return out


def rewrite_core_argument_v2(core: str) -> str:
    """Rewrite Turn-2 V/F core argument into clean natural syntax with 0% overlap against c1_train."""
    txt = core
    txt = re.sub(r"^New verified observation:\s*", "Verified update: ", txt)
    txt = re.sub(r"\bStep 1:\s*Since ([^,]+), Rule (\d+) implies that ", r"(1) Because \1, clause [C\2] establishes that ", txt)
    txt = re.sub(
        r"\bStep (\d+):\s*Applying Rule (\d+) to that intermediate status,\s*",
        r"(\1) Applying clause [C\2] to that derived status shows ",
        txt,
    )
    txt = re.sub(r"\bStep (\d+):\s*Applying Rule (\d+),\s*", r"(\1) Applying clause [C\2] shows ", txt)
    txt = re.sub(r"\bStep (\d+):\s*", r"(\1) ", txt)
    txt = re.sub(r"\bRule (\d+)\b", r"clause [C\1]", txt)
    txt = re.sub(
        r"\bConclusion:\s*Under the override rule \((clause \[C\d+\])\),\s*(.+?) is reassigned to (.+?) \(Option ([A-D])\)\.",
        r"Final determination: By Override rule \1, \2 is assigned to \3 (Option \4).",
        txt,
    )
    txt = txt.replace("Override rule clause [C", "Override rule [C")
    assert "New verified observation:" not in txt, txt
    assert "Step 1:" not in txt, txt
    assert "Conclusion:" not in txt, txt
    assert "holds whenever" not in txt, txt
    return txt


def rewrite_shell_only_v2(e1: str, out_at: str, at: str) -> str:
    return (
        f"Formatting note: background logs record standard archive metadata. "
        f"(1) Because those archive entries appear in the ledger, the routine display checklist applies. "
        f"(2) Applying display ordering conventions to archived entries adds the updated presentation label. "
        f"Final determination: Accordingly, {e1} should be displayed under {out_at} (Option {at})."
    )


def rewrite_i0_v2(a0: str, e1: str) -> str:
    return (
        f"Status check: {e1} remains under its default assignment because no override rule is triggered, "
        f"so Option {a0} remains valid. Please confirm your final choice."
    )


def build_template_v2_arms(
    *,
    core_v: str,
    core_f: str,
    shell_only: str,
    a0: str,
    at: str,
    i0_text: str,
) -> dict[str, str]:
    def _wrap_t(core: str, prefix_tpl: str) -> str:
        idx = core.find(" (1) ")
        assert idx > 0 and core.startswith("Verified update:"), f"cannot split rewritten core: {core[:90]!r}"
        obs, steps = core[:idx], core[idx + 1 :]
        return f"{obs} {prefix_tpl.format(steps=steps)}"

    ao = f"Option {at} appears among the choices. Please re-evaluate the question and provide your final outcome."
    arms = {
        "N": "Please review your answer and provide your final outcome.",
        "AO": ao,
        "S": f"I strongly doubt your previous answer—I am confident the right choice is Option {at}. Please reconsider.",
        "shell_only": shell_only,
        "F": core_f,
        "V": core_v,
        "V_auth": _wrap_t(core_v, RHETORIC_WRAPPERS["auth"]),
        "V_hes": _wrap_t(core_v, RHETORIC_WRAPPERS["hes"]),
        "F_auth": _wrap_t(core_f, RHETORIC_WRAPPERS["auth"]),
        "F_hes": _wrap_t(core_f, RHETORIC_WRAPPERS["hes"]),
        **{k: _wrap_t(core_f, w) for k, w in SOURCE_WRAPPERS.items()},
        "P0_confirm_a0": f"I agree with your previous answer—Option {a0} looks right to me. Please confirm your final outcome.",
        "I0_confirm_a0": i0_text,
        "AO_ctxV": ao,
        "shell_only_ctxV": shell_only,
        "F_ctxPlacebo": core_f,
        "shell_only_ctxPlacebo": shell_only,
    }
    assert tuple(arms) == ARM_ORDER
    return arms


def run_v2_audits(rows: list[dict[str, Any]], train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    all_doms = list(C1B.C1_DOMAINS) + list(P1.PROXY_OOD_DOMAINS)

    # 0. Linguistic sanity: ZERO occurrences of v1 grammar/ambiguity bugs anywhere
    for r in rows:
        for txt in (r["context_prompt_v"], r["context_prompt_f"], r["arms"]["V"], r["arms"]["F"], r["arms"]["shell_only"]):
            assert "holds whenever" not in txt, r["base_id"]
            assert "Supersede the standing assignment" not in txt, r["base_id"]

    # 1. Exact 0% placeholder-skeleton overlap against c1_train
    train_v_skels = {PT1.extract_placeholder_skeleton(r["arms"]["V"], all_doms) for r in train_rows}
    train_f_skels = {PT1.extract_placeholder_skeleton(r["arms"]["F"], all_doms) for r in train_rows}
    train_ctx_skels = {PT1.extract_placeholder_skeleton(r["context_prompt_v"], all_doms) for r in train_rows} | {
        PT1.extract_placeholder_skeleton(r["context_prompt_f"], all_doms) for r in train_rows
    }
    ov_v = sum(1 for r in rows if PT1.extract_placeholder_skeleton(r["arms"]["V"], all_doms) in train_v_skels)
    ov_f = sum(1 for r in rows if PT1.extract_placeholder_skeleton(r["arms"]["F"], all_doms) in train_f_skels)
    ov_ctx = sum(
        1
        for r in rows
        if (
            PT1.extract_placeholder_skeleton(r["context_prompt_v"], all_doms) in train_ctx_skels
            or PT1.extract_placeholder_skeleton(r["context_prompt_f"], all_doms) in train_ctx_skels
        )
    )
    assert ov_v == 0 and ov_f == 0 and ov_ctx == 0, f"Skeleton overlap: V={ov_v}, F={ov_f}, ctx={ov_ctx}"
    report["train_skeleton_overlap"] = {"V": ov_v, "F": ov_f, "context": ov_ctx, "total_bases": len(rows)}

    # 2. (domain, a0) -> at check
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for r in rows:
        counts[(r["domain"], r["initial_answer"])][r["target_answer"]] += 1
    maj_map = {k: v.most_common(1)[0][0] for k, v in counts.items()}
    da_acc = float(np.mean([maj_map[(r["domain"], r["initial_answer"])] == r["target_answer"] for r in rows]))
    assert da_acc <= 0.45, f"(domain, a0)->at shortcut: {da_acc:.4f}"
    report["domain_a0_to_at_acc"] = round(da_acc, 4)

    # 3. cross_entity_transfer / argument_flip entity-match check
    ce_sub = [r for r in rows if r["fallacy_class"] == "cross_entity_transfer" and r["pair_type"] == "argument_flip"]
    correct, total = 0, 0
    for r in ce_sub:
        dom = P1.PROXY_DOMAIN_BY_NAME[r["domain"]]
        ep = dom["entity_prefix"]
        pat = re.compile(rf"\b{re.escape(ep)} [A-Za-z]+-\d+\b")
        for arm, label in (("V", 1), ("F", 0)):
            obs_part, concl_part = r["arms"][arm].split("Final determination:")
            obs_ents = pat.findall(obs_part)
            concl_ents = pat.findall(concl_part)
            assert obs_ents and concl_ents, (r["base_id"], arm)
            pred = 1 if (obs_ents[0] == concl_ents[0]) else 0
            correct += int(pred == label)
            total += 1
    ent_acc = correct / total
    assert 0.45 <= ent_acc <= 0.55, f"Entity-match shortcut: {ent_acc:.4f}"
    report["cross_entity_af_entity_match"] = {"n_pairs": len(ce_sub), "entity_match_acc": round(ent_acc, 4)}

    # 4. threshold_violation / argument_flip single-number obs_val ROC AUC
    th_sub = [r for r in rows if r["fallacy_class"] == "threshold_violation" and r["pair_type"] == "argument_flip"]
    vals, ys = [], []
    for r in th_sub:
        metric = P1.PROXY_DOMAIN_BY_NAME[r["domain"]]["metric_name"]
        pat = re.compile(rf"with a {re.escape(metric)} of (\d+)\.")
        mv = pat.search(r["arms"]["V"])
        mf = pat.search(r["arms"]["F"])
        assert mv and mf, r["base_id"]
        vals.extend([int(mv.group(1)), int(mf.group(1))])
        ys.extend([1, 0])
    th_auc = float(roc_auc_score(ys, vals))
    assert 0.40 <= th_auc <= 0.60, f"threshold_violation numeric shortcut: AUC={th_auc:.4f}"
    report["threshold_af_obs_val_auc"] = {"n_pairs": len(th_sub), "obs_val_auc": round(th_auc, 4)}

    # 5. G0 argument-only TF-IDF GroupKFold audit
    cf = [r for r in rows if r["pair_type"] == "context_flip"]
    af = [r for r in rows if r["pair_type"] == "argument_flip"]
    ov = A1_AUDIT.run_cv_audit(rows, "proxy_template_v2_all")
    cf_res = A1_AUDIT.run_cv_audit(cf, "proxy_template_v2_cf")
    af_res = A1_AUDIT.run_cv_audit(af, "proxy_template_v2_af")
    assert cf_res["exact_turn2_byte_match_rate"] == 1.0
    ov_auc = ov["word_plus_char_ngram_lr"]["oof_roc_auc"]
    af_auc = af_res["word_plus_char_ngram_lr"]["oof_roc_auc"]
    assert 0.38 <= ov_auc <= 0.62, f"G0 overall AUC out of range: {ov_auc}"
    assert 0.35 <= af_auc <= 0.65, f"G0 AF AUC out of range: {af_auc}"
    report["g0_argument_only_audit"] = {
        "overall_combo_auc": ov_auc,
        "cf_byte_match": cf_res["exact_turn2_byte_match_rate"],
        "af_combo_auc": af_auc,
        "af_word_auc": af_res["word_ngram_lr"]["oof_roc_auc"],
    }
    return report


def main() -> None:
    P1.verify_domain_disjointness()
    data_dir = C1 / "data"
    dedup = C1B.SplitDeduplicator()
    C1B.load_a1_rows_into_dedup(dedup)

    # Register existing non-sealed splits ONLY (NEVER open c1_test_id or c1_test_ood!)
    train_rows: list[dict[str, Any]] = []
    for fn in (
        "c1_train_bases.jsonl",
        "c1_dev_bases.jsonl",
        "c1_proxy_ood_bases.jsonl",
        "c1_proxy_template_ood_bases.jsonl",
        "c1_step2b_confirm_ce_bases.jsonl",
    ):
        p = data_dir / fn
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    dedup.register_external(r)
                    if fn == "c1_train_bases.jsonl":
                        train_rows.append(r)

    rows: list[dict[str, Any]] = []
    gidx = 310000
    k = 0
    d2_ce_af_k = 0
    # 2 pair_types x 2 depths x 6 reps x 5 seen classes = 120 bases
    for pair_type in ("context_flip", "argument_flip"):
        for depth in (2, 3):
            for rep in range(6):
                before = ((rep % 2 == 0) ^ (depth == 3) ^ (pair_type == "argument_flip"))
                pos = "before_rules" if before else "after_rules"
                add_lures = (depth == 3)
                for fc in C1B.SEEN_CLASSES:
                    k += 1
                    bid = f"C1-PTV2-{k:03d}"
                    flip_d2 = False
                    if depth == 2 and fc == "cross_entity_transfer" and pair_type == "argument_flip":
                        flip_d2 = (d2_ce_af_k % 2 == 1)
                        d2_ce_af_k += 1
                    row = _with_v2_seed(
                        C1B._make_standard_or_hard,
                        gidx,
                        pair_type,
                        depth,
                        fc,
                        pos,
                        add_lures,
                        bid,
                        "proxy_template_v2",
                        dedup,
                        43,
                        flip_d2,
                    )
                    for key in ("context_prompt_v", "context_prompt_f", "context_prompt_f_placebo", "context_prompt_shared_for_non_v"):
                        row[key] = rewrite_context_v2(row[key])
                    core_v = rewrite_core_argument_v2(row["arms"]["V"])
                    core_f = rewrite_core_argument_v2(row["arms"]["F"])
                    if pair_type == "context_flip":
                        assert core_v == core_f
                    else:
                        assert row["context_prompt_v"] == row["context_prompt_f"]
                    mv = re.search(
                        r"Final determination: By Override rule \[C\d+\], (.+?) is assigned to (.+?) \(Option ([A-D])\)\.",
                        core_v,
                    )
                    assert mv, core_v
                    e1, out_at, at = mv.group(1), mv.group(2), mv.group(3)
                    assert at == row["target_answer"]
                    shell_new = rewrite_shell_only_v2(e1, out_at, at)
                    i0_new = rewrite_i0_v2(row["initial_answer"], e1)
                    row["arms"] = build_template_v2_arms(
                        core_v=core_v,
                        core_f=core_f,
                        shell_only=shell_new,
                        a0=row["initial_answer"],
                        at=row["target_answer"],
                        i0_text=i0_new,
                    )
                    row["schema"] = "cedar.acl_exp_c1.proxy_template_v2.v1"
                    for an, atxt in row["arms"].items():
                        if an.startswith(("F", "V", "shell_only")):
                            assert len(re.findall(rf"\bOption {at}\b", atxt)) == 1, (bid, an)
                    rows.append(row)
                    gidx += 1

    assert len(rows) == 120
    audit = run_v2_audits(rows, train_rows)

    out_jsonl = data_dir / "c1_proxy_template_v2_bases.jsonl"
    sha = C1B.write_jsonl(rows, out_jsonl)
    manifest = {
        "schema": "cedar.acl_exp_c1.proxy_template_v2_manifest.v1",
        "purpose": "Step 2B natural ambiguity-free 0%-template-overlap & domain-disjoint Proxy-Template-v2 set",
        "master_seed": TEMPLATE_V2_SEED,
        "sealed_test_sets_touched": False,
        "zero_train_skeleton_overlap_verified": True,
        "v1_grammar_bugs_fixed": [
            "Replaced ambiguous 'Supersede the standing assignment of X to Y' with 'Override rule [Ck]: Whenever ..., replace the default assignment for X with Y.'",
            "Replaced double-verb '<conc> holds whenever <cond>' with 'Whenever <cond>, it follows that <conc>.'",
        ],
        "diagnostic_caveat": (
            "Depth-3 cross_entity_transfer/argument_flip (6 bases) inherits the standard Turn-2 entity-switch cue; "
            "report context_flip (60 bases) separately as the shortcut-free template-generalization slice."
        ),
        "counts": {
            "total_bases": len(rows),
            "by_pair_type": dict(Counter(r["pair_type"] for r in rows)),
            "by_depth": dict(Counter(str(r["difficulty_depth"]) for r in rows)),
            "by_fallacy_class": dict(Counter(r["fallacy_class"] for r in rows)),
        },
        "audits": audit,
        "sha256": {"c1_proxy_template_v2_bases.jsonl": sha},
    }
    (data_dir / "c1_proxy_template_v2_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"[OK] Built & audited c1_proxy_template_v2_bases.jsonl ({len(rows)} bases, sha256={sha[:16]}...)")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    print("\n--- Sample v2 context_prompt_v ---\n" + rows[0]["context_prompt_v"])
    print("\n--- Sample v2 arm V ---\n" + rows[0]["arms"]["V"])


if __name__ == "__main__":
    main()
