#!/usr/bin/env python3
"""Stage 1 Probe 2 (Zero-Retraining): Cross-Letter Remapped 7-Way Causal Interchange Challenge.

Directly resolves Astra's crown-jewel mechanistic question:
  "Does patching B1 transmit an **ABSTRACT LOGICAL VALIDITY SIGNAL** (`valid -> switch to receptor's target`,
   `invalid -> stay on receptor's initial`), or does it merely **COPY THE SOURCE'S LITERAL ANSWER LETTER**?"

How the Cross-Letter Remapping Test works:
  For every base problem (with original letters `a0, at`, and unused options `o1, o2`), we build a
  **strictly disjoint deranged receptor copy** mapping `a0 -> o1`, `at -> o2`, `o1 -> a0`, `o2 -> at`!
  - Source valid letter = `at` (e.g. `B`), source invalid letter = `a0` (e.g. `A`).
  - Receptor valid letter = `o2` (e.g. `D`), receptor invalid letter = `o1` (e.g. `C`).
  - Note `{A, B} ∩ {C, D} == ∅`!
  When patching B1 from Source (`V`, gold=`B`) into Receptor (`F`, orig=`C`):
    1. Receptor outputs **`D`** (`tgt_cf_gold`) -> **`abstract_validity_flip`**! (Proves abstract validity routing!)
    2. Receptor outputs **`B`** (`src_lit_gold`) -> **`literal_letter_copy`**! (Proves literal letter copying!)
    3. Receptor outputs **`C`** (`tgt_orig_gold`) -> **`stay_orig`** (no flip)
    4. Receptor outputs **`A`** -> **`other_letter`**.

Runs ALL 7 Causal Component Decompositions across BOTH `same_mapping` AND `cross_letter_remapped`,
stratified across `context_flip`, `argument_flip`, AND `cross_entity_transfer` (Row A + Row B!):
  1. `b1_full`          : 21 B1 ver heads + 5 B1 ver MLPs
  2. `b1_mlp_only`      : ONLY 5 B1 ver MLPs (0 heads)
  3. `b1_heads_only`    : ONLY 21 B1 ver heads (0 MLPs)
  4. `b1_heads_rnd_mlp` : 21 B1 ver heads + 5 band-matched random MLPs
  5. `rnd_heads_b1_mlp` : 21 layer-matched random heads + 5 B1 ver MLPs
  6. `rnd_full`         : 21 random heads + 5 random MLPs
  7. `identity`         : Self-patch control (`src == tgt`, asserts 100.0% exact match & max_du < 1e-2!)

CPU Self-Test:
  python experiments/ACL-exp-c3.5/scripts/c35_eval_cross_letter_interchange.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
C35 = HERE.parents[0]
REPO = C35.parents[1]
C1 = REPO / "experiments" / "ACL-exp-c1"
C3 = REPO / "experiments" / "ACL-exp-c3"

sys.path.insert(0, str(C3 / "scripts"))
sys.path.insert(0, str(C1 / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-a1" / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-b1" / "scripts"))

import b1_lib as B1L  # noqa: E402
import c1_eval as C1EV  # noqa: E402
import c3_eval as C3EV  # noqa: E402
import c3_eval_interchange as IC3  # noqa: E402
import c3_modules as M3  # noqa: E402
from render_prompts import build_messages, load_config  # noqa: E402

OPTION_LETTERS = ("A", "B", "C", "D")


def build_disjoint_remapped_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Return a semantically identical copy of `row` whose `{initial, target}` letters are 100% DISJOINT from `row`."""
    a0, at = row["initial_answer"], row["target_answer"]
    others = [k for k in OPTION_LETTERS if k not in (a0, at)]
    assert len(others) == 2
    o1, o2 = others[0], others[1]
    # Exact involution swapping (a0 <-> o1) and (at <-> o2)!
    perm = {a0: o1, at: o2, o1: a0, o2: at}
    assert {perm[a0], perm[at]}.isdisjoint({a0, at})

    old_l2o = row["letter_to_outcome"]
    new_l2o = {perm[k]: v for k, v in old_l2o.items()}

    def _remap_ctx_options(ctx: str) -> str:
        # Rebuild option lines cleanly in A/B/C/D order matching either `A. Name` or `A) Name`!
        out_lines = []
        for line in ctx.splitlines():
            m_dot = re.match(r"^([A-D])\.\s+.+$", line)
            m_par = re.match(r"^([A-D])\)\s+.+$", line)
            if m_dot:
                k = m_dot.group(1)
                out_lines.append(f"{k}. {new_l2o[k]}")
            elif m_par:
                k = m_par.group(1)
                out_lines.append(f"{k}) {new_l2o[k]}")
            else:
                out_lines.append(line)
        return "\n".join(out_lines)

    def _remap_arm_letters(txt: str) -> str:
        # Remap every `Option <L>` token simultaneously via single-pass regex (zero double-swap!)
        return re.sub(r"\bOption ([A-D])\b", lambda m: f"Option {perm[m.group(1)]}", txt)

    new_row = copy.deepcopy(row)
    new_row["initial_answer"] = perm[a0]
    new_row["target_answer"] = perm[at]
    new_row["letter_to_outcome"] = new_l2o
    for k in ("context_prompt_v", "context_prompt_f", "context_prompt_f_placebo", "context_prompt_shared_for_non_v"):
        if k in new_row:
            new_row[k] = _remap_ctx_options(new_row[k])
    for arm_name, txt in list(new_row["arms"].items()):
        new_row["arms"][arm_name] = _remap_arm_letters(txt)

    # Strict invariants: semantic outcomes of initial & target are IDENTICAL while letters are 100% DISJOINT!
    assert new_row["letter_to_outcome"][new_row["initial_answer"]] == old_l2o[a0]
    assert new_row["letter_to_outcome"][new_row["target_answer"]] == old_l2o[at]
    assert {new_row["initial_answer"], new_row["target_answer"]}.isdisjoint({a0, at})
    return new_row, perm


def build_7way_specs(m_info: dict[str, Any]) -> dict[str, dict[str, Any]]:
    b1 = m_info["iit_targets"]["b1"]
    rh = m_info["iit_targets"]["rnd_head_1"]
    rf = m_info["iit_targets"]["rnd_full_1"]
    return {
        "b1_full": {"ver_heads": b1["ver_only_heads"], "ver_mlps": b1["ver_mlps"], "form_heads": b1["form_only_heads"]},
        "b1_mlp_only": {"ver_heads": [], "ver_mlps": b1["ver_mlps"], "form_heads": []},
        "b1_heads_only": {"ver_heads": b1["ver_only_heads"], "ver_mlps": [], "form_heads": b1["form_only_heads"]},
        "b1_heads_rnd_mlp": {"ver_heads": b1["ver_only_heads"], "ver_mlps": rf["ver_mlps"], "form_heads": b1["form_only_heads"]},
        "rnd_heads_b1_mlp": {"ver_heads": rh["ver_only_heads"], "ver_mlps": b1["ver_mlps"], "form_heads": rh["form_only_heads"]},
        "rnd_full": {"ver_heads": rf["ver_only_heads"], "ver_mlps": rf["ver_mlps"], "form_heads": rf["form_only_heads"]},
    }


def evaluate_cross_letter_suite(
    model,
    tok,
    a1_cfg: dict[str, Any],
    rows: list[dict[str, Any]],
    specs_6: dict[str, dict[str, Any]],
    head_dim: int,
    opt_tok_ids: dict[str, int],
    k_max: int = 8,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run BOTH `same_mapping` AND `cross_letter_remapped` interchange across all 6 decompositions + identity."""
    arms_needed = ("V", "F", "V_auth", "V_hes", "F_auth", "F_hes")
    # 4 balanced validity-flip pairs per base (2 plain + 1 auth + 1 hes) -> 320 pairs/spec on n=80!
    ver_pairs = [
        ("V", "F", +1.0),
        ("F", "V", -1.0),
        ("V_auth", "F_auth", +1.0),
        ("F_hes", "V_hes", -1.0),
    ]

    enc_orig: dict[str, dict[str, torch.Tensor]] = {}
    eval_orig: dict[str, dict[str, dict[str, Any]]] = {}
    enc_remap: dict[str, dict[str, torch.Tensor]] = {}
    eval_remap: dict[str, dict[str, dict[str, Any]]] = {}
    remap_rows: dict[str, dict[str, Any]] = {}

    for row in rows:
        bid = row["base_id"]
        rm_row, _ = build_disjoint_remapped_row(row)
        remap_rows[bid] = rm_row
        enc_orig[bid], eval_orig[bid] = {}, {}
        enc_remap[bid], eval_remap[bid] = {}, {}
        for arm in arms_needed:
            ids_o = B1L.encode_prompt(tok, build_messages(a1_cfg, row, arm, "no_think"), "Final:", enable_thinking=False)
            ids_r = B1L.encode_prompt(tok, build_messages(a1_cfg, rm_row, arm, "no_think"), "Final:", enable_thinking=False)
            enc_orig[bid][arm] = ids_o
            enc_remap[bid][arm] = ids_r
            eval_orig[bid][arm] = IC3._eval_4opt_logits(model, ids_o, opt_tok_ids, row["initial_answer"], row["target_answer"])
            eval_remap[bid][arm] = IC3._eval_4opt_logits(model, ids_r, opt_tok_ids, rm_row["initial_answer"], rm_row["target_answer"])

    # Identity control on b1_full (MUST be 100% exact!)
    b1_vh = [tuple(x) for x in specs_6["b1_full"]["ver_heads"]]
    b1_vm = [int(x) for x in specs_6["b1_full"]["ver_mlps"]]
    id_n = id_ok = 0
    id_max_du = 0.0
    for row in rows[: min(20, len(rows))]:
        bid = row["base_id"]
        for arm in ("V", "F"):
            res = IC3._interchange_4opt_logits(
                model, enc_orig[bid][arm], enc_orig[bid][arm],
                b1_vh, b1_vm, head_dim, opt_tok_ids, row["initial_answer"], row["target_answer"], k_max=k_max,
            )
            to = eval_orig[bid][arm]
            id_n += 1
            id_ok += int(res["pred"] == to["pred"])
            id_max_du = max(id_max_du, abs(res["u"] - to["u"]))
    assert id_ok == id_n and id_max_du < 1e-2, (id_ok, id_n, id_max_du)

    detailed: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {
        "identity_control": {"n_checks": id_n, "exact_match_rate": round(id_ok / max(1, id_n), 4), "max_abs_du": round(id_max_du, 6)},
        "same_mapping": {},
        "cross_letter_remapped": {},
    }

    for mode in ("same_mapping", "cross_letter_remapped"):
        for sname, sp in specs_6.items():
            vh = [tuple(x) for x in sp["ver_heads"]]
            vm = [int(x) for x in sp["ver_mlps"]]
            tot = both_ok = abs_flip_all = abs_flip_ok = lit_copy_ok = stay_ok = other_ok = 0
            signed_du_ok: list[float] = []
            by_pt = defaultdict(lambda: {"both_ok": 0, "abs_flip": 0, "lit_copy": 0})

            for row in rows:
                bid = row["base_id"]
                trow = row if mode == "same_mapping" else remap_rows[bid]
                t_enc = enc_orig[bid] if mode == "same_mapping" else enc_remap[bid]
                t_ev = eval_orig[bid] if mode == "same_mapping" else eval_remap[bid]

                src_a0, src_at = row["initial_answer"], row["target_answer"]
                tgt_a0, tgt_at = trow["initial_answer"], trow["target_answer"]

                for src_arm, tgt_arm, y_src in ver_pairs:
                    src_lit_gold = src_at if y_src > 0 else src_a0
                    tgt_cf_gold = tgt_at if y_src > 0 else tgt_a0      # receptor's OWN counterfactual valid/invalid option!
                    tgt_orig_gold = tgt_a0 if y_src > 0 else tgt_at

                    so = eval_orig[bid][src_arm]
                    to = t_ev[tgt_arm]
                    res = IC3._interchange_4opt_logits(
                        model, enc_orig[bid][src_arm], t_enc[tgt_arm],
                        vh, vm, head_dim, opt_tok_ids, tgt_a0, tgt_at, k_max=k_max,
                    )
                    src_ok = so["pred"] == src_lit_gold
                    tgt_ok = to["pred"] == tgt_orig_gold
                    is_abs_flip = res["pred"] == tgt_cf_gold
                    is_lit_copy = (mode == "cross_letter_remapped") and (res["pred"] == src_lit_gold)
                    is_stay = res["pred"] == tgt_orig_gold

                    tot += 1
                    abs_flip_all += int(is_abs_flip)
                    if src_ok and tgt_ok:
                        both_ok += 1
                        abs_flip_ok += int(is_abs_flip)
                        lit_copy_ok += int(is_lit_copy)
                        stay_ok += int(is_stay)
                        other_ok += int((not is_abs_flip) and (not is_lit_copy) and (not is_stay))
                        signed_du_ok.append(y_src * (res["u"] - to["u"]))
                        ptk = row["pair_type"]
                        by_pt[ptk]["both_ok"] += 1
                        by_pt[ptk]["abs_flip"] += int(is_abs_flip)
                        by_pt[ptk]["lit_copy"] += int(is_lit_copy)

                    detailed.append({
                        "mode": mode,
                        "spec": sname,
                        "base_id": bid,
                        "pair_type": row["pair_type"],
                        "fallacy_class": row["fallacy_class"],
                        "src_arm": src_arm,
                        "tgt_arm": tgt_arm,
                        "src_lit_gold": src_lit_gold,
                        "tgt_cf_gold": tgt_cf_gold,
                        "tgt_orig_gold": tgt_orig_gold,
                        "src_orig_pred": so["pred"],
                        "tgt_orig_pred": to["pred"],
                        "patch_pred": res["pred"],
                        "both_orig_ok": bool(src_ok and tgt_ok),
                        "abstract_validity_flip": bool(is_abs_flip),
                        "literal_letter_copy": bool(is_lit_copy),
                        "tgt_orig_u": round(to["u"], 4),
                        "patch_u": round(res["u"], 4),
                    })

            pt_summ = {
                k: {
                    "n_both_ok": v["both_ok"],
                    "abstract_flip_rate": round(v["abs_flip"] / v["both_ok"], 4) if v["both_ok"] else None,
                    "literal_copy_rate": round(v["lit_copy"] / v["both_ok"], 4) if v["both_ok"] else None,
                }
                for k, v in sorted(by_pt.items())
            }
            summaries[mode][sname] = {
                "n_pairs": tot,
                "n_both_orig_ok": both_ok,
                "abstract_flip_all": round(abs_flip_all / max(1, tot), 4),
                "abstract_flip_given_both_ok": round(abs_flip_ok / both_ok, 4) if both_ok else None,
                "literal_letter_copy_given_both_ok": round(lit_copy_ok / both_ok, 4) if (both_ok and mode == "cross_letter_remapped") else None,
                "stay_orig_given_both_ok": round(stay_ok / both_ok, 4) if both_ok else None,
                "other_given_both_ok": round(other_ok / both_ok, 4) if both_ok else None,
                "mean_signed_delta_u_given_both_ok": round(sum(signed_du_ok) / len(signed_du_ok), 4) if signed_du_ok else None,
                "by_pair_type": pt_summ,
            }

    return summaries, detailed


def select_balanced_probe_rows(all_rows: list[dict[str, Any]], n_cf: int = 30, n_af: int = 30, n_ce: int = 20) -> list[dict[str, Any]]:
    """Stratified selection covering context_flip, argument_flip, AND CE Row A/B cleanly."""
    cf_pool = [r for r in all_rows if r["pair_type"] == "context_flip" and r["fallacy_class"] != "cross_entity_transfer" and not r.get("paired_position_counterfactual")]
    af_pool = [r for r in all_rows if r["pair_type"] == "argument_flip" and r["fallacy_class"] != "cross_entity_transfer" and not r.get("paired_position_counterfactual")]
    ce_a = [r for r in all_rows if r["fallacy_class"] == "cross_entity_transfer" and r.get("cell_member") == "A"]
    ce_b = [r for r in all_rows if r["fallacy_class"] == "cross_entity_transfer" and r.get("cell_member") == "B"]

    def _round_robin(pool: list[dict[str, Any]], take: int) -> list[dict[str, Any]]:
        by_k: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
        for r in pool:
            by_k[(r["fallacy_class"], r.get("broken_step"))].append(r)
        out: list[dict[str, Any]] = []
        ks = sorted(by_k)
        ptr = 0
        while len(out) < min(take, len(pool)):
            prog = False
            for k in ks:
                if ptr < len(by_k[k]) and len(out) < take:
                    out.append(by_k[k][ptr])
                    prog = True
            if not prog:
                break
            ptr += 1
        return out

    return _round_robin(cf_pool, n_cf) + _round_robin(af_pool, n_af) + ce_a[: n_ce // 2] + ce_b[: n_ce // 2]


def run_selftest_tiny() -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM
    import c1_train as C1T

    rows_all = [json.loads(l) for l in (C35 / "data" / "c3_blind_confirm_v2_bases.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    picked = select_balanced_probe_rows(rows_all, n_cf=2, n_af=2, n_ce=2)
    assert len(picked) == 6
    assert sum(r["pair_type"] == "context_flip" for r in picked) == 2
    assert sum(r["pair_type"] == "argument_flip" for r in picked) == 2

    # Verify disjoint cross-letter remapping preserves semantics while making {a0, at} 100% disjoint!
    for r in picked:
        rm, perm = build_disjoint_remapped_row(r)
        assert {r["initial_answer"], r["target_answer"]}.isdisjoint({rm["initial_answer"], rm["target_answer"]})
        assert f"Option {rm['target_answer']}" in rm["arms"]["V"]
        assert f"{rm['target_answer']}. {r['letter_to_outcome'][r['target_answer']]}" in rm["context_prompt_v"] or (
            f"{rm['target_answer']}) {r['letter_to_outcome'][r['target_answer']]}" in rm["context_prompt_v"]
        )

    cfg = Qwen3Config(
        vocab_size=300, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    )
    tok = C1T._TinyTok()
    a1_cfg = load_config()
    opt_ids = {L: tok.encode(L)[0] for L in OPTION_LETTERS}
    model = Qwen3ForCausalLM(cfg).float().eval()
    tiny_specs = {
        "b1_full": {"ver_heads": [[1, 2], [2, 1]], "ver_mlps": [1, 2], "form_heads": [[1, 0]]},
        "b1_mlp_only": {"ver_heads": [], "ver_mlps": [1, 2], "form_heads": []},
        "b1_heads_only": {"ver_heads": [[1, 2], [2, 1]], "ver_mlps": [], "form_heads": [[1, 0]]},
        "b1_heads_rnd_mlp": {"ver_heads": [[1, 2], [2, 1]], "ver_mlps": [0, 3], "form_heads": [[1, 0]]},
        "rnd_heads_b1_mlp": {"ver_heads": [[1, 3], [2, 3]], "ver_mlps": [1, 2], "form_heads": [[1, 1]]},
        "rnd_full": {"ver_heads": [[1, 3], [2, 3]], "ver_mlps": [0, 3], "form_heads": [[1, 1]]},
    }
    orig_enc = B1L.encode_prompt
    def _tiny_enc(t, msgs, pref, enable_thinking=False):
        txt = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking) + pref
        return torch.tensor(t.encode(txt), dtype=torch.long)
    B1L.encode_prompt = _tiny_enc
    try:
        summ, recs = evaluate_cross_letter_suite(model, tok, a1_cfg, picked[:3], tiny_specs, head_dim=16, opt_tok_ids=opt_ids)
    finally:
        B1L.encode_prompt = orig_enc
    assert summ["identity_control"]["exact_match_rate"] == 1.0 and summ["identity_control"]["max_abs_du"] < 1e-5
    assert len(recs) == 2 * 6 * 3 * 4
    print(f"[OK] c35_eval_cross_letter_interchange self-test PASSED: identity={summ['identity_control']}, n_recs={len(recs)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-tiny", action="store_true")
    ap.add_argument("--model", type=str)
    ap.add_argument("--model-key", choices=["qwen3_8b", "qwen3_14b"], default="qwen3_8b")
    ap.add_argument("--alias", default="model")
    ap.add_argument("--arch", default=None, help="Omit for M0 base model")
    ap.add_argument("--adapter-state", type=Path, default=None)
    ap.add_argument("--components-file", type=Path, default=C3 / "data" / "c3_components_frozen.json")
    ap.add_argument("--dataset-file", type=Path, default=C35 / "data" / "c3_blind_confirm_v2_bases.jsonl")
    ap.add_argument("--n-cf", type=int, default=30)
    ap.add_argument("--n-af", type=int, default=30)
    ap.add_argument("--n-ce", type=int, default=20)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()

    if args.selftest_tiny:
        run_selftest_tiny()
        return

    assert args.model and args.out_dir
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    a1_cfg = load_config()
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    opt_ids = C1EV.get_cached_option_token_ids(tok)
    comp = json.loads(args.components_file.read_text(encoding="utf-8"))
    m_info = comp["models"][args.model_key]
    head_dim = int(m_info["model_spec"]["head_dim"])
    specs_6 = build_7way_specs(m_info)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    ).eval()
    if args.adapter_state is not None:
        assert args.arch
        arch_spec = m_info["architectures"][args.arch]
        scaling = C3EV._read_scaling(args.adapter_state)
        M3.apply_c3_adapters(model, arch_spec, **scaling)
        M3.load_c3_adapter_state(model, args.adapter_state)
        model.eval()

    all_rows = [json.loads(l) for l in args.dataset_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    chosen = select_balanced_probe_rows(all_rows, n_cf=args.n_cf, n_af=args.n_af, n_ce=args.n_ce)

    summaries, recs = evaluate_cross_letter_suite(
        model, tok, a1_cfg, chosen, specs_6, head_dim=head_dim, opt_tok_ids=opt_ids, k_max=args.k_max
    )

    out_doc = {
        "schema": "cedar.acl_exp_c35.cross_letter_interchange.v1",
        "alias": args.alias,
        "arch": args.arch or "M0",
        "dataset_file": str(args.dataset_file),
        "n_bases": len(chosen),
        "composition": {"n_cf": args.n_cf, "n_af": args.n_af, "n_ce": args.n_ce},
        "elapsed_seconds": round(time.time() - t0, 1),
        "summaries": summaries,
    }
    (args.out_dir / f"cross_letter_summary_{args.alias}.json").write_text(
        json.dumps(out_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    (args.out_dir / f"cross_letter_records_{args.alias}.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in recs) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"[OK] Cross-letter interchange ({args.alias}, n={len(chosen)}) in {out_doc['elapsed_seconds']}s:")
    for mode in ("same_mapping", "cross_letter_remapped"):
        print(f"  [{mode}]")
        for sn, sv in summaries[mode].items():
            print(
                f"    {sn:18s}: n_ok={sv['n_both_orig_ok']:3d} "
                f"abs_flip={sv['abstract_flip_given_both_ok']} lit_copy={sv['literal_letter_copy_given_both_ok']} "
                f"stay={sv['stay_orig_given_both_ok']} dU={sv['mean_signed_delta_u_given_both_ok']}"
            )


if __name__ == "__main__":
    main()
