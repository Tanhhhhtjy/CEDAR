#!/usr/bin/env python3
"""Held-Out Post-Hoc Causal Interchange Evaluator for ACL-exp-c3 (Astra's Mechanistic Evidence Upgrade).

Upgrades C2's training-window sign hit rate into true held-out causal mechanistic evidence on `c3_dev` or
`c3_blind_confirm` across `M0`, `Uniform` (`U1..U4`), and `M3-DAC` (`M1..M4`) under IDENTICAL interventions:
  1. `b1`         : B1 frozen `form_only_heads` (for tone-swap invariance) and `ver_only_heads + ver_mlps` (for counterfactual flip).
  2. `rnd_head_1` : Layer-matched disjoint random heads + FIXED B1 `ver_mlps` (pure head-localization test!).
  3. `rnd_full_1` : Disjoint random heads + band-matched disjoint random MLPs (full circuit-localization test!).
  4. `identity`   : Self-patching control (`src == tgt` on B1 `ver_only + ver_mlps`) -> verifies 100.0% exact match & zero hook drift!

For EVERY held-out base, records:
  - Unpatched source & target 4-option argmax answers (`src_orig_pred`, `tgt_orig_pred`) and `u_theta`,
  - Post-interchange 4-option argmax answer (`patch_pred`) and `patch_u`,
  - **Verification Counterfactual Flip** (`F_t <- V_t` & `V_t <- F_t` for `t in {plain, auth, hes}`):
      * `flip_4opt_all`: true 4-option argmax equals `src_gold`,
      * `flip_4opt_both_orig_ok`: true 4-option counterfactual flip conditioned on BOTH `src` and `tgt` originally correct,
      * `flip_sign_hit`: `y_src * patch_u > 0` (logged for transparent comparison with C2's training metric),
      * `mean_signed_delta_u`: `mean(y_src * (patch_u - tgt_orig_u))`.
  - **Rhetoric Tone-Swap Preservation** (`V_hes <-> V_auth`, `F_hes <-> F_auth` on `form_only_heads`):
      * `tone_preserve_same_pred`: `patch_pred == tgt_orig_pred`,
      * `tone_preserve_correct`: `patch_pred == tgt_gold`,
      * `mean_abs_delta_u`: `mean(|patch_u - tgt_orig_u|)`.

CPU Self-Test:
  python experiments/ACL-exp-c3/scripts/c3_eval_interchange.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import torch

HERE = Path(__file__).resolve().parent
C3 = HERE.parents[0]
REPO = C3.parents[1]
C1 = REPO / "experiments" / "ACL-exp-c1"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(C1 / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-a1" / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-b1" / "scripts"))

import b1_lib as B1L  # noqa: E402
import c1_eval as C1EV  # noqa: E402
import c3_eval as C3EV  # noqa: E402
import c3_modules as M3  # noqa: E402
from render_prompts import build_messages, load_config  # noqa: E402

OPTION_LETTERS = ("A", "B", "C", "D")


def _aligned_suffix_len_ids(ids_s: torch.Tensor, ids_t: torch.Tensor, k_max: int = 8) -> int:
    k = 0
    limit = min(k_max, int(ids_s.numel()), int(ids_t.numel()))
    while k < limit and int(ids_s[-1 - k]) == int(ids_t[-1 - k]):
        k += 1
    return max(1, k)


@torch.no_grad()
def _eval_4opt_logits(model, ids: torch.Tensor, opt_tok_ids: dict[str, int], a0: str, at: str) -> dict[str, Any]:
    dev = C1EV._input_device(model)
    last = model(input_ids=ids[None].to(dev), use_cache=False).logits[0, -1].float()
    opt_l = {k: float(last[tid]) for k, tid in opt_tok_ids.items()}
    pred = max(OPTION_LETTERS, key=lambda k: opt_l[k])
    return {"pred": pred, "u": opt_l[at] - opt_l[a0], "opt_logits": opt_l}


@torch.no_grad()
def _interchange_4opt_logits(
    model,
    ids_src: torch.Tensor,
    ids_tgt: torch.Tensor,
    patch_heads: list[tuple[int, int]],
    patch_mlps: list[int],
    head_dim: int,
    opt_tok_ids: dict[str, int],
    a0: str,
    at: str,
    k_max: int = 8,
) -> dict[str, Any]:
    dev = C1EV._input_device(model)
    k = _aligned_suffix_len_ids(ids_src, ids_tgt, k_max=k_max)
    Ls = int(ids_src.numel())
    Lt = int(ids_tgt.numel())
    sl_s = slice(Ls - k, Ls)
    sl_t = slice(Lt - k, Lt)

    heads_by_layer: dict[int, list[int]] = defaultdict(list)
    for l, h in patch_heads:
        heads_by_layer[int(l)].append(int(h))
    mlp_set = {int(l) for l in patch_mlps}

    cached_heads: dict[tuple[int, int], torch.Tensor] = {}
    cached_mlps: dict[int, torch.Tensor] = {}

    # 1. Source pass (no grad)
    with ExitStack() as rec:
        for l, hs in heads_by_layer.items():
            o_mod = model.model.layers[l].self_attn.o_proj

            def _mk_rec_o(li: int, h_list: list[int]):
                def _hk(_m, inp):
                    x = inp[0]
                    for h in h_list:
                        cached_heads[(li, h)] = x[:, sl_s, h * head_dim : (h + 1) * head_dim].detach().clone()
                return _hk

            rec.enter_context(o_mod.register_forward_pre_hook(_mk_rec_o(l, hs)))
        for l in mlp_set:
            mlp_mod = model.model.layers[l].mlp

            def _mk_rec_mlp(li: int):
                def _hk(_m, _inp, out):
                    t = out[0] if isinstance(out, tuple) else out
                    cached_mlps[li] = t[:, sl_s, :].detach().clone()
                return _hk

            rec.enter_context(mlp_mod.register_forward_hook(_mk_rec_mlp(l)))
        _ = model(input_ids=ids_src[None].to(dev), use_cache=False)

    # 2. Patched target pass (no grad)
    with ExitStack() as pat:
        for l, hs in heads_by_layer.items():
            o_mod = model.model.layers[l].self_attn.o_proj

            def _mk_pat_o(li: int, h_list: list[int]):
                def _hk(_m, inp):
                    x = inp[0].clone()
                    for h in h_list:
                        sl_h = slice(h * head_dim, (h + 1) * head_dim)
                        x[:, sl_t, sl_h] = cached_heads[(li, h)].to(x)
                    return (x, *inp[1:])
                return _hk

            pat.enter_context(o_mod.register_forward_pre_hook(_mk_pat_o(l, hs)))
        for l in mlp_set:
            mlp_mod = model.model.layers[l].mlp

            def _mk_pat_mlp(li: int):
                def _hk(_m, _inp, out):
                    t = out[0] if isinstance(out, tuple) else out
                    p = t.clone()
                    p[:, sl_t, :] = cached_mlps[li].to(p)
                    return (p, *out[1:]) if isinstance(out, tuple) else p
                return _hk

            pat.enter_context(mlp_mod.register_forward_hook(_mk_pat_mlp(l)))
        last = model(input_ids=ids_tgt[None].to(dev), use_cache=False).logits[0, -1].float()

    opt_l = {letter: float(last[tid]) for letter, tid in opt_tok_ids.items()}
    pred = max(OPTION_LETTERS, key=lambda letter: opt_l[letter])
    return {"pred": pred, "u": opt_l[at] - opt_l[a0], "opt_logits": opt_l, "k_aligned": k}


def evaluate_interchange_suite(
    model,
    tok,
    a1_cfg: dict[str, Any],
    rows: list[dict[str, Any]],
    iit_targets_map: dict[str, dict[str, Any]],
    head_dim: int,
    opt_tok_ids: dict[str, int],
    k_max: int = 8,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run held-out causal interchange across all targets (`b1`, `rnd_head_1`, `rnd_full_1`, `identity`)."""
    arms_needed = ("V", "F", "V_auth", "V_hes", "F_auth", "F_hes")
    ver_pairs = [
        ("V", "F", +1.0),
        ("F", "V", -1.0),
        ("V_auth", "F_auth", +1.0),
        ("F_auth", "V_auth", -1.0),
        ("V_hes", "F_hes", +1.0),
        ("F_hes", "V_hes", -1.0),
    ]
    form_pairs = [
        ("V_auth", "V_hes", +1.0),
        ("V_hes", "V_auth", +1.0),
        ("F_auth", "F_hes", -1.0),
        ("F_hes", "F_auth", -1.0),
    ]

    # Pre-encode prompts & unpatched 4-option baselines once per base
    encoded_ids: dict[str, dict[str, torch.Tensor]] = {}
    orig_eval: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        bid = row["base_id"]
        a0, at = row["initial_answer"], row["target_answer"]
        encoded_ids[bid] = {}
        orig_eval[bid] = {}
        for arm in arms_needed:
            msgs = build_messages(a1_cfg, row, arm, "no_think")
            ids = B1L.encode_prompt(tok, msgs, "Final:", enable_thinking=False)
            encoded_ids[bid][arm] = ids
            orig_eval[bid][arm] = _eval_4opt_logits(model, ids, opt_tok_ids, a0, at)

    detailed_records: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    for tname, spec in iit_targets_map.items():
        ver_heads = [tuple(x) for x in spec["ver_only_heads"]]
        ver_mlps = [int(x) for x in spec["ver_mlps"]]
        form_heads = [tuple(x) for x in spec["form_only_heads"]]

        v_tot = v_flip_all = v_both_ok = v_flip_both_ok = v_sign_hit = 0
        v_signed_du: list[float] = []
        f_tot = f_same = f_keep_ok = 0
        f_abs_du: list[float] = []

        for row in rows:
            bid = row["base_id"]
            a0, at = row["initial_answer"], row["target_answer"]
            # 1. Counterfactual Verification Flip
            for src_arm, tgt_arm, y_src in ver_pairs:
                src_gold = at if y_src > 0 else a0
                tgt_gold = a0 if y_src > 0 else at
                so = orig_eval[bid][src_arm]
                to = orig_eval[bid][tgt_arm]
                res = _interchange_4opt_logits(
                    model, encoded_ids[bid][src_arm], encoded_ids[bid][tgt_arm],
                    ver_heads, ver_mlps, head_dim, opt_tok_ids, a0, at, k_max=k_max,
                )
                src_ok = so["pred"] == src_gold
                tgt_ok = to["pred"] == tgt_gold
                flip_ok = res["pred"] == src_gold
                sign_ok = (y_src * res["u"]) > 0.0
                s_du = y_src * (res["u"] - to["u"])
                v_tot += 1
                v_flip_all += int(flip_ok)
                v_sign_hit += int(sign_ok)
                v_signed_du.append(s_du)
                if src_ok and tgt_ok:
                    v_both_ok += 1
                    v_flip_both_ok += int(flip_ok)
                detailed_records.append({
                    "target_spec": tname,
                    "kind": "ver_counterfactual",
                    "base_id": bid,
                    "fallacy_class": row["fallacy_class"],
                    "src_arm": src_arm,
                    "tgt_arm": tgt_arm,
                    "src_gold": src_gold,
                    "tgt_gold": tgt_gold,
                    "src_orig_pred": so["pred"],
                    "tgt_orig_pred": to["pred"],
                    "patch_pred": res["pred"],
                    "tgt_orig_u": round(to["u"], 4),
                    "patch_u": round(res["u"], 4),
                    "both_orig_ok": bool(src_ok and tgt_ok),
                    "flip_4opt_ok": bool(flip_ok),
                    "sign_hit": bool(sign_ok),
                })

            # 2. Form/Rhetoric Tone-Swap Preservation
            for src_arm, tgt_arm, y_tgt in form_pairs:
                tgt_gold = at if y_tgt > 0 else a0
                to = orig_eval[bid][tgt_arm]
                res = _interchange_4opt_logits(
                    model, encoded_ids[bid][src_arm], encoded_ids[bid][tgt_arm],
                    form_heads, [], head_dim, opt_tok_ids, a0, at, k_max=k_max,
                )
                same = res["pred"] == to["pred"]
                keep_ok = res["pred"] == tgt_gold
                f_tot += 1
                f_same += int(same)
                f_keep_ok += int(keep_ok)
                f_abs_du.append(abs(res["u"] - to["u"]))
                detailed_records.append({
                    "target_spec": tname,
                    "kind": "form_tone_swap",
                    "base_id": bid,
                    "fallacy_class": row["fallacy_class"],
                    "src_arm": src_arm,
                    "tgt_arm": tgt_arm,
                    "tgt_gold": tgt_gold,
                    "tgt_orig_pred": to["pred"],
                    "patch_pred": res["pred"],
                    "tgt_orig_u": round(to["u"], 4),
                    "patch_u": round(res["u"], 4),
                    "same_pred": bool(same),
                    "keep_correct": bool(keep_ok),
                })

        summaries[tname] = {
            "ver_n_pairs": v_tot,
            "ver_flip_4opt_all": round(v_flip_all / max(1, v_tot), 4),
            "ver_n_both_orig_ok": v_both_ok,
            "ver_flip_4opt_given_both_ok": round(v_flip_both_ok / v_both_ok, 4) if v_both_ok else None,
            "ver_sign_hit_rate": round(v_sign_hit / max(1, v_tot), 4),
            "ver_mean_signed_delta_u": round(sum(v_signed_du) / max(1, len(v_signed_du)), 4),
            "form_n_pairs": f_tot,
            "form_tone_preserve_same_pred": round(f_same / max(1, f_tot), 4),
            "form_tone_preserve_correct": round(f_keep_ok / max(1, f_tot), 4),
            "form_mean_abs_delta_u": round(sum(f_abs_du) / max(1, len(f_abs_du)), 4),
        }

    # 3. Identity Control (`src == tgt` on B1 ver_only + ver_mlps -> MUST be 100% exact!)
    b1_spec = iit_targets_map["b1"]
    b1_vh = [tuple(x) for x in b1_spec["ver_only_heads"]]
    b1_vm = [int(x) for x in b1_spec["ver_mlps"]]
    id_n = 0
    id_match = 0
    id_max_du = 0.0
    for row in rows[: min(20, len(rows))]:
        bid = row["base_id"]
        a0, at = row["initial_answer"], row["target_answer"]
        for arm in ("V", "F"):
            to = orig_eval[bid][arm]
            res = _interchange_4opt_logits(
                model, encoded_ids[bid][arm], encoded_ids[bid][arm],
                b1_vh, b1_vm, head_dim, opt_tok_ids, a0, at, k_max=k_max,
            )
            id_n += 1
            id_match += int(res["pred"] == to["pred"])
            id_max_du = max(id_max_du, abs(res["u"] - to["u"]))
    assert id_match == id_n and id_max_du < 1e-2, f"Identity self-patch control FAILED: {id_match}/{id_n}, max_du={id_max_du}"
    summaries["identity_control"] = {
        "n_checks": id_n,
        "exact_match_rate": round(id_match / max(1, id_n), 4),
        "max_abs_du": round(id_max_du, 6),
    }
    return summaries, detailed_records


def run_selftest_tiny() -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM
    import c1_train as C1T

    cfg = Qwen3Config(
        vocab_size=300, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    )
    tok = C1T._TinyTok()
    a1_cfg = load_config()
    opt_ids = {L: tok.encode(L)[0] for L in OPTION_LETTERS}
    model = Qwen3ForCausalLM(cfg).float().eval()
    rows = [json.loads(l) for l in (C3 / "data" / "c3_dev_bases.jsonl").read_text(encoding="utf-8").splitlines()[:2]]

    iit_map = {
        "b1": {"form_only_heads": [[1, 0]], "ver_only_heads": [[1, 2], [2, 1]], "hub_heads": [[3, 0]], "ver_mlps": [1, 2]},
        "rnd_head_1": {"form_only_heads": [[1, 1]], "ver_only_heads": [[1, 3], [2, 3]], "hub_heads": [[3, 1]], "ver_mlps": [1, 2]},
    }
    orig_enc = B1L.encode_prompt
    def _tiny_enc(t, msgs, pref, enable_thinking=False):
        txt = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking) + pref
        return torch.tensor(t.encode(txt), dtype=torch.long)
    B1L.encode_prompt = _tiny_enc
    try:
        summaries, recs = evaluate_interchange_suite(model, tok, a1_cfg, rows, iit_map, head_dim=16, opt_tok_ids=opt_ids)
    finally:
        B1L.encode_prompt = orig_enc
    assert summaries["identity_control"]["exact_match_rate"] == 1.0
    assert summaries["identity_control"]["max_abs_du"] < 1e-5
    assert len(recs) == 2 * 2 * (6 + 4)  # 2 targets * 2 bases * (6 ver + 4 form)
    print(f"[OK] c3_eval_interchange self-test PASSED: identity={summaries['identity_control']}, n_recs={len(recs)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-tiny", action="store_true")
    ap.add_argument("--model", type=str)
    ap.add_argument("--model-key", choices=["qwen3_8b", "qwen3_14b"], default="qwen3_8b")
    ap.add_argument("--alias", default="model")
    ap.add_argument("--arch", default=None, help="Omit for M0 base model")
    ap.add_argument("--adapter-state", type=Path, default=None)
    ap.add_argument("--components-file", type=Path, default=C3 / "data" / "c3_components_frozen.json")
    ap.add_argument("--dataset", default="c3_blind_confirm", choices=list(C3EV.C3_DATASETS.keys()))
    ap.add_argument("--limit", type=int, default=60, help="Stratified number of held-out bases to intervene on")
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()

    if args.selftest_tiny:
        run_selftest_tiny()
        return

    assert args.dataset not in C3EV.SEALED_DATASETS
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

    # Stratified sample across fallacy classes & broken steps
    all_rows = [json.loads(l) for l in C3EV.C3_DATASETS[args.dataset].read_text(encoding="utf-8").splitlines() if l.strip()]
    by_cls: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in all_rows:
        key = f"{r['fallacy_class']}_{r.get('cell_member', r.get('broken_step', 'x'))}"
        by_cls[key].append(r)
    chosen: list[dict[str, Any]] = []
    keys_sorted = sorted(by_cls.keys())
    idx_ptr = 0
    while len(chosen) < min(args.limit, len(all_rows)):
        progressed = False
        for k in keys_sorted:
            if idx_ptr < len(by_cls[k]) and len(chosen) < args.limit:
                chosen.append(by_cls[k][idx_ptr])
                progressed = True
        if not progressed:
            break
        idx_ptr += 1

    iit_targets = {k: m_info["iit_targets"][k] for k in ("b1", "rnd_head_1", "rnd_full_1")}
    summaries, recs = evaluate_interchange_suite(
        model, tok, a1_cfg, chosen, iit_targets, head_dim=head_dim, opt_tok_ids=opt_ids, k_max=args.k_max
    )

    out_report = {
        "schema": "cedar.acl_exp_c3.heldout_interchange.v1",
        "alias": args.alias,
        "arch": args.arch or "M0",
        "dataset": args.dataset,
        "n_bases": len(chosen),
        "elapsed_seconds": round(time.time() - t0, 1),
        "summaries": summaries,
    }
    (args.out_dir / f"interchange_summary_{args.alias}.json").write_text(
        json.dumps(out_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    (args.out_dir / f"interchange_records_{args.alias}.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in recs) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"[OK] Held-out interchange ({args.alias}, n={len(chosen)}): {json.dumps(summaries, indent=2)}")


if __name__ == "__main__":
    main()
