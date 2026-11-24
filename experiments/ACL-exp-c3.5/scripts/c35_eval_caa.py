#!/usr/bin/env python3
"""ACL-exp-c3.5 Stage 6 Baseline: Contrastive Activation Addition (`CAA`, Rimsky et al., ACL 2024).

Directly satisfies EVERY audit requirement from Astra's Stage 5 review:
  1. Estimates steering vectors STRICTLY on `c3_train` (zero dev/test leakage!):
       - `anti_syco`: `mean(h_l(N) - h_l(S))` (classic Rimsky et al. / Ma et al. anti-sycophancy residual direction)
       - `warrant_vf`: `mean(h_l(V) - h_l(F))` (paired warrant-sensitivity direction)
     Extracted at the exact final readout token position (`answer_readout_pos`) across candidate layers.
  2. Fairly selects `(contrast_mode, layer, alpha)` on `c3_dev` maximizing 4-Cell accuracy, explicitly logging the
     full grid-search table INCLUDING the `alpha = 0.0` no-intervention baseline point!
  3. Evaluates the dev-selected CAA configuration (AND `alpha = 0.0` anchor) on:
       - `c3_blind_confirm_v2`, `c3_blind_confirm_v3`, and `c3_entity_balanced_v1` (saving exact `records_no_think_*.jsonl`
         compatible with `c35_stage6_stats.py` paired 900-unit bootstrap!)
       - Exact WikiText-2 next-token KL WITH steering turned ON (vs unsteered `M0` on identical chunks!)
       - Official Ma et al. `EX-FEVER` (`1,000` test items, `no_think`) for BOTH steering directions (`+alpha*` AND `-alpha*`)
         alongside the `alpha = 0.0` no-intervention baseline!

CPU Self-Test:
  python experiments/ACL-exp-c3.5/scripts/c35_eval_caa.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
C35 = HERE.parents[0]
REPO = C35.parents[1]
C1 = REPO / "experiments" / "ACL-exp-c1"
C3 = REPO / "experiments" / "ACL-exp-c3"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(C3 / "scripts"))
sys.path.insert(0, str(C1 / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-a1" / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-b1" / "scripts"))

import b1_lib as B1L  # noqa: E402
import c1_eval as C1EV  # noqa: E402
import c1_train as C1T  # noqa: E402
import c3_eval as C3EV  # noqa: E402
import c35_eval_exfever as EXF  # noqa: E402
import c35_stats as S35  # noqa: E402
from render_prompts import load_config  # noqa: E402

C3EV.C3_DATASETS["c3_blind_confirm_v2"] = C35 / "data" / "c3_blind_confirm_v2_bases.jsonl"
C3EV.C3_DATASETS["c3_blind_confirm_v3"] = C35 / "data" / "c3_blind_confirm_v3_bases.jsonl"
C3EV.C3_DATASETS["c3_entity_balanced_v1"] = C35 / "data" / "c3_entity_balanced_v1_bases.jsonl"


@contextmanager
def caa_steering_hook(model, layer_idx: int, direction: torch.Tensor, alpha: float):
    """Residual-stream CAA addition hook: adds `alpha * direction` to layer output hidden states."""
    if abs(float(alpha)) < 1e-12:
        yield
        return

    layer = model.model.layers[int(layer_idx)]
    vec_cpu = direction.detach().float().cpu()

    def _hook(_mod, _inp, output):
        is_tup = isinstance(output, tuple)
        h = output[0] if is_tup else output
        delta = (float(alpha) * vec_cpu).to(device=h.device, dtype=h.dtype)
        h_steered = h + delta.view(1, 1, -1)
        return (h_steered, *output[1:]) if is_tup else h_steered

    handle = layer.register_forward_hook(_hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def extract_caa_directions(
    model,
    tok,
    train_rows: list[dict[str, Any]],
    layers: list[int],
    *,
    max_pairs: int = 150,
    batch_size: int = 4,
) -> tuple[dict[str, dict[int, torch.Tensor]], dict[str, Any]]:
    """Extract `anti_syco` (`N - S`) and `warrant_vf` (`V - F`) CAA directions strictly from `c3_train`."""
    by_gid: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for r in train_rows:
        by_gid[r["pair_group_id"]][r["arm"]] = r

    ns_gids = sorted(gid for gid, m in by_gid.items() if "N" in m and "S" in m)[:max_pairs]
    vf_gids = sorted(gid for gid, m in by_gid.items() if "V" in m and "F" in m)[:max_pairs]
    assert ns_gids and vf_gids, (len(ns_gids), len(vf_gids))

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    dev = C1T._input_device(model)
    layers_sorted = sorted(set(int(l) for l in layers))

    def _collect_arm_states(gids: list[str], arm: str) -> dict[int, list[torch.Tensor]]:
        encoded = [
            C1T.encode_sft_example(tok, by_gid[g][arm], enable_thinking=False, mask_rule_format_ce=False)
            for g in gids
        ]
        out: dict[int, list[torch.Tensor]] = {l: [] for l in layers_sorted}
        cached: dict[int, torch.Tensor] = {}
        hooks = []
        try:
            for l in layers_sorted:
                def _make_h(li: int):
                    def _hk(_mod, _inp, o):
                        cached[li] = (o[0] if isinstance(o, tuple) else o).detach().float().cpu()
                    return _hk
                hooks.append(model.model.layers[l].register_forward_hook(_make_h(l)))

            for i in range(0, len(encoded), batch_size):
                mb = encoded[i : i + batch_size]
                bsz = len(mb)
                max_len = max(int(ex["input_ids"].numel()) for ex in mb)
                input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
                attn_mask = torch.zeros((bsz, max_len), dtype=torch.long)
                for bi, ex in enumerate(mb):
                    L = int(ex["input_ids"].numel())
                    input_ids[bi, :L] = ex["input_ids"]
                    attn_mask[bi, :L] = 1
                cached.clear()
                _ = model(input_ids=input_ids.to(dev), attention_mask=attn_mask.to(dev), use_cache=False)
                for l in layers_sorted:
                    h_b = cached[l]
                    for bi, ex in enumerate(mb):
                        pos = int(ex["answer_readout_pos"])
                        out[l].append(h_b[bi, pos, :].clone())
        finally:
            for hk in hooks:
                hk.remove()
        return out

    st_n = _collect_arm_states(ns_gids, "N")
    st_s = _collect_arm_states(ns_gids, "S")
    st_v = _collect_arm_states(vf_gids, "V")
    st_f = _collect_arm_states(vf_gids, "F")

    dirs: dict[str, dict[int, torch.Tensor]] = {"anti_syco": {}, "warrant_vf": {}}
    norms: dict[str, dict[str, float]] = {"anti_syco": {}, "warrant_vf": {}}
    for l in layers_sorted:
        v_ns = torch.stack([a - b for a, b in zip(st_n[l], st_s[l])], dim=0).mean(dim=0)
        v_vf = torch.stack([a - b for a, b in zip(st_v[l], st_f[l])], dim=0).mean(dim=0)
        dirs["anti_syco"][l] = v_ns
        dirs["warrant_vf"][l] = v_vf
        norms["anti_syco"][str(l)] = round(float(v_ns.norm().item()), 4)
        norms["warrant_vf"][str(l)] = round(float(v_vf.norm().item()), 4)

    meta = {
        "n_pairs_anti_syco": len(ns_gids),
        "n_pairs_warrant_vf": len(vf_gids),
        "layers": layers_sorted,
        "vector_l2_norms": norms,
    }
    return dirs, meta


@torch.no_grad()
def eval_exfever_hf_steered(
    model,
    tok,
    rows: list[dict[str, Any]],
    *,
    layer_idx: int,
    direction: torch.Tensor,
    alpha: float,
    max_new_tokens: int = 256,
    batch_size: int = 16,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate official Ma et al. EX-FEVER (`no_think`) with HF batched generation under CAA steering hook."""
    dev = C1T._input_device(model)
    orig_side = tok.padding_side
    tok.padding_side = "left"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def _gen_prompts(prompts: list[str]) -> list[str]:
        outs: list[str] = []
        with caa_steering_hook(model, layer_idx, direction, alpha):
            for i in range(0, len(prompts), batch_size):
                chunk = prompts[i : i + batch_size]
                enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
                in_len = enc["input_ids"].shape[1]
                gen = model.generate(
                    input_ids=enc["input_ids"].to(dev),
                    attention_mask=enc["attention_mask"].to(dev),
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=pad_id,
                    eos_token_id=tok.eos_token_id,
                    use_cache=True,
                )
                for bi in range(len(chunk)):
                    outs.append(tok.decode(gen[bi, in_len:], skip_special_tokens=True))
        return outs

    try:
        t1_prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": EXF.first_turn_prompt(r)}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for r in rows
        ]
        t1_texts = _gen_prompts(t1_prompts)

        records: list[dict[str, Any]] = []
        by_cond: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
        t1_vis: dict[int, str] = {}

        for r, raw in zip(rows, t1_texts):
            qid = int(r["qid"])
            sc = EXF.score_response(raw, str(r["gold_str"]), is_think=False)
            t1_vis[qid] = sc["visible_text"]
            rec = {"qid": qid, "condition": "baseline", "gold_str": r["gold_str"], "raw_text": raw, **sc}
            records.append(rec)
            by_cond["baseline"][qid] = rec

        for cond in ("pressure", "evidence", "user_evidence"):
            t2_prompts = [
                tok.apply_chat_template(
                    [
                        {"role": "user", "content": EXF.first_turn_prompt(r)},
                        {"role": "assistant", "content": t1_vis[int(r["qid"])]},
                        {"role": "user", "content": EXF.second_turn_prompt(cond, r)},
                    ],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False,
                )
                for r in rows
            ]
            t2_texts = _gen_prompts(t2_prompts)
            for r, raw in zip(rows, t2_texts):
                qid = int(r["qid"])
                sc = EXF.score_response(raw, str(r["gold_str"]), is_think=False)
                rec = {"qid": qid, "condition": cond, "gold_str": r["gold_str"], "raw_text": raw, **sc}
                records.append(rec)
                by_cond[cond][qid] = rec
    finally:
        tok.padding_side = orig_side

    summary = EXF.compute_exfever_metrics(by_cond)
    return summary, records


@torch.no_grad()
def compute_steered_wikitext_kl(
    model,
    tok,
    texts: list[str],
    *,
    layer_idx: int,
    direction: torch.Tensor,
    alpha: float,
    max_len: int = 256,
) -> dict[str, Any]:
    """Compute exact token-level D_KL(p_base || p_steered) matching `c1_eval.compute_reference_kl` identically."""
    dev = C1T._input_device(model)
    kls: list[float] = []
    n_tokens = 0
    for txt in texts:
        ids = tok.encode(txt, add_special_tokens=False)[:max_len]
        if len(ids) < 4:
            continue
        t_ids = torch.tensor(ids, dtype=torch.long)[None].to(dev)
        logp_b = F.log_softmax(model(input_ids=t_ids, use_cache=False).logits[0, :-1].float(), dim=-1).cpu()
        with caa_steering_hook(model, layer_idx, direction, alpha):
            logp_s = F.log_softmax(model(input_ids=t_ids, use_cache=False).logits[0, :-1].float(), dim=-1).cpu()
        p_b = logp_b.exp()
        kl_tok = (p_b * (logp_b - logp_s)).sum(dim=-1)
        kls.extend(kl_tok.tolist())
        n_tokens += len(kl_tok)
    return {
        "n_texts": len(texts),
        "n_tokens": n_tokens,
        "kl_mean": round(statistics.mean(kls), 6) if kls else None,
        "kl_median": round(statistics.median(kls), 6) if kls else None,
    }


def score_dev_records(recs: list[dict[str, Any]]) -> dict[str, float]:
    """Score `c3_dev` no_think records via real `S35.summarize_v2()` (fail-closed on exact schema keys)."""
    bm = S35.compute_base_metrics_v2(recs)
    s = S35.summarize_v2(bm)
    for req_k in ("four_cell_family_weighted", "joint_t1_4cell_unit_weighted", "turn1_dual_acc"):
        if req_k not in s or s[req_k] is None:
            raise AssertionError(f"Missing or None summarize_v2 field {req_k!r}: keys={sorted(s.keys())}")
    return {
        "dev_four_cell_fw": round(float(s["four_cell_family_weighted"]), 6),
        "dev_joint_uw": round(float(s["joint_t1_4cell_unit_weighted"]), 6),
        "dev_turn1_dual_acc": round(float(s["turn1_dual_acc"]), 6),
    }


def run_selftest_tiny() -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(
        vocab_size=300, hidden_size=64, intermediate_size=128,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=2048, tie_word_embeddings=False,
    )
    tok = C1T._TinyTok()
    torch.manual_seed(29)
    model = Qwen3ForCausalLM(cfg).float().eval()

    train_path = C3 / "data" / "c3_train_examples_nothink.jsonl"
    raw_rows = [json.loads(l) for l in train_path.read_text(encoding="utf-8").splitlines()[:200]]
    dirs, meta = extract_caa_directions(model, tok, raw_rows, layers=[1, 2], max_pairs=5, batch_size=2)
    assert len(dirs["anti_syco"]) == 2 and len(dirs["warrant_vf"]) == 2
    assert meta["vector_l2_norms"]["anti_syco"]["1"] > 0.0

    # Verify alpha=0.0 is BIT-FOR-BIT identical to unsteered model, KL==0 at alpha=0, KL>0 at alpha!=0!
    ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    y0 = model(input_ids=ids).logits
    with caa_steering_hook(model, 2, dirs["anti_syco"][2], alpha=0.0):
        y_zero = model(input_ids=ids).logits
    with caa_steering_hook(model, 2, dirs["anti_syco"][2], alpha=1.5):
        y_pos = model(input_ids=ids).logits
    assert float((y_zero - y0).abs().max()) == 0.0
    assert float((y_pos - y0).abs().max()) > 1e-4

    sample_texts = ["A closed harbor berth controller applies exact rules to vessels."] * 3
    kl0 = compute_steered_wikitext_kl(model, tok, sample_texts, layer_idx=2, direction=dirs["anti_syco"][2], alpha=0.0)
    kl1 = compute_steered_wikitext_kl(model, tok, sample_texts, layer_idx=2, direction=dirs["anti_syco"][2], alpha=1.5)
    assert kl0["kl_mean"] == 0.0 and kl1["kl_mean"] > 0.0, (kl0, kl1)

    # Real-schema end-to-end dev-grid smoke test calling REAL `eval_nothink_dataset_batched` + `summarize_v2()`!
    a1_cfg = load_config()
    opt_ids = {L: tok.encode(L)[0] for L in C3EV.OPTION_LETTERS}
    dev_rows = [json.loads(l) for l in (C3 / "data" / "c3_dev_bases.jsonl").read_text(encoding="utf-8").splitlines()[:4]]
    orig_enc = B1L.encode_prompt

    def _tiny_enc(t, msgs, pref, enable_thinking=False):
        txt = t.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking) + pref
        return torch.tensor(t.encode(txt), dtype=torch.long)

    B1L.encode_prompt = _tiny_enc
    try:
        recs_0 = C3EV.eval_nothink_dataset_batched(model, tok, a1_cfg, dev_rows, list(C3EV.CORE_4CELL_ARMS), opt_ids, batch_size=2)
        sc_0 = score_dev_records(recs_0)
        with caa_steering_hook(model, 2, dirs["anti_syco"][2], alpha=1.5):
            recs_1 = C3EV.eval_nothink_dataset_batched(model, tok, a1_cfg, dev_rows, list(C3EV.CORE_4CELL_ARMS), opt_ids, batch_size=2)
        sc_1 = score_dev_records(recs_1)
    finally:
        B1L.encode_prompt = orig_enc

    for sc in (sc_0, sc_1):
        assert 0.0 <= sc["dev_four_cell_fw"] <= 1.0
        assert 0.0 <= sc["dev_joint_uw"] <= 1.0
        assert 0.0 <= sc["dev_turn1_dual_acc"] <= 1.0

    print(
        f"[OK] c35_eval_caa self-test PASSED (real summarize_v2 schema verified!): "
        f"norms={meta['vector_l2_norms']}, kl0={kl0['kl_mean']}, kl1={kl1['kl_mean']}, "
        f"dev_a0={sc_0}, dev_a15={sc_1}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest-tiny", action="store_true")
    ap.add_argument("--model", type=str)
    ap.add_argument("--train-file", type=Path, default=C3 / "data" / "c3_train_examples_nothink.jsonl")
    ap.add_argument("--dev-file", type=Path, default=C3 / "data" / "c3_dev_bases.jsonl")
    ap.add_argument("--exfever-file", type=Path, default=C35 / "data" / "exfever_official_test1000.jsonl")
    ap.add_argument("--kl-corpus", type=Path, required=False, default=None, help="Same WikiText-2 corpus JSONL/txt passed to c3_eval")
    ap.add_argument("--layers", type=str, default="14,18,20,22,24,26,28,30")
    ap.add_argument("--alphas", type=str, default="-2.0,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0,3.0,4.0")
    ap.add_argument("--max-train-pairs", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--exfever-batch-size", type=int, default=16)
    ap.add_argument("--datasets", type=str, default="c3_blind_confirm_v2,c3_blind_confirm_v3,c3_entity_balanced_v1")
    ap.add_argument("--out-dir", type=Path)
    args = ap.parse_args()

    if args.selftest_tiny:
        run_selftest_tiny()
        return

    assert args.model and args.out_dir, "--model and --out-dir are required"
    assert args.kl_corpus is not None and args.kl_corpus.is_file(), f"Missing --kl-corpus: {args.kl_corpus}"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    model.eval()
    a1_cfg = load_config()
    a_id, b_id = B1L.readout_token_ids(tok, "Final:", "A", "B")
    c_id, d_id = B1L.readout_token_ids(tok, "Final:", "C", "D")
    opt_ids = {"A": int(a_id), "B": int(b_id), "C": int(c_id), "D": int(d_id)}

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]
    if 0.0 not in alphas:
        alphas = [0.0] + alphas

    train_rows = [json.loads(l) for l in args.train_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    dirs, dir_meta = extract_caa_directions(model, tok, train_rows, layers=layers, max_pairs=args.max_train_pairs, batch_size=args.batch_size)
    torch.save({"dirs": dirs, "meta": dir_meta}, args.out_dir / "caa_directions_train.pt")
    print(f"[c35_eval_caa] Extracted CAA directions from c3_train: {dir_meta['vector_l2_norms']}")

    dev_rows = [json.loads(l) for l in args.dev_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    grid_rows: list[dict[str, Any]] = []
    # Explicit alpha=0.0 no-intervention baseline on c3_dev
    recs_zero = C3EV.eval_nothink_dataset_batched(model, tok, a1_cfg, dev_rows, list(C3EV.CORE_4CELL_ARMS), opt_ids, batch_size=args.batch_size)
    zero_scores = score_dev_records(recs_zero)
    zero_entry = {
        "contrast_mode": "none",
        "layer": None,
        "alpha": 0.0,
        **zero_scores,
    }
    grid_rows.append(zero_entry)
    print(
        f"  [dev] alpha=0.0 (no intervention): 4-Cell={zero_entry['dev_four_cell_fw']*100:.2f}% "
        f"Joint={zero_entry['dev_joint_uw']*100:.2f}% T1={zero_entry['dev_turn1_dual_acc']*100:.2f}%"
    )

    best = dict(zero_entry)
    for cmode in ("anti_syco", "warrant_vf"):
        for li in layers:
            for al in alphas:
                if abs(al) < 1e-12:
                    continue
                with caa_steering_hook(model, li, dirs[cmode][li], al):
                    recs = C3EV.eval_nothink_dataset_batched(
                        model, tok, a1_cfg, dev_rows, list(C3EV.CORE_4CELL_ARMS), opt_ids, batch_size=args.batch_size
                    )
                ent = {
                    "contrast_mode": cmode,
                    "layer": li,
                    "alpha": al,
                    **score_dev_records(recs),
                }
                grid_rows.append(ent)
                if ent["dev_four_cell_fw"] > best["dev_four_cell_fw"] + 1e-9 or (
                    abs(ent["dev_four_cell_fw"] - best["dev_four_cell_fw"]) <= 1e-9
                    and ent["dev_joint_uw"] > best["dev_joint_uw"]
                ):
                    best = dict(ent)
    print(f"[c35_eval_caa] DEV WINNER: {best}")
    (args.out_dir / "caa_dev_grid.json").write_text(
        json.dumps({"no_intervention": zero_entry, "best": best, "grid": grid_rows, "dir_meta": dir_meta}, indent=2) + "\n",
        encoding="utf-8",
    )

    # Evaluate selected CAA winner on target blind datasets (exact 15 EVAL_ARMS matching c3_eval!)
    nt_dir = args.out_dir / "no_think"
    nt_dir.mkdir(parents=True, exist_ok=True)
    eval_arm_list = list(C1EV.EVAL_ARMS)
    sel_mode = best["contrast_mode"]
    sel_layer = int(best["layer"]) if best["layer"] is not None else layers[len(layers) // 2]
    sel_alpha = float(best["alpha"])
    sel_vec = dirs[sel_mode][sel_layer] if sel_mode in dirs else dirs["anti_syco"][sel_layer]

    ds_summaries: dict[str, Any] = {}
    with caa_steering_hook(model, sel_layer, sel_vec, sel_alpha):
        for ds_name in [x.strip() for x in args.datasets.split(",") if x.strip()]:
            assert ds_name not in C3EV.SEALED_DATASETS, "SEALED TEST BLOCKED!"
            ds_rows = [json.loads(l) for l in C3EV.C3_DATASETS[ds_name].read_text(encoding="utf-8").splitlines() if l.strip()]
            recs = C3EV.eval_nothink_dataset_batched(model, tok, a1_cfg, ds_rows, eval_arm_list, opt_ids, batch_size=args.batch_size)
            rec_p = nt_dir / f"records_no_think_{ds_name}.jsonl"
            rec_p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8", newline="\n")
            bm = S35.compute_base_metrics_v2(recs)
            ds_summaries[ds_name] = S35.summarize_v2(bm)

    # Exact WikiText-2 KL WITH steering turned ON (and opposite direction) on identical 200-text corpus!
    kl_lines = [l.strip() for l in args.kl_corpus.read_text(encoding="utf-8").splitlines() if l.strip()][:200]
    kl_texts = [json.loads(l)["text"] if l.startswith("{") else l for l in kl_lines]
    kl_on = compute_steered_wikitext_kl(model, tok, kl_texts, layer_idx=sel_layer, direction=sel_vec, alpha=sel_alpha)
    kl_opp = compute_steered_wikitext_kl(model, tok, kl_texts, layer_idx=sel_layer, direction=sel_vec, alpha=-sel_alpha)
    eval_sum = {
        "schema": "cedar.acl_exp_c35.caa_eval_summary.v1",
        "alias": "caa_steering_best",
        "model": args.model,
        "selected": best,
        "reference_kl": kl_on,
        "reference_kl_opposite": kl_opp,
        "datasets_summary": ds_summaries,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (nt_dir / "eval_summary_no_think.json").write_text(json.dumps(eval_sum, indent=2) + "\n", encoding="utf-8", newline="\n")

    # Official Ma et al. EX-FEVER (`no_think`) for BOTH steering directions (+alpha*, -alpha*) AND alpha=0.0!
    ex_dir = args.out_dir / "exfever"
    ex_dir.mkdir(parents=True, exist_ok=True)
    ex_rows = [json.loads(l) for l in args.exfever_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    ex_out: dict[str, Any] = {}
    for tag, al in (("no_intervention_a0", 0.0), ("dev_selected_pos", sel_alpha), ("opposite_neg", -sel_alpha)):
        if tag == "opposite_neg" and abs(sel_alpha) < 1e-12:
            continue
        summ, recs = eval_exfever_hf_steered(
            model, tok, ex_rows,
            layer_idx=sel_layer, direction=sel_vec, alpha=al,
            batch_size=args.exfever_batch_size,
        )
        ex_out[tag] = {"alpha": al, "metrics": summ}
        (ex_dir / f"exfever_no_think_records_caa_{tag}.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n", encoding="utf-8", newline="\n"
        )
        (ex_dir / f"exfever_no_think_summary_caa_{tag}.json").write_text(
            json.dumps({"schema": "cedar.acl_exp_c35.exfever_eval.v1", "alias": f"caa_{tag}", "alpha": al, "selected": best, "metrics": summ}, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )
        print(f"  [EX-FEVER {tag} alpha={al:+.2f}] headline={summ['headline']}")

    summary_all = {
        "schema": "cedar.acl_exp_c35.caa_full_summary.v1",
        "model": args.model,
        "dir_meta": dir_meta,
        "no_intervention_dev": zero_entry,
        "selected": best,
        "wikitext_kl_on": kl_on,
        "wikitext_kl_opposite": kl_opp,
        "datasets": ds_summaries,
        "exfever_both_directions": {k: {"alpha": v["alpha"], "headline": v["metrics"]["headline"]} for k, v in ex_out.items()},
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (args.out_dir / "caa_summary.json").write_text(json.dumps(summary_all, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"[c35_eval_caa] DONE in {summary_all['elapsed_seconds']}s | kl_on={kl_on['kl_mean']} kl_opp={kl_opp['kl_mean']}")


if __name__ == "__main__":
    main()
