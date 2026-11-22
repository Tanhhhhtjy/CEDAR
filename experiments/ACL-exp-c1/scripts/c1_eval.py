#!/usr/bin/env python3
"""ACL-exp-c1 evaluation & KL-divergence harness (HF transformers / vLLM + CPU self-test).

Evaluates any base or merged C1 checkpoint (or live unmerged adapter_state.pt with pathway ablations) on:
  1. Dedicated Dev (`c1_dev`, 80 bases across 5 seen classes — Phase-0 selection only)
  2. Held-out ID (`c1_test_id`, 100 bases across 5 seen classes)
  3. Held-out OOD (`c1_test_ood`, 120 bases: 60 unseen converse_error + 60 true depth-4)
  4. Optional Historical Diagnostic (`natural_probe`: A1 dev_only 12 natural items — opt-in only, NOT primary held-out)
  5. Reference Next-Token KL Divergence D_KL(p_base || p_tuned) on an explicit frozen external corpus (wikitext-2)

Key guarantees (audited):
  - Fixed a1_parse.parse_output(raw_out, requires_verdict=False, hit_token_limit=...) returning dict keys
    `parsed_final_option` and `parse_status`; reports per-arm `parse_fail_rate` and `truncation_rate`
  - Evaluates BOTH `turn1_F` and `turn1_V` and supports freezing/reusing M0's dual-sided Turn-1 eligible denominator
    via `--eligible-ids-out` / `--eligible-ids-in` (reporting both full-set and shared-eligible summaries + post-tuning Turn-1 acc)
  - Supports `--backend vllm` for fast batched `think` generation (default max_new_tokens=3072)
  - Caches readout option token IDs once per tokenizer
  - Fails fast if reference KL is requested without an existing external `--kl-corpus`

Usage (GPU no_think):
  CUDA_VISIBLE_DEVICES=0 python experiments/ACL-exp-c1/scripts/c1_eval.py \
      --model checkpoints/qwen3_8b/Targeted-Dual/merged_hf --base-model /mnt/kcl/devuser/ckpts/Qwen3-8B \
      --kl-corpus experiments/ACL-exp-c1/data/c1_kl_wikitext2_200.jsonl \
      --eligible-ids-in results/qwen3_8b/M0/eligible_ids_no_think.json \
      --alias qwen3_8b__Targeted-Dual --regime no_think --out-dir results/qwen3_8b/Targeted-Dual

Usage (GPU think via vLLM):
  CUDA_VISIBLE_DEVICES=0 python experiments/ACL-exp-c1/scripts/c1_eval.py \
      --model checkpoints/qwen3_8b/Targeted-Dual/merged_hf --backend vllm --regime think \
      --max-new-tokens 3072 --eligible-ids-in results/qwen3_8b/M0/eligible_ids_think.json \
      --alias qwen3_8b__Targeted-Dual --out-dir results/qwen3_8b/Targeted-Dual

Usage (CPU self-test):
  python experiments/ACL-exp-c1/scripts/c1_eval.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
C1 = HERE.parents[0]
REPO = C1.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-a1" / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-b1" / "scripts"))

import a1_parse as P  # noqa: E402
import b1_lib as B1L  # noqa: E402
import c1_modules as M  # noqa: E402
from render_prompts import build_messages, load_config  # noqa: E402

EVAL_ARMS = [
    "turn1_F",
    "turn1_V",
    "N",
    "AO",
    "S",
    "shell_only",
    "F",
    "F_auth",
    "F_hes",
    "F_src_third_party",
    "V",
    "V_auth",
    "V_hes",
    "P0_confirm_a0",
    "I0_confirm_a0",
]

DEFAULT_DATASETS = {
    "c1_dev": C1 / "data" / "c1_dev_bases.jsonl",
    "c1_proxy_ood": C1 / "data" / "c1_proxy_ood_bases.jsonl",
    "c1_proxy_template_ood": C1 / "data" / "c1_proxy_template_ood_bases.jsonl",
    "c1_test_id": C1 / "data" / "c1_test_id_bases.jsonl",
    "c1_test_ood": C1 / "data" / "c1_test_ood_bases.jsonl",
    # Historical A1 dev_only diagnostic (opt-in only via --datasets natural_probe)
    "natural_probe": REPO / "experiments" / "ACL-exp-a1" / "data" / "natural_pilot_seed_12_v2_2.jsonl",
}


def _input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def get_cached_option_token_ids(tok) -> dict[str, int]:
    """Resolve and cache option token IDs ('A','B','C','D') after 'Final:' once per tokenizer."""
    return {
        letter: B1L.readout_token_ids(tok, "Final:", letter, "B" if letter == "A" else "A")[0]
        for letter in ("A", "B", "C", "D")
    }


def parse_think_generation(raw_out: str, hit_token_limit: bool) -> dict:
    """Wrapper around A1's reference parser for Final-only think outputs."""
    parsed = P.parse_output(raw_out, requires_verdict=False, hit_token_limit=hit_token_limit)
    return {
        "pred": parsed["parsed_final_option"],
        "parse_status": parsed["parse_status"],
        "parse_ok": parsed["parse_status"] == "ok",
        "truncated": parsed["parse_status"] == "truncated",
    }


@torch.no_grad()
def eval_nothink_item(model, tok, a1_cfg: dict, row: dict, arm: str, opt_tok_ids: dict[str, int]) -> dict:
    """Fast exact single-forward no_think evaluation: computes u_theta AND argmax option after 'Final:'."""
    msgs = build_messages(a1_cfg, row, arm, "no_think")
    ids = B1L.encode_prompt(tok, msgs, "Final:", enable_thinking=False)
    out = model(input_ids=ids[None].to(_input_device(model)))
    last = out.logits[0, -1].float()

    opt_logits = {letter: float(last[tid]) for letter, tid in opt_tok_ids.items()}
    pred = max(opt_logits, key=opt_logits.get)
    a0, at = row["initial_answer"], row["target_answer"]
    u_theta = opt_logits[at] - opt_logits[a0]
    return {
        "base_id": row["base_id"],
        "split": row.get("split", "unknown"),
        "pair_type": row["pair_type"],
        "fallacy_class": row["fallacy_class"],
        "difficulty_depth": row["difficulty_depth"],
        "arm": arm,
        "initial_answer": a0,
        "target_answer": at,
        "pred": pred,
        "parse_status": "ok",
        "parse_ok": True,
        "truncated": False,
        "is_target": pred == at,
        "is_initial": pred == a0,
        "u_theta": u_theta,
    }


def resolve_hub_heads(comp: dict, model_key: str) -> set[tuple[int, int]]:
    """Resolve frozen hub heads from c1_components_frozen.json (schema: models[model_key]['hub_heads'])."""
    m_info = comp["models"][model_key]
    raw = m_info["hub_heads"]
    return {(int(li), int(h)) for li, h in raw}


def checkpoint_scaling(state_path: Path, model_key: str, arm_name: str, arm_spec: dict) -> dict:
    config_path = state_path.with_name("adapter_config.json")
    meta_path = state_path.with_name("train_meta.json")
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        scaling = config["scaling"]
    elif meta_path.is_file():
        config = json.loads(meta_path.read_text(encoding="utf-8"))
        scaling = config["hyperparams"]
    else:
        return {}  # A standalone legacy state has no recorded scale. Use the protocol defaults.
    if (config["model_key"], config["arm"], config["arm_spec"]) != (model_key, arm_name, arm_spec):
        raise ValueError(f"Checkpoint arm or mask does not match {model_key}/{arm_name}: {state_path}")
    return {k: scaling[k] for k in ("scaling_mode", "base_scale", "anchor_rank")}


def generation_record(row, arm, text, n_tokens, finish_reason, *, enable_thinking):
    hit_limit = finish_reason == "length"
    if enable_thinking:
        parsed = parse_think_generation(text, hit_token_limit=hit_limit)
    else:
        # The no_think prompt requires exactly one Final line and a completed response.
        match = re.fullmatch(r"Final: ([A-D])(?:\r?\n)?", text)
        pred = match[1] if match and finish_reason == "stop" else None
        parsed = {
            "pred": pred,
            "parse_ok": pred is not None,
            "parse_status": "ok" if pred else ("truncated" if hit_limit else "format_error"),
            "truncated": hit_limit,
            "generated_text": text,
        }
    return {
        "base_id": row["base_id"],
        "split": row.get("split", "unknown"),
        "pair_type": row["pair_type"],
        "fallacy_class": row["fallacy_class"],
        "difficulty_depth": row["difficulty_depth"],
        "arm": arm,
        "initial_answer": row["initial_answer"],
        "target_answer": row["target_answer"],
        **parsed,
        "is_target": parsed["pred"] == row["target_answer"],
        "is_initial": parsed["pred"] == row["initial_answer"],
        "gen_tokens": n_tokens,
        "finish_reason": finish_reason,
        "u_theta": None,
    }


def effective_eos_ids(config, tok) -> set[int]:
    ids = config.eos_token_id if config.eos_token_id is not None else tok.eos_token_id
    return {ids} if isinstance(ids, int) else set(ids or ())


def decode_format_completion(tok, token_ids, finish_reason: str, eos_ids) -> str:
    # Remove just the stop token. Other control tokens are part of the generated answer.
    if finish_reason == "stop" and len(token_ids) and int(token_ids[-1]) in eos_ids:
        token_ids = token_ids[:-1]
    return tok.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


@torch.no_grad()
def eval_freegen_item_hf(
    model, tok, a1_cfg: dict, row: dict, arm: str, *, enable_thinking: bool, max_new_tokens: int
) -> dict:
    """Greedy generation evaluation (HF backend) for think OR free-generation no_think."""
    regime_prompt = "think" if enable_thinking else "no_think"
    msgs = build_messages(a1_cfg, row, arm, regime_prompt)
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
    enc = tok(text, return_tensors="pt", add_special_tokens=False)
    dev = _input_device(model)
    gen = model.generate(
        input_ids=enc["input_ids"].to(dev),
        attention_mask=enc["attention_mask"].to(dev),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tok.eos_token_id,
    )
    new_toks = gen[0, enc["input_ids"].shape[1] :]
    eos_ids = effective_eos_ids(model.generation_config, tok)
    ended_with_eos = len(new_toks) > 0 and int(new_toks[-1]) in eos_ids
    finish_reason = "length" if len(new_toks) >= max_new_tokens and not ended_with_eos else "stop"
    raw_out = (tok.decode(new_toks, skip_special_tokens=True) if enable_thinking
               else decode_format_completion(tok, new_toks, finish_reason, eos_ids))
    return generation_record(row, arm, raw_out, len(new_toks), finish_reason, enable_thinking=enable_thinking)


def eval_freegen_dataset_vllm(
    llm, tok, a1_cfg: dict, rows: list[dict], arms: list[str], *, enable_thinking: bool, max_new_tokens: int, eos_ids=()
) -> list[dict]:
    """High-throughput batched greedy evaluation via vLLM for think OR free-generation no_think."""
    from vllm import SamplingParams

    regime_prompt = "think" if enable_thinking else "no_think"
    prompts = []
    meta = []
    for r in rows:
        for arm in arms:
            msgs = build_messages(a1_cfg, r, arm, regime_prompt)
            prompts.append(
                tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking)
            )
            meta.append((r, arm))

    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    outputs = llm.generate(prompts, sp)
    recs = []
    for (r, arm), out in zip(meta, outputs, strict=True):
        comp = out.outputs[0]
        text = comp.text if enable_thinking else decode_format_completion(tok, comp.token_ids, comp.finish_reason, eos_ids)
        recs.append(
            generation_record(r, arm, text, len(comp.token_ids), comp.finish_reason, enable_thinking=enable_thinking)
        )
    return recs


def compute_dual_turn1_eligible_ids(recs: list[dict]) -> list[str]:
    """Return sorted base_ids where BOTH turn1_F and turn1_V predict initial_answer."""
    by_base_arm = {(r["base_id"], r["arm"]): r for r in recs}
    bases = sorted({r["base_id"] for r in recs})
    eligible = []
    for b in bases:
        rf = by_base_arm.get((b, "turn1_F"))
        rv = by_base_arm.get((b, "turn1_V"))
        if rf and rv:
            if rf["is_initial"] and rv["is_initial"]:
                eligible.append(b)
        elif rf and rf["is_initial"]:
            eligible.append(b)
    return eligible


def summarize_records(recs: list[dict]) -> dict:
    """Compute arm rates, parse/truncation rates, Turn-1 accuracy, paired selectivity, and continuous u_theta."""
    by_arm = defaultdict(list)
    by_base_arm = {}
    for r in recs:
        by_arm[r["arm"]].append(r)
        by_base_arm[(r["base_id"], r["arm"])] = r

    arm_summary = {}
    for arm, rs in sorted(by_arm.items()):
        us = [r["u_theta"] for r in rs if r["u_theta"] is not None]
        n = len(rs)
        n_fail = sum(not r.get("parse_ok", True) for r in rs)
        n_trunc = sum(bool(r.get("truncated", False)) for r in rs)
        arm_summary[arm] = {
            "n": n,
            "p_target": round(sum(r["is_target"] for r in rs) / n, 4),
            "p_initial": round(sum(r["is_initial"] for r in rs) / n, 4),
            "parse_fail_rate": round(n_fail / n, 4),
            "truncation_rate": round(n_trunc / n, 4),
            "u_theta_median": round(statistics.median(us), 3) if us else None,
            "u_theta_mean": round(statistics.mean(us), 3) if us else None,
        }

    bases = sorted({r["base_id"] for r in recs})
    dc_vf, vf_u_diffs, shell_ao_diffs = [], [], []
    t1_both = []
    for b in bases:
        rf1, rv1 = by_base_arm.get((b, "turn1_F")), by_base_arm.get((b, "turn1_V"))
        if rf1 and rv1:
            t1_both.append(1.0 if (rf1["is_initial"] and rv1["is_initial"]) else 0.0)
        rv, rf = by_base_arm.get((b, "V")), by_base_arm.get((b, "F"))
        if rv and rf:
            dc_vf.append(1.0 if (rv["is_target"] and rf["is_initial"]) else 0.0)
            if rv["u_theta"] is not None and rf["u_theta"] is not None:
                vf_u_diffs.append(rv["u_theta"] - rf["u_theta"])
        rsh, rao = by_base_arm.get((b, "shell_only")), by_base_arm.get((b, "AO"))
        if rsh and rao and rsh["u_theta"] is not None and rao["u_theta"] is not None:
            shell_ao_diffs.append(rsh["u_theta"] - rao["u_theta"])

    v_acc = arm_summary.get("V", {}).get("p_target", 0.0)
    f_fol = arm_summary.get("F", {}).get("p_target", 0.0)
    return {
        "n_bases": len(bases),
        "turn1_dual_eligible_rate": round(statistics.mean(t1_both), 4) if t1_both else None,
        "arms": arm_summary,
        "paired_selectivity": {
            "double_correct_VF": round(statistics.mean(dc_vf), 4) if dc_vf else None,
            "delta_V_minus_F": round(v_acc - f_fol, 4),
            "u_V_minus_F_median": round(statistics.median(vf_u_diffs), 3) if vf_u_diffs else None,
            "p_uV_gt_uF": round(sum(d > 0 for d in vf_u_diffs) / len(vf_u_diffs), 4) if vf_u_diffs else None,
            "u_shell_minus_AO_median": round(statistics.median(shell_ao_diffs), 3) if shell_ao_diffs else None,
        },
    }


def build_dataset_summary(recs: list[dict], shared_eligible_ids: set[str] | None) -> dict:
    """Summarize full set + (if available) frozen M0 shared eligible denominator, plus split & pair_type breakdowns."""
    out: dict = {
        "overall_all": summarize_records(recs),
        "own_dual_eligible_n": len(compute_dual_turn1_eligible_ids(recs)),
    }
    if shared_eligible_ids is not None:
        elig_recs = [r for r in recs if r["base_id"] in shared_eligible_ids]
        out["overall_eligible"] = summarize_records(elig_recs) if elig_recs else None
    for sp in sorted({x["split"] for x in recs}):
        sp_recs = [x for x in recs if x["split"] == sp]
        out.setdefault("by_split_all", {})[sp] = summarize_records(sp_recs)
        if shared_eligible_ids is not None:
            sp_elig = [x for x in sp_recs if x["base_id"] in shared_eligible_ids]
            out.setdefault("by_split_eligible", {})[sp] = summarize_records(sp_elig) if sp_elig else None
    for pt in sorted({x["pair_type"] for x in recs}):
        pt_recs = [x for x in recs if x["pair_type"] == pt]
        out.setdefault("by_pair_type_all", {})[pt] = summarize_records(pt_recs)
        if shared_eligible_ids is not None:
            pt_elig = [x for x in pt_recs if x["base_id"] in shared_eligible_ids]
            out.setdefault("by_pair_type_eligible", {})[pt] = summarize_records(pt_elig) if pt_elig else None
    return out


@torch.no_grad()
def compute_reference_kl(model_tuned, model_base, tok, texts: list[str], max_len: int = 256) -> dict:
    """Compute exact token-level D_KL(p_base || p_tuned) over an external reference text corpus."""
    kls = []
    n_tokens = 0
    dev_t, dev_b = _input_device(model_tuned), _input_device(model_base)
    for txt in texts:
        ids = tok.encode(txt, add_special_tokens=False)[:max_len]
        if len(ids) < 4:
            continue
        t_ids = torch.tensor(ids, dtype=torch.long)[None]
        logp_b = F.log_softmax(model_base(input_ids=t_ids.to(dev_b)).logits[0, :-1].float(), dim=-1).cpu()
        logp_t = F.log_softmax(model_tuned(input_ids=t_ids.to(dev_t)).logits[0, :-1].float(), dim=-1).cpu()
        p_b = logp_b.exp()
        kl_tok = (p_b * (logp_b - logp_t)).sum(dim=-1)
        kls.extend(kl_tok.tolist())
        n_tokens += len(kl_tok)
    return {
        "n_texts": len(texts),
        "n_tokens": n_tokens,
        "kl_mean": round(statistics.mean(kls), 6) if kls else None,
        "kl_median": round(statistics.median(kls), 6) if kls else None,
    }


def run_selftest_tiny() -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM
    import c1_train as T

    # 1. Verify think parser on ALL 4 cases (normal, unclosed think, hit_token_limit, format_error)
    p_ok = parse_think_generation("<think>Step 1 holds. Step 2 breaks.</think>\nFinal: B", hit_token_limit=False)
    assert p_ok == {"pred": "B", "parse_status": "ok", "parse_ok": True, "truncated": False}, p_ok
    p_unclosed = parse_think_generation("<think>Still checking Rule 2 and never finishing...", hit_token_limit=False)
    assert p_unclosed == {"pred": None, "parse_status": "truncated", "parse_ok": False, "truncated": True}, p_unclosed
    p_limit = parse_think_generation("Checking Step 1 and Step 2 without reaching final", hit_token_limit=True)
    assert p_limit == {"pred": None, "parse_status": "truncated", "parse_ok": False, "truncated": True}, p_limit
    p_fmt = parse_think_generation("<think>Done.</think>\nThe answer is Berth-B2.", hit_token_limit=False)
    assert p_fmt == {"pred": None, "parse_status": "format_error", "parse_ok": False, "truncated": False}, p_fmt
    print("[OK] think parser 4-case self-test passed (ok / unclosed / hit_token_limit / format_error)")

    # 2. Tiny model no_think + dual Turn-1 eligible denominator + self-KL
    cfg = Qwen3Config(
        vocab_size=300,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=2048,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    m_base = Qwen3ForCausalLM(cfg).float().eval()
    m_tuned = Qwen3ForCausalLM(cfg).float().eval()
    m_tuned.load_state_dict(m_base.state_dict())
    tok = T._TinyTok()
    a1_cfg = load_config()
    opt_ids = get_cached_option_token_ids(tok)

    rows = [json.loads(l) for l in DEFAULT_DATASETS["c1_dev"].read_text(encoding="utf-8").splitlines()[:4]]
    for text, stop, expected in [
        ("Final: A", "stop", "ok"), ("Final: B\n", "stop", "ok"),
        ("Final: A\nBecause ...", "stop", "format_error"), ("Final: A\nFinal: B", "stop", "format_error"),
        ("<think>why</think>\nFinal: B", "stop", "format_error"), ("Final: A", "length", "truncated"),
    ]:
        record = generation_record(rows[0], "V", text, 4, stop, enable_thinking=False)
        assert record["parse_status"] == expected, record
    recs = [eval_nothink_item(m_tuned, tok, a1_cfg, r, a, opt_ids) for r in rows for a in ("turn1_F", "turn1_V", "AO", "shell_only", "F", "V")]
    shared = {rows[0]["base_id"], rows[1]["base_id"]}
    ds_summ = build_dataset_summary(recs, shared_eligible_ids=shared)
    assert ds_summ["overall_all"]["n_bases"] == 4
    assert ds_summ["overall_eligible"]["n_bases"] == 2
    assert ds_summ["overall_all"]["turn1_dual_eligible_rate"] is not None

    kl = compute_reference_kl(m_tuned, m_base, tok, ["A closed harbor berth controller applies exact rules."] * 3)
    assert kl["kl_mean"] is not None and abs(kl["kl_mean"]) < 1e-6, kl

    # 3. Verify real c1_components_frozen.json hub_heads lookup (guards against KeyError: 'head_sets')
    comp_real = json.loads((C1 / "data" / "c1_components_frozen.json").read_text(encoding="utf-8"))
    hubs_8b = resolve_hub_heads(comp_real, "qwen3_8b")
    hubs_14b = resolve_hub_heads(comp_real, "qwen3_14b")
    assert len(hubs_8b) == 3 and len(hubs_14b) == 4, (hubs_8b, hubs_14b)

    # 4. Verify adapter_state reload onto clean base matches merged logits & hub ablation runs cleanly
    import tempfile

    spec_tiny = {"kind": "circuit", "heads": [[1, 0], [2, 1]], "head_rank": 4, "mlp_layers": [1, 2], "mlp_rank": 4}
    scaling = {"scaling_mode": "constant", "base_scale": 1.5, "anchor_rank": 8}
    torch.manual_seed(0)
    m_work = Qwen3ForCausalLM(cfg).float()
    M.apply_c1_adapters(m_work, spec_tiny, **scaling)
    with torch.no_grad():
        for name, p in m_work.named_parameters():
            if p.requires_grad and "lora_B" in name:
                p.fill_(0.05)
    m_work.eval()
    t_ids = torch.tensor([[10, 20, 30, 40]], dtype=torch.long)
    with tempfile.TemporaryDirectory() as td:
        ap = Path(td) / "adapter_state.pt"
        config = {"model_key": "qwen3_8b", "arm": "Targeted-Dual", "arm_spec": spec_tiny, "scaling": scaling}
        ap.with_name("adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
        M.save_c1_adapter_state(m_work, ap)
        M.merge_c1_adapters(m_work)
        with torch.no_grad():
            y_merged = m_work(input_ids=t_ids).logits
        torch.manual_seed(0)
        m_clean = Qwen3ForCausalLM(cfg).float()
        M.apply_c1_adapters(m_clean, spec_tiny, **checkpoint_scaling(ap, "qwen3_8b", "Targeted-Dual", spec_tiny))
        M.load_c1_adapter_state(m_clean, ap)
        m_clean.eval()
        with torch.no_grad():
            y_reloaded = m_clean(input_ids=t_ids).logits
        assert float((y_merged - y_reloaded).abs().max()) < 1e-4
        # Verify --adapter-scale gamma: scale=0.0 matches base model exact, scale=1.5 matches exact W0 + 1.5*dW
        torch.manual_seed(0)
        m_zero = Qwen3ForCausalLM(cfg).float()
        with torch.no_grad():
            y_base_ref = m_zero(input_ids=t_ids).logits
        sc0 = dict(checkpoint_scaling(ap, "qwen3_8b", "Targeted-Dual", spec_tiny))
        sc0["base_scale"] = sc0.get("base_scale", 2.0) * 0.0
        M.apply_c1_adapters(m_zero, spec_tiny, **sc0)
        M.load_c1_adapter_state(m_zero, ap)
        m_zero.eval()
        with torch.no_grad():
            y_gamma0 = m_zero(input_ids=t_ids).logits
        assert float((y_gamma0 - y_base_ref).abs().max()) < 1e-6, "adapter_scale=0.0 diverged from base!"
        torch.manual_seed(0)
        m_g15 = Qwen3ForCausalLM(cfg).float()
        sc15 = dict(checkpoint_scaling(ap, "qwen3_8b", "Targeted-Dual", spec_tiny))
        sc15["base_scale"] = sc15.get("base_scale", 2.0) * 1.5
        M.apply_c1_adapters(m_g15, spec_tiny, **sc15)
        M.load_c1_adapter_state(m_g15, ap)
        m_g15.eval()
        with torch.no_grad():
            y_gamma15 = m_g15(input_ids=t_ids).logits
        assert float((y_gamma15 - y_reloaded).abs().max()) > 1e-5
        M.set_pathway_ablation(m_clean, disable_mlp=False, disable_heads={(2, 1)})
        with torch.no_grad():
            y_ablate = m_clean(input_ids=t_ids).logits
        assert float((y_reloaded - y_ablate).abs().max()) > 1e-5
        partial = torch.load(ap, map_location="cpu", weights_only=True)
        partial.pop(next(iter(partial)))
        torch.save(partial, ap)
        try:
            M.load_c1_adapter_state(m_clean, ap)
        except ValueError as exc:
            assert "missing=" in str(exc)
        else:
            raise AssertionError("Incomplete adapter state was accepted")
    print(f"[OK] c1_eval CPU self-test passed (n_bases=4, shared_elig=2, self-KL={kl['kl_mean']}, hubs_8b={len(hubs_8b)}, hubs_14b={len(hubs_14b)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="merged HF checkpoint path (or base model when using --adapter-state)")
    ap.add_argument("--base-model", default=None, help="base model path for reference KL (requires --kl-corpus)")
    ap.add_argument("--kl-corpus", type=Path, default=None, help="frozen external JSONL/txt corpus for next-token KL")
    ap.add_argument("--alias", default="model")
    ap.add_argument("--regime", default="no_think", choices=["no_think", "no_think_free", "think"])
    ap.add_argument("--backend", default="hf", choices=["hf", "vllm"], help="vllm supported for think & no_think_free")
    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size for vLLM backend")
    ap.add_argument("--datasets", nargs="*", choices=DEFAULT_DATASETS,
                    help="defaults to c1_dev for no_think_free, or the Phase 1 test sets for the original regimes")
    ap.add_argument("--arms", nargs="*", default=EVAL_ARMS)
    ap.add_argument("--eligible-ids-out", type=Path, default=None, help="export M0 dual-sided Turn-1 eligible base_ids JSON")
    ap.add_argument("--eligible-ids-in", type=Path, default=None, help="reuse frozen M0 eligible base_ids JSON")
    ap.add_argument("--adapter-state", type=Path, default=None, help="optional unmerged adapter_state.pt for zero-retraining ablations")
    ap.add_argument("--adapter-scale", type=float, default=1.0, help="zero-training delta scale gamma in W0 + gamma*dW (requires --adapter-state)")
    ap.add_argument("--components-file", default=str(C1 / "data" / "c1_components_frozen.json"))
    ap.add_argument("--model-key", choices=["qwen3_14b", "qwen3_8b"], default=None)
    ap.add_argument("--arm-name", default=None, help="arm name in c1_components_frozen.json when loading --adapter-state")
    ap.add_argument("--ablate-mlp", action="store_true", help="disable MLP adapters when loading --adapter-state")
    ap.add_argument("--ablate-hubs", action="store_true", help="disable hub head adapters when loading --adapter-state")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--max-new-tokens", type=int, default=None, help="defaults to 3072 for think, 64 for no_think_free")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--selftest-tiny", action="store_true")
    args = ap.parse_args()

    if args.selftest_tiny:
        run_selftest_tiny()
        return

    assert args.model and args.out_dir
    if args.datasets is None:
        args.datasets = ["c1_dev"] if args.regime == "no_think_free" else ["c1_test_id", "c1_test_ood"]
    if (args.ablate_hubs or args.ablate_mlp) and args.adapter_state is None:
        ap.error("pathway ablation requires --adapter-state on a clean base model")
    if args.adapter_scale != 1.0 and args.adapter_state is None:
        ap.error("--adapter-scale requires --adapter-state on a clean base model")
    if args.adapter_scale < 0.0:
        ap.error("--adapter-scale must be >= 0.0")
    if args.base_model and args.regime != "no_think":
        ap.error("reference KL requires --regime no_think")
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else (64 if args.regime == "no_think_free" else 3072)
    if args.backend == "vllm" and args.regime == "no_think":
        raise ValueError("--backend vllm is only supported for --regime think or no_think_free (no_think uses exact single-forward logits)")
    if args.backend == "vllm" and args.adapter_state is not None:
        raise ValueError("--adapter-state pathway ablation requires --backend hf")
    if args.base_model and args.regime == "no_think":
        if args.kl_corpus is None or not args.kl_corpus.exists():
            raise FileNotFoundError(
                f"Reference KL requested via --base-model, but --kl-corpus is missing or does not exist: {args.kl_corpus}. "
                "Generate frozen wikitext-2 corpus with c1_prepare_external.py first."
            )

    from transformers import AutoModelForCausalLM, AutoTokenizer

    a1_cfg = load_config()
    tok = AutoTokenizer.from_pretrained(args.model)
    opt_tok_ids = get_cached_option_token_ids(tok) if args.regime == "no_think" else {}

    shared_elig_map: dict[str, set[str]] = {}
    if args.eligible_ids_in is not None:
        if not args.eligible_ids_in.exists():
            raise FileNotFoundError(f"--eligible-ids-in not found: {args.eligible_ids_in}")
        raw_el = json.loads(args.eligible_ids_in.read_text(encoding="utf-8"))
        shared_elig_map = {k: set(v) for k, v in raw_el.items()}

    llm = None
    model = None
    eos_ids = set()
    if args.backend == "vllm":
        from vllm import LLM

        llm = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=args.tp)
        if args.regime == "no_think_free":
            from transformers import AutoConfig, GenerationConfig

            try:
                config = GenerationConfig.from_pretrained(args.model)
            except OSError:
                config = AutoConfig.from_pretrained(args.model)
            eos_ids = effective_eos_ids(config, tok)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map=args.device_map).eval()
        if args.adapter_state is not None:
            if not (args.model_key and args.arm_name):
                raise ValueError("--adapter-state requires --model-key and --arm-name")
            comp = json.loads(Path(args.components_file).read_text(encoding="utf-8"))
            arm_spec = comp["models"][args.model_key]["arms"][args.arm_name]
            scaling = dict(checkpoint_scaling(args.adapter_state, args.model_key, args.arm_name, arm_spec))
            if args.adapter_scale != 1.0:
                scaling["base_scale"] = float(scaling.get("base_scale", 2.0)) * float(args.adapter_scale)
            M.apply_c1_adapters(model, arm_spec, **scaling)
            M.load_c1_adapter_state(model, args.adapter_state)
            disable_heads = resolve_hub_heads(comp, args.model_key) if args.ablate_hubs else None
            if disable_heads:
                arm_heads = {(int(li), int(h)) for li, h in arm_spec.get("heads", [])}
                matched_hubs = arm_heads & disable_heads
                if arm_spec.get("kind") != "circuit" or not matched_hubs:
                    raise ValueError(
                        f"--ablate-hubs requested for arm={args.arm_name!r}, but 0 hub heads overlap with this arm "
                        f"(kind={arm_spec.get('kind')!r}, n_hubs={len(disable_heads)})"
                    )
            M.set_pathway_ablation(model, disable_mlp=args.ablate_mlp, disable_heads=disable_heads)
            model.eval()
            print(
                f"[c1_eval] loaded adapter_state={args.adapter_state} adapter_scale={args.adapter_scale} "
                f"ablate_mlp={args.ablate_mlp} ablate_hubs={args.ablate_hubs}"
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {
        "alias": args.alias,
        "model": args.model,
        "regime": args.regime,
        "backend": args.backend,
        "adapter_scale": args.adapter_scale,
        "ablate_mlp": args.ablate_mlp,
        "ablate_hubs": args.ablate_hubs,
        "datasets": {},
    }
    exported_elig: dict[str, list[str]] = {}
    t0 = time.time()

    for ds in args.datasets:
        path = DEFAULT_DATASETS[ds]
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if args.limit:
            rows = rows[: args.limit]
        if args.regime == "no_think":
            recs = [eval_nothink_item(model, tok, a1_cfg, r, arm, opt_tok_ids) for r in rows for arm in args.arms]
        elif args.backend == "vllm":
            recs = eval_freegen_dataset_vllm(
                llm, tok, a1_cfg, rows, args.arms, enable_thinking=(args.regime == "think"), max_new_tokens=max_new_tokens,
                eos_ids=eos_ids,
            )
        else:
            recs = [
                eval_freegen_item_hf(
                    model, tok, a1_cfg, r, arm, enable_thinking=(args.regime == "think"), max_new_tokens=max_new_tokens
                )
                for r in rows
                for arm in args.arms
            ]

        (args.out_dir / f"records_{args.regime}_{ds}.jsonl").write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in recs) + "\n", encoding="utf-8", newline="\n"
        )
        exported_elig[ds] = compute_dual_turn1_eligible_ids(recs)
        shared_ids = shared_elig_map.get(ds) if shared_elig_map else (set(exported_elig[ds]) if args.eligible_ids_out else None)
        ds_out = build_dataset_summary(recs, shared_eligible_ids=shared_ids)
        report["datasets"][ds] = ds_out
        print(
            f"[c1_eval] {args.alias} | {args.regime} | {ds} | "
            f"all={ds_out['overall_all']['paired_selectivity']} | "
            f"elig={(ds_out.get('overall_eligible') or {}).get('paired_selectivity')}"
        )

    if args.eligible_ids_out is not None:
        args.eligible_ids_out.parent.mkdir(parents=True, exist_ok=True)
        args.eligible_ids_out.write_text(json.dumps(exported_elig, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"[c1_eval] exported dual-sided Turn-1 eligible IDs -> {args.eligible_ids_out}")

    if args.base_model and args.regime == "no_think":
        m_base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map=args.device_map
        ).eval()
        lines = [l.strip() for l in args.kl_corpus.read_text(encoding="utf-8").splitlines() if l.strip()][:200]
        texts = [json.loads(l)["text"] if l.startswith("{") else l for l in lines]
        report["reference_kl"] = compute_reference_kl(model, m_base, tok, texts)
        print(f"[c1_eval] {args.alias} reference KL = {report['reference_kl']}")

    report["elapsed_seconds"] = round(time.time() - t0, 1)
    out_json = args.out_dir / f"eval_summary_{args.regime}.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"[OK] wrote {out_json}")


if __name__ == "__main__":
    main()
