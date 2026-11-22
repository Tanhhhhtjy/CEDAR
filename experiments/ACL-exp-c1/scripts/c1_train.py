#!/usr/bin/env python3
"""ACL-exp-c1 trainer: Dual-Pathway Asymmetric Circuit LoRA (DAC-LoRA) with Signed Readout Margin Loss.

Supports all frozen arms from data/c1_components_frozen.json:
  Targeted-Dual | Targeted-Dual-NoMargin | Targeted-Form | Pinpoint-DLA | Pinpoint-AtP
  | Random-LM | Random-LM-seed2 | Uniform-LoRA | Hub-Only | Dual-wo-Hub

Key guarantees (audited):
  - rsLoRA rank-stabilized scaling s(r) = base_scale * sqrt(anchor_rank / r) (default 2 * sqrt(8 / r))
  - Exact readout_pos located at the LAST 'Final:' inside assistant_target (works for both no_think & Recipe B think)
  - Atomic group batching: never splits a 6-item rule base across gradient-accumulation boundaries;
    normalizes each window by its actual item count (no tail-batch shrink)
  - Padded microbatching (--microbatch 1/2/4) with exact per-example token-mean SFT loss (mathematically identical to mb=1)
  - Fails fast if --replay-file or --think-file is specified but missing
  - Always saves tiny unmerged `adapter_state.pt` alongside `merged_hf/` for zero-retraining post-hoc pathway ablations

Usage (GPU):
  CUDA_VISIBLE_DEVICES=0 python experiments/ACL-exp-c1/scripts/c1_train.py \
      --model /mnt/kcl/devuser/ckpts/Qwen3-8B --model-key qwen3_8b --arm Targeted-Dual \
      --replay-file experiments/ACL-exp-c1/data/c1_replay_norobots_529.jsonl \
      --microbatch 2 --out-dir checkpoints/qwen3_8b/Targeted-Dual

Usage (CPU self-test):
  python experiments/ACL-exp-c1/scripts/c1_train.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
C1 = HERE.parents[0]
REPO = C1.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "experiments" / "ACL-exp-b1" / "scripts"))
import b1_lib as B1L  # noqa: E402
import c1_modules as M  # noqa: E402


def _input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def _find_subseq_last(seq: list[int], sub: list[int], start: int = 0) -> int | None:
    if not sub or len(sub) > len(seq) - start:
        return None
    for i in range(len(seq) - len(sub), start - 1, -1):
        if seq[i : i + len(sub)] == sub:
            return i
    return None


def encode_sft_example(
    tok, ex: dict, enable_thinking: bool = False, mask_rule_format_ce: bool = False
) -> dict:
    """Encode (messages, assistant_target) with prompt masked (-100) and exact LAST 'Final:' readout index.

    When `mask_rule_format_ce=True`, for all no_think rule examples (including margin_sign == 0 items),
    masks the constant format tokens ('Final', ':', EOS) to -100 while preserving the exact original
    valid-token denominator (`ce_denom`, e.g., 4.0) so letter CE keeps its exact 1/4 coefficient.
    Replay targets and Recipe B reasoning traces keep their original labels.
    """
    prompt_text = tok.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    target_text = ex["assistant_target"]
    eos = getattr(tok, "eos_token", "") or ""
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    full_ids = tok.encode(prompt_text + target_text + eos, add_special_tokens=False)
    boundary_split = False
    if full_ids[: len(prompt_ids)] != prompt_ids:
        target_ids = tok.encode(target_text + eos, add_special_tokens=False)
        full_ids = prompt_ids + target_ids
        boundary_split = True
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    orig_valid_len = float(max(1, sum(1 for x in labels[1:] if x != -100)))

    is_replay = bool(ex.get("is_replay", ex.get("margin_sign", 0) == 0 and not ex.get("base_id")))
    margin_sign = int(ex.get("margin_sign", 0))

    # Locate the exact logit index predicting the option letter after the LAST "Final:" in assistant_target.
    # Runs for ALL non-replay rule examples (both margin_sign != 0 AND margin_sign == 0) so
    # --mask-rule-format-ce covers all 3,000 rule examples cleanly without missing the 291 margin_sign==0 items.
    answer_readout_pos = None
    pos_tok = neg_tok = None
    if not is_replay and ex.get("initial_answer") and ex.get("target_answer"):
        f_idx = target_text.rfind("Final:")
        if f_idx >= 0:
            prefix_in_target = target_text[: f_idx + len("Final:")]
            if boundary_split:
                anchor = prompt_ids + tok.encode(prefix_in_target, add_special_tokens=False)
                prefix_for_readout = prefix_in_target
            else:
                prefix_for_readout = prompt_text + prefix_in_target
                anchor = tok.encode(prefix_for_readout, add_special_tokens=False)
            pos_tok, neg_tok = B1L.readout_token_ids(
                tok, prefix_for_readout, ex["target_answer"], ex["initial_answer"]
            )
            gold_ans = ex.get("gold_answer")
            expected_gold_tok = pos_tok if gold_ans == ex["target_answer"] else neg_tok

            if len(anchor) < len(full_ids) and full_ids[: len(anchor)] == anchor:
                cand_pos = len(anchor) - 1
                if full_ids[cand_pos + 1] == expected_gold_tok:
                    answer_readout_pos = cand_pos
            if answer_readout_pos is None:
                # Fallback subsequence search strictly inside assistant target span
                key_ids = tok.encode("Final:", add_special_tokens=False)
                loc = _find_subseq_last(full_ids, key_ids, start=len(prompt_ids))
                if loc is not None and (loc + len(key_ids)) < len(full_ids):
                    cand_pos = loc + len(key_ids) - 1
                    if full_ids[cand_pos + 1] == expected_gold_tok:
                        answer_readout_pos = cand_pos
            assert answer_readout_pos is not None, (
                f"Failed to align Final: readout_pos for {ex.get('example_id')} "
                f"(gold={gold_ans}, target_text={target_text[:80]!r})"
            )

    if mask_rule_format_ce and not is_replay and not enable_thinking:
        assert answer_readout_pos is not None, (
            f"--mask-rule-format-ce requires aligned answer letter for rule item {ex.get('example_id')}"
        )
        ans_label_idx = answer_readout_pos + 1
        masked = [-100] * len(full_ids)
        masked[ans_label_idx] = full_ids[ans_label_idx]
        labels = masked

    readout_pos = answer_readout_pos if margin_sign != 0 else None

    return {
        "example_id": ex.get("example_id", ""),
        "pair_group_id": ex.get("pair_group_id", ex.get("example_id", "")),
        "is_replay": is_replay,
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "ce_denom": orig_valid_len,
        "margin_sign": margin_sign,
        "readout_pos": readout_pos,
        "answer_readout_pos": answer_readout_pos,
        "pos_tok": pos_tok if margin_sign != 0 else None,
        "neg_tok": neg_tok if margin_sign != 0 else None,
    }


def load_training_records(
    train_path: Path,
    think_path: Path | None,
    replay_path: Path | None,
    replay_ratio: float,
    seed: int,
    limit: int | None = None,
) -> list[tuple[dict, bool]]:
    """Return list of (example_dict, enable_thinking). Fails fast if optional paths do not exist."""
    if not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    rows = [json.loads(l) for l in train_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit:
        rows = rows[:limit]
    items: list[tuple[dict, bool]] = [(r, False) for r in rows]

    if think_path is not None:
        if not think_path.exists():
            raise FileNotFoundError(f"--think-file not found: {think_path}")
        trows = [json.loads(l) for l in think_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if limit:
            trows = trows[: max(1, limit // 3)]
        items.extend((r, True) for r in trows)

    if replay_path is not None and replay_ratio > 0:
        if not replay_path.exists():
            raise FileNotFoundError(f"--replay-file not found: {replay_path}")
        rrows = [json.loads(l) for l in replay_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        rng = random.Random(seed)
        n_want = int(round(len(items) * replay_ratio / max(1e-6, 1.0 - replay_ratio)))
        if len(rrows) > n_want:
            rrows = rng.sample(rrows, n_want)
        for r in rrows:
            r = dict(r)
            r.setdefault("margin_sign", 0)
            r["is_replay"] = True
            items.append((r, bool(r.get("enable_thinking", False))))
    return items


def build_atomic_windows(encoded: list[dict], grad_accum: int, seed: int) -> list[list[dict]]:
    """Pack groups into grad-accum windows without EVER splitting a paired rule group across a boundary.

    Replay singletons are packed into mini-groups of size <= min(6, grad_accum) and interleaved
    between rule groups so every 6-item paired rule base stays 100% atomic inside one optimizer step.
    """
    rule_groups: dict[str, list[dict]] = {}
    replay_items: list[dict] = []
    for ex in encoded:
        if ex.get("is_replay"):
            replay_items.append(ex)
        else:
            rule_groups.setdefault(ex["pair_group_id"], []).append(ex)

    rng = random.Random(seed)
    all_groups: list[list[dict]] = []
    for k in sorted(rule_groups.keys()):
        g = list(rule_groups[k])
        rng.shuffle(g)
        all_groups.append(g)

    rng.shuffle(replay_items)
    replay_chunk = max(1, min(6, grad_accum))
    for i in range(0, len(replay_items), replay_chunk):
        all_groups.append(replay_items[i : i + replay_chunk])

    rng.shuffle(all_groups)

    windows: list[list[dict]] = []
    cur: list[dict] = []
    for g in all_groups:
        if cur and len(cur) + len(g) > grad_accum:
            windows.append(cur)
            cur = []
        cur.extend(g)
        if len(cur) >= grad_accum:
            windows.append(cur)
            cur = []
    if cur:
        windows.append(cur)
    return windows


def compute_loss_microbatch(
    model,
    batch: list[dict],
    pad_token_id: int,
    lambda_margin: float,
    margin_m: float,
    zero_replay_loss: bool = False,
    return_source_losses: bool = False,
    accept_weight: float = 1.0,
) -> tuple[torch.Tensor, float, float, dict]:
    """Compute exact sum-of-per-example (SFT + Signed Readout Margin) loss over a padded microbatch.

    - Uses each example's `ce_denom` (original valid token count before optional format masking) so
      `--mask-rule-format-ce` preserves the exact 1/4 letter-CE weight instead of amplifying 4x.
    - When `zero_replay_loss=True`, still runs the exact forward pass (preserving dropout RNG, window
      denominator, and step count) and multiplies replay example SFT losses by 0.0.
    """
    dev = _input_device(model)
    bsz = len(batch)
    max_len = max(int(ex["input_ids"].numel()) for ex in batch)
    input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)
    attn_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    ce_denoms = torch.ones((bsz,), dtype=torch.float32)

    for bi, ex in enumerate(batch):
        L = int(ex["input_ids"].numel())
        input_ids[bi, :L] = ex["input_ids"]
        labels[bi, :L] = ex["labels"]
        attn_mask[bi, :L] = 1
        if "ce_denom" in ex:
            ce_denoms[bi] = ex["ce_denom"]
        else:
            ce_denoms[bi] = (ex["labels"][1:] != -100).sum().clamp_min(1)

    input_ids = input_ids.to(dev)
    attn_mask = attn_mask.to(dev)

    out = model(input_ids=input_ids, attention_mask=attn_mask, use_cache=False)
    logits = out.logits  # [B, T, V]

    # Exact per-example CE normalized by original valid token count `ce_denom` (identical to sequential mb=1)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous().to(logits.device)
    tok_losses = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(bsz, -1)
    per_ex_sft_raw = tok_losses.sum(dim=1) / ce_denoms.to(logits.device).clamp_min(1.0)
    replay_flags = [bool(ex.get("is_replay")) for ex in batch]
    is_replay = torch.tensor(replay_flags, device=logits.device, dtype=torch.bool)
    per_ex_sft = per_ex_sft_raw.masked_fill(is_replay, 0.0) if zero_replay_loss else per_ex_sft_raw
    if accept_weight != 1.0:
        acc_weights = torch.tensor(
            [accept_weight if (not ex.get("is_replay") and int(ex.get("margin_sign", 0)) > 0) else 1.0 for ex in batch],
            device=logits.device,
            dtype=per_ex_sft.dtype,
        )
        per_ex_sft = per_ex_sft * acc_weights

    total_sft = per_ex_sft.sum()
    total_margin = torch.tensor(0.0, device=logits.device, dtype=torch.float32)

    rule_sft = per_ex_sft[~is_replay].sum()
    replay_sft_raw = per_ex_sft_raw[is_replay].sum()
    n_replay = sum(replay_flags)
    n_rule = bsz - n_replay
    hinge_values = []

    for bi, ex in enumerate(batch):
        if ex["margin_sign"] != 0 and ex["readout_pos"] is not None:
            logits_at = logits[bi, ex["readout_pos"]].float()
            u_theta = logits_at[ex["pos_tok"]] - logits_at[ex["neg_tok"]]
            hinge = F.relu(margin_m - u_theta) if ex["margin_sign"] > 0 else F.relu(margin_m + u_theta)
            hinge_values.append(hinge.detach())
            if lambda_margin > 0.0:
                w_m = accept_weight if ex["margin_sign"] > 0 else 1.0
                total_margin = total_margin + w_m * hinge

    total_loss = total_sft + lambda_margin * total_margin
    n_active = (torch.stack(hinge_values) > 0).sum() if hinge_values else total_sft.new_zeros(())
    summed_sft, summed_margin, rule_value, replay_value, active = torch.stack(
        [total_sft, total_margin, rule_sft, replay_sft_raw, n_active]
    ).detach().tolist()
    stats = {
        "rule_sft_sum": rule_value,
        "replay_sft_sum": 0.0 if zero_replay_loss else replay_value,
        "replay_sft_raw_sum": replay_value,
        "n_rule": n_rule,
        "n_replay": n_replay,
        "n_margin_elig": len(hinge_values),
        "n_hinge_active": int(active),
    }
    if return_source_losses:
        stats["source_losses"] = {}
        if n_rule:
            stats["source_losses"]["rule"] = rule_sft + lambda_margin * total_margin
        if n_replay:
            stats["source_losses"]["replay"] = per_ex_sft[is_replay].sum()
    return total_loss, summed_sft, summed_margin, stats


def build_optimizer(model, lr: float, mlp_lr_mult: float, weight_decay: float) -> torch.optim.Optimizer:
    attn_params, mlp_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ".mlp." in name:
            mlp_params.append(p)
        else:
            attn_params.append(p)
    groups = []
    if attn_params:
        groups.append({"params": attn_params, "lr": lr, "weight_decay": weight_decay})
    if mlp_params:
        groups.append({"params": mlp_params, "lr": lr * mlp_lr_mult, "weight_decay": weight_decay})
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1e-8)


def lr_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


# ----------------------------------------------------------------------------------------------- CPU self-test


class _TinyTok:
    eos_token = " <|eos|>"
    eos_token_id = 1
    pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        s = "".join(f"<|{m['role']}|> {m['content']} <|end|> " for m in messages)
        if not add_generation_prompt:
            return s
        return s + ("<|assistant|> " if enable_thinking else "<|assistant|> <think> </think> ")

    def encode(self, text, add_special_tokens=False):
        return [5 + (sum(ord(ch) * 31 ** i for i, ch in enumerate(w)) % 290) for w in text.split()]


def run_selftest_tiny() -> None:
    from transformers import Qwen3Config, Qwen3ForCausalLM

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
    tok = _TinyTok()
    train_path = C1 / "data" / "c1_train_examples_nothink.jsonl"
    raw_rows = [json.loads(l) for l in train_path.read_text(encoding="utf-8").splitlines()[:12]]
    encoded = [encode_sft_example(tok, r, False) for r in raw_rows]
    assert any(e["margin_sign"] == 1 and e["readout_pos"] is not None for e in encoded)
    assert any(e["margin_sign"] == -1 and e["readout_pos"] is not None for e in encoded)

    # 1. Verify Recipe B multi-step think readout_pos alignment (finds LAST Final: inside target!)
    think_ex = dict(raw_rows[0])
    think_ex["assistant_target"] = (
        "<think> Step 1 checks Rule 1 and confirms the premise. Step 2 confirms Final: check is valid. </think>\n"
        f"Final: {think_ex['gold_answer']}"
    )
    enc_think = encode_sft_example(tok, think_ex, enable_thinking=True)
    assert torch.equal(enc_think["labels"], encode_sft_example(tok, think_ex, True, True)["labels"])
    assert enc_think["readout_pos"] is not None
    expected_gold_tok = enc_think["pos_tok"] if think_ex["gold_answer"] == think_ex["target_answer"] else enc_think["neg_tok"]
    assert int(enc_think["input_ids"][enc_think["readout_pos"] + 1]) == expected_gold_tok
    print(f"[OK] Recipe B multi-step think readout_pos alignment passed (pos={enc_think['readout_pos']})")

    # 2. Verify atomic group windowing with replay singletons mixed in (zero 6-item rule groups split!)
    replay_synth = []
    for k in range(5):
        r = {
            "example_id": f"REPLAY-{k}",
            "pair_group_id": f"REPLAY-{k}",
            "is_replay": True,
            "margin_sign": 0,
            "messages": [{"role": "user", "content": f"Hello {k}"}],
            "assistant_target": f"Reply {k}",
        }
        replay_synth.append(encode_sft_example(tok, r, False))
    windows = build_atomic_windows(encoded + replay_synth, grad_accum=12, seed=42)
    for w in windows:
        rule_gids = [e["pair_group_id"] for e in w if not e["is_replay"]]
        for gid in set(rule_gids):
            assert rule_gids.count(gid) == 6, f"Rule group {gid} split across windows! count={rule_gids.count(gid)}"
    assert sum(len(w) for w in windows) == len(encoded) + len(replay_synth)
    print(f"[OK] Atomic group windowing passed (windows={[len(w) for w in windows]}, zero split rule groups)")

    # 3. Verify microbatch=2 loss equivalence with microbatch=1
    torch.manual_seed(0)
    m_mb = Qwen3ForCausalLM(cfg).float().eval()
    l1_a, s1_a, m1_a, _ = compute_loss_microbatch(m_mb, [encoded[0]], tok.pad_token_id, 0.25, 4.0)
    l1_b, s1_b, m1_b, _ = compute_loss_microbatch(m_mb, [encoded[1]], tok.pad_token_id, 0.25, 4.0)
    l2, s2, m2, _ = compute_loss_microbatch(m_mb, encoded[:2], tok.pad_token_id, 0.25, 4.0)
    mb_diff = abs(float(l2.detach()) - float((l1_a + l1_b).detach()))
    assert mb_diff < 1e-4, f"microbatch=2 diverged from microbatch=1: diff={mb_diff}"
    print(f"[OK] microbatch=2 vs microbatch=1 equivalence passed (diff={mb_diff:.2e})")

    # 3b. Verify --mask-rule-format-ce on: (a) margin_sign != 0 rule, (b) margin_sign == 0 rule, (c) replay item!
    m0_rule_raw = next(r for r in [json.loads(l) for l in train_path.read_text(encoding="utf-8").splitlines()[:60]] if r["margin_sign"] == 0)
    rep_raw = {
        "example_id": "REPLAY-MASK-TEST",
        "pair_group_id": "REPLAY-MASK-TEST",
        "is_replay": True,
        "margin_sign": 0,
        "messages": [{"role": "user", "content": "Say hello"}],
        "assistant_target": "Hello there friend",
    }
    enc_v_unm = encode_sft_example(tok, raw_rows[0], False, mask_rule_format_ce=False)
    enc_v_msk = encode_sft_example(tok, raw_rows[0], False, mask_rule_format_ce=True)
    enc_m0_msk = encode_sft_example(tok, m0_rule_raw, False, mask_rule_format_ce=True)
    enc_rep_unm = encode_sft_example(tok, rep_raw, False, mask_rule_format_ce=False)
    enc_rep_msk = encode_sft_example(tok, rep_raw, False, mask_rule_format_ce=True)
    # Rule items (both margin_sign!=0 and margin_sign==0): exactly 1 supervised token (the answer letter), same ce_denom!
    assert int((enc_v_msk["labels"] != -100).sum()) == 1 and enc_v_msk["ce_denom"] == enc_v_unm["ce_denom"]
    assert int((enc_m0_msk["labels"] != -100).sum()) == 1 and enc_m0_msk["readout_pos"] is None and enc_m0_msk["answer_readout_pos"] is not None
    # Replay item: untouched!
    assert torch.equal(enc_rep_msk["labels"], enc_rep_unm["labels"]) and int((enc_rep_msk["labels"] != -100).sum()) >= 3
    # Verify exact 1/ce_denom coefficient (masked CE < unmasked CE and strictly equals letter_ce / ce_denom)
    _, s_unm, _, _ = compute_loss_microbatch(m_mb, [enc_v_unm], tok.pad_token_id, 0.0, 4.0)
    _, s_msk, _, _ = compute_loss_microbatch(m_mb, [enc_v_msk], tok.pad_token_id, 0.0, 4.0)
    pos = enc_v_msk["answer_readout_pos"]
    with torch.no_grad():
        letter_logits = m_mb(input_ids=enc_v_msk["input_ids"][None]).logits[0, pos]
        letter_ce = F.cross_entropy(letter_logits, enc_v_msk["labels"][pos + 1])
    assert 0.0 < s_msk < s_unm and abs(s_msk - float(letter_ce) / enc_v_unm["ce_denom"]) < 1e-5
    print(f"[OK] --mask-rule-format-ce passed (margin!=0, margin==0 rule, & replay; s_msk={s_msk:.4f} < s_unm={s_unm:.4f})")

    # 3c. Verify --zero-replay-loss on both mixed and pure-replay windows (exact 0.0 replay loss + valid zero backward)
    torch.manual_seed(0)
    m_zr = Qwen3ForCausalLM(cfg).float()
    M.apply_c1_adapters(m_zr, {"kind": "circuit", "heads": [[1, 0]], "head_rank": 4, "mlp_layers": [1], "mlp_rank": 4})
    m_zr.train()
    m_zr.zero_grad()
    loss_pr, sft_pr, marg_pr, st_pr = compute_loss_microbatch(
        m_zr, replay_synth[:2], tok.pad_token_id, 0.25, 4.0, zero_replay_loss=True
    )
    assert sft_pr == 0.0 and marg_pr == 0.0 and st_pr["replay_sft_sum"] == 0.0
    assert st_pr["replay_sft_raw_sum"] > 0
    (loss_pr / 2.0).backward()
    for p in m_zr.parameters():
        if p.requires_grad:
            assert p.grad is not None and float(p.grad.abs().max()) == 0.0
    print("[OK] --zero-replay-loss passed (pure-replay window exact 0.0 loss & exact 0.0 grad)")

    # 3d. Verify --accept-weight scales margin_sign>0 rule items by exact factor while leaving margin_sign<=0 & replay untouched
    m_zr.eval()
    _, s_v1, m_v1, _ = compute_loss_microbatch(m_zr, [enc_v_unm], tok.pad_token_id, 0.25, 4.0, accept_weight=1.0)
    _, s_v2, m_v2, _ = compute_loss_microbatch(m_zr, [enc_v_unm], tok.pad_token_id, 0.25, 4.0, accept_weight=1.7)
    assert abs(s_v2 - 1.7 * s_v1) < 1e-4 and abs(m_v2 - 1.7 * m_v1) < 1e-4, (s_v1, s_v2, m_v1, m_v2)
    f_raw = next(r for r in raw_rows if r["margin_sign"] < 0)
    enc_f_unm = encode_sft_example(tok, f_raw, False, mask_rule_format_ce=False)
    _, s_f1, m_f1, _ = compute_loss_microbatch(m_zr, [enc_f_unm], tok.pad_token_id, 0.25, 4.0, accept_weight=1.0)
    _, s_f2, m_f2, _ = compute_loss_microbatch(m_zr, [enc_f_unm], tok.pad_token_id, 0.25, 4.0, accept_weight=1.7)
    assert abs(s_f2 - s_f1) < 1e-6 and abs(m_f2 - m_f1) < 1e-6
    print("[OK] --accept-weight passed (exact 1.7x on margin_sign>0, exact 1.0x on margin_sign<0)")

    # 4. Verify all arm types: rsLoRA scaling, strict locality, merge equivalence, adapter_state save/reload & pathway ablation
    tiny_arms = {
        "Targeted-Dual": {"kind": "circuit", "heads": [[1, 0], [1, 2], [2, 1]], "head_rank": 4, "mlp_layers": [1, 2], "mlp_rank": 4},
        "Targeted-Form": {"kind": "circuit", "heads": [[2, 0], [3, 1]], "head_rank": 6, "mlp_layers": [], "mlp_rank": 0},
        "Pinpoint-DLA": {"kind": "circuit", "heads": [[3, 2], [3, 3]], "head_rank": 6, "mlp_layers": [], "mlp_rank": 0},
        "Uniform-LoRA": {"kind": "uniform", "uniform_rank": 2, "target_modules": ["q_proj", "o_proj", "gate_proj", "down_proj"]},
        "Hub-Only": {"kind": "circuit", "heads": [[2, 1]], "head_rank": 4, "mlp_layers": [], "mlp_rank": 0},
        "Dual-wo-Hub": {"kind": "circuit", "heads": [[1, 0], [1, 2]], "head_rank": 4, "mlp_layers": [1, 2], "mlp_rank": 4},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        for aname, spec in tiny_arms.items():
            torch.manual_seed(0)
            m = Qwen3ForCausalLM(cfg).float()
            orig_weights = {k: v.detach().clone() for k, v in m.named_parameters()}
            info = M.apply_c1_adapters(m, spec, scaling_mode="rslora", base_scale=2.0, anchor_rank=8)
            opt = build_optimizer(m, lr=5e-3, mlp_lr_mult=1.0, weight_decay=0.0)
            m.train()
            opt.zero_grad()
            win = encoded[:6]
            for j in range(0, len(win), 2):
                loss_sum, _, _, _ = compute_loss_microbatch(m, win[j : j + 2], tok.pad_token_id, 0.25, 4.0)
                (loss_sum / len(win)).backward()
            opt.step()

            norms = M.module_delta_norms(m)
            assert norms["attn_delta_frob"] > 0.0, (aname, norms)
            if spec.get("mlp_layers") or spec["kind"] == "uniform":
                assert norms["mlp_delta_frob"] > 0.0, (aname, norms)

            m.eval()
            test_ids = encoded[0]["input_ids"][None]
            with torch.no_grad():
                y_adapter = m(input_ids=test_ids).logits

            # Save unmerged adapter_state.pt & test zero-retraining pathway ablation on Targeted-Dual
            if aname == "Targeted-Dual":
                ap_path = Path(tmpdir) / "adapter_state.pt"
                M.save_c1_adapter_state(m, ap_path)
                # Test MLP ablation & Hub ablation on live adapters
                M.set_pathway_ablation(m, disable_mlp=True, disable_heads=None)
                with torch.no_grad():
                    y_nomlp = m(input_ids=test_ids).logits
                assert float((y_adapter - y_nomlp).abs().max()) > 1e-5, "MLP ablation had zero effect!"
                M.set_pathway_ablation(m, disable_mlp=False, disable_heads={(2, 1)})
                with torch.no_grad():
                    y_nohub = m(input_ids=test_ids).logits
                assert float((y_adapter - y_nohub).abs().max()) > 1e-5, "Hub ablation had zero effect!"
                M.set_pathway_ablation(m, disable_mlp=False, disable_heads=None)
                # Reload onto fresh base model & verify exact match
                torch.manual_seed(0)
                m_reload = Qwen3ForCausalLM(cfg).float()
                M.apply_c1_adapters(m_reload, spec, scaling_mode="rslora", base_scale=2.0, anchor_rank=8)
                M.load_c1_adapter_state(m_reload, ap_path)
                m_reload.eval()
                with torch.no_grad():
                    y_reload = m_reload(input_ids=test_ids).logits
                assert float((y_adapter - y_reload).abs().max()) < 1e-6, "adapter_state reload mismatch!"
                print("[OK] adapter_state.pt save/reload & zero-retraining MLP/Hub pathway ablations passed")

            M.merge_c1_adapters(m)
            with torch.no_grad():
                y_merged = m(input_ids=test_ids).logits
            max_err = float((y_adapter - y_merged).abs().max())
            assert max_err < 1e-4, (aname, max_err)

            new_weights = dict(m.named_parameters())
            assert torch.equal(new_weights["model.embed_tokens.weight"], orig_weights["model.embed_tokens.weight"])
            assert torch.equal(new_weights["lm_head.weight"], orig_weights["lm_head.weight"])
            if spec["kind"] == "circuit":
                sel_heads_by_l: dict[int, set[int]] = {}
                for li, h in spec["heads"]:
                    sel_heads_by_l.setdefault(li, set()).add(h)
                sel_mlps = set(spec["mlp_layers"])
                for li in range(4):
                    assert torch.equal(
                        new_weights[f"model.layers.{li}.self_attn.k_proj.weight"],
                        orig_weights[f"model.layers.{li}.self_attn.k_proj.weight"],
                    )
                    assert torch.equal(
                        new_weights[f"model.layers.{li}.self_attn.v_proj.weight"],
                        orig_weights[f"model.layers.{li}.self_attn.v_proj.weight"],
                    )
                    if li not in sel_mlps:
                        assert torch.equal(
                            new_weights[f"model.layers.{li}.mlp.gate_proj.weight"],
                            orig_weights[f"model.layers.{li}.mlp.gate_proj.weight"],
                        )
                        assert torch.equal(
                            new_weights[f"model.layers.{li}.mlp.down_proj.weight"],
                            orig_weights[f"model.layers.{li}.mlp.down_proj.weight"],
                        )
                    wq_new = new_weights[f"model.layers.{li}.self_attn.q_proj.weight"]
                    wq_old = orig_weights[f"model.layers.{li}.self_attn.q_proj.weight"]
                    wo_new = new_weights[f"model.layers.{li}.self_attn.o_proj.weight"]
                    wo_old = orig_weights[f"model.layers.{li}.self_attn.o_proj.weight"]
                    for h in range(4):
                        sl = slice(h * 16, (h + 1) * 16)
                        if h not in sel_heads_by_l.get(li, set()):
                            assert torch.equal(wq_new[sl, :], wq_old[sl, :]), (aname, li, h, "q_proj leaked!")
                            assert torch.equal(wo_new[:, sl], wo_old[:, sl]), (aname, li, h, "o_proj leaked!")
            print(f"[OK] tiny self-test {aname:15s}: params={info['trainable_params']:6d} merge_err={max_err:.2e} norms={norms}")
    print("[OK] ALL c1_train & c1_modules self-tests PASSED")


# ----------------------------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--model-key", choices=["qwen3_14b", "qwen3_8b"])
    ap.add_argument("--arm")
    ap.add_argument("--components-file", default=str(C1 / "data" / "c1_components_frozen.json"))
    ap.add_argument("--train-file", default=str(C1 / "data" / "c1_train_examples_nothink.jsonl"))
    ap.add_argument("--think-file", type=Path, default=None, help="optional Recipe B think-trace examples")
    ap.add_argument("--replay-file", type=Path, default=None, help="optional 15-20%% general instruction replay JSONL")
    ap.add_argument("--replay-ratio", type=float, default=0.15)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=None, help="optional early stop while preserving full --epochs LR/warmup horizon")
    ap.add_argument("--save-steps", default="", help="comma-separated optimizer steps to save adapter_state_stepNNNN.pt")
    ap.add_argument("--zero-replay-loss", action="store_true", help="keep replay windows/forward/RNG/denom identical while zeroing replay loss")
    ap.add_argument("--mask-rule-format-ce", action="store_true", help="keep only letter CE with its original weight on no_think rule items")
    ap.add_argument("--log-source-grad-norms", action="store_true", help="measure pre-clip rule and replay gradient contributions at logged steps")
    ap.add_argument("--grad-accum", type=int, default=12, help="multiple of 6 keeps paired groups atomic")
    ap.add_argument("--microbatch", type=int, default=1, help="padded microbatch size (1/2/4) inside each window")
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--mlp-lr-mult", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--scaling-mode", choices=["rslora", "constant"], default="rslora")
    ap.add_argument("--base-scale", type=float, default=2.0)
    ap.add_argument("--anchor-rank", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lambda-margin", type=float, default=None, help="defaults to arm_spec default_lambda_margin or 0.25")
    ap.add_argument("--margin-m", type=float, default=4.0)
    ap.add_argument("--accept-weight", type=float, default=1.0, help="loss multiplier on accept (margin_sign>0) rule items (default 1.0)")
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--no-merge-save", action="store_true")
    ap.add_argument("--selftest-tiny", action="store_true")
    args = ap.parse_args()

    if args.selftest_tiny:
        run_selftest_tiny()
        return

    assert args.model and args.model_key and args.arm and args.out_dir
    if args.max_steps is not None and args.max_steps < 1:
        ap.error("--max-steps must be positive")
    if args.accept_weight <= 0.0:
        ap.error("--accept-weight must be positive")
    try:
        save_steps = {int(s.strip()) for s in args.save_steps.split(",") if s.strip()}
    except ValueError:
        ap.error("--save-steps requires comma-separated optimizer step numbers")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    comp = json.loads(Path(args.components_file).read_text(encoding="utf-8"))
    arms_dict = comp["models"][args.model_key]["arms"]
    if args.arm not in arms_dict:
        raise KeyError(f"Unknown arm {args.arm!r}. Available: {sorted(arms_dict)}")
    arm_spec = arms_dict[args.arm]
    lambda_margin = float(args.lambda_margin) if args.lambda_margin is not None else float(arm_spec.get("default_lambda_margin", 0.25))

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device_map
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    attach_info = M.apply_c1_adapters(
        model,
        arm_spec,
        scaling_mode=args.scaling_mode,
        base_scale=args.base_scale,
        anchor_rank=args.anchor_rank,
        dropout=args.dropout,
    )
    assert attach_info["trainable_params"] == arm_spec["trainable_params"], (
        f"Param mismatch: actual={attach_info['trainable_params']} vs frozen={arm_spec['trainable_params']}"
    )
    print(
        f"[c1] {args.model_key} | {args.arm} | trainable={attach_info['trainable_params']:,} | "
        f"scaling={args.scaling_mode} | lambda_margin={lambda_margin} | microbatch={args.microbatch}"
    )

    raw_items = load_training_records(
        Path(args.train_file), args.think_file, args.replay_file, args.replay_ratio, args.seed, args.limit
    )
    encoded = [encode_sft_example(tok, r, thk, mask_rule_format_ce=args.mask_rule_format_ce) for r, thk in raw_items]
    n_margin = sum(e["readout_pos"] is not None for e in encoded)
    n_rule_aligned = sum(e["answer_readout_pos"] is not None for e in encoded)
    n_replay = sum(e["is_replay"] for e in encoded)
    if args.zero_replay_loss and not n_replay:
        ap.error("--zero-replay-loss requires replay samples in the training windows")
    print(
        f"[c1] encoded {len(encoded)} examples (rule_aligned={n_rule_aligned}, margin_active={n_margin}, "
        f"replay={n_replay}, mask_rule_format_ce={args.mask_rule_format_ce}, zero_replay_loss={args.zero_replay_loss})"
    )

    opt = build_optimizer(model, lr=args.lr, mlp_lr_mult=args.mlp_lr_mult, weight_decay=args.weight_decay)
    epoch_windows = [build_atomic_windows(encoded, args.grad_accum, seed=args.seed + ep) for ep in range(args.epochs)]
    total_steps = sum(len(ws) for ws in epoch_windows)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))
    stop_steps = min(total_steps, args.max_steps) if args.max_steps is not None else total_steps
    if any(step < 1 or step > stop_steps for step in save_steps):
        ap.error(f"--save-steps entries must be between 1 and {stop_steps}")
    trainable = [p for p in model.parameters() if p.requires_grad]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    adapter_config = {
        "model_key": args.model_key, "arm": args.arm, "arm_spec": arm_spec,
        "scaling": {"scaling_mode": args.scaling_mode, "base_scale": args.base_scale, "anchor_rank": args.anchor_rank},
    }
    (args.out_dir / "adapter_config.json").write_text(json.dumps(adapter_config, indent=2) + "\n", encoding="utf-8")
    log_rows = []
    saved_checkpoints = []
    t0 = time.time()
    global_step = 0
    model.train()

    for ep, windows in enumerate(epoch_windows):
        if global_step >= stop_steps:
            break
        for win in windows:
            if global_step >= stop_steps:
                break
            opt.zero_grad()
            win_n = len(win)
            next_step = global_step + 1
            log_step = next_step % 10 == 0 or next_step == stop_steps or next_step in save_steps
            measure_sources = args.log_source_grad_norms and log_step
            mixed_window = any(e["is_replay"] for e in win) and any(not e["is_replay"] for e in win)
            source_grads = {}
            if measure_sources and mixed_window and not args.zero_replay_loss:
                source_grads = {
                    source: [torch.zeros_like(p, dtype=torch.float32) for p in trainable]
                    for source in ("rule", "replay")
                }
            acc_sft = acc_m = 0.0
            acc_rule_sft = acc_rep_sft = acc_raw_replay = 0.0
            win_n_rule = win_n_rep = win_n_melig = win_n_hact = 0
            for j in range(0, win_n, args.microbatch):
                mb = win[j : j + args.microbatch]
                loss_sum, sft_sum, m_sum, st = compute_loss_microbatch(
                    model, mb, pad_id, lambda_margin, args.margin_m, zero_replay_loss=args.zero_replay_loss,
                    return_source_losses=bool(source_grads), accept_weight=args.accept_weight,
                )
                # Sample each source on the same dropout realization without writing optimizer .grad.
                if source_grads:
                    for source, objective in st.pop("source_losses").items():
                        grads = torch.autograd.grad(objective / win_n, trainable, retain_graph=True, allow_unused=True)
                        for buf, grad in zip(source_grads[source], grads):
                            if grad is not None:
                                buf.add_(grad.detach())
                (loss_sum / win_n).backward()
                acc_sft += sft_sum
                acc_m += m_sum
                acc_rule_sft += st["rule_sft_sum"]
                acc_rep_sft += st["replay_sft_sum"]
                acc_raw_replay += st["replay_sft_raw_sum"]
                win_n_rule += st["n_rule"]
                win_n_rep += st["n_replay"]
                win_n_melig += st["n_margin_elig"]
                win_n_hact += st["n_hinge_active"]

            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    trainable, args.max_grad_norm
                )
            )
            fac = lr_factor(global_step, total_steps, warmup_steps)
            for gi, pg in enumerate(opt.param_groups):
                base_lr = args.lr if gi == 0 else args.lr * args.mlp_lr_mult
                pg["lr"] = base_lr * fac
            opt.step()
            global_step += 1

            if global_step in save_steps:
                ck_path = args.out_dir / f"adapter_state_step{global_step:04d}.pt"
                M.save_c1_adapter_state(model, ck_path)
                saved_checkpoints.append({"step": global_step, "path": ck_path.name, "norms": M.module_delta_norms(model)})
                print(f"[c1] saved intermediate adapter checkpoint -> {ck_path}")

            if log_step:
                row = {
                    "epoch": ep + 1,
                    "step": global_step,
                    "window_n": win_n,
                    "n_rule": win_n_rule,
                    "n_replay": win_n_rep,
                    "sft_loss": round(acc_sft / win_n, 4),
                    "rule_sft_loss": round(acc_rule_sft / win_n_rule, 4) if win_n_rule else None,
                    "replay_sft_loss": round(acc_rep_sft / win_n_rep, 4) if win_n_rep else None,
                    "replay_sft_loss_unweighted": round(acc_raw_replay / win_n_rep, 4) if win_n_rep else None,
                    "margin_loss": round(acc_m / win_n, 4),
                    "hinge_active_rate": round(win_n_hact / win_n_melig, 4) if win_n_melig else None,
                    "grad_norm": round(grad_norm, 4),
                    "lr": opt.param_groups[0]["lr"],
                    "elapsed_s": round(time.time() - t0, 1),
                }
                if measure_sources:
                    if source_grads:
                        for source, buffers in source_grads.items():
                            norms = torch.stack([buf.norm().to(trainable[0].device) for buf in buffers])
                            row[f"{source}_grad_norm"] = float(norms.norm())
                    else:
                        row["rule_grad_norm"] = grad_norm if win_n_rule else 0.0
                        row["replay_grad_norm"] = 0.0 if win_n_rule else grad_norm
                log_rows.append(row)
                print(
                    f"[c1] ep={ep+1} step={global_step}/{stop_steps} (sched={total_steps}) "
                    f"sft={row['sft_loss']:.4f} margin={row['margin_loss']:.4f} gnorm={row['grad_norm']:.3f}"
                )
        norms = M.module_delta_norms(model)
        print(f"[c1] end epoch {ep+1} (step={global_step}): {norms}")

    final_norms = M.module_delta_norms(model)
    # Always save tiny unmerged adapter state for zero-retraining post-hoc pathway ablations
    adapter_state_path = args.out_dir / "adapter_state.pt"
    M.save_c1_adapter_state(model, adapter_state_path)
    print(f"[c1] saved unmerged adapter state -> {adapter_state_path}")

    meta = {
        "model": args.model,
        "model_key": args.model_key,
        "arm": args.arm,
        "arm_spec": arm_spec,
        "trainable_params": attach_info["trainable_params"],
        "hyperparams": {
            "epochs": args.epochs,
            "max_steps": args.max_steps,
            "stop_steps": stop_steps,
            "save_steps": sorted(save_steps),
            "zero_replay_loss": args.zero_replay_loss,
            "mask_rule_format_ce": args.mask_rule_format_ce,
            "log_source_grad_norms": args.log_source_grad_norms,
            "grad_accum": args.grad_accum,
            "microbatch": args.microbatch,
            "lr": args.lr,
            "mlp_lr_mult": args.mlp_lr_mult,
            "scaling_mode": args.scaling_mode,
            "base_scale": args.base_scale,
            "anchor_rank": args.anchor_rank,
            "lambda_margin": lambda_margin,
            "margin_m": args.margin_m,
            "accept_weight": args.accept_weight,
            "seed": args.seed,
            "n_examples": len(encoded),
            "n_rule_aligned": n_rule_aligned,
            "n_replay": n_replay,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
        },
        "saved_checkpoints": saved_checkpoints,
        "final_delta_norms": final_norms,
        "elapsed_seconds": round(time.time() - t0, 1),
        "log": log_rows,
    }
    (args.out_dir / "train_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if not args.no_merge_save:
        model.eval()
        M.merge_c1_adapters(model)
        model.config.use_cache = True
        merged_dir = args.out_dir / "merged_hf"
        model.save_pretrained(merged_dir, safe_serialization=True)
        tok.save_pretrained(merged_dir)
        print(f"[c1] saved merged HF checkpoint -> {merged_dir}")


if __name__ == "__main__":
    main()
