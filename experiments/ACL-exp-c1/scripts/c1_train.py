#!/usr/bin/env python3
"""ACL-exp-c1 trainer: Dual-Pathway Asymmetric Circuit LoRA with Paired Selective Margin Loss.

Supports all 8 frozen arms from data/c1_components_frozen.json:
  Targeted-Dual | Targeted-Form | Pinpoint-DLA | Pinpoint-AtP | Random-LM | Uniform-LoRA | Hub-Only | Dual-wo-Hub
Plus ablations:
  --lambda-margin 0.0  (WCA-unpaired / SFT-only ablation)
  --replay-file <jsonl> (15-20% general instruction/reasoning replay)
  --think-file  <jsonl> (Recipe B: mixed no_think + self-verified think traces)

Usage (GPU):
  CUDA_VISIBLE_DEVICES=0 python experiments/ACL-exp-c1/scripts/c1_train.py \
      --model /mnt/kcl/devuser/ckpts/Qwen3-8B --model-key qwen3_8b --arm Targeted-Dual --out-dir checkpoints/qwen3_8b/Targeted-Dual

Usage (CPU self-test):
  python experiments/ACL-exp-c1/scripts/c1_train.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
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


def encode_sft_example(tok, ex: dict, enable_thinking: bool = False) -> dict:
    """Encode (messages, assistant_target) with prompt tokens masked (-100) and exact Final: readout index."""
    prompt_text = tok.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking
    )
    target_text = ex["assistant_target"]
    eos = getattr(tok, "eos_token", "") or ""
    prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
    full_ids = tok.encode(prompt_text + target_text + eos, add_special_tokens=False)
    # Guard against BPE merge at boundary
    if full_ids[: len(prompt_ids)] != prompt_ids:
        target_ids = tok.encode(target_text + eos, add_special_tokens=False)
        full_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]

    # Locate the exact logit index predicting the option letter right after "Final:"
    readout_pos = None
    pos_tok = neg_tok = None
    if ex.get("margin_sign", 0) != 0 and ex.get("initial_answer") and ex.get("target_answer"):
        key_ids = tok.encode("Final:", add_special_tokens=False)
        anchor = tok.encode(prompt_text + "Final:", add_special_tokens=False)
        if anchor[-len(key_ids) :] == key_ids and len(anchor) < len(full_ids):
            # Logit at position (len(anchor) - 1) predicts token at len(anchor) (i.e. " <A|B|C|D>")
            readout_pos = len(anchor) - 1
            pos_tok, neg_tok = B1L.readout_token_ids(tok, "Final:", ex["target_answer"], ex["initial_answer"])

    return {
        "example_id": ex.get("example_id", ""),
        "pair_group_id": ex.get("pair_group_id", ex.get("example_id", "")),
        "input_ids": torch.tensor(full_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "margin_sign": int(ex.get("margin_sign", 0)),
        "readout_pos": readout_pos,
        "pos_tok": pos_tok,
        "neg_tok": neg_tok,
    }


def load_training_records(
    train_path: Path,
    think_path: Path | None,
    replay_path: Path | None,
    replay_ratio: float,
    seed: int,
    limit: int | None = None,
) -> list[tuple[dict, bool]]:
    """Return list of (example_dict, enable_thinking) grouped by pair_group_id with replay interleaved."""
    rows = [json.loads(l) for l in train_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if limit:
        rows = rows[:limit]
    items: list[tuple[dict, bool]] = [(r, False) for r in rows]

    if think_path and think_path.exists():
        trows = [json.loads(l) for l in think_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if limit:
            trows = trows[: max(1, limit // 3)]
        items.extend((r, True) for r in trows)

    if replay_path and replay_path.exists() and replay_ratio > 0:
        rrows = [json.loads(l) for l in replay_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        rng = random.Random(seed)
        n_want = int(len(items) * replay_ratio / max(1e-6, 1.0 - replay_ratio))
        if len(rrows) > n_want:
            rrows = rng.sample(rrows, n_want)
        for r in rrows:
            r.setdefault("margin_sign", 0)
            items.append((r, bool(r.get("enable_thinking", False))))
    return items


def build_group_shuffled_order(encoded: list[dict], seed: int) -> list[dict]:
    """Shuffle by pair_group_id so paired (V, V_hes, F, shell, S, P0) stay inside the same grad-accum step."""
    groups: dict[str, list[dict]] = {}
    for ex in encoded:
        groups.setdefault(ex["pair_group_id"], []).append(ex)
    keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)
    ordered = []
    for k in keys:
        g = list(groups[k])
        rng.shuffle(g)
        ordered.extend(g)
    return ordered


def compute_loss_one(model, ex: dict, lambda_margin: float, margin_m: float) -> tuple[torch.Tensor, float, float]:
    dev = _input_device(model)
    ids = ex["input_ids"][None].to(dev)
    labels = ex["labels"][None].to(dev)
    out = model(input_ids=ids, labels=labels)
    loss_sft = out.loss
    loss_m = torch.tensor(0.0, device=loss_sft.device, dtype=torch.float32)

    if lambda_margin > 0.0 and ex["margin_sign"] != 0 and ex["readout_pos"] is not None:
        logits_at = out.logits[0, ex["readout_pos"]].float()
        u_theta = logits_at[ex["pos_tok"]] - logits_at[ex["neg_tok"]]
        if ex["margin_sign"] > 0:
            loss_m = F.relu(margin_m - u_theta)
        else:
            loss_m = F.relu(margin_m + u_theta)
        total = loss_sft + lambda_margin * loss_m.to(loss_sft.dtype)
    else:
        total = loss_sft
    return total, float(loss_sft.detach()), float(loss_m.detach())


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

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        s = "".join(f"<|{m['role']}|> {m['content']} <|end|> " for m in messages)
        return s + ("<|assistant|> <think> </think> " if add_generation_prompt else "")

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

    tiny_arms = {
        "Targeted-Dual": {"kind": "circuit", "heads": [[1, 0], [1, 2], [2, 1]], "head_rank": 4, "mlp_layers": [1, 2], "mlp_rank": 4},
        "Targeted-Form": {"kind": "circuit", "heads": [[2, 0], [3, 1]], "head_rank": 6, "mlp_layers": [], "mlp_rank": 0},
        "Pinpoint-DLA": {"kind": "circuit", "heads": [[3, 2], [3, 3]], "head_rank": 6, "mlp_layers": [], "mlp_rank": 0},
        "Uniform-LoRA": {"kind": "uniform", "uniform_rank": 2, "target_modules": ["q_proj", "o_proj", "gate_proj", "down_proj"]},
        "Hub-Only": {"kind": "circuit", "heads": [[2, 1]], "head_rank": 4, "mlp_layers": [], "mlp_rank": 0},
        "Dual-wo-Hub": {"kind": "circuit", "heads": [[1, 0], [1, 2]], "head_rank": 4, "mlp_layers": [1, 2], "mlp_rank": 4},
    }

    for aname, spec in tiny_arms.items():
        torch.manual_seed(0)
        m = Qwen3ForCausalLM(cfg).float()
        orig_weights = {k: v.detach().clone() for k, v in m.named_parameters()}
        info = M.apply_c1_adapters(m, spec)
        opt = build_optimizer(m, lr=5e-3, mlp_lr_mult=1.0, weight_decay=0.0)
        m.train()
        opt.zero_grad()
        for ex in encoded[:6]:
            loss, _, _ = compute_loss_one(m, ex, lambda_margin=0.25, margin_m=4.0)
            (loss / 6.0).backward()
        opt.step()

        norms = M.module_delta_norms(m)
        assert norms["attn_delta_frob"] > 0.0, (aname, norms)
        if spec.get("mlp_layers") or spec["kind"] == "uniform":
            assert norms["mlp_delta_frob"] > 0.0, (aname, norms)

        # Check forward equivalence before vs after merge
        m.eval()
        test_ids = encoded[0]["input_ids"][None]
        with torch.no_grad():
            y_adapter = m(input_ids=test_ids).logits
        M.merge_c1_adapters(m)
        with torch.no_grad():
            y_merged = m(input_ids=test_ids).logits
        max_err = float((y_adapter - y_merged).abs().max())
        assert max_err < 1e-4, (aname, max_err)

        # Verify STRICT locality: untouched layers, K/V, embeddings, and untouched head slices have EXACTLY 0 diff!
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
    ap.add_argument("--grad-accum", type=int, default=12, help="multiple of 6 keeps paired groups together")
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--mlp-lr-mult", type=float, default=1.0)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--alpha-ratio", type=float, default=2.0)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lambda-margin", type=float, default=0.25, help="0.0 = WCA-unpaired ablation")
    ap.add_argument("--margin-m", type=float, default=4.0)
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
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    comp = json.loads(Path(args.components_file).read_text(encoding="utf-8"))
    arm_spec = comp["models"][args.model_key]["arms"][args.arm]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=args.device_map
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    attach_info = M.apply_c1_adapters(model, arm_spec, alpha_ratio=args.alpha_ratio, dropout=args.dropout)
    assert attach_info["trainable_params"] == arm_spec["trainable_params"], (
        f"Param mismatch: actual={attach_info['trainable_params']} vs frozen={arm_spec['trainable_params']}"
    )
    print(f"[c1] {args.model_key} | {args.arm} | trainable={attach_info['trainable_params']:,}")

    raw_items = load_training_records(
        Path(args.train_file), args.think_file, args.replay_file, args.replay_ratio, args.seed, args.limit
    )
    encoded = [encode_sft_example(tok, r, thk) for r, thk in raw_items]
    print(f"[c1] encoded {len(encoded)} training examples (margin_active={sum(e['readout_pos'] is not None for e in encoded)})")

    opt = build_optimizer(model, lr=args.lr, mlp_lr_mult=args.mlp_lr_mult, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(encoded) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_rows = []
    t0 = time.time()
    global_step = 0
    model.train()

    for ep in range(args.epochs):
        ordered = build_group_shuffled_order(encoded, seed=args.seed + ep)
        opt.zero_grad()
        acc_sft = acc_m = 0.0
        acc_n = 0
        for i, ex in enumerate(ordered):
            loss, lsft, lm = compute_loss_one(model, ex, args.lambda_margin, args.margin_m)
            (loss / args.grad_accum).backward()
            acc_sft += lsft
            acc_m += lm
            acc_n += 1
            if acc_n == args.grad_accum or (i + 1) == len(ordered):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.max_grad_norm
                )
                fac = lr_factor(global_step, total_steps, warmup_steps)
                for gi, pg in enumerate(opt.param_groups):
                    base_lr = args.lr if gi == 0 else args.lr * args.mlp_lr_mult
                    pg["lr"] = base_lr * fac
                opt.step()
                opt.zero_grad()
                global_step += 1
                if global_step % 10 == 0 or global_step == total_steps:
                    row = {
                        "epoch": ep + 1,
                        "step": global_step,
                        "sft_loss": round(acc_sft / acc_n, 4),
                        "margin_loss": round(acc_m / acc_n, 4),
                        "lr": opt.param_groups[0]["lr"],
                        "elapsed_s": round(time.time() - t0, 1),
                    }
                    log_rows.append(row)
                    print(f"[c1] ep={ep+1} step={global_step}/{total_steps} sft={row['sft_loss']:.4f} margin={row['margin_loss']:.4f}")
                acc_sft = acc_m = 0.0
                acc_n = 0
        norms = M.module_delta_norms(model)
        print(f"[c1] end epoch {ep+1}: {norms}")

    final_norms = M.module_delta_norms(model)
    meta = {
        "model": args.model,
        "model_key": args.model_key,
        "arm": args.arm,
        "arm_spec": arm_spec,
        "trainable_params": attach_info["trainable_params"],
        "hyperparams": {
            "epochs": args.epochs,
            "grad_accum": args.grad_accum,
            "lr": args.lr,
            "mlp_lr_mult": args.mlp_lr_mult,
            "lambda_margin": args.lambda_margin,
            "margin_m": args.margin_m,
            "seed": args.seed,
            "n_examples": len(encoded),
        },
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
