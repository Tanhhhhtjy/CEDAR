#!/usr/bin/env python3
"""Reproducible vLLM capability evaluator (IFEval + GSM8K) with native `--peft-lora-dir` support.

Strictly follows `experiments/ACL-exp-c1/results/phase0/capabilities/spec.json`:
  - IFEval (541 prompts): no_think (`enable_thinking=False`), full user prompt unchanged, NO system message,
    temperature=0, seed=42, max_tokens=3072, dtype=bfloat16.
  - GSM8K (1,319 questions): think (`enable_thinking=True`), zero-shot CoT suffix
    `\n\nSolve the problem step by step. On the last line, write #### followed by the final numerical answer.`,
    strict extraction of last numeric `####` answer line outside a completed `</think>` block.
  - Native LoRA (`--peft-lora-dir`): initializes `LLM(..., enable_lora=True, max_lora_rank=<MAX_R>)` and passes
    `LoRARequest("c1_lora", 1, str(peft_lora_dir))` to EVERY `llm.generate(...)` call.

Usage (GPU with frozen server data + optional server score.py):
  CUDA_VISIBLE_DEVICES=0 python experiments/ACL-exp-c1/scripts/c1_eval_capabilities.py \
      --model /mnt/kcl/devuser/ckpts/Qwen3-8B \
      --peft-lora-dir <PEFT_LORA_DIR> \
      --ifeval-jsonl <CAP_DIR>/data/ifeval.jsonl \
      --gsm8k-jsonl <CAP_DIR>/data/gsm8k.jsonl \
      --alias std_targeted_dual \
      --out-dir experiments/ACL-exp-c1/results/step2b_8b_showdown/std_targeted_dual/capabilities

Usage (CPU self-test):
  python experiments/ACL-exp-c1/scripts/c1_eval_capabilities.py --selftest-tiny
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c1_export_peft_lora as EXP  # noqa: E402

GSM8K_SUFFIX = "\n\nSolve the problem step by step. On the last line, write #### followed by the final numerical answer."


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_numeric_str(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = raw.strip().replace(",", "").replace("$", "").strip()
    if s.endswith("."):
        s = s[:-1].strip()
    try:
        d = Decimal(s)
    except InvalidOperation:
        return None
    if not d.is_finite():
        return None
    return format(d.normalize(), "f") if d != 0 else "0"


def extract_gsm8k_answer(raw_text: str) -> str | None:
    """Extract last numeric `#### <num>` outside a completed `<think>...</think>` block (exact spec.json rule)."""
    if "<think>" in raw_text:
        if "</think>" not in raw_text:
            return None  # unclosed think block -> unparsed / wrong
        tail = raw_text.rsplit("</think>", 1)[1]
    elif "</think>" in raw_text:
        tail = raw_text.rsplit("</think>", 1)[1]
    else:
        tail = raw_text
    matches = re.findall(r"####\s*([-+]?\$?\d[\d,]*(?:\.\d+)?)", tail)
    if not matches:
        return None
    return normalize_numeric_str(matches[-1])


def extract_gsm8k_gold(answer_field: str) -> str:
    m = re.findall(r"####\s*([-+]?\$?\d[\d,]*(?:\.\d+)?)", answer_field)
    val = normalize_numeric_str(m[-1] if m else answer_field)
    assert val is not None, f"Could not parse GSM8K gold answer: {answer_field!r}"
    return val


def score_ifeval_records(
    rows: list[dict[str, Any]],
    generations: list[dict[str, Any]],
    ifeval_lib_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Score IFEval prompts using official Google `evaluation_lib.test_instruction_following_strict`."""
    if ifeval_lib_dir is not None and str(ifeval_lib_dir) not in sys.path:
        sys.path.insert(0, str(ifeval_lib_dir))
    try:
        import evaluation_lib  # type: ignore
    except ImportError:
        from instruction_following_eval import evaluation_lib  # type: ignore

    items: list[dict[str, Any]] = []
    n_prompt_ok = 0
    n_inst_total = 0
    n_inst_ok = 0
    fr_counts: Counter[str] = Counter()

    for row, gen in zip(rows, generations, strict=True):
        inp = evaluation_lib.InputExample(
            key=row["key"],
            instruction_id_list=row["instruction_id_list"],
            prompt=row["prompt"],
            kwargs=row["kwargs"],
        )
        out = evaluation_lib.test_instruction_following_strict(inp, {row["prompt"]: gen["text"]})
        p_ok = bool(out.follow_all_instructions)
        i_list = [bool(x) for x in out.follow_instruction_list]
        n_prompt_ok += int(p_ok)
        n_inst_total += len(i_list)
        n_inst_ok += sum(i_list)
        fr_counts[gen["finish_reason"]] += 1
        items.append(
            {
                "task": "ifeval",
                "id": row["key"],
                "prompt_strict": p_ok,
                "instruction_strict": i_list,
                "finish_reason": gen["finish_reason"],
            }
        )

    n = len(rows)
    summary = {
        "n_prompts": n,
        "n_correct_prompts": n_prompt_ok,
        "prompt_strict": (n_prompt_ok / n) if n else 0.0,
        "n_instructions": n_inst_total,
        "n_correct_instructions": n_inst_ok,
        "instruction_strict": (n_inst_ok / n_inst_total) if n_inst_total else 0.0,
        "finish_reason": dict(sorted(fr_counts.items())),
    }
    return items, summary


def score_gsm8k_records(
    rows: list[dict[str, Any]],
    generations: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items: list[dict[str, Any]] = []
    n_correct = 0
    n_unparsed = 0
    fr_counts: Counter[str] = Counter()

    for idx, (row, gen) in enumerate(zip(rows, generations, strict=True)):
        gold = extract_gsm8k_gold(str(row["answer"]))
        pred = extract_gsm8k_answer(gen["text"])
        if pred is None:
            n_unparsed += 1
        ok = (pred is not None) and (pred == gold)
        n_correct += int(ok)
        fr_counts[gen["finish_reason"]] += 1
        items.append(
            {
                "task": "gsm8k",
                "id": row.get("id", idx),
                "prediction": pred,
                "correct": ok,
                "finish_reason": gen["finish_reason"],
            }
        )

    n = len(rows)
    summary = {
        "n": n,
        "n_correct": n_correct,
        "accuracy": (n_correct / n) if n else 0.0,
        "n_unparsed": n_unparsed,
        "finish_reason": dict(sorted(fr_counts.items())),
    }
    return items, summary


def run_selftest_tiny() -> None:
    assert extract_gsm8k_answer("<think>2+2=4</think>\n#### 1,234") == "1234"
    assert extract_gsm8k_answer("<think>still thinking #### 42") is None  # unclosed think block -> unparsed!
    assert extract_gsm8k_answer("<think>done</think>\nThe answer is 42.") is None  # missing #### -> unparsed!
    assert extract_gsm8k_answer("Step 1\n#### 18.00") == "18"
    rows = [
        {"id": 0, "question": "1+1?", "answer": "1+1=2\n#### 2"},
        {"id": 1, "question": "1000+234?", "answer": "#### 1,234"},
        {"id": 2, "question": "3+3?", "answer": "#### 6"},
    ]
    gens = [
        {"text": "<think>1+1=2</think>\n#### 2", "finish_reason": "stop"},
        {"text": "<think>ok</think>\n#### 1234", "finish_reason": "stop"},
        {"text": "<think>never closes #### 6", "finish_reason": "length"},
    ]
    items, summ = score_gsm8k_records(rows, gens)
    assert summ["n_correct"] == 2 and summ["n_unparsed"] == 1 and len(items) == 3, summ
    print("[OK] c1_eval_capabilities CPU self-test passed (GSM8K extraction & scoring verified)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="Base HF model path (e.g. Qwen3-8B)")
    ap.add_argument("--peft-lora-dir", type=Path, default=None, help="Optional PEFT LoRA dir for vLLM --enable-lora")
    ap.add_argument("--ifeval-jsonl", type=Path, default=None, help="Frozen IFEval JSONL (541 prompts)")
    ap.add_argument("--gsm8k-jsonl", type=Path, default=None, help="Frozen GSM8K JSONL (1319 questions)")
    ap.add_argument("--ifeval-lib-dir", type=Path, default=None, help="Optional directory containing Google evaluation_lib.py")
    ap.add_argument("--alias", default="model")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=None, help="Optional limit per task for fast pre-flight smoke test")
    ap.add_argument("--skip-ifeval-score", action="store_true", help="Only save raw IFEval generations (if scoring externally)")
    ap.add_argument("--selftest-tiny", action="store_true")
    args = ap.parse_args()

    if args.selftest_tiny:
        run_selftest_tiny()
        return

    assert args.model and args.out_dir
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)

    lora_request = None
    max_lora_rank = None
    peft_sha = None
    if args.peft_lora_dir is not None:
        from vllm.lora.request import LoRARequest

        cfg_path = args.peft_lora_dir / "adapter_config.json"
        if not cfg_path.is_file():
            raise FileNotFoundError(f"--peft-lora-dir missing adapter_config.json: {cfg_path}")
        peft_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        r_export = int(peft_cfg["r"])
        max_lora_rank = EXP.recommended_vllm_max_lora_rank(r_export)
        peft_sha = _sha256_file(args.peft_lora_dir / "adapter_model.safetensors")
        llm = LLM(
            model=args.model,
            dtype="bfloat16",
            tensor_parallel_size=args.tp,
            enable_lora=True,
            max_lora_rank=max_lora_rank,
        )
        lora_request = LoRARequest("c1_lora", 1, str(args.peft_lora_dir))
    else:
        llm = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=args.tp)

    sp = SamplingParams(temperature=0.0, seed=args.seed, max_tokens=args.max_tokens)

    def _run_gen(prompts: list[str]) -> list[dict[str, Any]]:
        outs = llm.generate(prompts, sp, lora_request=lora_request) if lora_request is not None else llm.generate(prompts, sp)
        return [
            {
                "text": o.outputs[0].text,
                "finish_reason": o.outputs[0].finish_reason,
                "n_tokens": len(o.outputs[0].token_ids),
            }
            for o in outs
        ]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    all_items: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "alias": args.alias,
        "model": args.model,
        "peft_lora_dir": str(args.peft_lora_dir) if args.peft_lora_dir else None,
        "peft_adapter_model_sha256": peft_sha,
        "vllm_max_lora_rank": max_lora_rank,
        "limit": args.limit,
    }

    # 1. IFEval (no_think, unchanged user prompt, NO system message)
    if args.ifeval_jsonl is not None:
        ifev_rows = [json.loads(l) for l in args.ifeval_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
        if args.limit:
            ifev_rows = ifev_rows[: args.limit]
        ifev_prompts = [
            tok.apply_chat_template([{"role": "user", "content": r["prompt"]}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for r in ifev_rows
        ]
        ifev_gens = _run_gen(ifev_prompts)
        (args.out_dir / "ifeval_raw.jsonl").write_text(
            "\n".join(json.dumps({"key": r["key"], "prompt": r["prompt"], "response": g["text"], "finish_reason": g["finish_reason"]}, ensure_ascii=False) for r, g in zip(ifev_rows, ifev_gens, strict=True)) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        if not args.skip_ifeval_score:
            ifev_items, ifev_summ = score_ifeval_records(ifev_rows, ifev_gens, ifeval_lib_dir=args.ifeval_lib_dir)
            all_items.extend(ifev_items)
            summary["ifeval"] = ifev_summ

    # 2. GSM8K (think, zero-shot CoT suffix)
    if args.gsm8k_jsonl is not None:
        gsm_rows = [json.loads(l) for l in args.gsm8k_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
        if args.limit:
            gsm_rows = gsm_rows[: args.limit]
        gsm_prompts = [
            tok.apply_chat_template([{"role": "user", "content": r["question"] + GSM8K_SUFFIX}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
            for r in gsm_rows
        ]
        gsm_gens = _run_gen(gsm_prompts)
        (args.out_dir / "gsm8k_raw.jsonl").write_text(
            "\n".join(json.dumps({"id": r.get("id", i), "response": g["text"], "finish_reason": g["finish_reason"]}, ensure_ascii=False) for i, (r, g) in enumerate(zip(gsm_rows, gsm_gens, strict=True))) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        gsm_items, gsm_summ = score_gsm8k_records(gsm_rows, gsm_gens)
        all_items.extend(gsm_items)
        summary["gsm8k"] = gsm_summ

    summary["elapsed_seconds"] = round(time.time() - t0, 1)
    (args.out_dir / "items.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in all_items) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"[OK] wrote {args.out_dir / 'summary.json'}: {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
