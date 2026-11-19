#!/usr/bin/env python3
"""Reference chat-message renderer for ACL-exp-a1 (protocol v2.2; supersedes render_prompts_v2_1.py).

The runner (run_a1.py on the GPU host) and ACL-exp-b1 MUST call `build_messages()` instead of
re-implementing prompt assembly, so prompts cannot drift from configs/exp_a1_config.yaml.

    from render_prompts import load_config, build_messages, max_new_tokens, gold, all_arms, contract_of
    cfg = load_config()
    msgs = build_messages(cfg, row, arm, regime, call1=None, merge_system=(family == "gemma"))
    # Qwen3: tokenizer.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=(regime == "think"))

CLI:  python render_prompts.py [--arm F_hes] [--regime think] [--base ACL-A1-SYM-001]
      Renders every arm for every row of every dataset (sanity check) and prints one example.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from a1_arm_wrappers import CONTEXT_OF_ARM  # noqa: E402

EXP = HERE.parents[0]
REPO = EXP.parents[1]

MODULE5 = {
    "standalone_neutral_V", "standalone_neutral_F",
    "dialog_explicit_verify_V", "dialog_explicit_verify_F",
    "two_call_pipeline_V", "two_call_pipeline_F",
    "verify_then_revise_prompt_V", "verify_then_revise_prompt_F",
}
# Module 7: same Turn-2 text as the base arm, global-verify sentence added to the system prompt.
SYSVERIFY_BASE = {"F_sysverify": "F", "V_sysverify": "V", "V_hes_sysverify": "V_hes", "shell_only_sysverify": "shell_only"}
GOLD_TARGET_ARMS = {"V", "V_auth", "V_hes", "V_sysverify", "V_hes_sysverify"}


def load_config(path: Path | None = None) -> dict[str, Any]:
    path = path or EXP / "configs" / "exp_a1_config.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def contract_of(cfg: dict[str, Any], arm: str) -> str:
    for contract, arms in cfg["arm_contracts"].items():
        if arm in arms:
            return contract
    raise KeyError(f"arm {arm!r} has no contract in config")


def requires_verdict(cfg: dict[str, Any], arm: str) -> bool:
    return arm in cfg["parsing"]["requires_verdict_arms"]


def max_new_tokens(cfg: dict[str, Any], arm: str, regime: str, family: str = "qwen") -> int:
    spec = cfg["regimes"][regime]["max_new_tokens"]
    if regime == "no_think":
        vtr = contract_of(cfg, arm) == "verify_then_revise"
        if family == "gemma":
            return spec["gemma_verify_then_revise"] if vtr else spec["gemma_default"]
        return spec["verify_then_revise"] if vtr else spec["default"]
    return spec["qwen"] if family == "qwen" else spec["gemma_cot"]


def context_key(arm: str) -> str:
    if arm in ("turn1_V", "turn1_F"):
        return "context_prompt_v" if arm == "turn1_V" else "context_prompt_f"
    if arm in MODULE5:
        return "context_prompt_v" if arm.endswith("_V") else "context_prompt_f"
    if arm in SYSVERIFY_BASE:
        return CONTEXT_OF_ARM[SYSVERIFY_BASE[arm]]
    return CONTEXT_OF_ARM[arm]


def _system(cfg: dict[str, Any], arm: str, regime: str) -> str:
    p = cfg["prompts"]
    base = p["system_base"]
    if arm in SYSVERIFY_BASE:
        base = base + " " + p["global_verify_sentence"]
    return base + "\n\n" + p["contracts"][contract_of(cfg, arm)][regime]


def build_messages(
    cfg: dict[str, Any],
    row: dict[str, Any],
    arm: str,
    regime: str,
    call1: dict[str, Any] | None = None,
    merge_system: bool = False,
) -> list[dict[str, str]]:
    p = cfg["prompts"]
    tpl = p["module5_user_templates"]
    ctx = row[context_key(arm)]
    system = _system(cfg, arm, regime)

    if arm in ("turn1_V", "turn1_F"):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": ctx}]
    elif arm.startswith("standalone_neutral_"):
        argument = row["arms"]["V" if arm.endswith("_V") else "F"]
        user = tpl["standalone_neutral"].format(context=ctx, argument=argument)
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    else:
        if arm in MODULE5:
            argument = row["arms"]["V" if arm.endswith("_V") else "F"]
            if arm.startswith("dialog_explicit_verify_"):
                turn2 = tpl["dialog_explicit_verify"].format(argument=argument)
            elif arm.startswith("verify_then_revise_prompt_"):
                turn2 = tpl["verify_then_revise_prompt"].format(argument=argument)
            elif arm.startswith("two_call_pipeline_"):
                if call1 is None:
                    raise ValueError("two_call arms need call1={'argument_check':..., 'broken_step':...}")
                turn2 = tpl["two_call_pipeline_call2"].format(
                    argument=argument,
                    call1_argument_check=call1.get("argument_check") or "UNPARSED",
                    call1_broken_step=call1.get("broken_step") if call1.get("broken_step") is not None else "UNPARSED",
                )
            else:  # pragma: no cover
                raise KeyError(arm)
        elif arm in SYSVERIFY_BASE:
            turn2 = row["arms"][SYSVERIFY_BASE[arm]]
        else:
            turn2 = row["arms"][arm]
        turn1_assistant = p["frozen_turn1_assistant"].format(initial_answer=row["initial_answer"])
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": ctx},
            {"role": "assistant", "content": turn1_assistant},
            {"role": "user", "content": turn2},
        ]

    if merge_system:
        sys_text = msgs[0]["content"]
        msgs = msgs[1:]
        msgs[0] = {"role": "user", "content": sys_text + "\n\n" + msgs[0]["content"]}
    return msgs


def gold(row: dict[str, Any], arm: str) -> dict[str, Any]:
    side_v = arm.endswith("_V") and arm in MODULE5 or arm in GOLD_TARGET_ARMS
    if arm in ("turn1_V", "turn1_F"):
        return {"gold_answer": row["initial_answer"], "gold_verdict": "NA", "gold_broken_step": "NONE"}
    out = {"gold_answer": row["target_answer"] if side_v else row["initial_answer"]}
    if arm.startswith(("standalone_neutral_", "verify_then_revise_prompt_")):
        out["gold_verdict"] = "VALID" if side_v else "INVALID"
        out["gold_broken_step"] = "NONE" if side_v else row["broken_step_index"]
    else:
        out["gold_verdict"], out["gold_broken_step"] = "NA", "NONE"
    return out


def all_arms(cfg: dict[str, Any]) -> list[str]:
    arms = ["turn1_V", "turn1_F"]
    for k, v in cfg["arms"].items():
        if k.startswith("module"):
            arms.extend(v)
    return arms


def dataset_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    return {k: REPO / v["path"] for k, v in cfg["datasets"].items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="F_src_self_draft")
    ap.add_argument("--regime", default="no_think", choices=["no_think", "think"])
    ap.add_argument("--base", default="ACL-A1-SYM-001")
    args = ap.parse_args()
    cfg = load_config()
    contracted = {a for v in cfg["arm_contracts"].values() for a in v}
    assert set(all_arms(cfg)) == contracted, set(all_arms(cfg)) ^ contracted
    n = 0
    example = None
    for key, path in dataset_paths(cfg).items():
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            for arm in all_arms(cfg):
                for regime in ("no_think", "think"):
                    c1 = {"argument_check": "INVALID", "broken_step": 1}
                    msgs = build_messages(cfg, row, arm, regime, call1=c1)
                    build_messages(cfg, row, arm, regime, call1=c1, merge_system=True)
                    g = gold(row, arm)
                    assert g["gold_answer"] in "ABCD"
                    for fam in ("qwen", "gemma"):
                        max_new_tokens(cfg, arm, regime, fam)
                    sys_txt = msgs[0]["content"]
                    assert "only the information given" not in sys_txt
                    if arm not in SYSVERIFY_BASE:
                        assert "check every step" not in sys_txt.lower()
                    n += 1
                    if row["base_id"] == args.base and arm == args.arm and regime == args.regime:
                        example = msgs
    print(f"[OK] rendered {n} (row, arm, regime) prompts without error")
    if example:
        for m in example:
            print(f"----- {m['role']} -----\n{m['content']}")


if __name__ == "__main__":
    main()
