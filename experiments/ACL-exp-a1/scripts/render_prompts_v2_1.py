#!/usr/bin/env python3
"""Reference chat-message renderer for ACL-exp-a1 patch v2.1.

The runner (run_a1.py on the GPU host) SHOULD call `build_messages()` instead of re-implementing
prompt assembly, so prompts cannot drift from configs/exp_a1_config.yaml.

    from render_prompts_v2_1 import load_config, build_messages, max_new_tokens, requires_verdict
    cfg = load_config()
    msgs = build_messages(cfg, row, arm, regime, call1=None)      # list[{"role","content"}]
    # Qwen3: tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
    #                                      enable_thinking=(regime == "think"))
    # Gemma-3: pass merge_system=True (system text is prepended to the first user turn).

CLI:  python render_prompts_v2_1.py [--arm F] [--regime think] [--base ACL-A1-SYM-001]
      Renders every arm for every row (sanity check) and prints one example.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[0]
REPO = EXP.parents[1]

MODULE5 = {
    "standalone_neutral_V", "standalone_neutral_F",
    "dialog_explicit_verify_V", "dialog_explicit_verify_F",
    "two_call_pipeline_V", "two_call_pipeline_F",
    "verify_then_revise_prompt_V", "verify_then_revise_prompt_F",
}
V_SIDE_DIALOG = {"V", "V_auth", "V_hes"}


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
        return spec["verify_then_revise"] if contract_of(cfg, arm) == "verify_then_revise" else spec["default"]
    return spec["qwen"] if family == "qwen" else spec["gemma_cot"]


def _system(cfg: dict[str, Any], arm: str, regime: str) -> str:
    p = cfg["prompts"]
    return p["system_base"] + "\n\n" + p["contracts"][contract_of(cfg, arm)][regime]


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
    side_v = arm.endswith("_V") or arm in V_SIDE_DIALOG
    ctx = row["context_prompt_v"] if side_v else row["context_prompt_f"]
    system = _system(cfg, arm, regime)

    if arm in ("turn1_V", "turn1_F"):
        ctx = row["context_prompt_v"] if arm == "turn1_V" else row["context_prompt_f"]
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": ctx}]
    elif arm.startswith("standalone_neutral_"):
        argument = row["arms"]["V" if side_v else "F"]
        user = tpl["standalone_neutral"].format(context=ctx, argument=argument)
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    else:
        if arm in MODULE5:
            argument = row["arms"]["V" if side_v else "F"]
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
    side_v = arm.endswith("_V") or arm in V_SIDE_DIALOG
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="F_src_self_draft")
    ap.add_argument("--regime", default="no_think", choices=["no_think", "think"])
    ap.add_argument("--base", default="ACL-A1-SYM-001")
    args = ap.parse_args()
    cfg = load_config()
    n = 0
    example = None
    for key in ("symbolic_pilot", "natural_pilot"):
        path = REPO / cfg["datasets"][key]["path"]
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            for arm in all_arms(cfg):
                for regime in ("no_think", "think"):
                    c1 = {"argument_check": "INVALID", "broken_step": 1}
                    msgs = build_messages(cfg, row, arm, regime, call1=c1)
                    build_messages(cfg, row, arm, regime, call1=c1, merge_system=True)
                    gold(row, arm)
                    max_new_tokens(cfg, arm, regime)
                    n += 1
                    if row["base_id"] == args.base and arm == args.arm and regime == args.regime:
                        example = msgs
    print(f"[OK] rendered {n} (row, arm, regime) prompts without error")
    if example:
        for m in example:
            print(f"----- {m['role']} -----\n{m['content']}")


if __name__ == "__main__":
    main()
