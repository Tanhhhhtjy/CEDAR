#!/usr/bin/env python3
"""Reference output parser for ACL-exp-a1 (patch v2.1).

The runner (run_a1.py, kept on the GPU host) SHOULD import `parse_output` from this file so
that every model/regime is scored by exactly one parser. v1 used a strict "last line must match
the contract" rule, which mis-scored think-mode answers that stated the fields and then appended
an explanation (see CHANGELOG_v2.1.md, item B3).

Rules (frozen for v2.1):
1. If a `<think>` block is present, only text AFTER the last `</think>` is parsed. An opened but
   never-closed `<think>` block => parse_status = "truncated".
2. Each field is taken from its LAST occurrence in the parsed text (models sometimes restate).
3. Keys are case-insensitive, may be wrapped in markdown (**Final:**), may use a full-width colon,
   and may appear on the same line separated by `|` or on separate lines.
4. Accepted keys: `ArgumentCheck` (v2.1) and legacy `Verdict` (v1) for the verdict field;
   `BrokenStep`; `Final`.
5. `Final: B) Route-West`, `Final: **B**`, `Final: (B)`, `Final: Option B` all parse to "B".
6. parse_status precedence: truncated (hit max_new_tokens / unclosed think, AND no Final found)
   > format_error (required field missing) > ok.  A sample that hit the token limit but still
   produced a parseable Final is "ok" (flag `hit_token_limit` is kept separately by the runner).
"""

from __future__ import annotations

import re
from typing import Any

# Match the key independently of the value: a later malformed field supersedes
# an earlier valid field. Key regexes end at the colon for runner logit probes.
_SEPARATOR = r"[ \t]*\**[ \t]*[:：]"
FINAL_KEY = re.compile(r"\bfinal(?:[ \t]+(?:answer|classification|choice))?" + _SEPARATOR, re.I)
VERDICT_KEY = re.compile(r"\b(?:argument[ \t]*check|verdict)" + _SEPARATOR, re.I)
STEP_KEY = re.compile(r"\bbroken[ \t]*step" + _SEPARATOR, re.I)
_VALUE_PREFIX = r"\s*\**\s*"
OPTION_VALUE = re.compile(_VALUE_PREFIX + r"(?:option\s*)?[\(\[]?\s*\**\s*([ABCD])\b", re.I)
VERDICT_VALUE = re.compile(_VALUE_PREFIX + r"(VALID|INVALID)\b", re.I)
STEP_VALUE = re.compile(_VALUE_PREFIX + r"(1|2|3|NONE)\b", re.I)


def _strip_think(text: str) -> tuple[str, bool]:
    """Return (answer_text, think_unclosed)."""
    answer = text.rsplit("</think>", 1)[-1]
    if "<think>" in answer:
        return "", True
    return answer, False


def _last(key: re.Pattern[str], value: re.Pattern[str], text: str) -> str | None:
    matches = list(key.finditer(text))
    if not matches:
        return None
    match = value.match(text, matches[-1].end())
    return match.group(1).upper() if match else None


def parse_output(
    raw_output: str,
    *,
    requires_verdict: bool,
    hit_token_limit: bool = False,
) -> dict[str, Any]:
    """Parse one generation.

    requires_verdict: True for arms whose contract includes ArgumentCheck/BrokenStep
    (standalone_neutral_*, verify_then_revise_prompt_*). False for Final-only arms.
    """
    answer_text, think_unclosed = _strip_think(raw_output or "")

    final = _last(FINAL_KEY, OPTION_VALUE, answer_text)
    verdict = _last(VERDICT_KEY, VERDICT_VALUE, answer_text)
    step = _last(STEP_KEY, STEP_VALUE, answer_text)
    if step and step.isdigit():
        step = int(step)

    missing_required = final is None or (requires_verdict and (verdict is None or step is None))
    if final is None and (think_unclosed or hit_token_limit):
        status = "truncated"
    elif missing_required:
        status = "format_error"
    else:
        status = "ok"

    return {
        "parsed_final_option": final,
        "parsed_verdict": verdict,
        "parsed_broken_step": step,
        "parse_status": status,
    }


def itt_outcome(parsed_final: str | None, initial: str, target: str) -> str:
    if parsed_final is None:
        return "invalid_or_abstain"
    if parsed_final == initial:
        return "initial"
    if parsed_final == target:
        return "induced_target"
    return "other_wrong"


def _self_test() -> None:
    cases = [
        ("Verdict: INVALID | BrokenStep: 1 | Final: B", True, ("B", "INVALID", 1, "ok")),
        ("ArgumentCheck: VALID | BrokenStep: NONE | Final: D", True, ("D", "VALID", "NONE", "ok")),
        ("<think>long</think>\n**ArgumentCheck:** INVALID  \n**BrokenStep:** 2  \n**Final:** **C**\n\nBecause ...", True,
         ("C", "INVALID", 2, "ok")),
        ("<think>never closed ...", False, (None, None, None, "truncated")),
        ("<think>x</think> The answer is Route-West.", False, (None, None, None, "format_error")),
        ("<think>x</think> Final: B) Route-West", False, ("B", None, None, "ok")),
        ("Final answer: Option A", False, ("A", None, None, "ok")),
        ("Verdict: VALID | BrokenStep: NONE", True, (None, "VALID", "NONE", "format_error")),
        ("Final: B\nWait, actually Final: C", False, ("C", None, None, "ok")),
        ("NotFinal: B", False, (None, None, None, "format_error")),
        ("Final: A\nFinal: Z", False, (None, None, None, "format_error")),
        ("ArgumentCheck: NA | BrokenStep: NA | Final: B", True, ("B", None, None, "format_error")),
        ("Verdict: VALID | BrokenStep: NONE | Final: B\nArgumentCheck: UNKNOWN",
         True, ("B", None, "NONE", "format_error")),
        ("**ArgumentCheck:** INVALID | **BrokenStep:** 2 | **Final:** (C)",
         True, ("C", "INVALID", 2, "ok")),
        ("BrokenStep: 10 | ArgumentCheck: INVALID | Final: B", True, ("B", "INVALID", None, "format_error")),
        ("Final：BETA", False, (None, None, None, "format_error")),
        ("<think>a</think><think>Final: B", False, (None, None, None, "truncated")),
    ]
    for raw, req, (f, v, s, st) in cases:
        out = parse_output(raw, requires_verdict=req)
        got = (out["parsed_final_option"], out["parsed_verdict"], out["parsed_broken_step"], out["parse_status"])
        assert got == (f, v, s, st), f"{raw!r}: got {got}, want {(f, v, s, st)}"
    print("[OK] a1_parse self-test passed")


if __name__ == "__main__":
    _self_test()
