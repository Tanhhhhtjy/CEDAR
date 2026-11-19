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

_OPT = r"(?:option\s*)?[\(\[]?\s*\**\s*([ABCD])\s*\**\s*[\)\]]?(?![A-Za-z])"
_SEP = r"\s*\**\s*[:：]\s*\**\s*"

FINAL_RE = re.compile(r"\**\s*final(?:\s+(?:answer|classification|choice))?" + _SEP + _OPT, re.IGNORECASE)
VERDICT_RE = re.compile(
    r"\**\s*(?:argument\s*check|argumentcheck|verdict)" + _SEP + r"(VALID|INVALID|NA|NONE|N/A)\b",
    re.IGNORECASE,
)
STEP_RE = re.compile(r"\**\s*broken\s*step" + _SEP + r"(1|2|3|NONE|N/A|NA)\b", re.IGNORECASE)


def _strip_think(text: str) -> tuple[str, bool]:
    """Return (answer_text, think_unclosed)."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1], False
    if "<think>" in text:
        return "", True
    return text, False


def _last(regex: re.Pattern[str], text: str) -> str | None:
    hits = regex.findall(text)
    return hits[-1] if hits else None


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

    final = _last(FINAL_RE, answer_text)
    verdict = _last(VERDICT_RE, answer_text)
    step = _last(STEP_RE, answer_text)

    final = final.upper() if final else None
    if verdict:
        verdict = verdict.upper()
        if verdict in ("NONE", "N/A"):
            verdict = "NA"
    if step:
        step = step.upper()
        step = int(step) if step.isdigit() else "NONE"

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
    ]
    for raw, req, (f, v, s, st) in cases:
        out = parse_output(raw, requires_verdict=req)
        got = (out["parsed_final_option"], out["parsed_verdict"], out["parsed_broken_step"], out["parse_status"])
        assert got == (f, v, s, st), f"{raw!r}: got {got}, want {(f, v, s, st)}"
    print("[OK] a1_parse self-test passed")


if __name__ == "__main__":
    _self_test()
