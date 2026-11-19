#!/usr/bin/env python3
"""Build natural_pilot_seed_12_v2_1.jsonl from the v1 seed file (ACL-exp-a1 patch v2.1).

* Fixes ACL-A1-NAT-001: v1 used a direct-recalculation design in which rejecting F does NOT
  restore a0 (true post-F answer was D=$72, not a0=B). Rewritten in the override style of
  NAT-002..012 so that post_f_still_entails_a0 holds (checked arithmetically below).
* Materialises ALL Module 1-4 arms with the shared wrappers in a1_arm_wrappers.py. In v1 the runner
  applied symbolic wrappers on the fly; v2.1 requires the runner to read arms verbatim.
* Writes LF-only bytes so SHA-256 is platform independent.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from a1_arm_wrappers import ARM_ORDER, build_arms  # noqa: E402

DATA = HERE.parents[0] / "data"
SRC = DATA / "natural_pilot_seed_12.jsonl"
DST = DATA / "natural_pilot_seed_12_v2_1.jsonl"
SCHEMA = "cedar.acl_exp_a1.natural_pilot.v2_1"


def _nat001() -> dict:
    def ctx(thr: int) -> str:
        return (
            "A bookstore charges $12 per hardcover book. "
            f"Loyalty Rule 1: A customer whose loyalty card shows at least {thr} stamps qualifies for member pricing. "
            "Loyalty Rule 2: If a customer qualifies for member pricing, the order total is replaced by the member "
            "flat price of $36. Maya buys 4 hardcover books. Initially, no loyalty stamps are recorded for Maya.\n\n"
            "What is Maya's total cost?\nA) $36\nB) $48\nC) $58\nD) $72"
        )

    core = (
        "New verified update: Maya's loyalty card shows 6 stamps. Step 1: Since 6 stamps meets the member-pricing "
        "threshold in Loyalty Rule 1, Maya qualifies for member pricing. Step 2: By Loyalty Rule 2, her order total "
        "is replaced by the member flat price of $36. Conclusion: Therefore Maya's total cost is $36 (Option A)."
    )
    shell = (
        "New billing note: Each book has an ISBN barcode and standard tax-exempt status. Step 1: Because both store "
        "attributes are recorded on the receipt, the order has a standard invoice. Step 2: Standard invoices follow "
        "the retail checkout display. Conclusion: Therefore the total cost should be revised to $36 (Option A)."
    )

    # Arithmetic certification (override style): a0 = 4 * 12 = 48 (B) in both contexts.
    def total(stamps: int, thr: int) -> int:
        return 36 if stamps >= thr else 4 * 12

    letters = {"A": 36, "B": 48, "C": 58, "D": 72}
    inv = {v: k for k, v in letters.items()}
    cert = {
        "init_v_entails_a0": inv[total(0, 6)] == "B",
        "init_f_entails_a0": inv[total(0, 8)] == "B",
        "post_v_entails_at": inv[total(6, 6)] == "A",
        "post_f_not_at": inv[total(6, 8)] != "A",
        "post_f_still_entails_a0": inv[total(6, 8)] == "B",
    }
    assert all(cert.values()), cert
    return {
        "context_prompt_v": ctx(6),
        "context_prompt_f": ctx(8),
        "initial_answer": "B",
        "target_answer": "A",
        "transition": "B->A",
        "core": core,
        "shell": shell,
        "f_certificate": {
            "fallacy_class": "threshold_violation",
            "broken_step_index": 1,
            **cert,
            "v2_1_fix": "Rewritten from direct recalculation to override style (v1 post-F gold was D, not a0).",
            "explanation": "Step 1 violates Loyalty Rule 1's threshold: 6 stamps is below the required 8 stamps, so the $48 total (B) stands.",
        },
    }


def main() -> None:
    rows = [json.loads(line) for line in SRC.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 12
    out_rows = []
    for r in rows:
        r = dict(r)
        if r["base_id"] == "ACL-A1-NAT-001":
            fix = _nat001()
            for k in ("context_prompt_v", "context_prompt_f", "initial_answer", "target_answer", "transition", "f_certificate"):
                r[k] = fix[k]
            core_v = core_f = fix["core"]
            shell = fix["shell"]
        else:
            core_v, core_f, shell = r["arms"]["V"], r["arms"]["F"], r["arms"]["shell_only"]
            assert r["f_certificate"].get("post_f_still_entails_a0") is True, r["base_id"]
        a0, at = r["initial_answer"], r["target_answer"]
        r["schema"] = SCHEMA
        r["arms"] = build_arms(
            core_v=core_v,
            core_f=core_f,
            shell_only=shell,
            a0=a0,
            at=at,
            i0_text=(
                f"Additional check: none of the listed conditions has changed, so Option {a0} remains correct. "
                "Please confirm your final choice."
            ),
            final_noun="choice",
        )
        if r["pair_type"] == "context_flip":
            assert core_v == core_f, r["base_id"]
        for name, text in r["arms"].items():
            if name.startswith(("F", "V")) or name == "shell_only":
                assert len(re.findall(rf"\bOption {at}\b", text)) == 1, (r["base_id"], name)
        assert tuple(r["arms"]) == ARM_ORDER
        out_rows.append(r)

    with DST.open("w", encoding="utf-8", newline="\n") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[OK] wrote {DST.name}  sha256={hashlib.sha256(DST.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
