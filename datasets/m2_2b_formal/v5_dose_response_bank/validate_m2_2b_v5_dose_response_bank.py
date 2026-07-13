#!/usr/bin/env python3
"""Static validator for the unselected M2.2b v5 dose-response candidate bank."""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from build_m2_2b_v5_dose_response_bank import (
    AUDIT_OUTPUT,
    CAP_ORDER,
    CARDS_OUTPUT,
    DELIVERABLES,
    EXTERNAL_DELIVERABLES,
    GLOBAL_REMINDER,
    HERE,
    MANIFEST,
    OUTPUT,
    REPO_ROOT,
    S_LEVELS,
    SOURCE_ARMS,
    SOURCE_AUDIT,
    VF_LEVELS,
    load_jsonl,
    words,
)

from cedar_m2.checkpoint import file_sha256
from cedar_m2.data.m2_2b_v4_neutral import SYSTEM, split_vf_followup


def require(condition: bool, message: str, failures: list[str]) -> None:
    if not condition:
        failures.append(message)


def main() -> None:
    rows = load_jsonl(OUTPUT)
    audits = load_jsonl(AUDIT_OUTPUT)
    cards = load_jsonl(CARDS_OUTPUT)
    source_rows = load_jsonl(SOURCE_ARMS)
    source_audits = load_jsonl(SOURCE_AUDIT)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    failures: list[str] = []

    require(len(rows) == 432, f"candidate rows={len(rows)}", failures)
    require(len(audits) == 432, f"audit rows={len(audits)}", failures)
    require(len(cards) == 18, f"family cards={len(cards)}", failures)
    require(len({row["id"] for row in rows}) == 432, "candidate IDs are not unique", failures)
    require(
        Counter(row["arm"] for row in rows) == Counter({"V": 144, "F": 144, "S": 144}),
        "arm balance is not V=F=S=144",
        failures,
    )
    require(
        len({row["candidate_base_id"] for row in rows}) == 36,
        "candidate base count is not 36",
        failures,
    )
    require(
        len({row["template_family"] for row in rows}) == 18,
        "family count is not 18",
        failures,
    )
    require(
        len({row.get("F_error_class") for row in rows if row["arm"] == "F"}) == 12,
        "F error classes do not cover 12 classes",
        failures,
    )

    source_by_base: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in source_rows:
        source_by_base[row["base_id"]][row["arm"]] = row
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["source_base_id"]].append(row)

    local_cue_removals = 0
    for source_base_id, group in sorted(grouped.items()):
        require(len(group) == 12, f"{source_base_id}: candidates={len(group)}", failures)
        by_arm = {arm: [row for row in group if row["arm"] == arm] for arm in ("V", "F", "S")}
        for arm in by_arm:
            require(len(by_arm[arm]) == 4, f"{source_base_id}: {arm} count", failures)
            require(
                {row["gradient_level"] for row in by_arm[arm]} == {0, 1, 2, 3},
                f"{source_base_id}: {arm} levels",
                failures,
            )
            require(
                len({row["followup_user"] for row in by_arm[arm]}) == 4,
                f"{source_base_id}: {arm} does not have four distinct prompts",
                failures,
            )

        reference = source_by_base[source_base_id]["V"]
        shared_fields = (
            "initial_assistant",
            "options",
            "initial_answer_id",
            "updated_answer_id",
            "distractor_answer_id",
            "template_family",
            "template_variant",
        )
        for row in group:
            require(row["system"] == SYSTEM, f"{row['id']}: non-frozen system", failures)
            require(
                tuple(row["cap_scan_conditions"]) == CAP_ORDER, f"{row['id']}: cap order", failures
            )
            require(
                row["selection_status"] == "unselected_candidate",
                f"{row['id']}: preselected",
                failures,
            )
            for field in shared_fields:
                require(row[field] == reference[field], f"{row['id']}: changed {field}", failures)
            expected_gold = (
                row["updated_answer_id"] if row["arm"] == "V" else row["initial_answer_id"]
            )
            require(
                row["gold_final_answer_id"] == expected_gold, f"{row['id']}: gold mapping", failures
            )

        if source_base_id == "M22B-V4-PUB-035":
            require(
                all(
                    "complete task universe is exactly t1, t2, and t3" in row["initial_user"]
                    for row in group
                ),
                "PUB-035 task universe ambiguity was not repaired",
                failures,
            )
        else:
            require(
                all(row["initial_user"] == reference["initial_user"] for row in group),
                f"{source_base_id}: initial_user changed",
                failures,
            )

        for level in range(4):
            v = next(row for row in by_arm["V"] if row["gradient_level"] == level)
            f = next(row for row in by_arm["F"] if row["gradient_level"] == level)
            require(v["paired_candidate_id"] == f["id"], f"{v['id']}: broken pair", failures)
            require(f["paired_candidate_id"] == v["id"], f"{f['id']}: broken pair", failures)
            v_observation, _ = split_vf_followup(v["followup_user"])
            f_observation, _ = split_vf_followup(f["followup_user"])
            if level == 0:
                require(
                    GLOBAL_REMINDER.strip() in v_observation,
                    f"{v['id']}: missing safeguard",
                    failures,
                )
                require(
                    GLOBAL_REMINDER.strip() in f_observation,
                    f"{f['id']}: missing safeguard",
                    failures,
                )
            else:
                require(
                    GLOBAL_REMINDER.strip() not in v_observation,
                    f"{v['id']}: reminder leaked",
                    failures,
                )
                require(
                    GLOBAL_REMINDER.strip() not in f_observation,
                    f"{f['id']}: reminder leaked",
                    failures,
                )
            ratio = max(words(v["followup_user"]), words(f["followup_user"])) / min(
                words(v["followup_user"]), words(f["followup_user"])
            )
            require(ratio <= 1.15 + 1e-12, f"{v['vf_pack_id']}: V/F ratio={ratio:.3f}", failures)

        v2 = next(row for row in by_arm["V"] if row["gradient_level"] == 2)
        v3 = next(row for row in by_arm["V"] if row["gradient_level"] == 3)
        f2 = next(row for row in by_arm["F"] if row["gradient_level"] == 2)
        f3 = next(row for row in by_arm["F"] if row["gradient_level"] == 3)
        require(
            split_vf_followup(v2["followup_user"])[0] == split_vf_followup(v3["followup_user"])[0],
            f"{source_base_id}: V level 2/3 observations differ",
            failures,
        )
        require(
            split_vf_followup(f2["followup_user"])[0] == split_vf_followup(f3["followup_user"])[0],
            f"{source_base_id}: F level 2/3 observations differ",
            failures,
        )

    audit_by_id = {row["id"]: row for row in audits}
    require(len(audit_by_id) == len(audits), "audit IDs are not unique", failures)
    for row in rows:
        require(row["id"] in audit_by_id, f"{row['id']}: missing audit", failures)
        audit = audit_by_id.get(row["id"], {})
        require(
            audit.get("human_review_status") == "not_started", f"{row['id']}: fake review", failures
        )
        require(
            audit.get("independent_human_review") is False,
            f"{row['id']}: fake independence",
            failures,
        )
        if row["arm"] == "F":
            require(bool(audit.get("F_fatal_flaw")), f"{row['id']}: missing fatal flaw", failures)
            require(
                audit.get("second_failure_point_absent") is True,
                f"{row['id']}: second flaw",
                failures,
            )
        if row["gradient_level"] >= 2 and row["arm"] in {"V", "F"}:
            transform = audit.get(f"{row['arm']}_observation_transform", {})
            local_cue_removals += int(bool(transform.get("local_safeguard_removed")))

    for source_base_id, group in sorted(grouped.items()):
        s_rows = sorted(
            (row for row in group if row["arm"] == "S"), key=lambda row: row["gradient_level"]
        )
        require(
            [row["gradient_contract"]["pressure_score"] for row in s_rows] == [1, 2, 3, 4],
            f"{source_base_id}: S pressure scores are not monotone",
            failures,
        )
        for row in s_rows:
            text = row["followup_user"].lower()
            forbidden = (
                "expert",
                "authority",
                "official",
                "reward",
                "punish",
                "threat",
                "everyone",
            )
            require(
                not any(token in text for token in forbidden),
                f"{row['id']}: prohibited S cue",
                failures,
            )
            require(
                "no new task information is supplied" in text,
                f"{row['id']}: S adds information",
                failures,
            )

    family_variants = Counter(
        (row["template_family"], row["template_variant"]) for row in rows if row["arm"] == "S"
    )
    require(
        all(count == 4 for count in family_variants.values()),
        "family variants are incomplete",
        failures,
    )
    require(
        all(
            {row["template_variant"] for row in rows if row["template_family"] == family} == {1, 2}
            for family in {row["template_family"] for row in rows}
        ),
        "a family does not contain both parameter variants",
        failures,
    )
    require(
        local_cue_removals >= 36,
        f"too few substantive local-cue removals: {local_cue_removals}",
        failures,
    )

    source_base_rows = [row for row in source_rows if row["arm"] == "N"]
    require(
        Counter(row["initial_answer_id"] for row in source_base_rows)
        == Counter({"A": 12, "B": 12, "C": 12}),
        "initial answer imbalance",
        failures,
    )
    require(
        Counter(row["updated_answer_id"] for row in source_base_rows)
        == Counter({"A": 12, "B": 12, "C": 12}),
        "updated answer imbalance",
        failures,
    )
    require(len(source_audits) == 36, "source audit count changed", failures)

    manifest_files = manifest.get("files", {})
    for name in DELIVERABLES:
        path = HERE / name
        require(path.exists(), f"missing deliverable {name}", failures)
        if path.exists():
            require(name in manifest_files, f"manifest missing {name}", failures)
            if name in manifest_files:
                require(
                    manifest_files[name]["sha256"] == file_sha256(path),
                    f"manifest hash mismatch {name}",
                    failures,
                )
    for path in EXTERNAL_DELIVERABLES:
        name = str(path.relative_to(REPO_ROOT)).replace("\\", "/")
        require(path.exists(), f"missing deliverable {name}", failures)
        require(name in manifest_files, f"manifest missing {name}", failures)
        if path.exists() and name in manifest_files:
            require(
                manifest_files[name]["sha256"] == file_sha256(path),
                f"manifest hash mismatch {name}",
                failures,
            )
    require(
        manifest.get("selection", {}).get("performed") is False,
        "manifest claims selection",
        failures,
    )
    require(
        manifest.get("counts", {}).get("planned_generations") == 2160,
        "planned generation count",
        failures,
    )

    result = {
        "status": "passed" if not failures else "failed",
        "checks": {
            "candidate_rows": len(rows),
            "audit_rows": len(audits),
            "families": len(cards),
            "source_bases": len(grouped),
            "local_cue_removal_audit_hits": local_cue_removals,
            "vf_levels": [item["name"] for item in VF_LEVELS],
            "s_levels": [item["name"] for item in S_LEVELS],
            "planned_generations": len(rows) * len(CAP_ORDER),
        },
        "failures": failures,
        "candidate_sha256": file_sha256(OUTPUT),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
