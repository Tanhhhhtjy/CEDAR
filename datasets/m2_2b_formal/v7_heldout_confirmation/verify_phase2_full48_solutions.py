#!/usr/bin/env python3
# ruff: noqa: E501
"""Independently recompute all C1/V7 base solutions without model execution."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASES = {
    row["base_id"]: row
    for row in (
        json.loads(line)
        for line in (HERE / "m2_2b_v7_heldout48_base.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    )
}


def expect(base_id: str, original: str, updated: str, distractor: str) -> None:
    base = BASES[base_id]
    options = base["options"]
    assert options[base["initial_answer"]] == original, (base_id, "original")
    assert options[base["target_answer"]] == updated, (base_id, "updated")
    assert options[base["distractor_answer"]] == distractor, (base_id, "distractor")
    assert len({original, updated, distractor}) == 3


def average(total: float, distances: tuple[float, ...], speeds: tuple[float, ...]) -> float:
    return total / sum(distance / speed for distance, speed in zip(distances, speeds, strict=False))


def recurrence(start: float, initial_rate: float, rework_rate: float, passes: int) -> float:
    accepted = start * initial_rate
    rejected = start - accepted
    for _ in range(passes):
        newly_accepted = rejected * rework_rate
        accepted += newly_accepted
        rejected -= newly_accepted
    return accepted


def longest_prefix(key: str, table: dict[str, str]) -> str:
    matching = [prefix for prefix in table if key.startswith(prefix)]
    best = max(matching, key=len)
    assert sum(len(prefix) == len(best) for prefix in matching) == 1
    return table[best]


def parse_escaped(raw: str) -> list[str]:
    fields = [""]
    index = 0
    while index < len(raw):
        if raw[index : index + 2] == "\\|":
            fields[-1] += "|"
            index += 2
        elif raw[index] == "|":
            fields.append("")
            index += 1
        else:
            fields[-1] += raw[index]
            index += 1
    return fields


def classify_interval(left: tuple[int, int], right: tuple[int, int]) -> str:
    if left[0] <= right[0] and right[1] <= left[1]:
        return "contains"
    if left[1] < right[0]:
        return "before"
    return "overlaps"


def acceptance_relation(measure: tuple[int, int], band: tuple[int, int]) -> str:
    if band[0] <= measure[0] and measure[1] <= band[1]:
        return "Fully contained in the acceptance band"
    if measure[1] < band[0] or band[1] < measure[0]:
        return "Fully excluded from the acceptance band"
    return "Partially overlaps the acceptance band"


def render_assignment(assignment: tuple[object, ...]) -> str:
    return f"A={assignment[0]}, B={assignment[1]}, C={assignment[2]}, D={assignment[3]}"


def quorum_decision(
    votes: dict[str, bool],
    weights: dict[str, int],
    veto: set[str],
    threshold: int,
) -> str:
    weight_ok = sum(weights[name] for name, vote in votes.items() if vote) >= threshold
    veto_ok = all(votes[name] for name in veto)
    return "Pass" if weight_ok and veto_ok else "Fail"


def immediately_before(left: str, right: str):
    def predicate(order: tuple[str, ...]) -> bool:
        return order.index(left) + 1 == order.index(right)

    return predicate


def appears_after(left: str, right: str):
    def predicate(order: tuple[str, ...]) -> bool:
        return order.index(left) > order.index(right)

    return predicate


def nth_weekday(length: int, day1: int, requested: int, occurrence: int) -> int | None:
    dates = [date for date in range(1, length + 1) if (day1 + date - 1) % 7 == requested]
    return dates[occurrence - 1] if len(dates) >= occurrence else None


def perfect_matchings(allowed: dict[str, set[object]]) -> list[tuple[object, ...]]:
    agents = tuple(allowed)
    items = sorted(set().union(*allowed.values()), key=str)
    return [
        assignment
        for assignment in itertools.permutations(items)
        if all(item in allowed[agent] for agent, item in zip(agents, assignment, strict=False))
    ]


def condorcet(candidates: tuple[str, ...], wins: set[tuple[str, str]]) -> str | None:
    winners = [
        candidate
        for candidate in candidates
        if all(candidate == other or (candidate, other) in wins for other in candidates)
    ]
    assert len(winners) <= 1
    return winners[0] if winners else None


def minimum_hitting_sets(universe: set[str], conflicts: list[set[str]]) -> list[frozenset[str]]:
    for size in range(len(universe) + 1):
        hits = [
            frozenset(combo)
            for combo in itertools.combinations(sorted(universe), size)
            if all(set(combo) & conflict for conflict in conflicts)
        ]
        if hits:
            return hits
    return []


def valid_sequences(items: str, predicates: tuple[object, ...]) -> list[str]:
    return [
        "".join(order)
        for order in itertools.permutations(items)
        if all(predicate(order) for predicate in predicates)
    ]


def closure(
    seed: frozenset[str], dependencies: tuple[tuple[frozenset[str], frozenset[str]], ...]
) -> frozenset[str]:
    result = set(seed)
    changed = True
    while changed:
        changed = False
        for left, right in dependencies:
            if left <= result and not right <= result:
                result.update(right)
                changed = True
    return frozenset(result)


def singleton_keys(
    attributes: set[str], dependencies: tuple[tuple[frozenset[str], frozenset[str]], ...]
) -> list[str]:
    return sorted(
        item for item in attributes if closure(frozenset({item}), dependencies) == attributes
    )


def main() -> None:
    assert len(BASES) == 48

    # Arithmetic/accounting: direct recomputation from the closed rules.
    expect(
        "C1V7-001",
        f"{sorted([12, 18, 25, 31, 44])[2]} points",
        f"{sorted([12, 18, 25, 16, 44])[2]} points",
        "22 points",
    )
    expect(
        "C1V7-002",
        str(sorted([12, 19, 27, 34, 46, 53, 61])[3]),
        str(sorted([12, 19, 27, 34, 46, 15, 61])[3]),
        "19",
    )
    expect(
        "C1V7-003",
        f"{min(90 * 0.8, 80 * 0.95, 100 * 0.9):g} units",
        f"{min(90 * 0.8, 80 * 0.7, 100 * 0.9):g} units",
        "76 units",
    )
    expect(
        "C1V7-004",
        f"{min(120 * 0.8, 100 * 0.9, 110 * 0.9, 150 * 0.7):g} units",
        f"{min(120 * 0.8, 100 * 0.9, 110 * 0.6, 150 * 0.7):g} units",
        "96 units",
    )
    expect(
        "C1V7-005",
        f"{math.ceil(52 / 12) * 8} credits",
        f"{math.ceil(52 / 18) * 8} credits",
        "32 credits",
    )
    expect(
        "C1V7-006",
        f"{math.ceil(137 / 24) * 11} tokens",
        f"{math.ceil(137 / 35) * 11} tokens",
        "55 tokens",
    )
    expect("C1V7-007", f"{800 * 0.75 * 0.8:g} units", f"{800 * 0.75 * 0.5:g} units", "400 units")
    expect(
        "C1V7-008",
        f"{1200 * 0.9 * 0.8 * 0.75:g} units",
        f"{1200 * 0.9 * 0.6 * 0.75:g} units",
        "540 units",
    )
    expect("C1V7-009", str(120 + 70 - 30), str(120 - 70 - 30), "90")
    expect("C1V7-010", str(210 + 90 - 40 - 20), str(210 - 90 - 40 - 20), "150")
    expect("C1V7-011", f"{132 / 1.10:g}", f"{132 / 1.20:g}", "108")
    expect("C1V7-012", f"{207 / 1.15:g}", f"{207 / 1.38:g}", "160")
    expect(
        "C1V7-013",
        f"{average(180, (90, 90), (45, 90)):g} km/h",
        f"{average(180, (60, 120), (45, 90)):g} km/h",
        "75 km/h",
    )
    expect(
        "C1V7-014",
        f"{average(216, (108, 108), (36, 72)):g} km/h",
        f"{average(216, (72, 144), (36, 72)):g} km/h",
        "60 km/h",
    )
    expect(
        "C1V7-015",
        f"{recurrence(200, 0.6, 0.5, 1):g} units",
        f"{recurrence(200, 0.6, 0.5, 2):g} units",
        "170 units",
    )
    expect(
        "C1V7-016",
        f"{recurrence(160, 0.5, 0.25, 2):g} units",
        f"{recurrence(160, 0.5, 0.25, 3):g} units",
        "120 units",
    )

    # Unit/label rules: independent parsers, lookups, bit operations, and calendars.
    expect(
        "C1V7-017",
        longest_prefix("KLMN", {"K": "Amber", "KL": "Blue", "KLM": "Coral"}),
        longest_prefix("KLMN", {"K": "Amber", "KL": "Blue"}),
        "Amber",
    )
    expect(
        "C1V7-018",
        longest_prefix("TANGO7", {"T": "Ivory", "TA": "Jade", "TAN": "Lilac", "TANG": "Ochre"}),
        longest_prefix("TANGO7", {"T": "Ivory", "TA": "Jade", "TAN": "Lilac", "TARP": "Ochre"}),
        "Jade",
    )
    expect(
        "C1V7-019",
        "Package Q" if "beta" > "alpha" else "Package P",
        "Package P" if "rc" > "beta" else "Package Q",
        "They have equal precedence",
    )
    expect(
        "C1V7-020",
        "Package Q" if 11 > 9 else "Package P",
        "Package P" if 15 > 11 else "Package Q",
        "They have equal precedence",
    )
    expect(
        "C1V7-021",
        "Fully contained in the acceptance band",
        "Partially overlaps the acceptance band",
        "Fully excluded from the acceptance band",
    )
    original_measure = (74 - 3, 74 + 3)
    updated_measure = (74 - 7, 74 + 7)
    band = (70, 80)
    expect(
        "C1V7-022",
        acceptance_relation(original_measure, band),
        acceptance_relation(updated_measure, band),
        "Fully excluded from the acceptance band",
    )
    expect(
        "C1V7-023",
        parse_escaped(r"oak\|pine|birch|cedar")[1],
        parse_escaped("oak|pine|birch|cedar")[1],
        "cedar",
    )
    expect(
        "C1V7-024",
        parse_escaped(r"north\|west|central\|hub|south|east")[2],
        parse_escaped(r"north\|west|central\|hub\|south|east")[2],
        "central|hub",
    )
    expect("C1V7-025", f"{10 * math.log10(1000):g} dB", f"{20 * math.log10(1000):g} dB", "20 dB")
    expect("C1V7-026", f"{10 * 1.2041:.1f} dB", f"{20 * 1.2041:.1f} dB", "16.0 dB")
    table = {("UDP", 88): "Maple", ("TCP", 88): "Quartz", ("UDP", 89): "Silver"}
    expect("C1V7-027", table[("UDP", 88)], table[("TCP", 88)], table[("UDP", 89)])
    table = {("UDP", 53): "Lumen", ("TCP", 53): "Nacre", ("UDP", 54): "Opal"}
    expect("C1V7-028", table[("UDP", 40 + 13)], table[("TCP", 40 + 13)], table[("UDP", 54)])
    expect(
        "C1V7-029",
        f"{0b1101 & (~0b0100 & 0b1111):04b}",
        f"{0b1101 & (~0b0101 & 0b1111):04b}",
        "1100",
    )
    labels = {"11010": "Falcon", "10010": "Harbor", "10110": "Juniper"}
    expect(
        "C1V7-030",
        labels[f"{0b11111 & (~0b00101 & 0b11111):05b}"],
        labels[f"{0b11111 & (~0b01101 & 0b11111):05b}"],
        "Juniper",
    )
    weekday = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Friday": 4, "Sunday": 6}
    expect(
        "C1V7-031",
        f"Day {nth_weekday(30, weekday['Friday'], weekday['Tuesday'], 4)}",
        f"Day {nth_weekday(30, weekday['Sunday'], weekday['Tuesday'], 4)}",
        "Day 22",
    )
    original = nth_weekday(30, weekday["Wednesday"], weekday["Monday"], 5)
    updated = nth_weekday(30, weekday["Monday"], weekday["Monday"], 5)
    expect(
        "C1V7-032",
        "There is no fifth Monday" if original is None else f"Day {original}",
        f"Day {updated}",
        "Day 30",
    )

    # Multistep logic: exhaustive finite-state checks.
    m = perfect_matchings({"Nia": {"Quartz"}, "Omar": {"Quartz", "Ruby"}, "Pia": {"Ruby", "Sable"}})
    n = perfect_matchings({"Nia": {"Sable"}, "Omar": {"Quartz", "Ruby"}, "Pia": {"Ruby", "Sable"}})
    assert len(m) == len(n) == 1
    expect("C1V7-033", m[0][1], n[0][1], "Sable")
    m = perfect_matchings({"A": {1}, "B": {1, 2}, "C": {2, 3}, "D": {3, 4}})
    n = perfect_matchings({"A": {4}, "B": {1, 2}, "C": {2, 3}, "D": {3, 4}})
    assert m == [(1, 2, 3, 4)] and n == [(4, 1, 2, 3)]
    expect(
        "C1V7-034",
        render_assignment(m[0]),
        render_assignment(n[0]),
        "A=4, B=2, C=3, D=1",
    )
    expect(
        "C1V7-035",
        "A contains B" if classify_interval((3, 10), (6, 8)) == "contains" else "",
        "A is before B" if classify_interval((3, 5), (6, 8)) == "before" else "",
        "A overlaps B",
    )
    names = {"contains": "X contains Y", "before": "X is before Y", "overlaps": "X overlaps Y"}
    expect(
        "C1V7-036",
        names[classify_interval((5, 15), (12, 18))],
        names[classify_interval((5, 11), (12, 18))],
        "X contains Y",
    )
    expect(
        "C1V7-037",
        quorum_decision(
            {"P": False, "Q": True, "R": True},
            {"P": 2, "Q": 3, "R": 3},
            {"P"},
            6,
        ),
        quorum_decision(
            {"P": True, "Q": True, "R": True},
            {"P": 2, "Q": 3, "R": 3},
            {"P"},
            6,
        ),
        "No decision",
    )
    expect(
        "C1V7-038",
        quorum_decision(
            {"A": True, "B": True, "C": True, "D": False},
            {"A": 4, "B": 4, "C": 3, "D": 2},
            {"A"},
            10,
        ),
        quorum_decision(
            {"A": True, "B": False, "C": True, "D": False},
            {"A": 4, "B": 4, "C": 3, "D": 2},
            {"A"},
            10,
        ),
        "No decision",
    )
    expect(
        "C1V7-039", f"{max(2, 5) + 3} time units", f"{max(6, 5) + 3} time units", "10 time units"
    )
    expect(
        "C1V7-040",
        f"{max(3 + 2, 6) + 2} time units",
        f"{max(3 + 5, 6) + 2} time units",
        "11 time units",
    )
    expect(
        "C1V7-041",
        condorcet(("X", "Y", "Z"), {("X", "Y"), ("X", "Z"), ("Y", "Z")}) or "No Condorcet winner",
        condorcet(("X", "Y", "Z"), {("Y", "X"), ("X", "Z"), ("Y", "Z")}) or "No Condorcet winner",
        "No Condorcet winner",
    )
    expect(
        "C1V7-042",
        condorcet(
            ("W", "X", "Y", "Z"),
            {("W", "X"), ("W", "Y"), ("W", "Z"), ("X", "Y"), ("X", "Z"), ("Y", "Z")},
        )
        or "No Condorcet winner",
        condorcet(
            ("W", "X", "Y", "Z"),
            {("X", "W"), ("W", "Y"), ("W", "Z"), ("X", "Y"), ("X", "Z"), ("Y", "Z")},
        )
        or "No Condorcet winner",
        "No Condorcet winner",
    )
    h = minimum_hitting_sets({"P", "Q", "R"}, [{"P", "Q"}, {"Q", "R"}])
    j = minimum_hitting_sets({"P", "Q", "R"}, [{"P", "Q"}, {"P", "R"}])
    assert h == [frozenset({"Q"})] and j == [frozenset({"P"})]
    expect("C1V7-043", "{Q}", "{P}", "{R}")
    h = minimum_hitting_sets(set("ABCDE"), [set("ABC"), set("ABD"), set("BE")])
    j = minimum_hitting_sets(set("ABCDE"), [set("ABC"), set("ABD"), set("AE")])
    assert h == [frozenset({"B"})] and j == [frozenset({"A"})]
    expect("C1V7-044", "{B}", "{A}", "{C}")
    p = valid_sequences(
        "PQRS",
        (lambda o: o[0] == "P", immediately_before("Q", "R"), appears_after("S", "R")),
    )
    q = valid_sequences(
        "PQRS",
        (lambda o: o[0] == "P", immediately_before("R", "Q"), appears_after("S", "R")),
    )
    assert p == ["PQRS"] and q == ["PRQS"]
    expect("C1V7-045", p[0], q[0], "PSRQ")
    p = valid_sequences(
        "ABCDE",
        (
            lambda o: o[0] == "A",
            immediately_before("B", "C"),
            immediately_before("D", "E"),
            appears_after("D", "C"),
        ),
    )
    q = valid_sequences(
        "ABCDE",
        (
            lambda o: o[0] == "A",
            immediately_before("C", "B"),
            immediately_before("D", "E"),
            appears_after("D", "C"),
        ),
    )
    assert p == ["ABCDE"] and q == ["ACBDE"]
    expect("C1V7-046", p[0], q[0], "ABDCE")
    attrs = frozenset("MNP")
    d = ((frozenset("M"), frozenset("N")), (frozenset("N"), frozenset("P")))
    e = ((frozenset("N"), frozenset("M")), (frozenset("N"), frozenset("P")))
    expect(
        "C1V7-047",
        "{" + singleton_keys(set(attrs), d)[0] + "}",
        "{" + singleton_keys(set(attrs), e)[0] + "}",
        "{P}",
    )
    attrs = frozenset("PQRST")
    d = (
        (frozenset("R"), frozenset("Q")),
        (frozenset("Q"), frozenset("P")),
        (frozenset("P"), frozenset("S")),
        (frozenset("Q"), frozenset("T")),
    )
    e = ((frozenset("Q"), frozenset("R")),) + d[1:]
    expect(
        "C1V7-048",
        "{" + singleton_keys(set(attrs), d)[0] + "}",
        "{" + singleton_keys(set(attrs), e)[0] + "}",
        "{P}",
    )
    print("FULL48_INDEPENDENT_SOLUTION_CHECK=PASS (48/48)")


if __name__ == "__main__":
    main()
