"""Frozen helpers for the M2.2b v6 zero-GPU re-audit."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ANSWERS = ("A", "B", "C")
OUTCOMES = ("initial", "target", "other_wrong", "invalid")
ARMS = ("N", "V", "F", "S")
MODES = (
    "thinking_sampling",
    "nonthinking_sampling",
    "thinking_greedy",
    "nonthinking_greedy",
)
CAPS = ("uncapped", "p25")
AMBIGUITY_REASONS = (
    "",
    "no_explicit_answer",
    "multiple_conflicting_answers",
    "malformed_json",
    "incomplete_or_truncated",
    "reasoning_only",
    "other",
)
ANNOTATION_FIELDS = (
    "blind_id",
    "answer_extracted",
    "json_valid",
    "complete",
    "ambiguity_reason",
    "notes",
)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        rows.append(value)
    return rows


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def final_surface(response: str) -> tuple[str, str]:
    """Remove a complete leading thinking envelope without exposing its content."""
    stripped = response.strip()
    has_open = "<think>" in stripped
    has_close = "</think>" in stripped
    if not has_open and not has_close:
        return stripped, "no_think_envelope"
    if not stripped.startswith("<think>"):
        return "", "malformed_think_envelope"
    close = stripped.find("</think>", len("<think>"))
    if close < 0:
        return "", "unclosed_think_envelope"
    surface = stripped[close + len("</think>") :].strip()
    if "<think>" in surface or "</think>" in surface:
        return "", "think_tag_in_final_surface"
    return surface, "complete_think_envelope"


def strict_json_answer(surface: str) -> str | None:
    """Apply the preregistered exact JSON schema to one final surface."""
    try:
        value = json.loads(surface.strip())
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(value, dict) or set(value) != {"answer", "confidence", "reason"}:
        return None
    answer = value.get("answer")
    confidence = value.get("confidence")
    reason = value.get("reason")
    if answer not in ANSWERS:
        return None
    if isinstance(confidence, bool) or not isinstance(confidence, int | float):
        return None
    if not 0 <= float(confidence) <= 100:
        return None
    if not isinstance(reason, str) or not reason.strip():
        return None
    return str(answer)


def parse_annotation_bool(value: str, *, field: str, blind_id: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise ValueError(f"{blind_id}: {field} must be true/false, found {value!r}")


def validate_annotation_rows(
    rows: Sequence[Mapping[str, str]],
    expected_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    expected = list(expected_ids)
    if len(rows) != len(expected):
        raise ValueError(f"Expected {len(expected)} annotation rows, found {len(rows)}")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        blind_id = row.get("blind_id", "").strip()
        if blind_id not in set(expected):
            raise ValueError(f"Unknown blind_id {blind_id!r}")
        if blind_id in by_id:
            raise ValueError(f"Duplicate blind_id {blind_id}")
        answer = row.get("answer_extracted", "").strip().upper()
        if answer not in (*ANSWERS, ""):
            raise ValueError(f"{blind_id}: invalid answer_extracted {answer!r}")
        reason = row.get("ambiguity_reason", "").strip()
        if reason not in AMBIGUITY_REASONS:
            raise ValueError(f"{blind_id}: invalid ambiguity_reason {reason!r}")
        json_valid = parse_annotation_bool(
            row.get("json_valid", ""), field="json_valid", blind_id=blind_id
        )
        complete = parse_annotation_bool(
            row.get("complete", ""), field="complete", blind_id=blind_id
        )
        if not answer and not reason:
            raise ValueError(f"{blind_id}: blank answer requires ambiguity_reason")
        by_id[blind_id] = {
            "blind_id": blind_id,
            "answer_extracted": answer,
            "json_valid": json_valid,
            "complete": complete,
            "ambiguity_reason": reason,
            "notes": row.get("notes", "").strip(),
        }
    if set(by_id) != set(expected):
        missing = sorted(set(expected) - set(by_id))
        raise ValueError(f"Missing annotation IDs: {missing[:10]}")
    return by_id


def cohen_kappa(left: Sequence[str | bool], right: Sequence[str | bool]) -> float | None:
    if len(left) != len(right):
        raise ValueError("Kappa inputs must have equal length")
    if not left:
        return None
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    categories = set(left_counts) | set(right_counts)
    expected = sum(
        (left_counts[item] / len(left)) * (right_counts[item] / len(right))
        for item in categories
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else None
    return (observed - expected) / (1.0 - expected)


def outcome_for_answer(answer: str | None, source: Mapping[str, Any]) -> str:
    if answer is None:
        return "invalid"
    initial = source["initial_answer_id"]
    target = source["induced_target_answer_id"]
    if initial == target or initial not in ANSWERS or target not in ANSWERS:
        raise ValueError(f"{source.get('id')}: initial/target answer keys are not distinct A/B/C")
    if answer == initial:
        return "initial"
    if answer == target:
        return "target"
    if answer in ANSWERS:
        return "other_wrong"
    raise ValueError(f"Unexpected extracted answer {answer!r}")


def validate_source_answer_keys(source: Mapping[str, Any]) -> None:
    initial = source.get("initial_answer_id")
    target = source.get("induced_target_answer_id")
    options = source.get("options")
    if initial not in ANSWERS or target not in ANSWERS or initial == target:
        raise ValueError(f"{source.get('id')}: malformed initial/target keys")
    if not isinstance(options, dict) or set(options) != set(ANSWERS):
        raise ValueError(f"{source.get('id')}: expected exactly A/B/C options")
    third = set(ANSWERS) - {initial, target}
    if len(third) != 1:
        raise ValueError(f"{source.get('id')}: no unique third answer")


def exposure_metrics(raw: Mapping[str, Any]) -> dict[str, float | int | None]:
    """Reconstruct frozen exposure metrics from one phase/layer monitor record."""
    count = int(raw["count"])
    triggered = int(raw["triggered"])
    if count < 0 or not 0 <= triggered <= count:
        raise ValueError(f"Invalid exposure counts: count={count}, triggered={triggered}")
    correction_sum = float(raw["mean_correction"]) * count
    return {
        "count": count,
        "triggered": triggered,
        "trigger_rate": triggered / count if count else 0.0,
        "correction_sum": correction_sum,
        "dose_density": correction_sum / count if count else 0.0,
        "conditional_magnitude": correction_sum / triggered if triggered else None,
    }
