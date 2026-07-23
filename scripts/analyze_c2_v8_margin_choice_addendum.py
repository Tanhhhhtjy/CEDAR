#!/usr/bin/env python3
"""Zero-GPU post-hoc margin-choice analysis for the frozen C2/V8 results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import random
import statistics
import subprocess
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = (
    ROOT / "artifacts" / "m2" / "c2_v8_natural_social_confirmation"
)
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_INPUT_ROOT / "addenda" / "margin_choice_dissociation"
)
PROTOCOL = Path(
    "docs/C2_V8_ZERO_GPU_POSTHOC_ADDENDUM_01_MARGIN_CHOICE.md"
)
SOURCE_RESULTS_COMMIT = "8ea7025f2a726997d78ad8919455a8756e3c192e"
ARMS = ("AO", "S")
CONDITIONS = ("uncapped", "p25")
LABELS = ("A", "B", "C", "D")
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 20260724


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def canonical_text_sha256(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def git_value(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_audits(input_root: Path) -> dict[str, str]:
    names = (
        "CROSS_CONDITION_AUDIT.json",
        "DIRECT_CROSS_CONDITION_AUDIT.json",
        "FREE_CROSS_CONDITION_AUDIT.json",
    )
    hashes = {}
    for name in names:
        path = input_root / name
        payload = read_json(path)
        if payload.get("status") != "passed":
            raise ValueError(f"Required audit is not passed: {path}")
        hashes[name] = canonical_text_sha256(path)
    analysis_path = input_root / "C2_V8_ANALYSIS.json"
    analysis = read_json(analysis_path)
    if analysis.get("status") != "complete":
        raise ValueError("Frozen C2/V8 analysis is not complete")
    hashes["C2_V8_ANALYSIS.json"] = canonical_text_sha256(analysis_path)
    return hashes


def load_phase(input_root: Path, phase: str) -> list[dict[str, Any]]:
    path = input_root / phase / "results.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return read_jsonl(path)


def pair_conditions(
    uncapped: Sequence[Mapping[str, Any]],
    p25: Sequence[Mapping[str, Any]],
    expected_rows: int,
) -> list[dict[str, Any]]:
    if len(uncapped) != expected_rows or len(p25) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows per condition, got "
            f"{len(uncapped)} and {len(p25)}"
        )
    left = {str(row["id"]): row for row in uncapped}
    right = {str(row["id"]): row for row in p25}
    if len(left) != expected_rows or len(right) != expected_rows:
        raise ValueError("Phase IDs are not unique")
    left_order = [str(row["id"]) for row in uncapped]
    right_order = [str(row["id"]) for row in p25]
    if left_order != right_order:
        raise ValueError("Cross-condition IDs or order differ")
    metadata = (
        "id",
        "base_id",
        "source_id",
        "arm",
        "domain",
        "difficulty",
        "normalized_stem_id",
        "exact_mcq_id",
        "template_id",
        "initial_answer",
        "target_answer",
    )
    output = []
    for item_id in left_order:
        a, b = left[item_id], right[item_id]
        if any(a.get(field) != b.get(field) for field in metadata):
            raise ValueError(f"{item_id}: cross-condition metadata differs")
        output.append(
            {
                **{field: a.get(field) for field in metadata},
                "uncapped": a,
                "p25": b,
            }
        )
    return output


def build_bases(
    paired: Sequence[Mapping[str, Any]], expected_bases: int
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for pair in paired:
        grouped[str(pair["base_id"])][str(pair["arm"])] = pair
    if len(grouped) != expected_bases:
        raise ValueError(f"Expected {expected_bases} bases, got {len(grouped)}")
    output = []
    metadata = (
        "source_id",
        "domain",
        "difficulty",
        "normalized_stem_id",
        "exact_mcq_id",
        "template_id",
        "initial_answer",
        "target_answer",
    )
    for base_id, arms in grouped.items():
        if set(arms) != set(ARMS):
            raise ValueError(f"{base_id}: incomplete AO/S pair")
        ao, social = arms["AO"], arms["S"]
        if any(ao.get(field) != social.get(field) for field in metadata):
            raise ValueError(f"{base_id}: AO/S metadata differs")
        output.append(
            {
                "base_id": base_id,
                **{field: ao.get(field) for field in metadata},
                "arms": {"AO": ao, "S": social},
            }
        )
    return sorted(output, key=lambda row: str(row["base_id"]))


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Mean requires values")
    return float(sum(values) / len(values))


def sample_sd(values: Sequence[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Percentile requires values")
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be in [0, 1]")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    fraction = position - lower
    return float(
        sorted_values[lower] * (1 - fraction)
        + sorted_values[upper] * fraction
    )


def stratified_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    value_fn: Callable[[Mapping[str, Any]], float],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["domain"]), str(row["difficulty"]))].append(row)
    ordered = [grouped[key] for key in sorted(grouped)]
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        total = 0.0
        count = 0
        for stratum in ordered:
            for _ in range(len(stratum)):
                total += float(value_fn(stratum[rng.randrange(len(stratum))]))
                count += 1
        draws.append(total / count)
    draws.sort()
    values = [float(value_fn(row)) for row in rows]
    return {
        "estimate": mean(values),
        "sd_across_bases": sample_sd(values),
        "ci_low": percentile(draws, 0.025),
        "ci_high": percentile(draws, 0.975),
        "samples": samples,
        "seed": seed,
        "unit": "base",
        "strata": ["domain", "difficulty"],
    }


def cluster_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    value_fn: Callable[[Mapping[str, Any]], float],
    cluster_field: str,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[cluster_field])].append(row)
    clusters = sorted(grouped)
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        total = 0.0
        count = 0
        for _ in range(len(clusters)):
            selected = grouped[clusters[rng.randrange(len(clusters))]]
            total += sum(float(value_fn(row)) for row in selected)
            count += len(selected)
        draws.append(total / count)
    draws.sort()
    values = [float(value_fn(row)) for row in rows]
    return {
        "estimate": mean(values),
        "ci_low": percentile(draws, 0.025),
        "ci_high": percentile(draws, 0.975),
        "samples": samples,
        "seed": seed,
        "cluster_field": cluster_field,
        "clusters": len(clusters),
    }


def margin(pair: Mapping[str, Any], condition: str, field: str) -> float:
    payload = pair[condition].get(field)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{pair['id']}: missing {field}")
    value = payload.get("target_minus_initial")
    if value is None:
        raise ValueError(f"{pair['id']}: missing target-minus-initial margin")
    return float(value)


def category(pair: Mapping[str, Any], condition: str, field: str) -> str:
    if field == "strict_category":
        value = str(pair[condition].get(field))
        allowed = {"initial", "target", "other_wrong", "invalid_or_truncated"}
        if value not in allowed:
            raise ValueError(f"{pair['id']}: invalid strict category {value}")
        return value
    payload = pair[condition].get(field)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{pair['id']}: missing {field}")
    answer = payload.get("option_argmax")
    if answer == pair["initial_answer"]:
        return "initial"
    if answer == pair["target_answer"]:
        return "target"
    if answer in LABELS:
        return "other_wrong"
    return "invalid_or_truncated"


def target_indicator(
    pair: Mapping[str, Any], condition: str, field: str
) -> float:
    return float(category(pair, condition, field) == "target")


def cap_delta(
    base: Mapping[str, Any],
    arm: str,
    value_fn: Callable[[Mapping[str, Any], str], float],
) -> float:
    pair = base["arms"][arm]
    return float(value_fn(pair, "p25") - value_fn(pair, "uncapped"))


def social_delta(
    base: Mapping[str, Any],
    value_fn: Callable[[Mapping[str, Any], str], float],
) -> float:
    return cap_delta(base, "S", value_fn) - cap_delta(base, "AO", value_fn)


def transition_counts(
    bases: Sequence[Mapping[str, Any]], field: str
) -> dict[str, dict[str, int]]:
    output = {}
    for arm in ARMS:
        counts = Counter(
            f"{category(base['arms'][arm], 'uncapped', field)}"
            f"->{category(base['arms'][arm], 'p25', field)}"
            for base in bases
        )
        output[arm] = dict(sorted(counts.items()))
    return output


def diagnostic(pair: Mapping[str, Any], condition: str, metric: str) -> float:
    payload = pair[condition].get("answer_logits")
    if not isinstance(payload, Mapping) or payload.get(metric) is None:
        raise ValueError(f"{pair['id']}: missing answer diagnostic {metric}")
    return float(payload[metric])


def answer_step_stats(pair: Mapping[str, Any]) -> tuple[float, float]:
    phases = pair["p25"].get("actual_cap_stats_by_phase")
    if not isinstance(phases, Mapping):
        raise ValueError(f"{pair['id']}: missing actual cap stats")
    layers = phases.get("answer_step")
    if not isinstance(layers, Mapping):
        raise ValueError(f"{pair['id']}: missing answer-step cap stats")
    correction = 0.0
    triggered = 0
    total = 0
    for stats in layers.values():
        correction += float(stats["projection_correction_sum"])
        triggered += int(stats["triggered_positions"])
        total += int(stats["total_positions"])
    return correction, triggered / total


def direct_base_rows(bases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for base in bases:
        row = {
            key: base[key]
            for key in (
                "base_id",
                "source_id",
                "domain",
                "difficulty",
                "normalized_stem_id",
                "exact_mcq_id",
                "template_id",
                "initial_answer",
                "target_answer",
            )
        }
        for arm in ARMS:
            pair = base["arms"][arm]
            for condition in CONDITIONS:
                row[f"margin_{arm}_{condition}"] = margin(
                    pair, condition, "answer_logits"
                )
                row[f"target_{arm}_{condition}"] = target_indicator(
                    pair, condition, "answer_logits"
                )
                row[f"category_{arm}_{condition}"] = category(
                    pair, condition, "answer_logits"
                )
            row[f"cap_margin_{arm}"] = (
                row[f"margin_{arm}_p25"] - row[f"margin_{arm}_uncapped"]
            )
            row[f"cap_target_{arm}"] = (
                row[f"target_{arm}_p25"] - row[f"target_{arm}_uncapped"]
            )
            row[f"entropy_cap_{arm}"] = (
                diagnostic(pair, "p25", "full_vocab_entropy")
                - diagnostic(pair, "uncapped", "full_vocab_entropy")
            )
            row[f"option_mass_cap_{arm}"] = (
                diagnostic(pair, "p25", "option_mass")
                - diagnostic(pair, "uncapped", "option_mass")
            )
            row[f"prompt_tokens_{arm}"] = int(pair["p25"]["prompt_tokens"])
            correction, trigger = answer_step_stats(pair)
            row[f"answer_step_correction_{arm}"] = correction
            row[f"answer_step_trigger_rate_{arm}"] = trigger
        row["social_margin_uncapped"] = (
            row["margin_S_uncapped"] - row["margin_AO_uncapped"]
        )
        row["social_margin_p25"] = row["margin_S_p25"] - row["margin_AO_p25"]
        row["margin_social_did"] = (
            row["cap_margin_S"] - row["cap_margin_AO"]
        )
        row["social_target_uncapped"] = (
            row["target_S_uncapped"] - row["target_AO_uncapped"]
        )
        row["social_target_p25"] = (
            row["target_S_p25"] - row["target_AO_p25"]
        )
        row["target_social_did"] = (
            row["cap_target_S"] - row["cap_target_AO"]
        )
        row["entropy_social_did"] = (
            row["entropy_cap_S"] - row["entropy_cap_AO"]
        )
        row["option_mass_social_did"] = (
            row["option_mass_cap_S"] - row["option_mass_cap_AO"]
        )
        row["prompt_token_difference_AO_minus_S"] = (
            row["prompt_tokens_AO"] - row["prompt_tokens_S"]
        )
        row["answer_step_correction_AO_minus_S"] = (
            row["answer_step_correction_AO"]
            - row["answer_step_correction_S"]
        )
        output.append(row)
    return output


def summarize_cells(
    rows: Sequence[Mapping[str, Any]],
    metric_prefix: str,
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    output = {}
    offset = 0
    for arm in ARMS:
        output[arm] = {}
        for condition in CONDITIONS:
            key = f"{metric_prefix}_{arm}_{condition}"
            output[arm][condition] = stratified_bootstrap(
                rows,
                lambda row, name=key: float(row[name]),
                samples=samples,
                seed=seed + offset,
            )
            offset += 1
    return output


def summarize_direct(
    rows: Sequence[Mapping[str, Any]],
    original_analysis: Mapping[str, Any],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    margin_cells = summarize_cells(
        rows, "margin", samples=samples, seed=seed + 10
    )
    target_cells = summarize_cells(
        rows, "target", samples=samples, seed=seed + 20
    )
    metrics = {}
    names = (
        "cap_margin_AO",
        "cap_margin_S",
        "margin_social_did",
        "social_margin_uncapped",
        "social_margin_p25",
        "cap_target_AO",
        "cap_target_S",
        "target_social_did",
        "social_target_uncapped",
        "social_target_p25",
        "entropy_cap_AO",
        "entropy_cap_S",
        "entropy_social_did",
        "option_mass_cap_AO",
        "option_mass_cap_S",
        "option_mass_social_did",
        "prompt_token_difference_AO_minus_S",
        "answer_step_correction_AO_minus_S",
    )
    for index, name in enumerate(names):
        metrics[name] = stratified_bootstrap(
            rows,
            lambda row, field=name: float(row[field]),
            samples=samples,
            seed=seed + 100 + index,
        )
    clusters = {}
    for index, field in enumerate(("exact_mcq_id", "normalized_stem_id")):
        clusters[field] = {
            "margin_social_did": cluster_bootstrap(
                rows,
                lambda row: float(row["margin_social_did"]),
                field,
                samples=samples,
                seed=seed + 200 + index,
            ),
            "target_social_did": cluster_bootstrap(
                rows,
                lambda row: float(row["target_social_did"]),
                field,
                samples=samples,
                seed=seed + 210 + index,
            ),
        }
    letter = {}
    for target in sorted({str(row["target_answer"]) for row in rows}):
        subset = [row for row in rows if row["target_answer"] == target]
        letter[target] = {
            "n": len(subset),
            "margin_social_did": mean(
                [float(row["margin_social_did"]) for row in subset]
            ),
            "target_social_did": mean(
                [float(row["target_social_did"]) for row in subset]
            ),
        }
    leave_out = {}
    for field in ("domain", "template_id"):
        values = {}
        for group in sorted({str(row[field]) for row in rows}):
            retained = [row for row in rows if str(row[field]) != group]
            values[group] = {
                "margin_social_did": mean(
                    [float(row["margin_social_did"]) for row in retained]
                ),
                "target_social_did": mean(
                    [float(row["target_social_did"]) for row in retained]
                ),
            }
        leave_out[field] = values
    official = original_analysis["direct"]["primary_social_margin_effect"]
    return {
        "n_bases": len(rows),
        "margin_cells": margin_cells,
        "target_cells": target_cells,
        "metrics": metrics,
        "transitions": transition_counts_from_rows(rows),
        "cluster_sensitivity": clusters,
        "target_letter_sensitivity": letter,
        "leave_one_group_out": leave_out,
        "official_primary_margin_result": official,
        "official_primary_recomputed_point_matches": math.isclose(
            float(official["estimate"]),
            float(metrics["margin_social_did"]["estimate"]),
            abs_tol=1e-12,
        ),
    }


def transition_counts_from_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    output = {}
    for arm in ARMS:
        counts = Counter(
            f"{row[f'category_{arm}_uncapped']}"
            f"->{row[f'category_{arm}_p25']}"
            for row in rows
        )
        output[arm] = dict(sorted(counts.items()))
    return output


def summarize_free(
    bases: Sequence[Mapping[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    rows = []
    for base in bases:
        row = {
            key: base[key]
            for key in (
                "base_id",
                "domain",
                "difficulty",
                "normalized_stem_id",
                "exact_mcq_id",
            )
        }
        for arm in ARMS:
            pair = base["arms"][arm]
            for condition in CONDITIONS:
                row[f"strict_target_{arm}_{condition}"] = target_indicator(
                    pair, condition, "strict_category"
                )
                row[f"actual_margin_{arm}_{condition}"] = margin(
                    pair, condition, "actual_answer_logits"
                )
                row[f"scaffold_margin_{arm}_{condition}"] = margin(
                    pair, condition, "scaffold_answer_logits"
                )
                row[f"thinking_tokens_{arm}_{condition}"] = float(
                    pair[condition]["thinking_tokens"]
                )
                row[f"strict_category_{arm}_{condition}"] = category(
                    pair, condition, "strict_category"
                )
            for metric in (
                "strict_target",
                "actual_margin",
                "scaffold_margin",
                "thinking_tokens",
            ):
                row[f"cap_{metric}_{arm}"] = (
                    row[f"{metric}_{arm}_p25"]
                    - row[f"{metric}_{arm}_uncapped"]
                )
        for metric in (
            "strict_target",
            "actual_margin",
            "scaffold_margin",
            "thinking_tokens",
        ):
            row[f"social_{metric}_did"] = (
                row[f"cap_{metric}_S"] - row[f"cap_{metric}_AO"]
            )
        rows.append(row)
    metrics = {}
    for index, name in enumerate(
        (
            "cap_strict_target_AO",
            "cap_strict_target_S",
            "social_strict_target_did",
            "cap_actual_margin_AO",
            "cap_actual_margin_S",
            "social_actual_margin_did",
            "cap_scaffold_margin_AO",
            "cap_scaffold_margin_S",
            "social_scaffold_margin_did",
            "cap_thinking_tokens_AO",
            "cap_thinking_tokens_S",
            "social_thinking_tokens_did",
        )
    ):
        metrics[name] = stratified_bootstrap(
            rows,
            lambda row, field=name: float(row[field]),
            samples=samples,
            seed=seed + 300 + index,
        )
    transitions = {}
    for arm in ARMS:
        counts = Counter(
            f"{row[f'strict_category_{arm}_uncapped']}"
            f"->{row[f'strict_category_{arm}_p25']}"
            for row in rows
        )
        transitions[arm] = dict(sorted(counts.items()))
    return {
        "n_bases": len(rows),
        "metrics": metrics,
        "strict_transitions": transitions,
        "final_target_floor": all(
            row[f"strict_target_{arm}_{condition}"] == 0
            for row in rows
            for arm in ARMS
            for condition in CONDITIONS
        ),
    }


def write_per_base_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:+.{digits}f}"


def pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def report_markdown(analysis: Mapping[str, Any]) -> str:
    direct = analysis["direct"]
    free = analysis["free"]
    cells_m = direct["margin_cells"]
    cells_t = direct["target_cells"]
    metrics = direct["metrics"]
    official = direct["official_primary_margin_result"]
    target_did = metrics["target_social_did"]
    lines = [
        "# C2/V8 Margin–Choice Dissociation：零 GPU 补充分析",
        "",
        "状态：完成；post-hoc 描述性补充，不替代预注册主终点。",
        "",
        "## 分析问题",
        "",
        "p25 对 S 相对 AO 的连续 margin 影响，与它对离散 target 选择的影响是否同向？",
        "",
        "## 主要结果",
        "",
        "| Arm | Uncapped margin | p25 margin | Cap margin effect | "
        "Uncapped target | p25 target | Cap target effect |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        lines.append(
            f"| {arm} | "
            f"{cells_m[arm]['uncapped']['estimate']:.4f} | "
            f"{cells_m[arm]['p25']['estimate']:.4f} | "
            f"{metrics[f'cap_margin_{arm}']['estimate']:+.4f} | "
            f"{pct(cells_t[arm]['uncapped']['estimate'])} | "
            f"{pct(cells_t[arm]['p25']['estimate'])} | "
            f"{100 * metrics[f'cap_target_{arm}']['estimate']:+.2f}pp |"
        )
    lines.extend(
        [
            "",
            f"- 预注册主 margin 交互：`{official['estimate']:+.5f}` logit，"
            f"95% CI `[{official['ci_low']:+.5f}, {official['ci_high']:+.5f}]`。",
            f"- 探索性离散 target 交互：`{100 * target_did['estimate']:+.2f}pp`，"
            f"post-hoc 95% CI `[{100 * target_did['ci_low']:+.2f}, "
            f"{100 * target_did['ci_high']:+.2f}]pp`。",
            f"- 600个base中，margin交互为负/零/正："
            f"`{analysis['direction_counts']['margin_negative']}/"
            f"{analysis['direction_counts']['margin_zero']}/"
            f"{analysis['direction_counts']['margin_positive']}`；"
            f"离散交互为负/零/正："
            f"`{analysis['direction_counts']['target_negative']}/"
            f"{analysis['direction_counts']['target_zero']}/"
            f"{analysis['direction_counts']['target_positive']}`。",
            "",
            "## 结果解释",
            "",
            "p25 在 AO 和 S 中都把连续答案边界推向 induced target，但 AO 的平均平移更大，"
            "所以 `S-AO` margin contrast 缩小。与此同时，S 的 uncapped margin 更接近零，"
            "较小的绝对平移也能让更多临界题跨越离散决策边界。因此，连续社会差异下降不保证"
            "离散错误跟随同步下降。",
            "",
            "这支持一个 `margin–choice dissociation` 描述，但不证明内部真实谄媚增加或减少。",
            "",
            "## Free-thinking 三层读出",
            "",
            f"- strict final target SocialExcess："
            f"`{100 * free['metrics']['social_strict_target_did']['estimate']:+.2f}pp`；",
            f"- actual-answer margin SocialExcess："
            f"`{free['metrics']['social_actual_margin_did']['estimate']:+.4f}`；",
            f"- scaffold margin SocialExcess："
            f"`{free['metrics']['social_scaffold_margin_did']['estimate']:+.4f}`；",
            f"- thinking-token SocialExcess："
            f"`{free['metrics']['social_thinking_tokens_did']['estimate']:+.2f}` tokens，"
            f"95% CI `["
            f"{free['metrics']['social_thinking_tokens_did']['ci_low']:+.2f}, "
            f"{free['metrics']['social_thinking_tokens_did']['ci_high']:+.2f}]`。",
            "",
            "最终 target-follow 四格均为零，属于地板效应，不能作为“thinking 完全消除副作用”"
            "的等价性证据。",
            "",
            "## 输出完整性与混淆",
            "",
            f"- entropy `S-AO` cap interaction："
            f"`{metrics['entropy_social_did']['estimate']:+.6f}`；",
            f"- option-mass `S-AO` cap interaction："
            f"`{metrics['option_mass_social_did']['estimate']:+.9f}`；",
            f"- AO−S 平均 prompt token 差："
            f"`{metrics['prompt_token_difference_AO_minus_S']['estimate']:+.2f}`；",
            f"- AO−S 平均 answer-step correction sum 差："
            f"`{metrics['answer_step_correction_AO_minus_S']['estimate']:+.2f}`；",
            f"- target字母只覆盖："
            f"`{', '.join(sorted(direct['target_letter_sensitivity']))}`。",
            "",
            "## Claim Candidates",
            "",
            "- Claim:",
            "  - Source evidence: Direct margin interaction and paired discrete "
            "target interaction.",
            "  - Allowed wording: p25 compressed the social-pressure margin contrast while "
            "increasing discrete target selections near the forced-choice boundary.",
            "  - Forbidden stronger wording: p25 reduces or increases genuine "
            "sycophancy in general.",
            "  - Uncertainty: post-hoc discrete endpoint; A/B-only targets; one model and dose.",
            "  - Next check: exact-token-matched, A/B/C/D-balanced held-out confirmation.",
            "  - Decision: keep, explicitly labeled post-hoc.",
            "",
            "- Claim:",
            "  - Source evidence: Free strict outputs remain at zero target-follow.",
            "  - Allowed wording: the direct boundary shift did not manifest in this easy "
            "free-thinking subset.",
            "  - Forbidden stronger wording: thinking universally protects against "
            "capping effects.",
            "  - Uncertainty: final-answer floor effect.",
            "  - Next check: harder natural items with non-zero uncapped target-follow.",
            "  - Decision: weaken.",
            "",
            "## 主要限制",
            "",
            "- 本分析在验收后提出，所有新增置信区间均为 post-hoc 描述性区间；",
            "- target只包含A/B，无法排除答案字母偏置对绝对arm effect的贡献；",
            "- AO/S没有做到exact-token-match，实际answer-step修正量也不同；",
            "- 600条只对应272个normalized stems；结构聚类敏感性应与base分析并列；",
            "- 单模型、单p25剂量、direct nonthinking和free thinking使用不同读出范式；",
            "- 单一数据负责人审核的协议偏离仍然存在。",
            "",
        ]
    )
    return "\n".join(lines)


def stats_markdown(analysis: Mapping[str, Any]) -> str:
    direct = analysis["direct"]
    metrics = direct["metrics"]
    clusters = direct["cluster_sensitivity"]
    lines = [
        "# C2/V8 Margin–Choice 补充统计附录",
        "",
        "## 统计口径",
        "",
        f"- Base单位：{direct['n_bases']}；",
        f"- Bootstrap：{analysis['bootstrap']['samples']}次；",
        f"- Seed：{analysis['bootstrap']['seed']}；",
        "- 分层：domain × difficulty；",
        "- 区间：percentile 95% CI；",
        "- 所有新增检验均为post-hoc且未做多重比较校正。",
        "",
        "## 核心差分之差",
        "",
        "| Metric | Estimate | SD across bases | 95% CI |",
        "|---|---:|---:|---:|",
    ]
    for name in ("margin_social_did", "target_social_did"):
        value = metrics[name]
        scale = 100 if name == "target_social_did" else 1
        suffix = "pp" if scale == 100 else " logit"
        lines.append(
            f"| {name} | {scale * value['estimate']:+.4f}{suffix} | "
            f"{scale * value['sd_across_bases']:.4f} | "
            f"[{scale * value['ci_low']:+.4f}, "
            f"{scale * value['ci_high']:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "## 结构聚类敏感性",
            "",
            "| Cluster | N clusters | Margin DID 95% CI | Target DID 95% CI |",
            "|---|---:|---:|---:|",
        ]
    )
    for field in ("exact_mcq_id", "normalized_stem_id"):
        margin_result = clusters[field]["margin_social_did"]
        target_result = clusters[field]["target_social_did"]
        lines.append(
            f"| {field} | {margin_result['clusters']} | "
            f"[{margin_result['ci_low']:+.4f}, "
            f"{margin_result['ci_high']:+.4f}] | "
            f"[{100 * target_result['ci_low']:+.2f}, "
            f"{100 * target_result['ci_high']:+.2f}]pp |"
        )
    lines.extend(
        [
            "",
            "## Target字母敏感性",
            "",
            "| Target | N | Margin DID | Target DID |",
            "|---|---:|---:|---:|",
        ]
    )
    for target, values in direct["target_letter_sensitivity"].items():
        lines.append(
            f"| {target} | {values['n']} | "
            f"{values['margin_social_did']:+.4f} | "
            f"{100 * values['target_social_did']:+.2f}pp |"
        )
    lines.extend(
        [
            "",
            "## 完整迁移",
            "",
            "```json",
            json.dumps(
                {
                    "direct": direct["transitions"],
                    "free_strict": analysis["free"]["strict_transitions"],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
            "```",
            "",
            "## 推断边界",
            "",
            "风险差本身是离散效应量；margin以logit差报告。由于新增分析是在结果验收后提出，"
            "不使用p值升级主张，也不把未校正区间解释成新的确认性Gate。",
            "",
        ]
    )
    return "\n".join(lines)


def figure_catalog_markdown() -> str:
    return """# C2/V8 补充图表目录

## figure-01-margin-choice-dissociation.svg

- Purpose：并列显示同一四格实验的连续 margin 与离散 target rate。
- Data source：600个严格配对的Direct base。
- Caption requirement：左图单位为target-minus-initial logit；右图为target选择率；连线只连接
  同一arm的uncapped/p25均值，不表示时间序列；本图为post-hoc补充。
- Observation：p25令AO/S margin均向target移动，但AO平移更大；离散target选择却在S中增加
  更多。
- Interpretation：相对margin压缩与临界选择恶化可以同时发生。
- Implication：不能把负的social margin interaction直接称为行为安全收益。
- Caveat：target只覆盖A/B，四格置信区间为未校正描述性区间。

## figure-02-boundary-transitions.svg

- Purpose：检查S arm的离散翻转是否集中于uncapped margin接近零的题。
- Data source：全部600个S-arm Direct base。
- Caption requirement：横轴为uncapped margin，纵轴为p25 margin；虚线为零边界；颜色表示
  完整迁移类别；不删除other-wrong或极端点。
- Observation：initial-to-target翻转集中在决策边界附近，而大多数题保持原类别。
- Interpretation：离散结果可由连续边界平移与阈值共同产生。
- Implication：后续实验必须同时报告连续margin和四态选择。
- Caveat：相关图不能确定内部机制，也不能区分字母偏置与语义偏移。
"""


def svg_text(x: float, y: float, text: str, **attrs: Any) -> str:
    properties = {
        "x": f"{x:.2f}",
        "y": f"{y:.2f}",
        "font-family": "Arial, sans-serif",
        "font-size": "13",
        "fill": "#222222",
    }
    properties.update({key.replace("_", "-"): str(value) for key, value in attrs.items()})
    joined = " ".join(
        f'{key}="{html.escape(value)}"' for key, value in properties.items()
    )
    return f"<text {joined}>{html.escape(text)}</text>"


def write_figure_one(path: Path, analysis: Mapping[str, Any]) -> None:
    direct = analysis["direct"]
    margin_cells = direct["margin_cells"]
    target_cells = direct["target_cells"]
    width, height = 1120, 430
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(280, 28, "A. Target-minus-initial margin", text_anchor="middle", font_size=17),
        svg_text(840, 28, "B. Discrete target selection rate", text_anchor="middle", font_size=17),
    ]
    colors = {"AO": "#0072B2", "S": "#D55E00"}
    left = (70, 520, 55, 350)
    right = (630, 1080, 55, 350)
    margin_min, margin_max = -9.0, 0.0
    target_min, target_max = 0.0, 0.45

    def scale_y(
        value: float,
        bounds: tuple[float, float, float, float],
        low: float,
        high: float,
    ) -> float:
        _, _, top, bottom = bounds
        return bottom - (value - low) / (high - low) * (bottom - top)

    for bounds, low, high, ticks in (
        (left, margin_min, margin_max, (-9, -6, -3, 0)),
        (right, target_min, target_max, (0.0, 0.15, 0.30, 0.45)),
    ):
        x0, x1, top, bottom = bounds
        lines.append(f'<line x1="{x0}" y1="{bottom}" x2="{x1}" y2="{bottom}" stroke="#222"/>')
        lines.append(f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{bottom}" stroke="#222"/>')
        for tick in ticks:
            y = scale_y(float(tick), bounds, low, high)
            lines.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x1}" y2="{y:.2f}" stroke="#dddddd"/>')
            label = f"{100 * tick:.0f}%" if bounds == right else f"{tick:.0f}"
            lines.append(svg_text(x0 - 10, y + 4, label, text_anchor="end", font_size=11))
        for index, condition in enumerate(CONDITIONS):
            x = x0 + 145 + index * 190
            lines.append(svg_text(x, bottom + 28, condition, text_anchor="middle", font_size=12))
    x_positions_left = {"uncapped": 215, "p25": 405}
    x_positions_right = {"uncapped": 775, "p25": 965}
    for arm in ARMS:
        margin_points = []
        target_points = []
        for condition in CONDITIONS:
            margin_value = float(margin_cells[arm][condition]["estimate"])
            target_value = float(target_cells[arm][condition]["estimate"])
            mx = x_positions_left[condition] + (-18 if arm == "AO" else 18)
            tx = x_positions_right[condition] + (-18 if arm == "AO" else 18)
            my = scale_y(margin_value, left, margin_min, margin_max)
            ty = scale_y(target_value, right, target_min, target_max)
            margin_points.append((mx, my))
            target_points.append((tx, ty))
            lines.append(f'<circle cx="{mx}" cy="{my:.2f}" r="7" fill="{colors[arm]}"/>')
            lines.append(f'<circle cx="{tx}" cy="{ty:.2f}" r="7" fill="{colors[arm]}"/>')
            lines.append(
                svg_text(
                    mx,
                    my - 12,
                    f"{margin_value:.2f}",
                    text_anchor="middle",
                    font_size=10,
                )
            )
            lines.append(
                svg_text(
                    tx,
                    ty - 12,
                    f"{100 * target_value:.1f}%",
                    text_anchor="middle",
                    font_size=10,
                )
            )
        for points in (margin_points, target_points):
            lines.append(
                f'<line x1="{points[0][0]}" y1="{points[0][1]:.2f}" '
                f'x2="{points[1][0]}" y2="{points[1][1]:.2f}" '
                f'stroke="{colors[arm]}" stroke-width="2"/>'
            )
    lines.extend(
        [
            '<circle cx="445" cy="390" r="6" fill="#0072B2"/>',
            svg_text(458, 395, "AO", font_size=12),
            '<circle cx="505" cy="390" r="6" fill="#D55E00"/>',
            svg_text(518, 395, "S", font_size=12),
            svg_text(
                560,
                420,
                "n=600 paired bases; post-hoc descriptive addendum",
                text_anchor="middle",
                font_size=11,
                fill="#555555",
            ),
            "</svg>",
        ]
    )
    atomic_write_text(path, "\n".join(lines) + "\n")


def write_figure_two(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    width, height = 760, 650
    margins = [
        float(row["margin_S_uncapped"])
        for row in rows
    ] + [
        float(row["margin_S_p25"])
        for row in rows
    ]
    low = min(-20.0, math.floor(min(margins)))
    high = max(12.0, math.ceil(max(margins)))
    plot_left, plot_right, plot_top, plot_bottom = 85, 710, 45, 565

    def sx(value: float) -> float:
        return plot_left + (value - low) / (high - low) * (plot_right - plot_left)

    def sy(value: float) -> float:
        return plot_bottom - (value - low) / (high - low) * (plot_bottom - plot_top)

    colors = {
        "initial->initial": "#0072B2",
        "initial->target": "#D55E00",
        "target->initial": "#009E73",
        "target->target": "#777777",
        "other": "#CC79A7",
    }
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        svg_text(
            397,
            25,
            "S-arm decision boundary transitions",
            text_anchor="middle",
            font_size=17,
        ),
        f'<rect x="{plot_left}" y="{plot_top}" width="{plot_right - plot_left}" '
        f'height="{plot_bottom - plot_top}" fill="#fafafa" stroke="#222"/>',
        f'<line x1="{sx(0):.2f}" y1="{plot_top}" x2="{sx(0):.2f}" '
        f'y2="{plot_bottom}" stroke="#555" stroke-dasharray="5 4"/>',
        f'<line x1="{plot_left}" y1="{sy(0):.2f}" x2="{plot_right}" '
        f'y2="{sy(0):.2f}" stroke="#555" stroke-dasharray="5 4"/>',
        f'<line x1="{sx(low):.2f}" y1="{sy(low):.2f}" x2="{sx(high):.2f}" '
        f'y2="{sy(high):.2f}" stroke="#bbbbbb"/>',
    ]
    for tick in range(int(math.ceil(low / 5) * 5), int(high) + 1, 5):
        lines.append(
            svg_text(
                sx(tick),
                plot_bottom + 20,
                str(tick),
                text_anchor="middle",
                font_size=10,
            )
        )
        lines.append(
            svg_text(
                plot_left - 10,
                sy(tick) + 4,
                str(tick),
                text_anchor="end",
                font_size=10,
            )
        )
    counts = Counter()
    for row in rows:
        before = str(row["category_S_uncapped"])
        after = str(row["category_S_p25"])
        transition = f"{before}->{after}"
        key = transition if transition in colors else "other"
        counts[key] += 1
        lines.append(
            f'<circle cx="{sx(float(row["margin_S_uncapped"])):.2f}" '
            f'cy="{sy(float(row["margin_S_p25"])):.2f}" r="3.2" '
            f'fill="{colors[key]}" fill-opacity="0.55"/>'
        )
    lines.extend(
        [
            svg_text(
                397,
                610,
                "Uncapped target-minus-initial margin",
                text_anchor="middle",
                font_size=13,
            ),
            '<text x="22" y="305" transform="rotate(-90 22 305)" '
            'font-family="Arial, sans-serif" font-size="13" fill="#222" '
            'text-anchor="middle">p25 target-minus-initial margin</text>',
        ]
    )
    legend_x, legend_y = 105, 635
    for index, key in enumerate(
        ("initial->initial", "initial->target", "target->initial", "target->target", "other")
    ):
        x = legend_x + index * 125
        lines.append(f'<circle cx="{x}" cy="{legend_y}" r="5" fill="{colors[key]}"/>')
        lines.append(svg_text(x + 10, legend_y + 4, f"{key} ({counts[key]})", font_size=10))
    lines.append("</svg>")
    atomic_write_text(path, "\n".join(lines) + "\n")


def delivery_manifest(output_root: Path) -> dict[str, Any]:
    files = {}
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "delivery_manifest.json":
            files[path.relative_to(output_root).as_posix()] = {
                "bytes": path.stat().st_size,
                "canonical_text_sha256": canonical_text_sha256(path),
            }
    return {
        "status": "complete",
        "analysis": "C2/V8 margin-choice dissociation post-hoc addendum",
        "files": files,
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_samples <= 0:
        raise ValueError("Bootstrap samples must be positive")
    audit_hashes = validate_audits(args.input_root)
    original_analysis = read_json(args.input_root / "C2_V8_ANALYSIS.json")
    direct_pairs = pair_conditions(
        load_phase(args.input_root, "direct-uncapped"),
        load_phase(args.input_root, "direct-p25"),
        1200,
    )
    free_pairs = pair_conditions(
        load_phase(args.input_root, "free-uncapped"),
        load_phase(args.input_root, "free-p25"),
        480,
    )
    direct_bases = build_bases(direct_pairs, 600)
    free_bases = build_bases(free_pairs, 240)
    per_base = direct_base_rows(direct_bases)
    direct = summarize_direct(
        per_base,
        original_analysis,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    free = summarize_free(
        free_bases,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    direction_counts = {
        "margin_negative": sum(row["margin_social_did"] < 0 for row in per_base),
        "margin_zero": sum(row["margin_social_did"] == 0 for row in per_base),
        "margin_positive": sum(row["margin_social_did"] > 0 for row in per_base),
        "target_negative": sum(row["target_social_did"] < 0 for row in per_base),
        "target_zero": sum(row["target_social_did"] == 0 for row in per_base),
        "target_positive": sum(row["target_social_did"] > 0 for row in per_base),
    }
    analysis = {
        "status": "complete",
        "analysis_type": "zero_gpu_posthoc_addendum",
        "source_results_commit": SOURCE_RESULTS_COMMIT,
        "analysis_commit": git_value("rev-parse", "HEAD"),
        "protocol": str(PROTOCOL),
        "protocol_canonical_text_sha256": canonical_text_sha256(ROOT / PROTOCOL),
        "script_canonical_text_sha256": canonical_text_sha256(Path(__file__)),
        "input_hashes": audit_hashes,
        "bootstrap": {
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "unit": "base",
            "stratification": ["domain", "difficulty"],
            "interval": "percentile_95",
            "multiplicity": "unadjusted_posthoc_descriptive",
        },
        "direct": direct,
        "free": free,
        "direction_counts": direction_counts,
        "interpretation": {
            "label": "margin_choice_dissociation",
            "allowed": (
                "p25 compressed the S-versus-AO margin contrast while discrete "
                "target selections moved in the opposite normative direction "
                "near the forced-choice boundary"
            ),
            "forbidden": (
                "No result establishes a general increase or decrease in "
                "internal genuine sycophancy"
            ),
        },
        "forced_stop": True,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.output_root / "margin_choice_addendum.json", analysis)
    write_per_base_csv(args.output_root / "per_base_margin_choice.csv", per_base)
    atomic_write_text(args.output_root / "analysis-report.md", report_markdown(analysis))
    atomic_write_text(args.output_root / "stats-appendix.md", stats_markdown(analysis))
    atomic_write_text(
        args.output_root / "figure-catalog.md", figure_catalog_markdown()
    )
    figures = args.output_root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    write_figure_one(figures / "figure-01-margin-choice-dissociation.svg", analysis)
    write_figure_two(figures / "figure-02-boundary-transitions.svg", per_base)
    atomic_write_json(
        args.output_root / "delivery_manifest.json",
        delivery_manifest(args.output_root),
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "output": str(args.output_root),
                "margin_social_did": direct["metrics"]["margin_social_did"],
                "target_social_did": direct["metrics"]["target_social_did"],
                "free_final_target_floor": free["final_target_floor"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
