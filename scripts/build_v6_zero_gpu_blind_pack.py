#!/usr/bin/env python3
"""Build the frozen, condition-blind 1152-record v6 annotation packet."""

from __future__ import annotations

import argparse
import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cedar_m2.eval.v6_zero_gpu import (
    ANNOTATION_FIELDS,
    CAPS,
    MODES,
    file_sha256,
    final_surface,
    read_jsonl,
    sha256_text,
    validate_source_answer_keys,
    write_csv,
    write_json,
    write_jsonl,
)

ROOT = Path(__file__).resolve().parents[1]
V6_ROOT = ROOT / "artifacts" / "m2" / "m2_2b_v6_factorial_diagnostic"
V5_ROOT = ROOT / "artifacts" / "m2" / "m2_2b_v5_dose_response_bank"
DATA_PATH = (
    ROOT
    / "datasets"
    / "m2_2b_formal"
    / "v6_factorial_diagnostic"
    / "m2_2b_v6_factorial144.jsonl"
)
DEFAULT_OUTPUT = ROOT / "artifacts" / "m2" / "m2_2b_v6_zero_gpu_reaudit"
PROTOCOL_COMMIT = "cdeff7b19c64b43d4d01dfdc70a1fb85ba61f3ce"
SOURCE_COMMITS = (
    "548e5eb1db7e5fccb9b968b77da8c05c1c573503",
    "bb404c4abea59e3e71b087d0c9ce62ebb240e4c6",
    "3cbed5db704abb1edde1da424000d3ddcc76ad60",
    "0ae91534759cf8b34413d3244393456f86aeb72c",
)
SHUFFLE_SEED = 20260718


@dataclass(frozen=True)
class FrozenFile:
    label: str
    path: Path
    rows: int
    sha256: str
    source_commit: str


FORMAL_HASHES = {
    ("thinking_sampling", "uncapped"): (
        "23cec34bc643ff0a30682a4d5e0fbd80e311dd15aa2f05f430460604129f9240"
    ),
    ("thinking_sampling", "p25"): (
        "6ce1c71f73b9b26704dc56aebf8d3679306f0cd01a129294b7f3e03727618335"
    ),
    ("nonthinking_sampling", "uncapped"): (
        "6077682fa516711efe9ed97a0d7f08f5c0d806a20e95b6020cf252fb5b984787"
    ),
    ("nonthinking_sampling", "p25"): (
        "8070ee8d3db35a50de5a40cd0a1f25fe512cb51583b8bffa69d1776d700de69a"
    ),
    ("thinking_greedy", "uncapped"): (
        "980da4d48f0c1e74b53bbb5663c360779913ef2678e5738d924a0c9a285bfb3d"
    ),
    ("thinking_greedy", "p25"): "b96e209992f2a8d93347c673e3020aac00dfe28f15c2db5241384c7ee4041659",
    ("nonthinking_greedy", "uncapped"): (
        "4fb844b1bb96120d84823bee561e89515cc7d4879354f73fc281218d256b8411"
    ),
    ("nonthinking_greedy", "p25"): (
        "faeafd4ceb31dddd803e5ce00ba349bd50f7f8c35320b580133e0a71e89f02ef"
    ),
}
AXIS_HASHES = {
    ("thinking_sampling", "uncapped"): (
        "18f6a175322b650664141a5ffd590c7a4b3cd239fd12b88d0ec08abbfaa0fbed"
    ),
    ("thinking_sampling", "p25"): (
        "a9750995e369f4c2560e817ce50050c2e30bb1b6a44d54b811b691d894f6355d"
    ),
    ("nonthinking_sampling", "uncapped"): (
        "e7f0c065e5f96d2f8f2f33c4d2f2476f3ee7114f56c1128b46b4a139fc650868"
    ),
    ("nonthinking_sampling", "p25"): (
        "12971d2a4fb887db6eb4b20760e11304005a064d5f59259a742173b375450872"
    ),
}
V5_HASHES = {
    "uncapped": "d88e0dcff8a91a34d3c7a6e8c8b71b89a7b98fb46e68cdb441e5873f80f2a546",
    "p75": "bbfa172df8af80efd51043d58ac92828832bd7c711cf9ba88573a3255ea66318",
    "p50": "939feef134aa81745cfc86b7fc50963b5ce418d808062baa854e6f1c3566eac6",
    "p25": "da6fffe86ce0d2b9e8d326b35183def5e41b0fbdd36b179e756c7cfc6b7a5f07",
    "p1": "9dea7cd17f6ce1fd62192b7377f7747debf7909269a21af8fd4babc7129bcf93",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def require_protocol_commit() -> str:
    head = git_output("rev-parse", "HEAD")
    for commit in (PROTOCOL_COMMIT, *SOURCE_COMMITS):
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, head],
            cwd=ROOT,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Frozen commit {commit} is not an ancestor of {head}")
    return head


def formal_file(mode: str, cap: str) -> FrozenFile:
    run = V6_ROOT / f"qwen3_32b_v6_formal_{mode}_{cap}_seed0_development"
    commit = (
        "548e5eb1db7e5fccb9b968b77da8c05c1c573503"
        if mode.endswith("sampling")
        else "3cbed5db704abb1edde1da424000d3ddcc76ad60"
    )
    return FrozenFile(
        f"formal/{mode}/{cap}", run / "results.jsonl", 144, FORMAL_HASHES[(mode, cap)], commit
    )


def other_frozen_files() -> list[FrozenFile]:
    files: list[FrozenFile] = []
    for mode in ("thinking_sampling", "nonthinking_sampling"):
        for cap in CAPS:
            run = V6_ROOT / f"qwen3_32b_v6_axis_{mode}_{cap}_seed0_development"
            files.append(
                FrozenFile(
                    f"axis/{mode}/{cap}",
                    run / "results.jsonl",
                    24,
                    AXIS_HASHES[(mode, cap)],
                    "bb404c4abea59e3e71b087d0c9ce62ebb240e4c6",
                )
            )
    for cap, digest in V5_HASHES.items():
        run = V5_ROOT / f"qwen3_32b_v5_dose_bank_{cap}_thinking_greedy_seed0_development"
        files.append(
            FrozenFile(
                f"v5/{cap}",
                run / "qwen_thinking_results.jsonl",
                468,
                digest,
                "0ae91534759cf8b34413d3244393456f86aeb72c",
            )
        )
    return files


def verify_frozen_file(spec: FrozenFile) -> list[dict[str, Any]]:
    if not spec.path.exists():
        raise FileNotFoundError(spec.path)
    actual_hash = file_sha256(spec.path)
    if actual_hash != spec.sha256:
        raise ValueError(f"{spec.label}: SHA-256 {actual_hash} != frozen {spec.sha256}")
    rows = read_jsonl(spec.path)
    if len(rows) != spec.rows:
        raise ValueError(f"{spec.label}: rows={len(rows)} != {spec.rows}")
    return rows


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected object")
    return value


def audit_cap_pair_inputs(mode: str) -> dict[str, Any]:
    signatures: dict[str, dict[str, Any]] = {}
    for cap in CAPS:
        run = V6_ROOT / f"qwen3_32b_v6_formal_{mode}_{cap}_seed0_development"
        manifest = load_json(run / "manifest.json")
        runtime = load_json(run / "runtime.json")
        configuration = manifest["configuration"]
        signatures[cap] = {
            "dataset_sha256": configuration["dataset"]["sha256"],
            "row_ids_sha256": configuration["dataset"]["row_ids_sha256"],
            "mode": configuration["mode"],
            "model": configuration["model"],
            "generation": configuration["generation"],
            "rendered_prompts_sha256": runtime["rendered_prompts_sha256"],
            "tokenized_input_ids_sha256": runtime["tokenized_input_ids_sha256"],
            "padding_side": runtime["tokenizer_padding_side"],
            "pad_token_id": runtime["tokenizer_pad_token_id"],
        }
    if signatures["uncapped"] != signatures["p25"]:
        raise ValueError(f"{mode}: uncapped/p25 inputs differ")
    return signatures["uncapped"]


def shuffled_without_adjacent_source(
    entries: list[dict[str, Any]], seed: int
) -> tuple[list[dict[str, Any]], int]:
    rng = random.Random(seed)
    for attempt in range(1, 100_001):
        candidate = list(entries)
        rng.shuffle(candidate)
        if all(
            left["source_id"] != right["source_id"]
            for left, right in zip(candidate, candidate[1:], strict=False)
        ):
            return candidate, attempt
    raise RuntimeError("Could not construct a non-adjacent deterministic blind ordering")


def annotation_instructions() -> str:
    return """# v6 零 GPU 复核：盲提取说明

你只需要抄录模型最终明确给出的选项，不需要解题。

每条填写：

- `answer_extracted`：A、B、C；若没有唯一明确答案则留空；
- `json_valid`：最终文本整体是否为合法 JSON 对象；
- `complete`：最终回答是否完整；
- `ambiguity_reason`：无歧义留空，否则只用下列枚举；
- `notes`：可选。

歧义枚举：

```text
no_explicit_answer
multiple_conflicting_answers
malformed_json
incomplete_or_truncated
reasoning_only
other
```

规则：

1. 不查看仓库、prompt、原始结果或 private key；
2. 不判断答案是否正确、是否更新、是否谄媚或论证是否有效；
3. malformed JSON 若仍有唯一明确 A/B/C，可以抄录答案，同时 `json_valid=false`；
4. 多个冲突答案不能自行选择，答案留空；
5. 空的 final surface 记为答案空、`json_valid=false`、`complete=false`、
   `ambiguity_reason=incomplete_or_truncated`；
6. 独立完成全部 1152 条后再交回，不与另一位标注者讨论单条结果。
"""


def main() -> None:
    args = parse_args()
    head = require_protocol_commit()
    sources = {row["id"]: row for row in read_jsonl(DATA_PATH)}
    if len(sources) != 144:
        raise ValueError(f"Expected 144 source rows, found {len(sources)}")
    for source in sources.values():
        validate_source_answer_keys(source)

    frozen_inventory: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    input_signatures = {mode: audit_cap_pair_inputs(mode) for mode in MODES}
    for mode in MODES:
        for cap in CAPS:
            spec = formal_file(mode, cap)
            rows = verify_frozen_file(spec)
            ids = [row.get("id") for row in rows]
            if ids != list(sources):
                raise ValueError(f"{spec.label}: result ID order differs from frozen dataset")
            for index, row in enumerate(rows):
                if row.get("mode") != mode or row.get("cap") != cap:
                    raise ValueError(f"{spec.label}:{index}: embedded condition mismatch")
                response = row.get("response")
                if not isinstance(response, str):
                    raise ValueError(f"{spec.label}:{index}: response is not text")
                surface, envelope_status = final_surface(response)
                max_tokens = int(sources[row["id"]]["recommended_max_new_tokens"])
                response_tokens = int(row.get("response_tokens", -1))
                entries.append(
                    {
                        "source_id": row["id"],
                        "mode": mode,
                        "cap": cap,
                        "source_index": index,
                        "response": response,
                        "final_surface": surface,
                        "surface_status": envelope_status,
                        "response_tokens": response_tokens,
                        "max_new_tokens": max_tokens,
                        "truncated": response_tokens >= max_tokens,
                        "result_path": str(spec.path.relative_to(ROOT)),
                    }
                )
            frozen_inventory.append(
                {
                    "label": spec.label,
                    "path": str(spec.path.relative_to(ROOT)),
                    "rows": spec.rows,
                    "sha256": spec.sha256,
                    "source_commit": spec.source_commit,
                }
            )

    for spec in other_frozen_files():
        verify_frozen_file(spec)
        frozen_inventory.append(
            {
                "label": spec.label,
                "path": str(spec.path.relative_to(ROOT)),
                "rows": spec.rows,
                "sha256": spec.sha256,
                "source_commit": spec.source_commit,
            }
        )

    if len(entries) != 1152:
        raise ValueError(f"Expected 1152 formal outputs, found {len(entries)}")
    truncated = [entry for entry in entries if entry["truncated"]]
    if len(truncated) != 1:
        raise ValueError(f"Expected exactly one max-token record, found {len(truncated)}")

    shuffled, shuffle_attempts = shuffled_without_adjacent_source(entries, SHUFFLE_SEED)
    packet_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    workbook_rows: list[dict[str, Any]] = []
    for index, entry in enumerate(shuffled, start=1):
        blind_id = f"V6ZR-{index:04d}"
        packet_rows.append({"blind_id": blind_id, "final_surface": entry["final_surface"]})
        workbook_rows.append(
            {
                "blind_id": blind_id,
                "final_surface": entry["final_surface"],
                "answer_extracted": "",
                "json_valid": "",
                "complete": "",
                "ambiguity_reason": "",
                "notes": "",
            }
        )
        key_rows.append(
            {
                "blind_id": blind_id,
                "source_id": entry["source_id"],
                "mode": entry["mode"],
                "cap": entry["cap"],
                "source_index": entry["source_index"],
                "result_path": entry["result_path"],
                "raw_response_sha256": sha256_text(entry["response"]),
                "final_surface_sha256": sha256_text(entry["final_surface"]),
                "surface_status": entry["surface_status"],
                "response_tokens": entry["response_tokens"],
                "max_new_tokens": entry["max_new_tokens"],
                "truncated": entry["truncated"],
            }
        )

    output = args.output_root
    packet_path = output / "blind" / "annotation_packet.jsonl"
    key_path = output / "private" / "blind_key.jsonl"
    a_path = output / "blind" / "annotator_a.csv"
    b_path = output / "blind" / "annotator_b.csv"
    write_jsonl(packet_path, packet_rows)
    write_jsonl(key_path, key_rows)
    workbook_fields = ("blind_id", "final_surface", *ANNOTATION_FIELDS[1:])
    write_csv(a_path, workbook_rows, workbook_fields)
    write_csv(b_path, workbook_rows, workbook_fields)
    instructions_path = output / "blind" / "ANNOTATION_INSTRUCTIONS_ZH.md"
    instructions_path.write_text(annotation_instructions(), encoding="utf-8")

    manifest = {
        "status": "ready_for_blind_annotation",
        "scope": "zero-GPU extraction only; no behavior scoring performed",
        "branch": git_output("branch", "--show-current"),
        "head": head,
        "protocol_commit": PROTOCOL_COMMIT,
        "shuffle_seed": SHUFFLE_SEED,
        "shuffle_attempts": shuffle_attempts,
        "formal_records": len(entries),
        "known_max_token_records": len(truncated),
        "packet_sha256": file_sha256(packet_path),
        "blind_key_sha256": file_sha256(key_path),
        "annotator_template_sha256": file_sha256(a_path),
        "instructions_sha256": file_sha256(instructions_path),
        "input_signatures": input_signatures,
        "frozen_inputs": frozen_inventory,
        "packet_visible_fields": ["blind_id", "final_surface"],
        "private_key_fields": sorted(key_rows[0]),
        "gpu_or_model_execution": False,
    }
    write_json(output / "input_freeze_manifest.json", manifest)
    write_json(
        output / "private" / "blind_manifest.json",
        {
            "status": "private_do_not_distribute_to_annotators",
            "packet_sha256": manifest["packet_sha256"],
            "blind_key_sha256": manifest["blind_key_sha256"],
            "known_max_token_blind_ids": [
                row["blind_id"] for row in key_rows if row["truncated"]
            ],
        },
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "records": len(entries),
                "packet": str(packet_path),
                "private_key": str(key_path),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
