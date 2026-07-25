#!/usr/bin/env python3
"""Build the deterministic 192-trace E2B fixed-trace bank without using a GPU."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SCRIPTS = ROOT / "scripts"
E2A_DATA = HERE.parent / "e2_phase_validity_choice"
for path in (HERE, SCRIPTS, E2A_DATA):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_prover_generator_step5_uncapped as base  # noqa: E402
from e2b_common import (  # noqa: E402
    ANALYSIS_PLAN,
    ANCHOR_RESULTS,
    EXPECTED_FORMAL_CELLS,
    EXPECTED_PROMPTS,
    EXPECTED_TRACES,
    FORWARD_CONDITIONS,
    LABELS,
    MODEL_REPO,
    MODEL_REVISION,
    SOURCE_CONDITION,
    TRACE_SOURCES,
    canonical_text_sha256,
    object_sha256,
    read_jsonl,
    reconstruct_response_ids,
    require,
    smoke_prompt_ids,
    validate_anchor,
    validate_frozen_protocol_hashes,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hf-home", type=Path, default=Path(os.environ.get("HF_HOME", "/data1/WuMingrui/HF_HOME"))
    )
    parser.add_argument("--output-dir", type=Path, default=HERE)
    return parser.parse_args()


def _load_tokenizer(hf_home: Path) -> Any:
    os.environ["HF_HOME"] = str(hf_home.resolve())
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        MODEL_REPO,
        revision=MODEL_REVISION,
        trust_remote_code=False,
        local_files_only=True,
    )


def _source_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompts = read_jsonl(E2A_DATA / "answer_inputs96.jsonl")
    require(len(prompts) == EXPECTED_PROMPTS, "Frozen E2A Answer prompt count changed")
    require(len({row["id"] for row in prompts}) == EXPECTED_PROMPTS, "Prompt IDs are not unique")
    anchor = [
        row
        for row in read_jsonl(ANCHOR_RESULTS)
        if row.get("assay") == "answer"
        and row.get("condition") in {"uncapped", "full_p25"}
    ]
    require(len(anchor) == EXPECTED_TRACES, "Frozen E2A source trace count changed")
    return prompts, anchor


def build(output_dir: Path, tokenizer: Any) -> dict[str, Any]:
    validate_frozen_protocol_hashes()
    anchor_validation = validate_anchor()
    prompts, anchor = _source_rows()
    prompt_by_id = {row["id"]: row for row in prompts}
    mode = base.MODES["thinking"]

    observed_label_ids = {
        label: tokenizer.encode(label, add_special_tokens=False) for label in LABELS
    }
    require(
        all(len(token_ids) == 1 for token_ids in observed_label_ids.values()),
        "Pinned A/B/C/D labels are not single tokens",
    )

    traces: list[dict[str, Any]] = []
    reconstruction_methods: Counter[str] = Counter()
    for source in anchor:
        source_id = str(source["id"])
        prompt_row = prompt_by_id[source_id]
        require(source.get("truncated") is False, f"{source_id}: source trace is truncated")
        require(
            source.get("final_answer_parse_status") == "passed",
            f"{source_id}: source trace did not parse",
        )
        require(source.get("answer_logits") is not None, f"{source_id}: source logits missing")

        prompt = base.render_prompt(tokenizer, prompt_row, mode)
        prompt_ids = [int(value) for value in tokenizer.encode(prompt, add_special_tokens=False)]
        require(object_sha256(prompt) == source["prompt_sha256"], f"{source_id}: prompt hash drift")
        require(
            object_sha256(prompt_ids) == source["prompt_token_ids_sha256"],
            f"{source_id}: prompt-token hash drift",
        )
        require(len(prompt_ids) == source["prompt_tokens"], f"{source_id}: prompt length drift")

        response_ids, reconstruction_method = reconstruct_response_ids(tokenizer, source)
        reconstruction_methods[reconstruction_method] += 1
        require(
            object_sha256(response_ids) == source["response_token_ids_sha256"],
            f"{source_id}: response-token hash drift",
        )
        cut = source.get("selected_answer_logit_index")
        require(isinstance(cut, int) and 0 < cut < len(response_ids), f"{source_id}: invalid cut")
        boundary = source.get("answer_boundary") or {}
        require(boundary.get("answer_choice_index") == cut, f"{source_id}: boundary/cut mismatch")
        label = str(source["chosen_option"])
        label_token_id = int(source["candidate_token_ids"][label])
        require(response_ids[cut] == label_token_id, f"{source_id}: generated label-token mismatch")
        require(
            observed_label_ids[label] == [label_token_id],
            f"{source_id}: pinned candidate token changed",
        )
        candidates = source.get("answer_logit_candidates") or []
        boundary_matches = [
            item
            for item in candidates
            if item.get("generated_label") == label and item.get("response_index") == cut
        ]
        require(len(boundary_matches) == 1, f"{source_id}: selected answer boundary is ambiguous")
        require(not boundary.get("errors"), f"{source_id}: stored answer boundary has errors")

        prefix_ids = response_ids[:cut]
        prefix_text = tokenizer.decode(
            prefix_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        require(
            source["response"].startswith(prefix_text),
            f"{source_id}: trace text is not a prefix",
        )
        character_cut = len(prefix_text)
        require(
            source["response"][character_cut : character_cut + 1] == label,
            f"{source_id}: character answer boundary mismatch",
        )
        trace_source = (
            "e2a_uncapped" if source["condition"] == "uncapped" else "e2a_full_p25"
        )
        trace_id = f"{source_id}::{trace_source}"
        traces.append(
            {
                "schema": "cedar.e2b_fixed_trace.v1",
                "trace_id": trace_id,
                "source_id": source_id,
                "trace_source": trace_source,
                "source_condition": source["condition"],
                "family_id": source["family_id"],
                "base_id": source["base_id"],
                "structure_class": source["structure_class"],
                "variant": source["variant"],
                "arm": source["arm"],
                "initial_answer": source["initial_answer"],
                "target_answer": source["target_answer"],
                "gold_answer": source["gold_answer"],
                "prompt_sha256": source["prompt_sha256"],
                "prompt_token_ids_sha256": source["prompt_token_ids_sha256"],
                "prompt_tokens": len(prompt_ids),
                "response_text_sha256": object_sha256(source["response"]),
                "response_token_ids_sha256": source["response_token_ids_sha256"],
                "response_tokens": len(response_ids),
                "response_reconstruction_method": reconstruction_method,
                "trace_prefix_text": prefix_text,
                "trace_prefix_text_sha256": object_sha256(prefix_text),
                "trace_prefix_token_ids": prefix_ids,
                "trace_prefix_token_ids_sha256": object_sha256(prefix_ids),
                "trace_prefix_tokens": len(prefix_ids),
                "teacher_forced_input_token_ids_sha256": object_sha256([*prompt_ids, *prefix_ids]),
                "teacher_forced_input_tokens": len(prompt_ids) + len(prefix_ids),
                "selected_answer_logit_index": cut,
                "answer_choice_character_index": character_cut,
                "generated_label": label,
                "generated_label_token_id": label_token_id,
                "candidate_token_ids": source["candidate_token_ids"],
                "anchor_answer_logits": source["answer_logits"],
                "anchor_finish_reason": source["finish_reason"],
                "anchor_record_seed": source["record_seed"],
                "anchor_result_commit": "0c8517bb5fb4640f7d5e4d51356d0cf167046643",
            }
        )

    traces.sort(key=lambda row: (row["source_id"], TRACE_SOURCES.index(row["trace_source"])))
    require(len(traces) == EXPECTED_TRACES, "Built trace count changed")
    require(len({row["trace_id"] for row in traces}) == EXPECTED_TRACES, "Trace IDs collide")
    require(
        Counter(row["trace_source"] for row in traces)
        == Counter({"e2a_uncapped": 96, "e2a_full_p25": 96}),
        "Trace-source balance changed",
    )
    for source_id in prompt_by_id:
        paired = [trace for trace in traces if trace["source_id"] == source_id]
        require(
            {trace["trace_source"] for trace in paired} == set(TRACE_SOURCES),
            f"{source_id}: trace pair incomplete",
        )

    smoke_ids = smoke_prompt_ids(prompts)
    matrix = [
        {
            "schema": "cedar.e2b_fixed_trace_cell.v1",
            "cell_index": index,
            "cell_id": f"{trace['trace_id']}::{condition}",
            "source_id": trace["source_id"],
            "trace_id": trace["trace_id"],
            "trace_source": trace["trace_source"],
            "forward_condition": condition,
            "is_diagonal_replay": SOURCE_CONDITION[trace["trace_source"]] == condition,
        }
        for index, (trace, condition) in enumerate(
            (trace, condition) for trace in traces for condition in FORWARD_CONDITIONS
        )
    ]
    require(len(matrix) == EXPECTED_FORMAL_CELLS, "Formal E2B matrix changed")
    require(len({row["cell_id"] for row in matrix}) == EXPECTED_FORMAL_CELLS, "Cell IDs collide")

    output_dir.mkdir(parents=True, exist_ok=True)
    traces_path = output_dir / "frozen_traces192.jsonl"
    matrix_path = output_dir / "formal_matrix384.jsonl"
    write_jsonl(traces_path, traces)
    write_jsonl(matrix_path, matrix)
    report = {
        "schema": "cedar.e2b_fixed_trace_cross.build_report.v1",
        "status": "passed",
        "gpu_used": False,
        "model": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "prompt_rows": len(prompts),
        "frozen_traces": len(traces),
        "formal_cells": len(matrix),
        "smoke_prompt_ids": smoke_ids,
        "smoke_cells": len(smoke_ids) * 2 * 2,
        "reconstruction_methods": dict(sorted(reconstruction_methods.items())),
        "anchor": anchor_validation,
        "frozen_protocol_hashes": validate_frozen_protocol_hashes(),
        "analysis_plan_canonical_lf_sha256": canonical_text_sha256(ANALYSIS_PLAN),
        "outputs": {
            "frozen_traces192.jsonl": canonical_text_sha256(traces_path),
            "formal_matrix384.jsonl": canonical_text_sha256(matrix_path),
        },
    }
    write_json(output_dir / "build_report.json", report)
    return report


def main() -> None:
    args = parse_args()
    tokenizer = _load_tokenizer(args.hf_home)
    report = build(args.output_dir.resolve(), tokenizer)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
