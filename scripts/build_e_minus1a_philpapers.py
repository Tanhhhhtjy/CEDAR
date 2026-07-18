"""Build the frozen E-1a PhilPapers manifests from the authors' public protocol.

This is a data-only builder. It does not load Qwen, steering vectors, or a GPU.
The sampling and counterbalancing logic intentionally follows the public
``kelkarI/sycophancy-qwen`` implementation at commit
270889c1a876a2c9ac384a6e4a1273f027e3e349.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

BUILDER_VERSION = "e_minus1a_philpapers_v1"
SOURCE_COMMIT = "84fcc677e52e1902d696c32cd1a6b663e70d3993"
SOURCE_BLOB_SHA1 = "5525210614d4f26b1732042e7dcb7210d23fe5aa"
SOURCE_SHA256 = "2f112b35334fbec0b16dc755df60349fb2b2bf00d4dbaae47175519bee7d37dd"
SOURCE_URL = (
    "https://raw.githubusercontent.com/anthropics/evals/"
    f"{SOURCE_COMMIT}/sycophancy/sycophancy_on_philpapers2020.jsonl"
)

AUTHOR_QWEN_COMMIT = "270889c1a876a2c9ac384a6e4a1273f027e3e349"
AUTHOR_GEMMA_COMMIT = "aa0d78ea9680cb53f61098e86e9f174940562fc7"
AUTHOR_CODE_URL = (
    "https://github.com/kelkarI/sycophancy-qwen/blob/"
    f"{AUTHOR_QWEN_COMMIT}/scripts/00b_rebuild_eval.py"
)
AUTHOR_CODE_SHA256 = {
    "scripts/00b_rebuild_eval.py": (
        "7bf3494ea8079f236344c85c8e56a6fb50b215c771df768a6c3bfd6dcd534f43"
    ),
    "scripts/02_evaluate_steering.py": (
        "4118d9ea5533f5f72fea93c5421cee80ef70b75f97daac4c6c339bc1a448f979"
    ),
    "scripts/run_multiseed.py": (
        "d4eb2db6e9103de055175e84c8bc3ef29333fe15328080de6a3f7421175e5c8a"
    ),
    "scripts/build_vectors_from_official.py": (
        "2e2a0a59bb456b082d0c895e6f1daaa4b969615cffba34dd6c00d35e748cae83"
    ),
    "scripts/config.py": "9e3693289c88cfccc7acf782fa486290c8142c6016b6f66094d88542e88f2d26",
}
AXIS_REPOSITORY = "lu-christina/assistant-axis-vectors"
AXIS_REVISION = "3b3b788432ad33e3a28d9ff08e88a530c0740814"
AXIS_PATH = "qwen-3-32b/assistant_axis.pt"
AXIS_SHA256 = "a207fe7a36563280b7b29010880aa0082bd8e3113c141cb4a2eed6b46c140211"
PUBLIC_AUTHOR_MANIFESTS = (
    {
        "seed": 42,
        "label": "public_gemma_seed42",
        "url": (
            "https://raw.githubusercontent.com/kelkarI/sycophancy-gemma/"
            f"{AUTHOR_GEMMA_COMMIT}/experiment-main/data/eval_data.json"
        ),
    },
    {
        "seed": 42,
        "label": "paper_anonymous_snapshot_seed42",
        "url": (
            "https://anonymous.4open.science/api/repo/"
            "Sycophancy-Steering-9DF0/file/gemma/data/eval_data.json"
        ),
    },
    {
        "seed": 123,
        "label": "public_qwen_seed123",
        "url": (
            "https://raw.githubusercontent.com/kelkarI/sycophancy-qwen/"
            f"{AUTHOR_QWEN_COMMIT}/data/eval_data.json"
        ),
    },
)

SAMPLE_SEEDS = (42, 7, 123)
TUNE_TEST_SEED = 99
N_BASE = 300
EXPECTED_SOURCE_ROWS = 9867
EXPECTED_ELIGIBLE_ROWS = 4026
EXPECTED_AUTHOR_CANONICAL_HASHES = {
    42: "67fe2511f3b4a7e362f49e32680059cb0bba631d0d094120fdf33c7d669944a2",
    7: "2a1eb56667de659309de450a3b394d02ca52ce1b11d015b7515003c521547d87",
    123: "106d01b6006d4235c80aeb715ca1fd05ee3bfd7bfb7a88f2857a7cd1e8ba4c0e",
}

AB_PATTERN = re.compile(
    r"\n \(A\) (.*?)\n \(B\) (.*?)(\n \(C\)|\n\nAnswer:)",
    re.DOTALL,
)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode()
    return hashlib.sha1(header + payload).hexdigest()  # noqa: S324


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    return sha256_bytes(payload)


def fetch(url: str, attempts: int = 4) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "CEDAR-E-1a-builder/1.0"})
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except Exception as error:  # network stacks expose several transient exception types
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def normalize_answer(answer: str | list[str]) -> str:
    if isinstance(answer, list):
        answer = answer[0]
    return answer.strip().strip("()")


def parse_source(payload: bytes) -> tuple[list[dict[str, Any]], int, int]:
    raw = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line]
    parsed: list[dict[str, Any]] = []
    skipped = 0
    for orig_id, row in enumerate(raw):
        syc = normalize_answer(row["answer_matching_behavior"])
        honest = normalize_answer(row["answer_not_matching_behavior"])
        if syc not in {"A", "B"} or honest not in {"A", "B"} or syc == honest:
            skipped += 1
            continue
        match = AB_PATTERN.search(row["question"])
        if match is None or match.group(3).startswith("\n (C)"):
            skipped += 1
            continue
        parsed.append(
            {
                "orig_id": orig_id,
                "preamble": row["question"][: match.start()],
                "a_text": match.group(1),
                "b_text": match.group(2),
                "tail": match.group(3),
                "syc_letter": syc,
            }
        )
    return parsed, len(raw), skipped


def build_author_rows(parsed: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    selected = list(parsed)
    random.Random(seed).shuffle(selected)
    selected = selected[:N_BASE]
    rows: list[dict[str, Any]] = []
    for base_id, item in enumerate(selected):
        preamble = item["preamble"]
        a_text = item["a_text"]
        b_text = item["b_text"]
        tail = item["tail"]
        original_question = f"{preamble}\n (A) {a_text}\n (B) {b_text}{tail}"
        swapped_question = f"{preamble}\n (A) {b_text}\n (B) {a_text}{tail}"
        original_syc = item["syc_letter"]
        swapped_syc = "B" if original_syc == "A" else "A"
        for variant, question, syc in (
            ("original", original_question, original_syc),
            ("swapped", swapped_question, swapped_syc),
        ):
            rows.append(
                {
                    "question_id": len(rows),
                    "base_id": base_id,
                    "orig_id": item["orig_id"],
                    "variant": variant,
                    "question_text": question,
                    "sycophantic_answer": syc,
                    "non_sycophantic_answer": "B" if syc == "A" else "A",
                }
            )
    return rows


def split_base_ids() -> tuple[set[int], set[int]]:
    base_ids = list(range(N_BASE))
    random.Random(TUNE_TEST_SEED).shuffle(base_ids)
    midpoint = int(N_BASE * 0.5)
    return set(base_ids[:midpoint]), set(base_ids[midpoint:])


def enrich_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    tune_ids, test_ids = split_base_ids()
    enriched = []
    for row in rows:
        split = "tune" if row["base_id"] in tune_ids else "test"
        assert row["base_id"] in tune_ids | test_ids
        enriched.append(
            {
                "dataset_version": BUILDER_VERSION,
                "sample_seed": seed,
                "pair_id": f"seed{seed}_base{row['base_id']:03d}",
                "split": split,
                **row,
            }
        )
    return enriched


def validate_seed(author_rows: list[dict[str, Any]], enriched: list[dict[str, Any]]) -> None:
    assert len(author_rows) == 600
    assert len(enriched) == 600
    assert len({row["base_id"] for row in author_rows}) == 300
    assert len({row["orig_id"] for row in author_rows}) == 300
    assert sum(row["sycophantic_answer"] == "A" for row in author_rows) == 300
    assert sum(row["split"] == "tune" for row in enriched) == 300
    assert sum(row["split"] == "test" for row in enriched) == 300
    for base_id in range(N_BASE):
        pair = [row for row in author_rows if row["base_id"] == base_id]
        assert [row["variant"] for row in pair] == ["original", "swapped"]
        assert pair[0]["orig_id"] == pair[1]["orig_id"]
        assert pair[0]["sycophantic_answer"] != pair[1]["sycophantic_answer"]
        assert pair[0]["non_sycophantic_answer"] != pair[1]["non_sycophantic_answer"]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def make_readme() -> str:
    return """# E-1a PhilPapers frozen manifests

This directory is a deterministic reconstruction of the public protocol in
`kelkarI/sycophancy-qwen` at commit
`270889c1a876a2c9ac384a6e4a1273f027e3e349`.

Important: the paper's seeds 42, 7, and 123 select three different sets of 300
base questions. `philpapers600.jsonl` is the seed-42 convenience manifest only.
The full three-seed reproduction must use all three `philpapers600_seed*.jsonl`
files (1,800 rows before the 50/50 tune/test split; 900 test rows).

Each base appears twice with counterbalanced A/B order. The statistical unit is
the base pair, not an individual row. The `split` field is assigned at the base
level with seed 99.

Public author artifacts were found for seed 42 and seed 123 and are checked
object-for-object by the builder. No standalone seed-7 manifest was found in
the paper snapshot, the public Qwen/Gemma repositories, their public branches,
or the clean-results repository. Seed 7 is nevertheless exactly reproducible
from the authors' immutable source blob and public builder code.

Build without loading a model:

```bash
python scripts/build_e_minus1a_philpapers.py
```

Build the deterministic engineering-only 64-row smoke subset from the frozen
combined manifest:

```bash
python scripts/build_e_minus1a_smoke.py
```

The smoke subset uses 32 complete seed-42 tune pairs. It must not be used for
effect estimation, coefficient selection, or post-hoc item selection. The GPU
protocol and release gates are frozen in `docs/E_MINUS1A_PROTOCOL.md`.

See `docs/E_MINUS1A_AUTHOR_SOURCE_AUDIT.md` for provenance and implementation
details. Do not call these files a byte-identical copy of every author runtime
artifact: the CEDAR JSONL rows add `dataset_version`, `sample_seed`, `pair_id`,
and `split` for auditability.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, help="optional local copy of the source JSONL")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "datasets/e_minus1a_philpapers",
    )
    parser.add_argument(
        "--skip-public-author-check",
        action="store_true",
        help="build offline without downloading the public seed-42/123 author manifests",
    )
    parser.add_argument(
        "--public-seed42",
        type=Path,
        help="local copy of the immutable public seed-42 author manifest",
    )
    parser.add_argument(
        "--public-seed123",
        type=Path,
        help="local copy of the immutable public seed-123 author manifest",
    )
    args = parser.parse_args()
    if bool(args.public_seed42) != bool(args.public_seed123):
        parser.error("--public-seed42 and --public-seed123 must be supplied together")
    if args.skip_public_author_check and args.public_seed42:
        parser.error("local public manifests cannot be combined with --skip-public-author-check")

    source_payload = args.source.read_bytes() if args.source else fetch(SOURCE_URL)
    assert sha256_bytes(source_payload) == SOURCE_SHA256
    assert git_blob_sha1(source_payload) == SOURCE_BLOB_SHA1

    parsed, source_rows, skipped = parse_source(source_payload)
    assert source_rows == EXPECTED_SOURCE_ROWS
    assert len(parsed) == EXPECTED_ELIGIBLE_ROWS
    assert skipped == EXPECTED_SOURCE_ROWS - EXPECTED_ELIGIBLE_ROWS

    args.out_dir.mkdir(parents=True, exist_ok=True)
    author_rows_by_seed: dict[int, list[dict[str, Any]]] = {}
    enriched_by_seed: dict[int, list[dict[str, Any]]] = {}
    for seed in SAMPLE_SEEDS:
        author_rows = build_author_rows(parsed, seed)
        assert canonical_hash(author_rows) == EXPECTED_AUTHOR_CANONICAL_HASHES[seed]
        enriched = enrich_rows(author_rows, seed)
        validate_seed(author_rows, enriched)
        author_rows_by_seed[seed] = author_rows
        enriched_by_seed[seed] = enriched

    public_checks = []
    if args.public_seed42 and args.public_seed123:
        local_artifacts = (
            (PUBLIC_AUTHOR_MANIFESTS[0], args.public_seed42),
            (PUBLIC_AUTHOR_MANIFESTS[2], args.public_seed123),
        )
        for artifact, local_path in local_artifacts:
            payload = local_path.read_bytes()
            public_rows = json.loads(payload)
            seed = artifact["seed"]
            matches = public_rows == author_rows_by_seed[seed]
            assert matches
            public_checks.append(
                {
                    **artifact,
                    "verification_input": "local_copy_of_immutable_public_artifact",
                    "rows": len(public_rows),
                    "file_sha256": sha256_bytes(payload),
                    "canonical_sha256": canonical_hash(public_rows),
                    "object_for_object_match": matches,
                }
            )
    elif not args.skip_public_author_check:
        for artifact in PUBLIC_AUTHOR_MANIFESTS:
            payload = fetch(artifact["url"])
            public_rows = json.loads(payload)
            seed = artifact["seed"]
            matches = public_rows == author_rows_by_seed[seed]
            assert matches
            public_checks.append(
                {
                    **artifact,
                    "verification_input": "downloaded_from_public_url",
                    "rows": len(public_rows),
                    "file_sha256": sha256_bytes(payload),
                    "canonical_sha256": canonical_hash(public_rows),
                    "object_for_object_match": matches,
                }
            )

    all_rows = [row for seed in SAMPLE_SEEDS for row in enriched_by_seed[seed]]
    output_files: list[Path] = []
    seed42_path = args.out_dir / "philpapers600.jsonl"
    write_jsonl(seed42_path, enriched_by_seed[42])
    output_files.append(seed42_path)
    for seed in SAMPLE_SEEDS:
        path = args.out_dir / f"philpapers600_seed{seed}.jsonl"
        write_jsonl(path, enriched_by_seed[seed])
        output_files.append(path)
    combined_path = args.out_dir / "philpapers1800_all_seeds.jsonl"
    write_jsonl(combined_path, all_rows)
    output_files.append(combined_path)

    split_manifest = {
        "dataset_version": BUILDER_VERSION,
        "sample_seeds": list(SAMPLE_SEEDS),
        "tune_test_seed": TUNE_TEST_SEED,
        "per_seed": {},
    }
    for seed in SAMPLE_SEEDS:
        rows = enriched_by_seed[seed]
        split_manifest["per_seed"][str(seed)] = {
            split: {
                "base_ids": sorted({row["base_id"] for row in rows if row["split"] == split}),
                "orig_ids": [
                    row["orig_id"]
                    for row in rows
                    if row["split"] == split and row["variant"] == "original"
                ],
                "n_base": len(
                    {row["base_id"] for row in rows if row["split"] == split}
                ),
                "n_rows": sum(row["split"] == split for row in rows),
            }
            for split in ("tune", "test")
        }
    split_path = args.out_dir / "split_manifest.json"
    write_json(split_path, split_manifest)
    output_files.append(split_path)

    audit = {
        "review_type": "public_author_artifact_reconstruction_audit",
        "builder_version": BUILDER_VERSION,
        "author_code": AUTHOR_CODE_URL,
        "public_manifest_checks_skipped": args.skip_public_author_check,
        "public_manifest_checks": public_checks,
        "standalone_public_manifest_found": {"42": True, "7": False, "123": True},
        "author_canonical_hashes": {
            str(seed): canonical_hash(author_rows_by_seed[seed]) for seed in SAMPLE_SEEDS
        },
        "conclusion": (
            "Seeds 42 and 123 match public author artifacts object-for-object. "
            "Seed 7 is an exact deterministic reconstruction from the same public code."
        ),
    }
    audit_path = args.out_dir / "author_manifest_audit.json"
    write_json(audit_path, audit)
    output_files.append(audit_path)

    readme_path = args.out_dir / "README.md"
    readme_path.write_text(make_readme(), encoding="utf-8", newline="\n")
    output_files.append(readme_path)

    source_manifest = {
        "dataset_version": BUILDER_VERSION,
        "status": "frozen_cedar_reconstruction_of_published_protocol",
        "source": {
            "repository": "anthropics/evals",
            "commit": SOURCE_COMMIT,
            "path": "sycophancy/sycophancy_on_philpapers2020.jsonl",
            "url": SOURCE_URL,
            "rows": source_rows,
            "eligible_two_choice_rows": len(parsed),
            "sha256": SOURCE_SHA256,
            "git_blob_sha1": SOURCE_BLOB_SHA1,
        },
        "author_protocol": {
            "qwen_repository": "kelkarI/sycophancy-qwen",
            "qwen_commit": AUTHOR_QWEN_COMMIT,
            "code_url": AUTHOR_CODE_URL,
            "code_file_sha256": AUTHOR_CODE_SHA256,
            "sample_seeds": list(SAMPLE_SEEDS),
            "n_base_per_seed": N_BASE,
            "rows_per_seed": 2 * N_BASE,
            "tune_test_seed": TUNE_TEST_SEED,
            "tune_fraction": 0.5,
        },
        "assistant_axis": {
            "repository": AXIS_REPOSITORY,
            "revision": AXIS_REVISION,
            "path": AXIS_PATH,
            "sha256": AXIS_SHA256,
            "shape": [64, 5120],
            "stored_dtype": "torch.bfloat16",
            "target_layer": 32,
            "layer_32_raw_norm_after_float": 22.655502319335938,
        },
        "builder": {
            "path": "scripts/build_e_minus1a_philpapers.py",
            "sha256": sha256_bytes(Path(__file__).read_bytes()),
        },
        "files": {
            path.name: {"sha256": sha256_bytes(path.read_bytes()), "bytes": path.stat().st_size}
            for path in output_files
        },
    }
    source_path = args.out_dir / "source_manifest.json"
    write_json(source_path, source_manifest)

    print(f"Built {len(all_rows)} rows across seeds {SAMPLE_SEEDS} in {args.out_dir}")
    print("Public author checks:", "skipped" if args.skip_public_author_check else "passed")
    print("Source SHA-256:", SOURCE_SHA256)


if __name__ == "__main__":
    main()
