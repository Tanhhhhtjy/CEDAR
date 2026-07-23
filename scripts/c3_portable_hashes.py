"""Portable file identities for the C3 SycoBench experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from cedar_m2.checkpoint import file_sha256, object_sha256

TEXT_HASH_MODE = "utf8_lf_canonical_v1"
SEMANTIC_HASH_MODE = "canonical_json_object_v1"
PORTABLE_IDENTITY_KEYS = (
    "hash_mode",
    "git_blob_id",
    "canonical_text_sha256",
    "semantic_hash_mode",
    "semantic_sha256",
)


def canonical_text_sha256(path: str | Path) -> str:
    """Hash UTF-8 text after canonicalizing CRLF and CR newlines to LF."""
    text = Path(path).read_text(encoding="utf-8")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def git_blob_id(path: str | Path, *, root: str | Path) -> str:
    """Compute the filter-aware Git blob ID for a worktree file."""
    repository = Path(root).resolve()
    target = Path(path).resolve()
    relative = target.relative_to(repository).as_posix()
    return subprocess.run(
        ["git", "hash-object", f"--path={relative}", str(target)],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Expected JSON objects in {path}")
    return rows


def semantic_sha256(path: str | Path) -> str | None:
    """Hash parsed JSON/JSONL/CSV content; return None for unstructured text."""
    target = Path(path)
    suffix = target.suffix.lower()
    if suffix == ".jsonl":
        value: object = read_jsonl(target)
    elif suffix == ".json":
        value = json.loads(target.read_text(encoding="utf-8"))
    elif suffix == ".csv":
        with target.open(encoding="utf-8", newline="") as handle:
            value = list(csv.DictReader(handle))
    else:
        return None
    return object_sha256(value)


def file_identity(path: str | Path, *, root: str | Path) -> dict[str, Any]:
    """Return all frozen cross-platform identities required by C3."""
    target = Path(path)
    semantic = semantic_sha256(target)
    return {
        "hash_mode": TEXT_HASH_MODE,
        "git_blob_id": git_blob_id(target, root=root),
        "raw_sha256_diagnostic": file_sha256(target),
        "bytes_diagnostic": target.stat().st_size,
        "canonical_text_sha256": canonical_text_sha256(target),
        "semantic_hash_mode": SEMANTIC_HASH_MODE if semantic is not None else None,
        "semantic_sha256": semantic,
    }


def portable_identity_matches(observed: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Compare hard-gate identities while excluding raw bytes and byte length."""
    return all(observed.get(key) == expected.get(key) for key in PORTABLE_IDENTITY_KEYS)
