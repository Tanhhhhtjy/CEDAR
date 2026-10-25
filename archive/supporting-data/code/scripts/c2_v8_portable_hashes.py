"""Portable text identities used by the C2/V8 execution hard gates."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

TEXT_HASH_MODE = "utf8_lf_canonical_v1"


def canonical_text_sha256(path: str | Path) -> str:
    """Hash UTF-8 text after canonicalizing CRLF/CR newlines to LF."""
    text = Path(path).read_text(encoding="utf-8")
    canonical = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def git_blob_id(path: str | Path, *, root: str | Path) -> str:
    """Compute the filter-aware Git blob ID for a tracked-text candidate."""
    repository = Path(root)
    target = Path(path)
    relative = target.relative_to(repository).as_posix()
    return subprocess.run(
        ["git", "hash-object", f"--path={relative}", str(target)],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
