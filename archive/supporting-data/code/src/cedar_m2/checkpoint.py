"""Small atomic record store for resumable local experiment runs."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any


def canonical_json(value: object) -> str:
    """Serialize a JSON-compatible value with stable ordering."""
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def object_sha256(value: object) -> str:
    """Hash a JSON-compatible object deterministically."""
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    """Hash a file without loading it wholly into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: str | Path, value: object) -> None:
    """Write JSON through a sibling temporary file and atomic rename."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def initialize_run(run_dir: str | Path, configuration: Mapping[str, Any]) -> dict[str, Any]:
    """Create or validate an immutable run manifest."""
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    normalized = json.loads(canonical_json(configuration))
    fingerprint = object_sha256(normalized)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("fingerprint") != fingerprint
            or manifest.get("configuration") != normalized
        ):
            raise ValueError(
                f"Run directory {root} belongs to a different configuration; choose a new run name"
            )
        return manifest

    manifest = {
        "created_at": datetime.now().astimezone().isoformat(),
        "fingerprint": fingerprint,
        "configuration": normalized,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest


class RecordStore:
    """One atomically written JSON file per deterministic experiment key."""

    def __init__(self, path: str | Path, *, run_fingerprint: str) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.run_fingerprint = run_fingerprint

    @staticmethod
    def record_id(key: Mapping[str, Any]) -> str:
        return object_sha256(key)

    def _path_for(self, key: Mapping[str, Any]) -> Path:
        return self.path / f"{self.record_id(key)}.json"

    def read(self, key: Mapping[str, Any]) -> dict[str, Any] | None:
        path = self._path_for(key)
        if not path.exists():
            return None
        envelope = json.loads(path.read_text(encoding="utf-8"))
        if envelope.get("run_fingerprint") != self.run_fingerprint:
            raise ValueError(f"Record {path} has the wrong run fingerprint")
        if envelope.get("key") != json.loads(canonical_json(key)):
            raise ValueError(f"Record {path} does not match its requested key")
        payload = envelope.get("payload")
        if not isinstance(payload, dict):
            raise ValueError(f"Record {path} has a malformed payload")
        return payload

    def contains(self, key: Mapping[str, Any]) -> bool:
        return self.read(key) is not None

    def write(self, key: Mapping[str, Any], payload: Mapping[str, Any]) -> None:
        envelope = {
            "run_fingerprint": self.run_fingerprint,
            "key": json.loads(canonical_json(key)),
            "payload": json.loads(canonical_json(payload)),
        }
        atomic_write_json(self._path_for(key), envelope)

    def payloads(self) -> Iterable[dict[str, Any]]:
        for path in sorted(self.path.glob("*.json")):
            envelope = json.loads(path.read_text(encoding="utf-8"))
            if envelope.get("run_fingerprint") != self.run_fingerprint:
                raise ValueError(f"Record {path} has the wrong run fingerprint")
            payload = envelope.get("payload")
            if not isinstance(payload, dict):
                raise ValueError(f"Record {path} has a malformed payload")
            yield payload
