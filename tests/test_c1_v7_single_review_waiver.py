from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_c1_v7_single_review_waiver.py"
DATA_DIR = ROOT / "datasets" / "m2_2b_formal" / "v7_heldout_confirmation"


def load_module():
    spec = importlib.util.spec_from_file_location("c1_v7_single_review_waiver", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_review_waiver_is_explicit_and_hash_bound() -> None:
    report = load_module().validate(DATA_DIR)
    assert report["status"] == "passed"
    assert report["failures"] == []
    assert report["checks"]["reviewer_count_one"] is True
    assert report["checks"]["second_reviewer_false"] is True
    assert report["checks"]["item_labels_false"] is True
    assert report["checks"]["base_hash_bound"] is True
    assert report["checks"]["arms_hash_bound"] is True

