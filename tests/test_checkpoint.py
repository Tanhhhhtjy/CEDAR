from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from cedar_m2.checkpoint import RecordStore, initialize_run


class CheckpointTest(unittest.TestCase):
    def test_manifest_is_immutable_and_records_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = initialize_run(root, {"model": "pinned", "seed": 7})
            repeated = initialize_run(root, {"seed": 7, "model": "pinned"})
            self.assertEqual(manifest["fingerprint"], repeated["fingerprint"])

            store = RecordStore(root / "records", run_fingerprint=manifest["fingerprint"])
            key = {"condition": "baseline", "item": "one"}
            self.assertFalse(store.contains(key))
            store.write(key, {"score": 1})
            self.assertTrue(store.contains(key))
            self.assertEqual(store.read(key), {"score": 1})
            self.assertEqual(list(store.payloads()), [{"score": 1}])

            with self.assertRaisesRegex(ValueError, "different configuration"):
                initialize_run(root, {"model": "changed", "seed": 7})

    def test_wrong_run_fingerprint_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = RecordStore(root, run_fingerprint="first")
            key = {"item": 1}
            first.write(key, {"value": 2})
            second = RecordStore(root, run_fingerprint="second")
            with self.assertRaisesRegex(ValueError, "wrong run fingerprint"):
                second.read(key)


if __name__ == "__main__":
    unittest.main()
