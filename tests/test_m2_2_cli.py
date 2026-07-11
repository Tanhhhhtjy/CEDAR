from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.run_belief_m2_2 import validate_authorization


def args_for(
    split: str,
    *,
    allow_pilot: bool = False,
    allow_extension: bool = False,
    prompt_sha256: str | None = None,
    authorization_summary: Path | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        split=split,
        allow_pilot=allow_pilot,
        allow_extension=allow_extension,
        prompt_sha256=prompt_sha256,
        authorization_summary=authorization_summary,
    )


class M22AuthorizationTest(unittest.TestCase):
    prompt_hash = "frozen-prompt"

    def _summary(self, root: Path, split: str, statistics: dict[str, object]) -> Path:
        path = root / f"{split}.json"
        path.write_text(
            json.dumps(
                {
                    "split": split,
                    "prompt_sha256": self.prompt_hash,
                    "statistics": statistics,
                }
            ),
            encoding="utf-8",
        )
        return path

    def test_development_rejects_later_stage_flags(self) -> None:
        validate_authorization(args_for("development"), self.prompt_hash)
        with self.assertRaisesRegex(ValueError, "must not be supplied"):
            validate_authorization(
                args_for("development", allow_pilot=True), self.prompt_hash
            )

    def test_pilot_requires_explicit_flag_prompt_and_passing_development(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = self._summary(
                Path(tmp), "development", {"development_gate": {"prompt_valid": True}}
            )
            with self.assertRaisesRegex(PermissionError, "remains sealed"):
                validate_authorization(
                    args_for(
                        "pilot",
                        prompt_sha256=self.prompt_hash,
                        authorization_summary=summary,
                    ),
                    self.prompt_hash,
                )

            validate_authorization(
                args_for(
                    "pilot",
                    allow_pilot=True,
                    prompt_sha256=self.prompt_hash,
                    authorization_summary=summary,
                ),
                self.prompt_hash,
            )

            with self.assertRaisesRegex(ValueError, "does not apply"):
                validate_authorization(
                    args_for(
                        "pilot",
                        allow_pilot=True,
                        allow_extension=True,
                        prompt_sha256=self.prompt_hash,
                        authorization_summary=summary,
                    ),
                    self.prompt_hash,
                )

    def test_extension_requires_pilot_expand_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            allowed = self._summary(
                root, "pilot", {"pilot_gate": {"next_step": "expand_bu"}}
            )
            validate_authorization(
                args_for(
                    "extension",
                    allow_extension=True,
                    prompt_sha256=self.prompt_hash,
                    authorization_summary=allowed,
                ),
                self.prompt_hash,
            )

            blocked = self._summary(
                root, "pilot", {"pilot_gate": {"next_step": "cost_established"}}
            )
            with self.assertRaisesRegex(PermissionError, "requesting BU expansion"):
                validate_authorization(
                    args_for(
                        "extension",
                        allow_extension=True,
                        prompt_sha256=self.prompt_hash,
                        authorization_summary=blocked,
                    ),
                    self.prompt_hash,
                )


if __name__ == "__main__":
    unittest.main()
