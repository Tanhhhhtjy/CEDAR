"""Load and render category-level persona templates for M2.1."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from cedar_m2.data.harmbench import HARMBENCH_SEMANTIC_CATEGORIES


@dataclass(frozen=True)
class PersonaTemplate:
    name: str
    identity: str


@dataclass(frozen=True)
class PersonaConfig:
    """Validated persona configuration whose source file is fingerprinted."""

    version: int
    status: str
    system_template: str
    priming_user_template: str
    priming_assistant_template: str
    user_template: str
    personas: dict[str, PersonaTemplate]
    source_path: Path
    sha256: str

    def render_messages(self, category: str, behavior: str) -> list[dict[str, str]]:
        """Render one system/user conversation without changing behavior text."""
        try:
            persona = self.personas[category]
        except KeyError as error:
            raise ValueError(f"No persona configured for category {category!r}") from error
        values = {"name": persona.name, "identity": persona.identity}
        return [
            {"role": "system", "content": self.system_template.format(**values)},
            {"role": "user", "content": self.priming_user_template.format(**values)},
            {
                "role": "assistant",
                "content": self.priming_assistant_template.format(**values),
            },
            {
                "role": "user",
                "content": self.user_template.format(behavior=behavior, **values),
            },
        ]

    def require_frozen(self) -> None:
        """Reject use on a sealed split until the template is explicitly frozen."""
        if self.status != "frozen":
            raise PermissionError(
                f"Persona config status is {self.status!r}; HarmBench test requires 'frozen'"
            )


def _require_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return value


def load_persona_config(path: str | Path) -> PersonaConfig:
    """Load a YAML config and bind it to the SHA-256 of its exact bytes."""
    source_path = Path(path)
    raw = source_path.read_bytes()
    document = _require_mapping(yaml.safe_load(raw), "Persona config")
    template = _require_mapping(document.get("template"), "template")
    raw_personas = _require_mapping(document.get("personas"), "personas")

    categories = set(raw_personas)
    expected = set(HARMBENCH_SEMANTIC_CATEGORIES)
    if categories != expected:
        raise ValueError(
            f"Persona categories differ from HarmBench categories; "
            f"missing={sorted(expected - categories)}, extra={sorted(categories - expected)}"
        )

    personas: dict[str, PersonaTemplate] = {}
    for category, value in raw_personas.items():
        record = _require_mapping(value, f"personas.{category}")
        name = str(record.get("name", "")).strip()
        identity = str(record.get("identity", "")).strip()
        if not name or not identity:
            raise ValueError(f"Persona {category!r} requires non-empty name and identity")
        personas[category] = PersonaTemplate(name=name, identity=identity)

    status = str(document.get("status", "")).strip()
    if status not in {"development", "frozen"}:
        raise ValueError("Persona status must be 'development' or 'frozen'")
    system_template = str(template.get("system", "")).strip()
    priming_user_template = str(template.get("priming_user", "")).strip()
    priming_assistant_template = str(template.get("priming_assistant", "")).strip()
    user_template = str(template.get("user", "")).strip()
    required_templates = (
        system_template,
        priming_user_template,
        priming_assistant_template,
        user_template,
    )
    if not all(required_templates) or "{behavior}" not in user_template:
        raise ValueError(
            "Persona templates must be non-empty and the final user template "
            "must contain {behavior}"
        )

    return PersonaConfig(
        version=int(document["version"]),
        status=status,
        system_template=system_template,
        priming_user_template=priming_user_template,
        priming_assistant_template=priming_assistant_template,
        user_template=user_template,
        personas=personas,
        source_path=source_path,
        sha256=hashlib.sha256(raw).hexdigest(),
    )
