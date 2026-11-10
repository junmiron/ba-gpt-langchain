"""Audit the configuration surface: nothing declared may be unread, and config.yaml may not drift.

Two failure modes this catches, both of which have bitten this project:

1. **Dead config** — a field declared on AppConfig that nothing ever reads. It looks like a knob the
   user can tune; in fact editing it does nothing. (``subject_coverage_threshold`` shipped this way.)
2. **Drift** — config.yaml carries a value that differs from the AppConfig default, or omits a field
   the code defaults elsewhere. Whichever file a reader trusts, they get the wrong number.

Run it as a gate::

    uv run python scripts/config_audit.py

Exit code 1 on any finding, so CI can fail on it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .config import PROJECT_ROOT, AppConfig


def config_sections() -> dict[str, list[str]]:
    """section name -> its field names, walking one level into nested pydantic models."""
    sections: dict[str, list[str]] = {}
    for name, model in AppConfig.model_fields.items():
        fields = getattr(model.annotation, "model_fields", None)
        if fields:
            sections[name] = list(fields)
    return sections


def dead_config_fields() -> list[str]:
    """Fields declared on AppConfig that no source file anywhere in src/ ever mentions.

    config.py is part of the haystack on purpose: a field read by ``AppConfig.db_path()`` is used even
    though no role or server module touches it. A declaration line such as ``state_dir: str = "state"``
    matches none of the patterns below, so including config.py cannot mask a genuinely dead field.
    """
    src = Path(PROJECT_ROOT) / "src"
    haystack = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in src.rglob("*.py"))

    dead: list[str] = []
    for section, fields in config_sections().items():
        for field in fields:
            # `.field` (a read), `"field"` (a key in a SETTABLE tuple or a dict), `field=` (a kwarg).
            if not any(pattern in haystack for pattern in (f".{field}", f'"{field}"', f"{field}=")):
                dead.append(f"{section}.{field}")
    return dead


def config_drift() -> list[str]:
    """Differences between config.yaml and the AppConfig defaults it is meant to mirror."""
    path = Path(PROJECT_ROOT) / "config.yaml"
    if not path.exists():
        return ["config.yaml is missing"]

    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config = AppConfig()
    drift: list[str] = []

    for section, values in raw.items():
        # A top-level scalar (e.g. `provider: strata`) is a value, not a section, and AppConfig's
        # field of the same name is a plain str — there is nothing nested to compare here.
        if not isinstance(values, dict):
            if section in AppConfig.model_fields and getattr(config, section, None) != values:
                drift.append(f"{section}: yaml={values!r} default={getattr(config, section, None)!r}")
            elif section not in AppConfig.model_fields:
                drift.append(f"config.yaml has a '{section}' key with no matching AppConfig field")
            continue

        model = getattr(config, section, None)
        if model is None:
            drift.append(f"config.yaml has a '{section}' section with no matching AppConfig field")
            continue
        if not isinstance(values, dict):
            drift.append(f"config.yaml section '{section}' is not a mapping")
            continue
        fields = getattr(type(model), "model_fields", None)
        if not fields:
            continue

        for key, value in values.items():
            if key not in fields:
                drift.append(f"{section}.{key}: present in config.yaml, absent from AppConfig")
            elif getattr(model, key) != value:
                drift.append(f"{section}.{key}: yaml={value!r} default={getattr(model, key)!r}")

        for key, field in fields.items():
            if key in values:
                continue
            # A default_factory means the value is computed or deliberately empty, so leaving it out of
            # the YAML is not drift.
            if field.default_factory is not None:
                continue
            drift.append(f"{section}.{key}: AppConfig default {field.default!r} not written to config.yaml")

    for section in config_sections():
        if section not in raw:
            drift.append(f"AppConfig section '{section}' is absent from config.yaml")

    return drift


def report() -> dict[str, Any]:
    total = sum(len(v) for v in config_sections().values())
    dead = dead_config_fields()
    drift = config_drift()
    return {"declared_fields": total, "dead": dead, "drift": drift, "ok": not dead and not drift}


def main() -> int:
    result = report()
    print(f"config surface: {result['declared_fields']} declared fields")
    print(f"dead config: {len(result['dead'])}")
    for item in result["dead"]:
        print("  ", item)
    print(f"config.yaml drift: {len(result['drift'])}")
    for item in result["drift"]:
        print("  ", item)
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
