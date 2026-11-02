"""Prompt templates live in ``prompts/*.md.j2`` (Jinja2) so they are editable without touching code.

This fixes the v1 duplication problem: prompts existed both in ``agents_config.yaml`` and inline in
``app2.py``, already out of sync. Here there is exactly one file per model step, versioned with the code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from .config import get_config


@lru_cache(maxsize=1)
def _env() -> Environment:
    prompts_dir = _prompts_dir()
    return Environment(
        loader=FileSystemLoader(str(prompts_dir)),
        undefined=StrictUndefined,
        autoescape=select_autoescape(default_for_string=False, default=False),
        trim_blocks=True,
        lstrip_blocks=True,
        keep_trailing_newline=False,
    )


def _prompts_dir() -> Path:
    cfg = get_config()
    directory = Path(cfg.paths.prompts_dir)
    if not directory.is_absolute():
        directory = cfg.root / directory
    if not directory.exists():  # fall back to templates shipped inside the package
        directory = Path(__file__).resolve().parent / "prompts"
    return directory


def render(name: str, **context) -> tuple[str, dict[str, str]]:
    """Render ``<name>`` into ``(system_prompt, user_prompt)``.

    Templates are split with a literal line ``---USER---`` so both halves stay in one readable file.
    """
    template = _env().get_template(f"{name}.md.j2")
    text = template.render(**context).strip()
    if "---USER---" in text:
        system_part, _, user_part = text.partition("---USER---")
        return system_part.strip(), user_part.strip()
    return "You are a helpful assistant.", text.strip()


def available_prompts() -> list[str]:
    return sorted(p.stem.replace(".md", "") for p in _prompts_dir().glob("*.md.j2"))
