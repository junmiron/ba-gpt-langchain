"""Path sandbox for anything a model can name.

v1's file tools accepted arbitrary strings (and the terminate agent carried hard-coded absolute paths from
another OS). Here every path is resolved and checked against an allow-list of roots, symlinks included — so a
hallucinated ``../../Windows/system32/x`` is refused instead of written.
"""

from __future__ import annotations

from pathlib import Path

from ..config import AppConfig, get_config


class SandboxError(PermissionError):
    """Raised when a requested path escapes the allowed roots."""


class Sandbox:
    def __init__(self, writable_roots: list[Path], readable_roots: list[Path] | None = None) -> None:
        self.writable_roots = [root.resolve() for root in writable_roots]
        # Order is preserved (not sorted): a relative path resolves against the first matching root, and callers
        # pass their primary directory first on purpose.
        combined: list[Path] = []
        for root in [*self.writable_roots, *[(r.resolve() if hasattr(r, "resolve") else r) for r in (readable_roots or [])]]:
            if root not in combined:
                combined.append(root)
        self.readable_roots = combined

    def _inside(self, path: Path, roots: list[Path]) -> bool:
        try:
            resolved = path.resolve()
        except OSError as exc:  # pragma: no cover - defensive
            raise SandboxError(f"Cannot resolve path: {path}") from exc
        for root in roots:
            try:
                resolved.relative_to(root)
                return True
            except ValueError:
                continue
        return False

    def resolve(self, path: str | Path, *, write: bool = False) -> Path:
        """Anchor *path* inside the allow-list.

        Relative paths are anchored on the first root (the caller's primary directory); for reads, a root where
        the file actually exists wins, so ``notes.md`` is found whether it lives in outputs or the docs folder.
        """
        candidate = Path(str(path)).expanduser()
        roots = self.writable_roots if write else self.readable_roots
        if not candidate.is_absolute():
            candidates = [root / candidate for root in roots]
            if not write:
                existing = [found for found in candidates if found.exists()]
                if existing:
                    candidate = existing[0]
                else:
                    candidate = candidates[0] if candidates else candidate
            else:
                candidate = candidates[0] if candidates else candidate
        if not self._inside(candidate, roots):
            raise SandboxError(
                f"Path '{path}' is outside the allowed directories "
                f"({', '.join(str(r) for r in roots)})."
            )
        return candidate

    def read_text(self, path: str | Path, max_chars: int = 20_000) -> str:
        resolved = self.resolve(path, write=False)
        if not resolved.exists() or not resolved.is_file():
            raise FileNotFoundError(f"No such file: {path}")
        text = resolved.read_text(encoding="utf-8", errors="replace")
        return text if len(text) <= max_chars else text[:max_chars] + "\n…[truncated]"

    def write_text(self, path: str | Path, content: str) -> Path:
        resolved = self.resolve(path, write=True)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return resolved


def get_sandbox(config: AppConfig | None = None) -> Sandbox:
    cfg = config or get_config()
    outputs = (cfg.root / cfg.paths.outputs_dir).resolve()
    data = (cfg.root / cfg.knowledge.docs_dir).resolve()
    uploads = (cfg.root / "uploads").resolve()
    return Sandbox(writable_roots=[outputs, data, uploads], readable_roots=[outputs, data, cfg.root / "docs"])
