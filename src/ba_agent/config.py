"""Configuration: ``config.yaml`` + environment overrides, validated with pydantic.

Precedence (lowest → highest): dataclass defaults → ``config.yaml`` → ``.env`` → real env vars.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

log = logging.getLogger(__name__)


def configure_logging(config: AppConfig | None = None, level: str = "INFO") -> None:
    """Put the library loggers on one handler: console for a human watching, a file for post-mortems.

    Nothing else in the project calls `basicConfig`, so without this every `log.warning` goes to
    stderr with no timestamp and no trace afterwards — which is exactly what you need when a run
    failed an hour ago and you are trying to work out why.
    """
    config = config or get_config()
    root = logging.getLogger()
    if getattr(root, "_ba_agent_configured", False):
        return

    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")

    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(formatter)
    root.addHandler(console)

    log_file = config._abs(config.paths.logs_dir) / "ba_agent.log"
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(log_file, maxBytes=2_000_000, backupCount=3)
    except Exception:  # noqa: BLE001 - an unwritable log dir must not stop the run
        file_handler = None
    if file_handler is not None:
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    root._ba_agent_configured = True  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- paths


def _find_project_root() -> Path:
    """Project root = the folder holding ``pyproject.toml``/``config.yaml`` above this file."""
    here = Path(__file__).resolve()
    for candidate in [here.parent.parent.parent, *here.parents]:
        if (candidate / "config.yaml").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


PROJECT_ROOT = _find_project_root()


def load_dotenv(path: Path) -> None:
    """Minimal ``.env`` loader (no dependency); never overrides real environment variables."""
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# --------------------------------------------------------------------------- sections


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765
    # Comma separated list, or ["*"]. The app is local-first; keep CORS tight by default.
    cors_origins: list[str] = Field(default_factory=lambda: ["http://127.0.0.1:8765", "http://localhost:8765"])
    #: Optional bearer token. Empty (the default) means no auth — correct for a localhost-only tool.
    #: Set it (or BA_AGENT_TOKEN) the moment the server is reachable from anything other than your own machine.
    auth_token: str = ""

    @field_validator("auth_token")
    @classmethod
    def _reject_weak_token(cls, value: str) -> str:
        value = value.strip()
        if value and len(value) < 16:
            raise ValueError("server.auth_token must be at least 16 characters, or empty to disable auth")
        return value


class LMStudioConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    base_url: str = "http://127.0.0.1:1234/v1"
    admin_url: str = "http://127.0.0.1:1234/api/v0"
    api_key: str = "lm-studio"
    request_timeout_s: float = 600.0
    connect_timeout_s: float = 10.0
    max_concurrency: int = 1
    default_model: str | None = None
    embedding_model: str | None = "text-embedding-nomic-embed-text-v1.5"
    # auto | json_schema | function_calling | prompt
    structured_output_method: str = "auto"
    #: Explicit rung order for this profile, e.g. ["json_schema", "prompt"]. Empty = probe the model.
    structured_output_rungs: list[str] = Field(default_factory=list)
    max_retries: int = 2
    #: A batched tool call on a local Qwen gets truncated mid-payload, so the AutoGen reference
    # implementation turns parallel tool calls off on both local lanes.
    parallel_tool_calls: bool = False

    @field_validator("structured_output_rungs")
    @classmethod
    def _normalise_rungs(cls, value: list[str]) -> list[str]:
        allowed = {"json_schema", "function_calling", "prompt"}
        return [rung.strip().lower() for rung in value if rung.strip().lower() in allowed]

    @field_validator("base_url", "admin_url")
    @classmethod
    def _strip_slash(cls, v: str) -> str:
        return v.rstrip("/")


class StrataConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    base_url: str = "http://127.0.0.1:47831/v1"  # local Strata server (no key required)
    api_key: str = "strata"  # placeholder; never a real credential
    # Read timeout per chunk, not total wall time. Kept tight because a hung generation costs this
    # whole value for every attempt the ladder makes.
    request_timeout_s: float = 120.0
    connect_timeout_s: float = 10.0
    max_concurrency: int = 1
    default_model: str | None = "q2_0"
    embedding_model: str | None = None  # no embeddings served -> BM25-only retrieval
    # auto | json_schema | function_calling | prompt
    structured_output_method: str = "auto"
    #: Explicit rung order for this profile, e.g. ["json_schema", "prompt"]. Empty = probe the model.
    structured_output_rungs: list[str] = Field(default_factory=list)
    max_retries: int = 2
    #: Local Qwen lanes spend most of the budget on reasoning before the first tool call; letting the
    # backend batch tool calls in parallel truncates the payload mid-call and the tool never runs.
    # The AutoGen reference implementation sets this to false for both local lanes.
    parallel_tool_calls: bool = False

    @field_validator("structured_output_rungs")
    @classmethod
    def _normalise_rungs(cls, value: list[str]) -> list[str]:
        allowed = {"json_schema", "function_calling", "prompt"}
        return [rung.strip().lower() for rung in value if rung.strip().lower() in allowed]

    @field_validator("base_url")
    @classmethod
    def _strip_slash(cls, v: str) -> str:
        return v.rstrip("/")


class RoleParams(BaseModel):
    model: str | None = None
    temperature: float = 0.4
    max_output_tokens: int | None = None
    # Extra context tokens this role may use on top of the shared ledger digest.
    context_budget: int = 3_000


# Writer caps are generous on purpose: thinking models (e.g. Qwen3 via LM Studio) spend thousands of tokens
# on internal reasoning before any visible output, so a small cap starves them into an empty section.
# If a capped call still returns nothing visible the gateway retries once without a cap (see models.py).
def _default_roles() -> dict[str, RoleParams]:
    return {
        "plan": RoleParams(temperature=0.2),
        "interview": RoleParams(temperature=0.6),
        "triage": RoleParams(temperature=0.1),
        "summary": RoleParams(temperature=0.3),
        "writer": RoleParams(temperature=0.4, max_output_tokens=16384),
        "use_cases": RoleParams(temperature=0.5, max_output_tokens=16384),
        "glossary": RoleParams(temperature=0.3),
        "critic": RoleParams(temperature=0.1),
        "diagrammer": RoleParams(temperature=0.3),
        "mermaid_reviewer": RoleParams(temperature=0.1),
        "human": RoleParams(temperature=0.2),
    }


class InterviewConfig(BaseModel):
    max_questions_per_subject: int = 2
    min_subjects: int = 4
    max_subjects: int = 10
    max_gap_rounds: int = 2
    max_gap_questions_per_round: int = 3
    # Ceiling on "request changes" at the approval gate, so summary ⇄ approval cannot ping-pong forever.
    max_summary_rejections: int = 2
    # Hard ceiling on interview questions so a run can never stall.
    max_total_questions: int = 40
    # Transcript ingestion (Phase 3): how much of an uploaded transcript the engine will triage, and how
    # much evidence a subject needs before the transcript counts as having answered it. Evidence is counted
    # as (excerpt x distinct subject term) pairs, so a single rich chunk can cover a subject.
    max_transcript_chunks: int = 24
    transcript_chunk_chars: int = 1200
    subject_coverage_threshold: int = 2


class ReviewConfig(BaseModel):
    max_review_rounds: int = 2
    min_functional_requirements: int = 5
    require_acceptance_criteria: bool = True


class DiagramConfig(BaseModel):
    max_revision_rounds: int = 1
    max_diagrams: int = 3
    # Optional external renderer/parser (e.g. https://kroki.io) used as an extra syntax check and to
    # produce an SVG for the deliverable bundle. OFF by default: it would send your specification to a
    # third party. Rendering failures never block the pipeline — the browser renders mermaid locally.
    validator_url: str | None = None
    #: Render SVG/PNG for the PDF deliverable with the local mermaid-cli (`mmdc`) instead of a remote
    #: endpoint. Requires Node + mermaid-cli on PATH; silently skipped when absent.
    local_renderer: bool = True
    render_timeout_seconds: float = 120.0
    #: Also emit a raster PNG next to each SVG. Costs a second headless-browser render per diagram, so it
    #: is off unless a stakeholder actually wants the bitmap.
    emit_png: bool = False


class KnowledgeConfig(BaseModel):
    docs_dir: str = "data"
    #: A chunk_size of 0 (or a negative one) makes the packer append empty blocks forever, so the bounds
    #: are enforced here rather than relying on the UI's BOUNDS table — config.yaml and env are also input.
    chunk_size: int = Field(default=900, ge=100, le=8000)
    chunk_overlap: int = Field(default=140, ge=0, le=2000)
    #: Chunk on markdown headings (langchain-text-splitters). Falls back to paragraph packing if the
    #: optional dependency is missing.
    heading_aware: bool = True
    top_k: int = 6
    min_score: float = 0.05
    embed_batch_size: int = 16


class PathsConfig(BaseModel):
    state_dir: str = "state"
    outputs_dir: str = "outputs"
    logs_dir: str = "logs"
    prompts_dir: str = "prompts"


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    # Which backend the agents talk to: lmstudio | strata.
    # Strata is the shipped default because it is the lane the AutoGen version runs on (q2_0 on the 5090),
    # so an out-of-the-box run here exercises the same model as the reference implementation.
    provider: str = "strata"
    lmstudio: LMStudioConfig = Field(default_factory=LMStudioConfig)
    strata: StrataConfig = Field(default_factory=StrataConfig)
    roles: dict[str, RoleParams] = Field(default_factory=_default_roles)
    interview: InterviewConfig = Field(default_factory=InterviewConfig)
    review: ReviewConfig = Field(default_factory=ReviewConfig)
    diagram: DiagramConfig = Field(default_factory=DiagramConfig)
    knowledge: KnowledgeConfig = Field(default_factory=KnowledgeConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    root: Path = PROJECT_ROOT

    # -- helpers -------------------------------------------------------------

    @field_validator("provider")
    @classmethod
    def _known_provider(cls, v: str) -> str:
        v = (v or "strata").strip().lower()
        if v not in ("lmstudio", "strata"):
            raise ValueError(f"provider must be 'lmstudio' or 'strata', got {v!r}")
        return v

    @property
    def active_profile(self) -> LMStudioConfig | StrataConfig:
        """The model profile the gateway actually uses."""
        return self.strata if self.provider == "strata" else self.lmstudio

    def role(self, name: str) -> RoleParams:
        return self.roles.get(name) or RoleParams()

    @property
    def db_path(self) -> Path:
        return self._abs(self.paths.state_dir) / "ba_agent.db"

    def _abs(self, value: str) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else (self.root / p)

    @property
    def outputs_root(self) -> Path:
        return self._abs(self.paths.outputs_dir)

    def session_dir(self, session_id: str) -> Path:
        """Where a session's artefacts live. Does **not** create the directory."""
        return self.outputs_root / session_id

    def out_dir(self, session_id: str) -> Path:
        d = self.session_dir(session_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def ensure_dirs(self) -> None:
        for value in (
            self.paths.state_dir,
            self.paths.outputs_dir,
            self.paths.logs_dir,
            self.knowledge.docs_dir,
            "uploads",
        ):
            self._abs(value).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- loading

_ENV_MAP: dict[str, tuple[str | None, str, Callable[[str], Any]]] = {
    "BA_HOST": ("server", "host", str),
    "BA_PORT": ("server", "port", int),
    "BA_CORS_ORIGINS": ("server", "cors_origins", lambda v: [x.strip() for x in v.split(",") if x.strip()]),
    "LMSTUDIO_BASE_URL": ("lmstudio", "base_url", str),
    "LMSTUDIO_ADMIN_URL": ("lmstudio", "admin_url", str),
    "LMSTUDIO_API_KEY": ("lmstudio", "api_key", str),
    "BA_MODEL": ("lmstudio", "default_model", str),
    "BA_EMBEDDING_MODEL": ("lmstudio", "embedding_model", str),
    "BA_STRUCTURED_METHOD": ("lmstudio", "structured_output_method", str),
    "BA_STRUCTURED_RUNGS": (
        "lmstudio",
        "structured_output_rungs",
        lambda value: [rung.strip() for rung in value.split(",") if rung.strip()],
    ),
    "BA_MAX_CONCURRENCY": ("lmstudio", "max_concurrency", int),
    "BA_REQUEST_TIMEOUT_S": ("lmstudio", "request_timeout_s", float),
    "BA_PROVIDER": (None, "provider", str),
    "STRATA_BASE_URL": ("strata", "base_url", str),
    "STRATA_MODEL": ("strata", "default_model", str),
    "STRATA_STRUCTURED_RUNGS": (
        "strata",
        "structured_output_rungs",
        lambda value: [rung.strip() for rung in value.split(",") if rung.strip()],
    ),
    "BA_MAX_QUESTIONS_PER_SUBJECT": ("interview", "max_questions_per_subject", int),
    "BA_MAX_GAP_ROUNDS": ("interview", "max_gap_rounds", int),
    "BA_MAX_SUMMARY_REJECTIONS": ("interview", "max_summary_rejections", int),
    "BA_MAX_REVIEW_ROUNDS": ("review", "max_review_rounds", int),
    "BA_KROKI_URL": ("diagram", "validator_url", str),
    "BA_AGENT_TOKEN": ("server", "auth_token", str),
}


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load configuration from YAML + env. Missing file is fine (defaults are usable)."""
    load_dotenv(PROJECT_ROOT / ".env")

    raw: dict[str, Any] = {}
    cfg_path = Path(path) if path else Path(os.environ.get("BA_CONFIG", PROJECT_ROOT / "config.yaml"))
    if cfg_path.exists():
        loaded = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{cfg_path} must contain a YAML mapping")
        raw = loaded

    raw.pop("root", None)
    # Merge role defaults with whatever the file provides.
    roles_raw = raw.get("roles") or {}
    merged_roles = _default_roles()
    for name, params in roles_raw.items():
        base = merged_roles.get(name, RoleParams())
        if isinstance(params, dict):
            merged_roles[name] = base.model_copy(update=params)
        elif params is None:
            merged_roles[name] = base
        else:  # `writer: qwen3-32b` shorthand → just a model override
            merged_roles[name] = base.model_copy(update={"model": str(params)})
    raw["roles"] = merged_roles

    cfg = AppConfig.model_validate(raw)
    if cfg.root != PROJECT_ROOT:  # keep resolved root stable when config lives elsewhere
        cfg.root = PROJECT_ROOT

    for env_key, (section, field_name, caster) in _ENV_MAP.items():
        value = os.environ.get(env_key)
        if value in (None, ""):
            continue
        try:
            cast_value = caster(value)
        except (TypeError, ValueError) as exc:
            # A typo in an environment variable silently falling back to config.yaml is the hardest
            # kind of misconfiguration to spot, so name the key that was ignored.
            log.warning(
                "ignoring %s=%r for %s.%s: %s",
                env_key,
                value,
                section or "config",
                field_name,
                exc,
            )
            continue
        section_obj = cfg if section is None else getattr(cfg, section)
        setattr(section_obj, field_name, cast_value)

    cfg.ensure_dirs()
    return cfg


_config: AppConfig | None = None


def get_config(reload: bool = False) -> AppConfig:
    global _config
    if _config is None or reload:
        _config = load_config()
    return _config
