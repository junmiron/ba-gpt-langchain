"""FastAPI application: REST commands + SSE event stream + the static web UI."""

from .app import create_app, run

__all__ = ["create_app", "run"]
