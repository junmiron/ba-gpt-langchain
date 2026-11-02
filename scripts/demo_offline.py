"""Run the real web UI with a scripted gateway — no LM Studio, no VRAM.

Useful for reviewing or demoing the interface (timeline, question cards, tabs, SSE replay, downloads) while a
model is busy elsewhere, and for frontend work in general::

    python scripts/demo_offline.py [--port 8765]

Every answer produces plausible scripted content, so the whole pipeline runs to completion in about a second.
Nothing here ships in the package: it lives outside ``src/`` and imports the test double from ``tests/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

import uvicorn  # noqa: E402

from ba_agent.cli import configure_console  # noqa: E402
from ba_agent.config import get_config  # noqa: E402
from ba_agent.knowledge import KnowledgeService  # noqa: E402
from ba_agent.server.app import create_app  # noqa: E402
from ba_agent.services import Services  # noqa: E402
from conftest import FakeGateway  # noqa: E402


def main() -> int:
    configure_console()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    config = get_config()
    real_create = Services.create
    gateway = FakeGateway()

    def create_with_fake(cfg=None):
        services = real_create(cfg or config)
        services.gateway = gateway
        services.knowledge = KnowledgeService(services.store, gateway, config)
        return services

    # The app builds its services inside the lifespan handler, so swap the factory it calls.
    import ba_agent.server.app as app_module

    app_module.Services = SimpleNamespace(create=create_with_fake)  # type: ignore[assignment]

    print("=" * 78)
    print("  OFFLINE DEMO — scripted answers, no model is loaded and LM Studio is never called.")
    print(f"  UI: http://{args.host}:{args.port}   (sessions still persist to state/ba_agent.db)")
    print("=" * 78)

    uvicorn.run(create_app(config), host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
