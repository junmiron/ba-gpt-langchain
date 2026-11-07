"""Command line interface: ``ba-agent serve | doctor | index | sessions | export``."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from .config import get_config


def configure_console() -> None:
    """Windows consoles default to a legacy codepage (cp1252); reports contain →, — and other
    characters that would raise UnicodeEncodeError. Reconfigure to UTF-8 with replacement so the
    CLI never crashes while printing a report. Also used by scripts/ so their logs stay readable."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except Exception:  # pragma: no cover - already-wrapped or exotic streams
            pass


def main(argv: list[str] | None = None) -> int:
    configure_console()
    parser = argparse.ArgumentParser(prog="ba-agent", description="Business Analyst interview agent (LangChain + LM Studio)")
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="start the web UI and API")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)

    doctor = sub.add_parser("doctor", help="check LM Studio connectivity, models and knowledge base")
    doctor.add_argument("--json", action="store_true")

    index = sub.add_parser("index", help="ingest the knowledge base (documents + URLs)")
    index.add_argument("--dir", default=None)
    index.add_argument("--url", action="append", default=[], dest="urls")

    sessions = sub.add_parser("sessions", help="list stored sessions")
    sessions.add_argument("--limit", type=int, default=20)

    export = sub.add_parser("export", help="re-assemble deliverables for a session without any model call")
    export.add_argument("session_id")

    args = parser.parse_args(argv)
    config = get_config()

    if args.command == "serve":
        from .server.app import run

        print(f"BA agent (LangChain) → http://{args.host or config.server.host}:{args.port or config.server.port}")
        run(config, host=args.host, port=args.port)
        return 0

    if args.command == "doctor":
        report = asyncio.run(_doctor(config))
        print(json.dumps(report, indent=2, ensure_ascii=False) if args.json else _format_doctor(report))
        return 0 if report["backend"]["reachable"] else 1

    if args.command == "index":
        from .services import Services

        services = Services.create(config)
        result: dict = {}
        if args.dir or not args.urls:
            result["directory"] = asyncio.run(services.knowledge.index_directory(args.dir))
        if args.urls:
            result["urls"] = asyncio.run(services.knowledge.index_urls(args.urls))
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    if args.command == "sessions":
        from .services import Services

        services = Services.create(config)
        for row in services.store.list_sessions(limit=args.limit):
            print(f"{row['id']}  {row['phase']:<10} {row['status']:<13} {row['updated_at']}  {row['title'][:60]}")
        return 0

    if args.command == "export":
        from .export import write_deliverables
        from .services import Services

        services = Services.create(config)
        state = services.store.load_state(args.session_id)
        if state is None:
            print(f"unknown session {args.session_id}", file=sys.stderr)
            return 2
        written = write_deliverables(services, state)
        for item in written:
            print(f"{item.bytes:>9} bytes  {item.path}")
        return 0

    return 1


async def _doctor(config) -> dict:
    from .services import Services

    services = Services.create(config)
    health = await services.gateway.health()
    knowledge = services.knowledge.status()
    models = [m for m in health["chat_models"]]
    loaded = [m["id"] for m in models if m.get("state") == "loaded"]
    tool_capable = [m["id"] for m in models if "tool_use" in (m.get("capabilities") or [])]
    return {
        "backend": {"provider": config.provider, "base_url": config.active_profile.base_url, "reachable": health["reachable"],
                     "chat_models": len(models), "loaded": loaded, "tool_capable": tool_capable,
                     "default_model": health.get("default_model")},
        "embeddings": {"model": knowledge["embedding_model"], "available": bool(health["embedding_models"]),
                       "vector_search_enabled": knowledge["vector_search"]},
        "knowledge": {"documents": len(knowledge["documents"]), "chunks": knowledge["chunks"],
                      "bm25": knowledge["bm25"]},
        "paths": {"project_root": str(config.root), "db": str(config.db_path),
                  "outputs": str(config.root / config.paths.outputs_dir)},
    }


def _format_doctor(report: dict) -> str:
    lm = report["backend"]
    lines = [
        f"Backend        : {'reachable' if lm['reachable'] else 'NOT REACHABLE'} ({lm['provider']}) at {lm['base_url']}",
        f"Chat models    : {lm['chat_models']} available, loaded: {', '.join(lm['loaded']) or 'none'}",
        f"Tool calling   : {', '.join(lm['tool_capable']) or 'no model reports tool_use (fine — the workflow does not need it)'}",
        f"Default model  : {lm['default_model'] or '(unset — first loaded chat model will be used)'}",
        f"Embeddings     : {report['embeddings']['model']} "
        f"({'available' if report['embeddings']['available'] else 'not served by the backend'}) → "
        f"vector search {'on' if report['embeddings']['vector_search_enabled'] else 'off (BM25 only)'}",
        f"Knowledge base : {report['knowledge']['documents']} document(s), {report['knowledge']['chunks']} chunk(s)",
        f"Database       : {report['paths']['db']}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
