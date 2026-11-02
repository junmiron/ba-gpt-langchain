"""SQLite persistence — the reason a run survives a crash, a restart or a closed browser tab.

Everything the workflow needs to resume lives here: ``sessions.state_json`` (phase machine),
``qa_pairs``, ``ledger`` is embedded in state (it is small and always needed together with it),
``events`` (append-only, doubles as the SSE replay log) and ``artifacts``. The knowledge base has its
own tables so re-indexing can be incremental and deduplicated by content hash.

Writes are short (sub-millisecond) and serialised with a lock; heavy work never happens here.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from .schemas import ArtifactRef, LedgerItem, QAPair, SessionState

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT '',
    brief       TEXT NOT NULL DEFAULT '',
    phase       TEXT NOT NULL DEFAULT 'intake',
    status      TEXT NOT NULL DEFAULT 'idle',
    awaiting    TEXT NOT NULL DEFAULT 'none',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    state_json  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS qa_pairs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    q_index     INTEGER NOT NULL,
    subject     TEXT NOT NULL DEFAULT '',
    question    TEXT NOT NULL,
    answer      TEXT,
    skipped     INTEGER NOT NULL DEFAULT 0,
    na          INTEGER NOT NULL DEFAULT 0,
    rationale   TEXT NOT NULL DEFAULT '',
    round       INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_qa_session ON qa_pairs(session_id, q_index);

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    ts          TEXT NOT NULL,
    type        TEXT NOT NULL,
    role        TEXT NOT NULL DEFAULT '',
    step        TEXT NOT NULL DEFAULT '',
    payload     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, id);

CREATE TABLE IF NOT EXISTS artifacts (
    name        TEXT NOT NULL,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    kind        TEXT NOT NULL DEFAULT 'file',
    path        TEXT NOT NULL DEFAULT '',
    bytes       INTEGER NOT NULL DEFAULT 0,
    sha256      TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    PRIMARY KEY (session_id, name)
);

CREATE TABLE IF NOT EXISTS kb_docs (
    id          TEXT PRIMARY KEY,
    source      TEXT NOT NULL,
    uri         TEXT NOT NULL DEFAULT '',
    sha256      TEXT NOT NULL,
    title       TEXT NOT NULL DEFAULT '',
    bytes       INTEGER NOT NULL DEFAULT 0,
    chunks      INTEGER NOT NULL DEFAULT 0,
    embedded    INTEGER NOT NULL DEFAULT 0,
    indexed_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS kb_chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id      TEXT NOT NULL REFERENCES kb_docs(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    meta        TEXT NOT NULL DEFAULT '{}',
    embedding   BLOB,
    dim         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON kb_chunks(doc_id);

CREATE TABLE IF NOT EXISTS settings (
    key         TEXT PRIMARY KEY,
    value       TEXT NOT NULL
);
"""


def new_session_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"ba-{stamp}-{uuid.uuid4().hex[:6]}"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


class Store:
    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- sessions -----------------------------------------------------------

    def create_session(self, brief: str, title: str = "", model_overrides: dict[str, str] | None = None) -> SessionState:
        state = SessionState(
            session_id=new_session_id(),
            title=title or (brief.strip().splitlines()[0][:80] if brief.strip() else "Untitled interview"),
            brief=brief.strip(),
            model_overrides=dict(model_overrides or {}),
        )
        with self._lock:
            self._conn.execute(
                "INSERT INTO sessions (id,title,brief,phase,status,awaiting,created_at,updated_at,state_json)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (state.session_id, state.title, state.brief, state.phase.value, state.status.value,
                 state.awaiting.value, state.created_at, state.updated_at, state.model_dump_json()),
            )
            self._conn.commit()
        return state

    def load_state(self, session_id: str) -> SessionState | None:
        with self._lock:
            row = self._conn.execute("SELECT state_json FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if not row:
            return None
        try:
            return SessionState.model_validate(json.loads(row["state_json"]))
        except Exception:
            return None

    def save_state(self, state: SessionState) -> None:
        state.touch()
        payload = state.model_dump_json()
        with self._lock:
            self._conn.execute(
                "UPDATE sessions SET title=?, brief=?, phase=?, status=?, awaiting=?, updated_at=?, state_json=?"
                " WHERE id=?",
                (state.title, state.brief, state.phase.value, state.status.value, state.awaiting.value,
                 state.updated_at, payload, state.session_id),
            )
            self._conn.commit()

    def list_sessions(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id,title,brief,phase,status,awaiting,created_at,updated_at FROM sessions"
                " ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            cur = self._conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            self._conn.commit()
        return cur.rowcount > 0

    def rename_session(self, session_id: str, title: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE sessions SET title=? WHERE id=?", (title, session_id))
            self._conn.commit()

    # -- Q&A ----------------------------------------------------------------

    def add_qa(self, pair: QAPair) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO qa_pairs (session_id,q_index,subject,question,answer,skipped,na,rationale,round,created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (pair.session_id, pair.q_index, pair.subject, pair.question, pair.answer,
                 int(pair.skipped), int(pair.not_applicable), pair.rationale, pair.round, pair.created_at),
            )
            self._conn.commit()

    def update_qa_answer(self, session_id: str, q_index: int, answer: str | None,
                         skipped: bool = False, not_applicable: bool = False) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE qa_pairs SET answer=?, skipped=?, na=? WHERE session_id=? AND q_index=?",
                (answer, int(skipped), int(not_applicable), session_id, q_index),
            )
            self._conn.commit()

    def list_qa(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM qa_pairs WHERE session_id=? ORDER BY q_index", (session_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    # -- events -------------------------------------------------------------

    def log_event(self, session_id: str, event_type: str, payload: dict[str, Any] | None = None, *,
                  role: str = "", step: str = "") -> int:
        ts = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO events (session_id,ts,type,role,step,payload) VALUES (?,?,?,?,?,?)",
                (session_id, ts, event_type, role, step, json.dumps(payload or {}, ensure_ascii=False, default=str)),
            )
            self._conn.commit()
        return int(cur.lastrowid or 0)

    def get_events(self, session_id: str, after_id: int = 0, limit: int = 2000) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events WHERE session_id=? AND id>? ORDER BY id LIMIT ?",
                (session_id, after_id, limit),
            ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            try:
                item["payload"] = json.loads(item.get("payload") or "{}")
            except Exception:
                item["payload"] = {}
            out.append(item)
        return out

    def trim_events(self, session_id: str, keep: int = 5000) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM events WHERE session_id=? AND id NOT IN"
                " (SELECT id FROM events WHERE session_id=? ORDER BY id DESC LIMIT ?)",
                (session_id, session_id, keep),
            )
            self._conn.commit()

    # -- artifacts ----------------------------------------------------------

    def add_artifact(self, session_id: str, artifact: ArtifactRef) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO artifacts (session_id,name,kind,path,bytes,sha256,created_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (session_id, artifact.name, artifact.kind, artifact.path, artifact.bytes,
                 artifact.sha256, artifact.created_at),
            )
            self._conn.commit()

    def list_artifacts(self, session_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM artifacts WHERE session_id=? ORDER BY name", (session_id,)
            ).fetchall()
        return [dict(row) for row in rows]

    # -- knowledge base -----------------------------------------------------

    def kb_hashes(self) -> dict[str, str]:
        """``source key → sha256`` used to decide what needs (re-)indexing."""
        with self._lock:
            rows = self._conn.execute("SELECT source, uri, sha256 FROM kb_docs").fetchall()
        return {f"{row['source']}::{row['uri']}": row["sha256"] for row in rows}

    def kb_upsert_doc(self, *, doc_id: str, source: str, uri: str, sha256: str, title: str,
                      size_bytes: int, chunks: int, embedded: bool) -> None:
        ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kb_docs (id,source,uri,sha256,title,bytes,chunks,embedded,indexed_at)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (doc_id, source, uri, sha256, title, size_bytes, chunks, int(embedded), ts),
            )
            self._conn.commit()

    def kb_replace_chunks(self, doc_id: str, chunks: Sequence[tuple[str, dict[str, Any], list[float] | None]]) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM kb_chunks WHERE doc_id=?", (doc_id,))
            rows = []
            for index, (text, meta, embedding) in enumerate(chunks):
                blob = sqlite3.Binary(_pack_vector(embedding)) if embedding else None
                rows.append((doc_id, index, text, json.dumps(meta, ensure_ascii=False), blob,
                             len(embedding) if embedding else 0))
            self._conn.executemany(
                "INSERT INTO kb_chunks (doc_id,chunk_index,text,meta,embedding,dim) VALUES (?,?,?,?,?,?)", rows
            )
            self._conn.commit()

    def kb_delete_doc(self, doc_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM kb_docs WHERE id=?", (doc_id,))
            self._conn.commit()

    def kb_all_chunks(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT c.id, c.doc_id, c.chunk_index, c.text, c.meta, c.embedding, d.title, d.uri, d.source"
                " FROM kb_chunks c JOIN kb_docs d ON d.id = c.doc_id ORDER BY c.id"
            ).fetchall()
        out = []
        for row in rows:
            item = dict(row)
            item["vector"] = _unpack_vector(item.pop("embedding"))
            try:
                item["meta"] = json.loads(item.get("meta") or "{}")
            except Exception:
                item["meta"] = {}
            out.append(item)
        return out

    def kb_docs(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM kb_docs ORDER BY indexed_at DESC").fetchall()
        return [dict(row) for row in rows]

    # -- settings -----------------------------------------------------------

    def get_setting(self, key: str, default: Any = None) -> Any:
        with self._lock:
            row = self._conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        if not row:
            return default
        try:
            return json.loads(row["value"])
        except Exception:
            return row["value"]

    def set_setting(self, key: str, value: Any) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO settings (key,value) VALUES (?,?)",
                (key, json.dumps(value, ensure_ascii=False, default=str)),
            )
            self._conn.commit()


# --------------------------------------------------------------------------- vectors


def _pack_vector(vector: Iterable[float] | None) -> bytes:
    if not vector:
        return b""
    import struct

    values = [float(v) for v in vector]
    return struct.pack(f"<{len(values)}f", *values)


def _unpack_vector(blob: Any) -> list[float]:
    if not blob:
        return []
    import struct

    data = bytes(blob)
    count = len(data) // 4
    return list(struct.unpack(f"<{count}f", data[: count * 4]))


def ledger_from_items(items: Iterable[LedgerItem]) -> list[dict[str, Any]]:
    return [item.model_dump() for item in items]
