"""
backend/app/memory.py
----------------------
SQLite-backed session memory for multi-turn query continuity.

Table schema
------------
session_memory:
    session_id           TEXT  PRIMARY KEY
    last_query           TEXT  NOT NULL
    cached_evidence_json TEXT  NOT NULL   -- JSON list of RetrievalResult dicts
    timestamp            DATETIME DEFAULT CURRENT_TIMESTAMP

Public API
----------
init_memory(db_path)
    Create the table if it does not exist.  Called once at application startup.

get_session_context(session_id, db_path) -> dict
    Return the stored last_query and cached_evidence_json for a session.
    Returns empty defaults when the session has never been seen.

save_session(session_id, query, chunks, db_path)
    Upsert the session row: serialise chunks to JSON and store alongside query.

load_cached_chunks(session_id, db_path) -> List[RetrievalResult]
    Deserialise cached_evidence_json back to RetrievalResult objects.
    Returns [] if the session has no cached evidence.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import List

from app.retrieval.dense import RetrievalResult

logger = logging.getLogger(__name__)


# ---- Schema ------------------------------------------------------------------

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS session_memory (
    session_id           TEXT     PRIMARY KEY,
    last_query           TEXT     NOT NULL DEFAULT '',
    cached_evidence_json TEXT     NOT NULL DEFAULT '[]',
    timestamp            DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


# ---- Serialisation helpers ---------------------------------------------------

def _chunks_to_json(chunks: List[RetrievalResult]) -> str:
    """Serialise a list of RetrievalResult objects to a JSON string."""
    return json.dumps(
        [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "metadata": c.metadata,
                "score": c.score,
            }
            for c in chunks
        ]
    )


def _json_to_chunks(raw: str) -> List[RetrievalResult]:
    """Deserialise a JSON string back to a list of RetrievalResult objects."""
    try:
        items = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    return [
        RetrievalResult(
            chunk_id=item["chunk_id"],
            text=item["text"],
            metadata=item.get("metadata", {}),
            score=item.get("score", 0.0),
        )
        for item in items
        if isinstance(item, dict)
    ]


# ---- Connection factory -------------------------------------------------------

def _connect(db_path: str) -> sqlite3.Connection:
    """Return a sqlite3 connection.  Supports the special :memory: path for tests."""
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


# ---- Public API --------------------------------------------------------------

def init_memory(db_path: str) -> None:
    """
    Create the session_memory table if it does not exist.

    Parameters
    ----------
    db_path:
        Filesystem path to the SQLite database file, or ``:memory:`` for tests.
        Parent directories are created automatically.
    """
    with _connect(db_path) as conn:
        conn.execute(_CREATE_TABLE)
    logger.info("Session memory initialised at %s", db_path)


def get_session_context(session_id: str, db_path: str) -> dict:
    """
    Return the stored context for *session_id*.

    Returns
    -------
    dict with keys:
        last_query           (str)  -- '' if no prior session
        cached_evidence_json (str)  -- '[]' if no cached evidence
        has_cache            (bool) -- True if there is usable evidence
    """
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT last_query, cached_evidence_json FROM session_memory WHERE session_id = ?",
            (session_id,),
        ).fetchone()

    if row is None:
        return {"last_query": "", "cached_evidence_json": "[]", "has_cache": False}

    evidence_json = row["cached_evidence_json"] or "[]"
    has_cache = bool(_json_to_chunks(evidence_json))
    return {
        "last_query": row["last_query"] or "",
        "cached_evidence_json": evidence_json,
        "has_cache": has_cache,
    }


def save_session(
    session_id: str,
    query: str,
    chunks: List[RetrievalResult],
    db_path: str,
) -> None:
    """
    Upsert the session row with the latest query and evidence chunks.

    Parameters
    ----------
    session_id:
        Unique session identifier.
    query:
        The user's query text.
    chunks:
        Evidence chunks to cache (serialised as JSON).
    db_path:
        Path to the SQLite database.
    """
    evidence_json = _chunks_to_json(chunks)
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO session_memory (session_id, last_query, cached_evidence_json)
            VALUES (?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                last_query           = excluded.last_query,
                cached_evidence_json = excluded.cached_evidence_json,
                timestamp            = CURRENT_TIMESTAMP
            """,
            (session_id, query, evidence_json),
        )
    logger.debug("Session %s: saved %d evidence chunks.", session_id, len(chunks))


def load_cached_chunks(session_id: str, db_path: str) -> List[RetrievalResult]:
    """
    Deserialise and return cached evidence chunks for *session_id*.

    Returns an empty list when no cached evidence exists.
    """
    ctx = get_session_context(session_id, db_path)
    return _json_to_chunks(ctx["cached_evidence_json"])
