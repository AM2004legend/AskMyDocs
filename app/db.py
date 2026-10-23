"""Postgres connection pool. Plain SQL everywhere, no ORM.

`register_vector` teaches psycopg to send/receive numpy arrays as pgvector values.
"""

from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.config import get_settings

SCHEMA_FILE = Path(__file__).resolve().parent.parent / "sql" / "schema.sql"

_pool: ConnectionPool | None = None


def init_schema() -> None:
    # runs before the pool exists: register_vector needs the extension to already be there
    with psycopg.connect(get_settings().database_url, autocommit=True) as conn:
        conn.execute(SCHEMA_FILE.read_text(encoding="utf-8"))


def _configure(conn: psycopg.Connection) -> None:
    register_vector(conn)
    conn.row_factory = dict_row
    # The HNSW index finds nearest neighbours across ALL documents, then the WHERE document_id
    # filter runs. Iterative scan (pgvector >= 0.8) keeps searching until enough rows survive
    # the filter, so a small document among big ones still gets its full top-k.
    conn.execute("SET hnsw.iterative_scan = strict_order")
    conn.commit()


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(get_settings().database_url, min_size=1, max_size=10, configure=_configure, open=True)
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
