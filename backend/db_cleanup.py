#!/usr/bin/env python
"""
Lightweight SQLite cleanup for CrownTALK.

- Keeps only the last N rows in `comments`
- Deletes old rows in:
    - comments_seen
    - comments_openers_seen
    - comments_ngrams_seen
    - comments_templates_seen

All thresholds are tunable via env vars:

    DB_PATH (default: /app/crowntalk.db)
    DB_COMMENTS_MAX_ROWS (default: 50000)
    DB_RETENTION_DAYS (default: 60)

Usage (inside container):

    python db_cleanup.py

You can also wire this into a scheduled job or run it periodically.
"""

from __future__ import annotations

import os
import sqlite3
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("db_cleanup")

DB_PATH = os.environ.get("DB_PATH", "/app/crowntalk.db")

# Keep at most this many comments (most recent by id)
DB_COMMENTS_MAX_ROWS = int(os.environ.get("DB_COMMENTS_MAX_ROWS", "50000"))

# For the auxiliary tables that store hashes/openers/ngrams/templates,
# we delete entries older than this many days.
DB_RETENTION_DAYS = int(os.environ.get("DB_RETENTION_DAYS", "60"))


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30, isolation_level=None)
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


def prune_comments(conn: sqlite3.Connection) -> None:
    if DB_COMMENTS_MAX_ROWS <= 0:
        logger.info("DB_COMMENTS_MAX_ROWS <= 0, skipping comments pruning")
        return

    cur = conn.cursor()
    cur.execute("SELECT MAX(id) FROM comments;")
    row = cur.fetchone()
    max_id = row[0] if row and row[0] is not None else None

    if max_id is None:
        logger.info("No rows in comments, nothing to prune.")
        return

    cutoff_id = max_id - DB_COMMENTS_MAX_ROWS
    if cutoff_id <= 0:
        logger.info(
            "comments rows (%s) <= DB_COMMENTS_MAX_ROWS (%s), nothing to prune.",
            max_id,
            DB_COMMENTS_MAX_ROWS,
        )
        return

    logger.info("Pruning comments with id <= %s", cutoff_id)
    cur.execute("DELETE FROM comments WHERE id <= ?;", (cutoff_id,))
    logger.info("Pruned %s rows from comments", conn.total_changes)


def prune_aux_tables(conn: sqlite3.Connection) -> None:
    """
    For *_seen tables that use integer created_at timestamps (epoch seconds),
    delete entries older than N days.
    """
    if DB_RETENTION_DAYS <= 0:
        logger.info("DB_RETENTION_DAYS <= 0, skipping aux tables pruning")
        return

    cutoff_ts = int(time.time()) - DB_RETENTION_DAYS * 86400
    logger.info(
        "Pruning aux tables with created_at < %s (≈ %s days ago)",
        cutoff_ts,
        DB_RETENTION_DAYS,
    )

    tables = [
        "comments_seen",
        "comments_openers_seen",
        "comments_ngrams_seen",
        "comments_templates_seen",
    ]
    cur = conn.cursor()
    for tbl in tables:
        try:
            logger.info("Pruning table %s ...", tbl)
            cur.execute(f"DELETE FROM {tbl} WHERE created_at < ?;", (cutoff_ts,))
            logger.info("Table %s: %s rows deleted", tbl, conn.total_changes)
        except sqlite3.OperationalError as e:
            logger.warning("Skipping table %s due to error: %s", tbl, e)


def main() -> None:
    logger.info("Starting DB cleanup against %s", DB_PATH)
    conn = get_conn()
    try:
        prune_comments(conn)
        prune_aux_tables(conn)
        logger.info("DB cleanup completed successfully.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
