#!/usr/bin/env python3
"""Search the local qoppac blog SQLite archive."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path


DEFAULT_DB = Path(__file__).resolve().parents[1] / "blog" / "qoppac_blog.sqlite"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", help="FTS query, e.g. 'carry OR momentum'")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        rows = conn.execute(
            """
            SELECT
                substr(posts.published, 1, 10) AS published,
                posts.title,
                posts.url,
                snippet(posts_fts, 2, '[', ']', '...', 18) AS excerpt
            FROM posts_fts
            JOIN posts ON posts.id = posts_fts.rowid
            WHERE posts_fts MATCH ?
            ORDER BY bm25(posts_fts)
            LIMIT ?
            """,
            (args.query, args.limit),
        ).fetchall()
    finally:
        conn.close()

    for index, (published, title, url, excerpt) in enumerate(rows, 1):
        print(f"{index}. {published} | {title}")
        print(f"   {url}")
        print(f"   {excerpt}")
        print()


if __name__ == "__main__":
    main()
