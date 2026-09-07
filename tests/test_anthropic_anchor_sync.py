from __future__ import annotations

import sqlite3
from unittest import TestCase
from unittest.mock import patch

from anthropic_arr import db
from anthropic_arr.db import upsert_arr_known_from_parent


class AnthropicAnchorSyncTest(TestCase):
    def test_sync_prunes_retired_code_rows_but_preserves_local_rows(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        conn.execute(
            """
            CREATE TABLE anthropic_arr_known (
                month TEXT PRIMARY KEY,
                arr_b_usd REAL NOT NULL,
                source TEXT,
                notes TEXT,
                added_at TEXT NOT NULL,
                is_local INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.executemany(
            "INSERT INTO anthropic_arr_known VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("2020-01", 1.0, "retired", "", "2020-01-01", 0),
                ("2020-02", 2.0, "local", "", "2020-02-01", 1),
            ],
        )

        upsert_arr_known_from_parent(
            conn,
            [
                {
                    "month": "2020-03",
                    "arr_b_usd": 3.0,
                    "source": "current",
                    "notes": "",
                    "added_at": "2020-03-01",
                }
            ],
        )

        rows = conn.execute(
            "SELECT month, is_local FROM anthropic_arr_known ORDER BY month"
        ).fetchall()
        self.assertEqual(
            [(row["month"], row["is_local"]) for row in rows],
            [("2020-02", 1), ("2020-03", 0)],
        )
        conn.close()

    def test_schema_startup_reconciles_stale_code_owned_rows(self) -> None:
        conn = sqlite3.connect(":memory:", isolation_level=None)
        conn.row_factory = sqlite3.Row
        current = [
            {
                "month": "2090-01",
                "arr_b_usd": 1.0,
                "source": "current",
                "notes": "",
                "added_at": "2090-01-01",
            }
        ]
        with (
            patch("anthropic_arr.sources.known_arr_rows", return_value=current),
            patch.object(db, "_seed_openai_arr"),
            patch.object(db, "_seed_models_curated"),
        ):
            db.init_schema(conn)
            conn.execute(
                "INSERT INTO anthropic_arr_known VALUES (?, ?, ?, ?, ?, ?)",
                ("2090-02", 2.0, "stale", "", "2090-02-01", 0),
            )
            db.init_schema(conn)

        rows = conn.execute(
            "SELECT month FROM anthropic_arr_known ORDER BY month"
        ).fetchall()
        self.assertEqual([row["month"] for row in rows], ["2090-01"])
        conn.close()
