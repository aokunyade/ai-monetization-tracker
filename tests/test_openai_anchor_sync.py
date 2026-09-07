from __future__ import annotations

import sqlite3
from unittest import TestCase
from unittest.mock import patch

from anthropic_arr import db
from anthropic_arr.db import sync_openai_checkpoints_from_parent


class OpenAiAnchorSyncTest(TestCase):
    def _connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(":memory:", isolation_level=None)
        self.addCleanup(conn.close)
        conn.row_factory = sqlite3.Row
        conn.executescript(
            """
            CREATE TABLE openai_arr_checkpoints (
                kind TEXT NOT NULL,
                date TEXT NOT NULL,
                arr_bn REAL NOT NULL,
                source TEXT,
                url TEXT,
                note TEXT,
                classification TEXT NOT NULL DEFAULT 'reported',
                is_local INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (kind, date)
            );
            CREATE TABLE openai_arr_known (
                month TEXT PRIMARY KEY,
                arr_b_usd REAL NOT NULL,
                source TEXT,
                notes TEXT,
                added_at TEXT NOT NULL,
                is_local INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE openai_arr_extrapolation (
                as_of TEXT PRIMARY KEY,
                to_date TEXT NOT NULL,
                low_bn REAL NOT NULL,
                high_bn REAL NOT NULL
            );
            """
        )
        return conn

    def test_seed_removes_legacy_internal_extrapolation(self) -> None:
        conn = self._connection()
        conn.execute(
            "INSERT INTO openai_arr_extrapolation VALUES (?, ?, ?, ?)",
            ("2020-01-01", "2020-12-31", 1.0, 2.0),
        )

        db._seed_openai_arr(conn)

        count = conn.execute(
            "SELECT COUNT(*) FROM openai_arr_extrapolation"
        ).fetchone()[0]
        self.assertEqual(count, 0)

    def test_sync_prunes_retired_rows_and_rebuilds_the_materialized_series(
        self,
    ) -> None:
        conn = self._connection()
        conn.executescript(
            """
            INSERT INTO openai_arr_checkpoints
                (kind, date, arr_bn, source, is_local)
            VALUES
                ('checkpoint', '2026-07-15', 50, 'retired managed row', 0),
                ('checkpoint', '2026-07-20', 41, 'local override', 1),
                ('estimate', '2026-07-25', 42.6, 'third-party estimate', 0);
            INSERT INTO openai_arr_known
                (month, arr_b_usd, source, notes, added_at, is_local)
            VALUES
                ('2026-05', 33, 'retired', NULL, '2026-06-01', 0),
                ('2026-07', 52, 'stale', NULL, '2026-08-01', 0),
                ('2026-08', 55, 'local override', NULL, '2026-08-01', 1);
            """
        )

        changed, known = sync_openai_checkpoints_from_parent(
            conn,
            [
                {
                    "date": "2026-07-29",
                    "arr_bn": 42.6,
                    "kind": "checkpoint",
                    "classification": "third_party_estimate",
                    "source": "TickerTrends estimate",
                    "url": "https://example.test/source",
                    "note": None,
                }
            ],
        )

        rows = conn.execute(
            "SELECT kind, date, classification, is_local "
            "FROM openai_arr_checkpoints ORDER BY date"
        ).fetchall()
        self.assertEqual(
            [tuple(row) for row in rows],
            [
                ("checkpoint", "2026-07-20", "reported", 1),
                ("checkpoint", "2026-07-29", "third_party_estimate", 0),
            ],
        )
        self.assertGreater(changed, 0)
        self.assertGreater(known, 0)
        materialized = conn.execute(
            "SELECT month, arr_b_usd, source FROM openai_arr_known ORDER BY month"
        ).fetchall()
        self.assertEqual(
            [tuple(row) for row in materialized],
            [
                (
                    "2026-07",
                    42.6,
                    "third_party_estimate:TickerTrends estimate",
                )
            ],
        )

    def test_sync_rolls_back_checkpoint_changes_when_materialization_fails(
        self,
    ) -> None:
        conn = self._connection()
        conn.execute(
            "INSERT INTO openai_arr_checkpoints "
            "(kind, date, arr_bn, source) VALUES (?, ?, ?, ?)",
            ("checkpoint", "2026-06-25", 37.3, "existing"),
        )

        with patch.object(
            db, "rebuild_openai_arr_known", side_effect=RuntimeError("boom")
        ):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                sync_openai_checkpoints_from_parent(
                    conn,
                    [
                        {
                            "date": "2026-07-29",
                            "arr_bn": 42.6,
                            "kind": "checkpoint",
                            "classification": "third_party_estimate",
                            "source": "TickerTrends estimate",
                            "url": "https://example.test/source",
                            "note": None,
                        }
                    ],
                )

        rows = conn.execute(
            "SELECT date, arr_bn FROM openai_arr_checkpoints ORDER BY date"
        ).fetchall()
        self.assertEqual([tuple(row) for row in rows], [("2026-06-25", 37.3)])

    def test_manifest_reclaims_a_local_row_at_the_same_coordinate(self) -> None:
        conn = self._connection()
        conn.execute(
            "INSERT INTO openai_arr_checkpoints "
            "(kind, date, arr_bn, source, classification, is_local) "
            "VALUES ('checkpoint', '2026-07-29', 999, 'local override', 'reported', 1)"
        )

        sync_openai_checkpoints_from_parent(
            conn,
            [
                {
                    "date": "2026-07-29",
                    "arr_bn": 42.6,
                    "kind": "checkpoint",
                    "classification": "third_party_estimate",
                    "source": "TickerTrends estimate",
                    "url": "https://example.test/source",
                    "note": None,
                }
            ],
        )

        checkpoint = conn.execute(
            "SELECT arr_bn, source, classification, is_local "
            "FROM openai_arr_checkpoints WHERE kind='checkpoint' AND date='2026-07-29'"
        ).fetchone()
        self.assertEqual(
            tuple(checkpoint),
            (42.6, "TickerTrends estimate", "third_party_estimate", 0),
        )
        materialized = conn.execute(
            "SELECT month, arr_b_usd FROM openai_arr_known"
        ).fetchone()
        self.assertEqual(tuple(materialized), ("2026-07", 42.6))
