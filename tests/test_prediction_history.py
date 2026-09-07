from __future__ import annotations

import sqlite3
from contextlib import closing
from unittest import TestCase
from unittest.mock import patch

from anthropic_arr.payload import prediction_history
from scripts.publish_snapshot import (
    persist_published_prediction,
    record_published_prediction,
)
from scripts.snapshot import add_current_prediction_history


def prediction_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE arr_predictions_history (
            forecast_date TEXT NOT NULL,
            target_month TEXT NOT NULL,
            method TEXT NOT NULL,
            signal TEXT NOT NULL,
            predicted_arr_b REAL NOT NULL,
            ci_low REAL,
            ci_high REAL,
            n_train INTEGER,
            r2 REAL,
            rmse_loo REAL,
            PRIMARY KEY (forecast_date, target_month, method, signal)
        )
        """
    )
    return conn


class PredictionHistoryTest(TestCase):
    def test_published_history_persistence_failure_is_nonfatal(self) -> None:
        with patch(
            "scripts.publish_snapshot.db_ctx", side_effect=sqlite3.ProgrammingError
        ):
            self.assertEqual(persist_published_prediction({}), 0)

    def test_transaction_failure_is_rolled_back_before_connection_closes(self) -> None:
        conn = prediction_connection()
        with (
            patch("scripts.publish_snapshot.db_ctx", return_value=closing(conn)),
            patch(
                "scripts.publish_snapshot.record_published_prediction",
                side_effect=RuntimeError("boom"),
            ),
        ):
            self.assertEqual(persist_published_prediction({}), 0)

    def test_published_snapshot_records_its_current_ensemble_headline(self) -> None:
        conn = prediction_connection()
        snapshot = {
            "captured_at": "2026-09-07T09:30:00+00:00",
            "endpoints": {
                "dashboard": {
                    "body": {
                        "predictor": {
                            "target_month": "2026-09",
                            "ensemble": {"n_models": 4},
                        },
                        "predictions_history": [
                            {
                                "as_of": "2026-09-07",
                                "target_month": "2026-09",
                                "arr_b": 76.5,
                                "ci_low": 70.0,
                                "ci_high": 83.0,
                            }
                        ],
                    }
                }
            },
        }

        self.assertEqual(record_published_prediction(snapshot, conn), 1)

        row = conn.execute(
            "SELECT target_month, method, signal, predicted_arr_b, ci_low, "
            "ci_high, n_train FROM arr_predictions_history"
        ).fetchone()
        self.assertEqual(
            dict(row),
            {
                "target_month": "2026-09",
                "method": "hero",
                "signal": "ensemble",
                "predicted_arr_b": 76.5,
                "ci_low": 70.0,
                "ci_high": 83.0,
                "n_train": 4,
            },
        )
        conn.close()

    def test_capture_adds_the_current_headline_without_persisting_it(self) -> None:
        snapshot = {
            "captured_at": "2026-09-07T09:30:00+00:00",
            "endpoints": {
                "dashboard": {
                    "body": {
                        "public_meta": {
                            "hero_arr_b": 76.5,
                            "hero_ci_low": 70.0,
                            "hero_ci_high": 83.0,
                        },
                        "predictor": {"target_month": "2026-09"},
                        "predictions_history": [
                            {
                                "as_of": "2026-08-31",
                                "target_month": "2026-08",
                                "arr_b": 79.0,
                                "ci_low": None,
                                "ci_high": None,
                            }
                        ],
                    }
                }
            },
        }

        add_current_prediction_history(snapshot)

        history = snapshot["endpoints"]["dashboard"]["body"]["predictions_history"]
        self.assertEqual(
            [row["target_month"] for row in history], ["2026-08", "2026-09"]
        )
        self.assertEqual(history[-1]["as_of"], "2026-09-07")
        self.assertEqual(history[-1]["arr_b"], 76.5)

    def test_payload_keeps_the_last_published_headline_for_each_month(self) -> None:
        conn = prediction_connection()
        rows = [
            ("2026-07-15", "2026-07", "hero", "ensemble", 70.0, 60.0, 80.0),
            ("2026-07-31", "2026-07", "hero", "ensemble", 72.98, 63.5, 82.46),
            ("2026-08-31", "2026-08", "hero", "ensemble", 79.0, None, None),
            ("2026-08-31", "2026-08", "hero", "secondary", 999.0, None, None),
            ("2026-09-01", "2026-09", "signal", "npm", 75.0, None, None),
        ]
        conn.executemany(
            "INSERT INTO arr_predictions_history "
            "(forecast_date, target_month, method, signal, predicted_arr_b, "
            "ci_low, ci_high) VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )

        self.assertEqual(
            prediction_history(conn),
            [
                {
                    "as_of": "2026-07-31",
                    "target_month": "2026-07",
                    "arr_b": 72.98,
                    "ci_low": 63.5,
                    "ci_high": 82.46,
                },
                {
                    "as_of": "2026-08-31",
                    "target_month": "2026-08",
                    "arr_b": 79.0,
                    "ci_low": None,
                    "ci_high": None,
                },
            ],
        )
        conn.close()
