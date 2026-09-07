import sqlite3
import unittest
from datetime import date, timedelta
from unittest.mock import call, patch

from anthropic_arr import arr_curve, compute, routes_sdk, sources

CONTROL_PACKAGES = ["requests", "certifi", "urllib3", "six", "boto3"]


def _signals_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE anthropic_signals (
            date TEXT NOT NULL,
            signal_name TEXT NOT NULL,
            value REAL NOT NULL,
            captured_at TEXT NOT NULL,
            PRIMARY KEY (date, signal_name)
        )
        """
    )
    return conn


class PypiStepCorrectionTests(unittest.TestCase):
    def test_model_signal_value_rejects_invalid_dates(self):
        helper = getattr(compute, "model_signal_value", None)
        self.assertIsNotNone(helper)

        with self.assertRaises(ValueError):
            helper("pypi:openai", "2026-08-XX", 100.0)

    def test_daily_series_corrects_only_selected_signals_at_and_after_break(self):
        conn = _signals_conn()
        self.addCleanup(conn.close)
        signals = {
            "pypi:anthropic": 100.0,
            "pypi:openai": 200.0,
            "pypi:langchain-anthropic": 300.0,
            "npm:@anthropic-ai/sdk": 400.0,
            "pypictl:requests": 500.0,
        }
        dates = ["2026-08-24", "2026-08-25", "2026-08-26"]
        rows = [
            (day, signal, value, "2026-08-27T00:00:00+00:00")
            for signal, value in signals.items()
            for day in dates
        ]
        conn.executemany(
            "INSERT INTO anthropic_signals VALUES (?, ?, ?, ?)",
            rows,
        )

        anthropic = compute._daily_series(conn, "pypi:anthropic")
        openai = compute._daily_series(conn, "pypi:openai")

        self.assertEqual(anthropic["2026-08-24"], 100.0)
        self.assertEqual(anthropic["2026-08-25"], 3_000_100.0)
        self.assertEqual(anthropic["2026-08-26"], 3_000_100.0)
        self.assertEqual(openai["2026-08-24"], 200.0)
        self.assertEqual(openai["2026-08-25"], 6_700_200.0)
        self.assertEqual(openai["2026-08-26"], 6_700_200.0)

        for signal in (
            "pypi:langchain-anthropic",
            "npm:@anthropic-ai/sdk",
            "pypictl:requests",
        ):
            value = signals[signal]
            self.assertEqual(
                compute._daily_series(conn, signal),
                {day: value for day in dates},
                msg=f"unexpected correction for {signal}",
            )

        stored = conn.execute(
            "SELECT date, signal_name, value FROM anthropic_signals "
            "ORDER BY date, signal_name"
        ).fetchall()
        self.assertEqual(
            [(r["date"], r["signal_name"], r["value"]) for r in stored],
            sorted(
                (day, signal, value)
                for signal, value in signals.items()
                for day in dates
            ),
        )

    def test_monthly_predictor_paths_share_the_daily_series_correction(self):
        conn = _signals_conn()
        self.addCleanup(conn.close)
        dates = ["2026-08-24", "2026-08-25", "2026-08-26"]
        for signal, value in (
            ("pypi:openai", 200.0),
            ("pypi:langchain-anthropic", 300.0),
        ):
            conn.executemany(
                "INSERT INTO anthropic_signals VALUES (?, ?, ?, ?)",
                [(day, signal, value, "2026-08-27T00:00:00+00:00") for day in dates],
            )

        corrected_daily_sum = sum(compute._daily_series(conn, "pypi:openai").values())
        monthly_sum = compute._monthly_sum(conn, "pypi:openai", "2026-08")
        projected, _ = compute._project_monthly_sum(
            conn, "pypi:openai", "2026-08", cutoff_date="2026-08-27"
        )

        self.assertEqual(corrected_daily_sum, 13_400_600.0)
        self.assertEqual(monthly_sum, corrected_daily_sum)
        self.assertEqual(projected, corrected_daily_sum * 31 / 3)
        self.assertEqual(
            compute._monthly_sum(conn, "pypi:langchain-anthropic", "2026-08"),
            900.0,
        )
        other_projected, _ = compute._project_monthly_sum(
            conn,
            "pypi:langchain-anthropic",
            "2026-08",
            cutoff_date="2026-08-27",
        )
        self.assertEqual(other_projected, 900.0 * 31 / 3)

    def test_openai_fallback_growth_uses_corrected_pypi_series(self):
        conn = _signals_conn()
        self.addCleanup(conn.close)
        start = date(2026, 8, 1)
        raw_points = [
            ((start + timedelta(days=offset)).isoformat(), 200.0)
            for offset in range(40)
        ]
        conn.executemany(
            "INSERT INTO anthropic_signals VALUES (?, ?, ?, ?)",
            [
                (day, "pypi:openai", value, "2026-09-10T00:00:00+00:00")
                for day, value in raw_points
            ],
        )

        raw_expected = arr_curve._series_monthly_growth(
            raw_points[:-1],
            arr_curve._OPENAI_SIG_LOOKBACK_DAYS,
            arr_curve._OPENAI_SIG_MIN_POINTS,
        )
        corrected_points = [
            (
                day,
                200.0 + (6_700_000.0 if day >= "2026-08-25" else 0.0),
            )
            for day, _ in raw_points[:-1]
        ]
        corrected_expected = arr_curve._series_monthly_growth(
            corrected_points,
            arr_curve._OPENAI_SIG_LOOKBACK_DAYS,
            arr_curve._OPENAI_SIG_MIN_POINTS,
        )

        result = arr_curve._openai_signals(conn)

        self.assertNotEqual(raw_expected, corrected_expected)
        self.assertAlmostEqual(result["sdk_downloads"], corrected_expected)

    def test_fetch_signals_fetches_control_basket_with_separate_namespace(self):
        client = object()
        fetched = {
            package: [{"date": "2026-08-26", "value": float(index)}]
            for index, package in enumerate(CONTROL_PACKAGES, start=1)
        }

        with (
            patch.object(sources, "NPM_PACKAGES", []),
            patch.object(sources, "PYPI_PACKAGES", []),
            patch.object(
                sources, "PYPI_CONTROL_PACKAGES", CONTROL_PACKAGES, create=True
            ),
            patch.object(
                sources,
                "fetch_pypi_overall",
                side_effect=lambda _client, package: fetched[package],
            ) as fetch_pypi,
            patch.object(sources.time, "sleep") as sleep,
        ):
            rows = sources.fetch_signals(client, end="2026-08-26")

        self.assertEqual(
            getattr(sources, "PYPI_CONTROL_PACKAGES", None), CONTROL_PACKAGES
        )
        self.assertEqual(
            fetch_pypi.call_args_list,
            [call(client, package) for package in CONTROL_PACKAGES],
        )
        self.assertEqual(sleep.call_args_list, [call(1.5)] * 5)
        self.assertEqual(
            [row["signal_name"] for row in rows],
            [f"pypictl:{package}" for package in CONTROL_PACKAGES],
        )

    def test_controls_are_not_registered_as_model_signals(self):
        registered = (
            set(compute.SIGNAL_REGISTRY)
            | set(compute.SIGNAL_REGISTRY_OPENAI)
            | set(compute.SHADOW_SIGNALS)
            | set(compute.SHADOW_SIGNALS_OPENAI)
            | set(compute.PROXY_SIGNALS)
            | set(compute.PROXY_SIGNALS_OPENAI)
        )

        self.assertFalse(any(signal.startswith("pypictl:") for signal in registered))

    def test_sdk_downloads_payload_excludes_control_signals(self):
        conn = _signals_conn()
        rows = [
            ("npm:@anthropic-ai/sdk", 1.0),
            ("npm:openai", 2.0),
            ("pypi:anthropic", 3.0),
            ("pypi:openai", 4.0),
            ("pypictl:requests", 999.0),
        ]
        conn.executemany(
            "INSERT INTO anthropic_signals VALUES (?, ?, ?, ?)",
            [
                ("2026-08-26", signal, value, "2026-08-27T00:00:00+00:00")
                for signal, value in rows
            ],
        )

        with patch.object(routes_sdk, "get_db", return_value=conn):
            payload = routes_sdk.sdk_downloads()

        self.assertEqual(set(payload["npm"]), {"@anthropic-ai/sdk", "openai"})
        self.assertEqual(set(payload["pypi"]), {"anthropic", "openai"})
        self.assertNotIn("pypictl", repr(payload))
        self.assertNotIn("999.0", repr(payload))


if __name__ == "__main__":
    unittest.main()
