from __future__ import annotations

from unittest import TestCase

from scripts.refresh_snapshot_data import refresh_snapshot_data


class SnapshotRefreshTest(TestCase):
    def test_refreshes_models_before_daily_sources(self) -> None:
        calls: list[str] = []

        def models() -> dict[str, int]:
            calls.append("models")
            return {"model_meta": 107, "provider_throughput_history": 940}

        def daily() -> dict[str, int]:
            calls.append("daily")
            return {"signals": 12}

        result = refresh_snapshot_data(models, daily)

        self.assertEqual(calls, ["models", "daily"])
        self.assertEqual(result["models"]["model_meta"], 107)
        self.assertEqual(result["daily"]["signals"], 12)

    def test_refuses_daily_refresh_when_models_are_empty(self) -> None:
        daily_called = False

        def daily() -> dict[str, int]:
            nonlocal daily_called
            daily_called = True
            return {}

        with self.assertRaisesRegex(RuntimeError, "no publishable models"):
            refresh_snapshot_data(lambda: {}, daily)

        self.assertFalse(daily_called)
