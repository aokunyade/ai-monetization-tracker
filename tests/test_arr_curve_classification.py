from __future__ import annotations

import sqlite3
from unittest import TestCase

from anthropic_arr.arr_curve import _ms, _openai_curve, build_openai_signal_breakdown
from anthropic_arr.db import init_schema


class ArrCurveClassificationTest(TestCase):
    def test_openai_curve_preserves_estimate_classification(self) -> None:
        conn = sqlite3.connect(":memory:", isolation_level=None)
        self.addCleanup(conn.close)
        conn.row_factory = sqlite3.Row
        init_schema(conn)

        curve = _openai_curve(conn)
        self.assertEqual(curve["hist"][0][0], curve["cps"][0]["t"])
        july = next(point for point in curve["cps"] if point["t"] == _ms("2026-07-29"))

        self.assertEqual(july["v"], 42.6)
        self.assertEqual(july["classification"], "third_party_estimate")

        breakdown = build_openai_signal_breakdown(conn)
        self.assertTrue(breakdown["available"])
