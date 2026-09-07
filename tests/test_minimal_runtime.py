from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping
from unittest import IsolatedAsyncioTestCase, TestCase
from unittest.mock import Mock, patch

from anthropic_arr import PROJECT_ROOT, api_server, db, sources


def _contains_nested_text(
    value: object, needle: str, seen: set[int] | None = None
) -> bool:
    if isinstance(value, str):
        return needle in value
    if isinstance(value, (bytes, bytearray)):
        return needle.encode() in value
    if seen is None:
        seen = set()
    identity = id(value)
    if identity in seen:
        return False
    seen.add(identity)
    if isinstance(value, Mapping):
        return any(
            _contains_nested_text(item, needle, seen)
            for pair in value.items()
            for item in pair
        )
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_contains_nested_text(item, needle, seen) for item in value)
    if isinstance(value, BaseException):
        return _contains_nested_text(value.args, needle, seen) or _contains_nested_text(
            value.__dict__, needle, seen
        )
    return False


class MinimalRuntimeTest(TestCase):
    def test_retired_auth_tables_and_invite_codes_are_not_shipped(self) -> None:
        for table in ("users", "invite_codes", "sessions", "access_log"):
            self.assertNotIn(f"CREATE TABLE IF NOT EXISTS {table}", db.SCHEMA)
        self.assertFalse(hasattr(db, "INVITE_CODES"))

    def test_legacy_internal_arr_seed_is_not_shipped(self) -> None:
        self.assertFalse(
            (PROJECT_ROOT / "anthropic_arr" / "data" / "openai_arr_seed.json").exists()
        )

    def test_daily_pipeline_uses_snapshot_refresh_orchestrator(self) -> None:
        script = (PROJECT_ROOT / "scripts" / "daily_sync_snapshot.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("-m scripts.refresh_snapshot_data", script)
        self.assertLess(
            script.index("ANTHROPIC_ARR_PRIVATE_ANCHORS_JSON"),
            script.index("-m scripts.refresh_snapshot_data"),
        )

    def test_publish_pipeline_always_starts_an_isolated_local_server(self) -> None:
        script = (PROJECT_ROOT / "scripts" / "publish_snapshot.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("SNAPSHOT_SERVER_PORT", script)
        self.assertIn("ANTHROPIC_ARR_PORT", script)
        self.assertIn("TRACKER_API_BASE_URL", script)
        self.assertIn("starting isolated dashboard API server", script)

    def test_publish_pipeline_requires_private_anchor_injection(self) -> None:
        script = (PROJECT_ROOT / "scripts" / "publish_snapshot.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("ANTHROPIC_ARR_PRIVATE_ANCHORS_JSON", script)

    def test_private_arr_anchors_are_loaded_only_from_environment(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(sources.load_private_arr_anchors(), [])

        payload = [
            {"month": "2025-01", "arr_b_usd": 1.25},
            {"month": "2025-02", "arr_b_usd": 1.5},
        ]
        with patch.dict(
            os.environ,
            {"ANTHROPIC_ARR_PRIVATE_ANCHORS_JSON": json.dumps(payload)},
        ):
            rows = sources.load_private_arr_anchors()

        self.assertEqual(
            [row[:2] for row in rows], [("2025-01", 1.25), ("2025-02", 1.5)]
        )
        self.assertTrue(all(row[2] == "funda_research_private" for row in rows))

    def test_private_arr_anchors_may_precede_a_later_public_disclosure(self) -> None:
        payload = [
            {"month": "2025-01", "arr_b_usd": 1.25},
            {"month": "2025-02", "arr_b_usd": 1.5},
        ]
        with patch.dict(
            os.environ,
            {"ANTHROPIC_ARR_PRIVATE_ANCHORS_JSON": json.dumps(payload)},
        ):
            rows = sources.load_private_arr_anchors()

        self.assertEqual([row[0] for row in rows], ["2025-01", "2025-02"])

    def test_private_arr_anchor_input_rejects_public_month_conflicts(self) -> None:
        payload = [
            {
                "month": sources._PUBLIC_KNOWN_ARR[0][0],
                "arr_b_usd": 1.25,
            }
        ]
        with patch.dict(
            os.environ,
            {"ANTHROPIC_ARR_PRIVATE_ANCHORS_JSON": json.dumps(payload)},
        ):
            with self.assertRaisesRegex(ValueError, "conflicts with a public"):
                sources.load_private_arr_anchors()

    def test_module_merges_public_and_private_anchors_in_month_order(self) -> None:
        payload = [
            {"month": "2025-01", "arr_b_usd": 1.25},
            {"month": "2025-02", "arr_b_usd": 1.5},
        ]
        env = os.environ.copy()
        env["ANTHROPIC_ARR_PRIVATE_ANCHORS_JSON"] = json.dumps(payload)
        code = """
from anthropic_arr import sources
months = [row[0] for row in sources.KNOWN_ARR]
assert months == sorted(months)
assert sources.HIDDEN_ARR_MONTHS == frozenset({'2025-01', '2025-02'})
"""

        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_public_arr_anchors_include_the_latest_reuters_disclosure(self) -> None:
        self.assertIn(
            ("2026-07", 65.0, "reuters_2026-08-17"),
            [row[:3] for row in sources._PUBLIC_KNOWN_ARR],
        )

    def test_private_arr_anchor_input_rejects_unsorted_months(self) -> None:
        payload = [
            {"month": "2020-02", "arr_b_usd": 1.5},
            {"month": "2020-01", "arr_b_usd": 1.25},
        ]
        with patch.dict(
            os.environ,
            {"ANTHROPIC_ARR_PRIVATE_ANCHORS_JSON": json.dumps(payload)},
        ):
            with self.assertRaisesRegex(ValueError, "strictly increasing"):
                sources.load_private_arr_anchors()

    def test_private_arr_anchor_input_rejects_non_finite_values(self) -> None:
        for value in (float("nan"), float("inf")):
            with self.subTest(value=value):
                payload = [{"month": "2025-01", "arr_b_usd": value}]
                with (
                    patch.dict(
                        os.environ,
                        {"ANTHROPIC_ARR_PRIVATE_ANCHORS_JSON": json.dumps(payload)},
                    ),
                    self.assertRaisesRegex(ValueError, "invalid private ARR value"),
                ):
                    sources.load_private_arr_anchors()

    def test_private_arr_anchor_requires_a_later_public_anchor(self) -> None:
        latest_public = max(row[0] for row in sources._PUBLIC_KNOWN_ARR)
        year, month = map(int, latest_public.split("-"))
        if month == 12:
            year, month = year + 1, 1
        else:
            month += 1
        payload = [{"month": f"{year:04d}-{month:02d}", "arr_b_usd": 1.25}]

        with (
            patch.dict(
                os.environ,
                {"ANTHROPIC_ARR_PRIVATE_ANCHORS_JSON": json.dumps(payload)},
            ),
            self.assertRaisesRegex(ValueError, "requires a later public anchor"),
        ):
            sources.load_private_arr_anchors()

    def test_private_arr_anchor_errors_do_not_echo_secret_fields(self) -> None:
        cases = [
            ([{"month": "2025-04", "arr_b_usd": -12345.67}], "2025-04"),
            (
                [{"month": "not-a-secret-month", "arr_b_usd": 1.25}],
                "not-a-secret-month",
            ),
            (
                [
                    {"month": "2025-02", "arr_b_usd": 1.25},
                    {"month": "2025-01", "arr_b_usd": 1.5},
                ],
                "2025-01",
            ),
        ]
        for payload, secret_month in cases:
            with self.subTest(secret_month=secret_month):
                with patch.dict(
                    os.environ,
                    {"ANTHROPIC_ARR_PRIVATE_ANCHORS_JSON": json.dumps(payload)},
                ):
                    with self.assertRaises(ValueError) as raised:
                        sources.load_private_arr_anchors()
                message = str(raised.exception)
                self.assertNotIn(secret_month, message)
                self.assertNotIn("-12345.67", message)
                self.assertIn("entry #", message)
                self.assertIsNone(raised.exception.__cause__)

    def test_malformed_private_arr_json_does_not_retain_secret_document(self) -> None:
        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_ARR_PRIVATE_ANCHORS_JSON": (
                    '[{"traceback-secret-fragment":"raw-document"'
                )
            },
        ):
            try:
                sources.load_private_arr_anchors()
            except ValueError as raised:
                self.assertEqual(
                    str(raised),
                    "ANTHROPIC_ARR_PRIVATE_ANCHORS_JSON must be valid JSON",
                )
                self.assertIsNone(raised.__cause__)
                self.assertIsNone(raised.__context__)
                self.assertFalse(hasattr(raised, "doc"))
                traceback = raised.__traceback__
                self.assertIsNotNone(traceback)
                while traceback is not None:
                    frame_locals = traceback.tb_frame.f_locals
                    self.assertFalse(
                        _contains_nested_text(frame_locals, "traceback-secret-fragment")
                    )
                    self.assertFalse(
                        _contains_nested_text(
                            frame_locals,
                            '[{"traceback-secret-fragment":"raw-document"',
                        )
                    )
                    traceback = traceback.tb_next
            else:
                self.fail("malformed private ARR JSON was accepted")


class ServerLifecycleTest(IsolatedAsyncioTestCase):
    async def test_startup_initializes_schema_without_implicitly_refreshing(
        self,
    ) -> None:
        initialize = Mock()
        refresh = Mock(return_value={})

        with (
            patch.object(api_server, "init_schema", initialize),
            patch.object(api_server, "refresh", refresh),
        ):
            async with api_server._lifespan(None):
                pass

        initialize.assert_called_once_with()
        refresh.assert_not_called()
