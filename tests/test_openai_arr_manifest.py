import importlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from anthropic_arr import openai_anchors
from anthropic_arr.openai_anchors import openai_checkpoint_rows
from scripts import update_data

MANIFEST = Path(__file__).resolve().parents[1] / "config" / "openai_arr_anchors.json"
TRACKER_CONFIG = Path(__file__).resolve().parents[1] / "config" / "tracker_config.json"


class OpenAiArrManifestTest(TestCase):
    def test_approved_historical_reported_anchors(self) -> None:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        by_date = {row["date"]: row for row in payload["anchors"]}

        june = by_date["2025-06-09"]
        self.assertEqual(june["arr_bn"], 10.0)
        self.assertEqual(june["evidence_type"], "reported")
        self.assertEqual(june["source_title"], "CNBC")
        self.assertEqual(
            june["url"],
            "https://www.cnbc.com/2025/06/09/"
            "openai-hits-10-billion-in-annualized-revenue-fueled-by-chatgpt-growth.html",
        )

        july = by_date["2025-07-31"]
        self.assertEqual(july["arr_bn"], 12.0)
        self.assertEqual(july["currency"], "USD")
        self.assertEqual(july["unit"], "billions")
        self.assertEqual(july["evidence_type"], "reported")
        self.assertEqual(july["source_title"], "The Information")
        self.assertEqual(
            july["url"],
            "https://www.theinformation.com/articles/"
            "openai-hits-12-billion-annualized-revenue-breaks-700-million-"
            "chatgpt-weekly-active-users",
        )

    def test_tickertrends_estimates_remain_unchanged(self) -> None:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        tickertrends = {
            row["date"]: (row["arr_bn"], row["evidence_type"], row["url"])
            for row in payload["anchors"]
            if row["source_title"] == "TickerTrends"
        }

        url = "https://blog.tickertrends.io/p/openai-arr-growth-accelerated-july-2026"
        self.assertEqual(
            tickertrends,
            {
                "2026-05-30": (33.0, "third_party_estimate", url),
                "2026-06-25": (37.3, "third_party_estimate", url),
                "2026-07-29": (42.6, "third_party_estimate", url),
            },
        )

    def test_unapproved_2026_anchors_are_absent(self) -> None:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        dates = {row["date"] for row in payload["anchors"]}

        self.assertFalse(any(date.startswith("2026-03-") for date in dates))
        self.assertFalse(any(date.startswith("2026-08-") for date in dates))

    def test_july_anchor_is_dated_and_classified_as_an_estimate(self) -> None:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        july = next(row for row in payload["anchors"] if row["date"] == "2026-07-29")

        self.assertEqual(july["arr_bn"], 42.6)
        self.assertEqual(july["evidence_type"], "third_party_estimate")
        self.assertEqual(
            july["url"],
            "https://blog.tickertrends.io/p/openai-arr-growth-accelerated-july-2026",
        )

    def test_every_anchor_carries_reviewable_provenance(self) -> None:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))

        self.assertEqual(payload["schema_version"], 2)
        for anchor in payload["anchors"]:
            self.assertIn(anchor["evidence_type"], {"reported", "third_party_estimate"})
            self.assertEqual(anchor["currency"], "USD")
            self.assertEqual(anchor["unit"], "billions")
            self.assertTrue(anchor["source_title"])
            self.assertTrue(anchor["source"])
            self.assertTrue(anchor["url"].startswith("https://"))

    def test_model_rows_preserve_manifest_classifications(self) -> None:
        rows = {row["date"]: row for row in openai_checkpoint_rows()}

        self.assertEqual(rows["2025-06-09"]["kind"], "checkpoint")
        self.assertEqual(rows["2025-06-09"]["classification"], "reported")
        self.assertEqual(rows["2025-07-31"]["classification"], "reported")
        self.assertEqual(rows["2026-07-29"]["classification"], "third_party_estimate")

    def test_future_anchor_is_rejected(self) -> None:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        payload["anchors"][-1]["date"] = "2100-01-01"
        with TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "anchors.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with patch.object(openai_anchors, "OPENAI_ANCHORS_PATH", manifest):
                with self.assertRaisesRegex(ValueError, "future"):
                    openai_anchors.load_openai_anchor_records()

    def test_anchor_dates_must_be_strictly_increasing(self) -> None:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        payload["anchors"][0], payload["anchors"][1] = (
            payload["anchors"][1],
            payload["anchors"][0],
        )
        with TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "anchors.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with patch.object(openai_anchors, "OPENAI_ANCHORS_PATH", manifest):
                with self.assertRaisesRegex(ValueError, "strictly increasing"):
                    openai_anchors.load_openai_anchor_records()

    def test_anchor_arr_must_be_positive_and_finite(self) -> None:
        for value in (float("nan"), float("inf"), 0, -1):
            with self.subTest(value=value), TemporaryDirectory() as temp_dir:
                payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
                payload["anchors"][0]["arr_bn"] = value
                manifest = Path(temp_dir) / "anchors.json"
                manifest.write_text(json.dumps(payload), encoding="utf-8")

                with patch.object(openai_anchors, "OPENAI_ANCHORS_PATH", manifest):
                    with self.assertRaisesRegex(ValueError, "positive finite"):
                        openai_anchors.load_openai_anchor_records()

    def test_anchor_primitives_are_not_coerced(self) -> None:
        mutations = {
            "date": lambda anchor: anchor.update(date=20250609),
            "source": lambda anchor: anchor.update(source=123),
            "source title": lambda anchor: anchor.update(source_title=123),
            "url": lambda anchor: anchor.update(url=["https://example.test/source"]),
            "boolean ARR": lambda anchor: anchor.update(arr_bn=True),
        }
        for field, mutate in mutations.items():
            with self.subTest(field=field), TemporaryDirectory() as temp_dir:
                payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
                mutate(payload["anchors"][0])
                manifest = Path(temp_dir) / "anchors.json"
                manifest.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaises((TypeError, ValueError)):
                    openai_anchors.load_openai_anchor_records(manifest)

    def test_anchor_url_must_use_https(self) -> None:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        payload["anchors"][0]["url"] = "http://example.test/source"
        with TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "anchors.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with patch.object(openai_anchors, "OPENAI_ANCHORS_PATH", manifest):
                with self.assertRaisesRegex(ValueError, "HTTPS"):
                    openai_anchors.load_openai_anchor_records()

    def test_anchor_url_must_have_a_host(self) -> None:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        payload["anchors"][0]["url"] = "https:///source"
        with TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "anchors.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "HTTPS"):
                openai_anchors.load_openai_anchor_records(manifest)

    def test_static_tracker_uses_the_validated_manifest_loader(self) -> None:
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        payload["schema_version"] = 1
        with TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "anchors.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            try:
                with (
                    patch.object(openai_anchors, "OPENAI_ANCHORS_PATH", manifest),
                    self.assertRaisesRegex(ValueError, "schema_version must be 2"),
                ):
                    importlib.reload(update_data)
            finally:
                importlib.reload(update_data)

    def test_static_tracker_projects_openai_anchors_from_the_manifest(self) -> None:
        tracker_config = json.loads(TRACKER_CONFIG.read_text(encoding="utf-8"))
        self.assertNotIn("checkpoints", tracker_config["arr"]["companies"]["openai"])
        self.assertNotIn("start", tracker_config["arr"]["companies"]["openai"])

        projected = update_data.CFG["arr"]["companies"]["openai"]
        self.assertEqual(projected["start"]["date"], "2025-06-09")
        self.assertTrue(projected["start"]["src"].startswith("reported:"))
        self.assertEqual(projected["start"]["classification"], "reported")
        projected_by_date = {row["date"]: row for row in projected["checkpoints"]}
        self.assertIn("2025-07-31", projected_by_date)
        historical = projected_by_date["2025-07-31"]
        self.assertEqual(historical["value_b"], 12.0)
        self.assertTrue(historical["src"].startswith("reported:"))
        july = next(
            row for row in projected["checkpoints"] if row["date"] == "2026-07-29"
        )
        self.assertEqual(july["value_b"], 42.6)
        self.assertTrue(july["src"].startswith("third_party_estimate:"))

        static_curve = update_data.build_arr({})["companies"]["openai"]
        first = next(
            row
            for row in static_curve["cps"]
            if row["t"] == update_data.ms("2025-06-09")
        )
        self.assertTrue(first["src"].startswith("reported:"))
        self.assertEqual(first["classification"], "reported")
