from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from scripts import snapshot
from scripts.publish_snapshot import REQUIRED_ENDPOINTS
from scripts.snapshot import ENDPOINTS, PTH_BY_MODEL_KEY, capture_by_model


class _Response:
    def __init__(self, status: int) -> None:
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return {"top_providers": [], "metrics": {}}


class _Client:
    def get(self, _url: str, *, params: dict[str, str]) -> _Response:
        return _Response(503 if params["model"] == "broken-model" else 200)


class _CaptureResponse(_Response):
    headers = {"content-type": "application/json"}

    def __init__(self, status: int, body: dict) -> None:
        super().__init__(status)
        self._body = body
        self.is_success = status < 400

    def json(self) -> dict:
        return self._body


class _CaptureClient:
    def __enter__(self) -> _CaptureClient:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def get(
        self,
        _url: str,
        *,
        params: dict[str, str] | None = None,
    ) -> _CaptureResponse:
        if params:
            return _CaptureResponse(503, {})
        return _CaptureResponse(200, {"models": ["broken-model"]})


class SnapshotContractTest(TestCase):
    def test_hidden_interior_span_is_scrubbed_between_public_checkpoints(self) -> None:
        before, hidden_one, hidden_two, after = (
            "2024-12",
            "2025-01",
            "2025-02",
            "2025-03",
        )
        times = {
            month: snapshot._ms(snapshot._month_end(month))
            for month in (before, hidden_one, hidden_two, after)
        }
        company = {
            "cps": [
                {"t": times[before], "v": 10.0},
                {"t": times[hidden_one], "v": 20.0},
                {"t": times[hidden_two], "v": 30.0},
                {"t": times[after], "v": 40.0},
            ],
            "hist": [
                [times[before], 10.0],
                [times[hidden_one], 20.0],
                [times[hidden_two], 30.0],
                [times[after], 40.0],
            ],
        }
        hidden = frozenset({hidden_one, hidden_two})

        span_model = snapshot._scrub_curve_company(company, hidden)

        self.assertEqual(
            [checkpoint["t"] for checkpoint in company["cps"]],
            [times[before], times[after]],
        )
        values_by_time = dict(company["hist"])
        self.assertNotEqual(values_by_time[times[hidden_one]], 20.0)
        self.assertNotEqual(values_by_time[times[hidden_two]], 30.0)
        self.assertEqual(values_by_time[times[after]], 40.0)

        predictor = {
            "all_arr_known": [
                {"month": hidden_one, "arr_b_usd": 20.0},
                {"month": after, "arr_b_usd": 40.0},
            ],
            "mom_growth": [
                {
                    "from_month": before,
                    "to_month": hidden_one,
                    "from_arr_b": 10.0,
                    "to_arr_b": 20.0,
                    "gap_months": 1,
                    "mom_pct": 999.0,
                    "annualized_pct": 999.0,
                },
                {
                    "from_month": hidden_one,
                    "to_month": hidden_two,
                    "from_arr_b": 20.0,
                    "to_arr_b": 30.0,
                    "gap_months": 1,
                    "mom_pct": 999.0,
                    "annualized_pct": 999.0,
                },
                {
                    "from_month": hidden_two,
                    "to_month": after,
                    "from_arr_b": 30.0,
                    "to_arr_b": 40.0,
                    "gap_months": 1,
                    "mom_pct": 999.0,
                    "annualized_pct": 999.0,
                },
            ],
            "per_signal": [
                {
                    "points": [
                        {"month": hidden_one, "value": 20.0},
                        {"month": after, "value": 40.0},
                    ],
                    "forecast_method": (f"ARR[{hidden_one}]=20B ARR[{hidden_two}]=30B"),
                }
            ],
        }

        snapshot._scrub_predictor(predictor, hidden, span_model)

        self.assertEqual(
            predictor["all_arr_known"], [{"month": after, "arr_b_usd": 40.0}]
        )
        self.assertEqual(
            predictor["per_signal"][0]["points"],
            [{"month": after, "value": 40.0}],
        )
        self.assertNotIn(hidden_one, predictor["per_signal"][0]["forecast_method"])
        self.assertNotIn(hidden_two, predictor["per_signal"][0]["forecast_method"])
        self.assertNotEqual(predictor["mom_growth"][0]["mom_pct"], 999.0)
        self.assertNotEqual(predictor["mom_growth"][1]["mom_pct"], 999.0)
        self.assertNotEqual(predictor["mom_growth"][2]["mom_pct"], 999.0)

    def test_expired_arr_pin_does_not_override_the_ensemble(self) -> None:
        from anthropic_arr import compute

        ramped = {
            "ensemble": {
                "predicted_arr_b": 70.0,
                "ci_low": 60.0,
                "ci_high": 80.0,
                "weighting": "ensemble",
            }
        }

        compute._apply_pinned_arr(ramped, "2026-08")

        self.assertEqual(compute.PINNED_ARR, {})
        self.assertEqual(ramped["ensemble"]["predicted_arr_b"], 70.0)
        self.assertNotIn("unpinned_arr_b", ramped)

    def test_forecast_method_is_fully_redacted_when_training_points_are_hidden(
        self,
    ) -> None:
        hidden_month = "2025-01"
        public_month = "2025-03"
        private_derived_method = (
            f"momentum: ARR[{public_month}]=40.00B × (2.000)^(1/1mo)"
        )
        public_method = "OLS over public inputs"
        predictor = {
            "per_signal": [
                {
                    "points": [
                        {"month": hidden_month, "value": 20.0},
                        {"month": public_month, "value": 40.0},
                    ],
                    "forecast_method": private_derived_method,
                },
                {
                    "points": [{"month": public_month, "value": 40.0}],
                    "forecast_method": public_method,
                },
            ]
        }

        snapshot._scrub_predictor(predictor, frozenset({hidden_month}), None)

        private_signal, public_signal = predictor["per_signal"]
        self.assertEqual(
            private_signal["forecast_method"],
            "Forecast uses redacted private ARR anchors.",
        )
        self.assertEqual(public_signal["forecast_method"], public_method)

    def test_capture_declares_the_supported_schema_version(self) -> None:
        self.assertEqual(snapshot.SNAPSHOT_SCHEMA_VERSION, 1)

    def test_capture_emits_every_endpoint_the_publisher_requires(self) -> None:
        captured = {name for name, _path in ENDPOINTS}
        captured.discard("provider_throughput_history")
        captured.add(PTH_BY_MODEL_KEY)

        self.assertLessEqual(REQUIRED_ENDPOINTS, captured)

    def test_by_model_capture_reports_every_failed_model(self) -> None:
        body, failures = capture_by_model(
            _Client(),
            {"models": ["good-model", "broken-model"]},
        )

        self.assertEqual(set(body["by_model"]), {"good-model"})
        self.assertEqual(len(failures), 1)
        self.assertIn("broken-model", failures[0])

    def test_empty_model_roster_is_incomplete(self) -> None:
        _body, failures = capture_by_model(_Client(), {"models": []})

        self.assertEqual(failures, ["provider throughput returned no models"])

    def test_capture_process_fails_when_any_required_fetch_is_incomplete(self) -> None:
        with TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "capture.json"
            with (
                patch.object(
                    snapshot,
                    "ENDPOINTS",
                    [("provider_throughput_history", "/provider_throughput_history")],
                ),
                patch.object(snapshot.httpx, "Client", return_value=_CaptureClient()),
                patch("sys.argv", ["snapshot.py", str(destination)]),
            ):
                exit_code = snapshot.main()

        self.assertEqual(exit_code, 1)
