from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import Mock, patch

from anthropic_arr.openai_anchors import load_openai_anchor_records
from scripts import publish_snapshot as publish_snapshot_module
from scripts.publish_snapshot import (
    ANCHOR_MANIFEST_PATH,
    MAX_SNAPSHOT_BYTES,
    _load_complete_snapshot,
    publish_snapshot,
)

COMPLETE_ENDPOINTS = {
    "dashboard",
    "predictor_openai",
    "arr_curve",
    "arr_nowcast",
    "openrouter_market",
    "openrouter_market_fable",
    "vercel_gateway",
    "vercel_gateway_history",
    "sdk_downloads",
    "codex_wau",
    "curated_signals",
    "models_meta",
    "models_scores",
    "models_curated",
    "provider_throughput_by_model",
}


def write_capture(path: Path) -> None:
    records = load_openai_anchor_records(ANCHOR_MANIFEST_PATH)
    openai_known = [
        {
            "month": row["date"][:7],
            "arr_b": row["arr_bn"],
            "source": f"{row['classification']}:{row['source']}",
            "notes": row["url"],
        }
        for row in records
    ]
    curve_company = {
        "hist": [[1, 1.0]],
        "ext": [],
        "fanLo": [],
        "fanHi": [],
        "cps": [
            {
                "t": 1,
                "v": 1.0,
                "src": "source",
                "classification": "reported",
            }
        ],
        "counter": {"tLast": 1, "vLast": 1.0, "rMs": 0.0, "rMsExt": 0.0},
    }
    openai_curve_company = {
        **curve_company,
        "cps": [
            {
                "t": int(
                    datetime.strptime(row["date"], "%Y-%m-%d")
                    .replace(tzinfo=timezone.utc)
                    .timestamp()
                    * 1000
                ),
                "v": row["arr_bn"],
                "src": row["source"],
                "classification": row["classification"],
            }
            for row in records
        ],
    }
    dated_series = [[f"2026-08-{day:02d}", float(day)] for day in range(1, 9)]
    throughput_metric = {
        "unit": "unit",
        "providers": {
            "provider-a": [["2026-09-01", 1.0, 0.9, 1.1, 1]],
        },
        "market_median": [["2026-09-01", 1.0]],
    }
    bodies = {name: {"data": True} for name in COMPLETE_ENDPOINTS}
    bodies.update(
        {
            "dashboard": {
                "public_meta": {
                    "hero_arr_b": 1.0,
                    "hero_ci_low": 0.9,
                    "hero_ci_high": 1.1,
                },
                "predictor": {
                    "target_month": "2026-09",
                    "mom_growth": [
                        {
                            "from_month": "2026-08",
                            "to_month": "2026-09",
                            "gap_months": 1,
                            "mom_pct": 0.1,
                            "annualized_pct": 2.14,
                            "from_arr_b": 1.0,
                            "to_arr_b": 1.1,
                            "source": "predicted",
                        }
                    ],
                },
                "known_arr": [{"month": "2026-08", "arr_b": 1.0}],
            },
            "predictor_openai": {
                "all_arr_known": openai_known,
            },
            "arr_curve": {
                "companies": {
                    "anthropic": curve_company,
                    "openai": openai_curve_company,
                }
            },
            "arr_nowcast": {
                "available": True,
                "companies": {
                    "anthropic": {"available": True, "nowcast": {"rMs": 0.01}},
                    "openai": {"available": True, "nowcast": {"rMs": 0.01}},
                },
            },
            "openrouter_market": {
                "latest_date": "2026-09-01",
                "daily_totals": [["2026-09-01", 1.0]],
                "ma7": [["2026-09-01", 1.0]],
                "watchlist": {"openai/gpt": [["2026-09-01", 1.0]]},
                "top_models_latest": [{"slug": "model-a", "tokens_b": 1.0}],
                "top_models_7d": [{"slug": "model-a", "tokens_b": 1.0}],
            },
            "openrouter_market_fable": {
                "prefix": "anthropic/claude-5-fable",
                "series": [["2026-09-01", 1.0]],
            },
            "vercel_gateway": {
                "as_of": "2026-09-01",
                "snapshots": [
                    {
                        "date": "2026-09-01",
                        "token_share": [["model-a", 1.0]],
                        "spend_share": [["model-a", 1.0]],
                    }
                ],
            },
            "vercel_gateway_history": {
                "history": {
                    "cost": {
                        "days": ["2026-09-01"],
                        "series": {"model-a": [1.0]},
                    },
                    "tokens": {
                        "days": ["2026-09-01"],
                        "series": {"model-a": [1.0]},
                    },
                }
            },
            "sdk_downloads": {
                "as_of": "2026-09-01",
                "npm": {"@anthropic-ai/sdk": dated_series, "openai": dated_series},
                "pypi": {"anthropic": dated_series, "openai": dated_series},
            },
            "codex_wau": {
                "as_of": "2026-09-01",
                "source": "reviewed public disclosures",
                "series": [["2026-09-01", 1.0]],
                "events": [["2026-09-01", "launch"]],
            },
            "curated_signals": {"as_of": None, "kol": [], "reports": []},
            "models_meta": {
                "as_of": "2026-09-01",
                "models": {
                    "model-a": {
                        "lab": "openai",
                        "display": "Model A",
                        "ctx": 1,
                        "modalities_in": ["text"],
                        "price_in": 1.0,
                        "price_out": 1.0,
                        "providers": [],
                    }
                },
            },
            "models_scores": {
                "as_of": "2026-09-01",
                "scores": {
                    "arena": {"model-a": {"overall": {"rating": 1.0}}},
                    "aa": {},
                    "vals": {},
                },
            },
            "models_curated": {
                "as_of": "2026-09-01",
                "method": "reviewed public results",
                "benchmarks": {
                    "benchmark-a": {
                        "label": "Benchmark A",
                        "unit": "%",
                        "cat": "Code",
                        "note": "",
                    }
                },
                "scores": [
                    {
                        "slug": "model-a",
                        "bench": "benchmark-a",
                        "v": 1.0,
                        "cfg": "default",
                        "comparable": True,
                        "official": True,
                        "src": "https://example.test/benchmark",
                        "date": "2026-09-01",
                        "note": "reviewed result",
                    }
                ],
                "extra_models": {},
            },
            "provider_throughput_by_model": {
                "as_of": "2026-09-01",
                "granularity": "day",
                "models": ["model-a"],
                "by_model": {
                    "model-a": {
                        "metrics": {
                            name: throughput_metric
                            for name in ("tps", "latency", "tail")
                        }
                    }
                },
            },
        }
    )
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "captured_at": "2026-09-01T06:00:00+00:00",
                "endpoints": {
                    name: {"status": 200, "ok": True, "body": bodies[name]}
                    for name in COMPLETE_ENDPOINTS
                },
            }
        ),
        encoding="utf-8",
    )


class PlaySnapshotPublisherTest(TestCase):
    def test_publish_requires_an_https_api_base(self) -> None:
        with TemporaryDirectory() as temp_dir:
            capture = Path(temp_dir) / "capture.json"
            write_capture(capture)
            send = Mock()
            read = Mock()

            with self.assertRaisesRegex(ValueError, "HTTPS"):
                publish_snapshot(
                    capture,
                    api_base="http://api.example.test",
                    api_key="admin-key",
                    dry_run_only=True,
                    send=send,
                    read=read,
                )

        send.assert_not_called()
        read.assert_not_called()

    def test_publisher_uses_the_validated_manifest_loader(self) -> None:
        with TemporaryDirectory() as temp_dir:
            capture = Path(temp_dir) / "capture.json"
            write_capture(capture)
            payload = json.loads(ANCHOR_MANIFEST_PATH.read_text(encoding="utf-8"))
            payload["schema_version"] = 1
            manifest = Path(temp_dir) / "anchors.json"
            manifest.write_text(json.dumps(payload), encoding="utf-8")

            with (
                patch.object(publish_snapshot_module, "ANCHOR_MANIFEST_PATH", manifest),
                self.assertRaisesRegex(ValueError, "schema_version must be 2"),
            ):
                _load_complete_snapshot(capture)

    def test_openai_curve_anchors_must_exactly_match_the_manifest(self) -> None:
        mutations = {
            "timestamp": lambda cps: cps[0].update(t=cps[0]["t"] + 1),
            "value": lambda cps: cps[0].update(v=cps[0]["v"] + 0.1),
            "source": lambda cps: cps[0].update(src="stale source"),
            "classification": lambda cps: cps[0].update(
                classification=(
                    "third_party_estimate"
                    if cps[0]["classification"] == "reported"
                    else "reported"
                )
            ),
            "complete set": lambda cps: cps.pop(0),
        }
        for field, mutate in mutations.items():
            with self.subTest(field=field), TemporaryDirectory() as temp_dir:
                capture = Path(temp_dir) / "capture.json"
                write_capture(capture)
                payload = json.loads(capture.read_text(encoding="utf-8"))
                curve_anchors = payload["endpoints"]["arr_curve"]["body"]["companies"][
                    "openai"
                ]["cps"]
                mutate(curve_anchors)
                capture.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaisesRegex(ValueError, "curve anchors.*manifest"):
                    _load_complete_snapshot(capture)

    def test_dashboard_requires_complete_finite_arr_ui_fields(self) -> None:
        mutations = {
            "hero ARR": lambda body: body["public_meta"].update(
                hero_arr_b=float("nan")
            ),
            "hero low CI": lambda body: body["public_meta"].pop("hero_ci_low"),
            "hero high CI": lambda body: body["public_meta"].update(
                hero_ci_high=float("inf")
            ),
            "Anthropic predictor": lambda body: body.update(predictor={}),
            "Anthropic predictor row": lambda body: body["predictor"].update(
                mom_growth=[None]
            ),
            "known ARR row": lambda body: body.update(
                known_arr=[{"month": "2026-08", "arr_b": float("nan")}]
            ),
        }
        for field, mutate in mutations.items():
            with self.subTest(field=field), TemporaryDirectory() as temp_dir:
                capture = Path(temp_dir) / "capture.json"
                write_capture(capture)
                payload = json.loads(capture.read_text(encoding="utf-8"))
                mutate(payload["endpoints"]["dashboard"]["body"])
                capture.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaisesRegex(ValueError, "dashboard"):
                    _load_complete_snapshot(capture)

    def test_models_meta_requires_the_fields_consumed_by_the_ui(self) -> None:
        mutations = {
            "context": lambda model: model.pop("ctx"),
            "modalities": lambda model: model.update(modalities_in=[None]),
            "input price": lambda model: model.update(price_in=float("nan")),
            "output price": lambda model: model.update(price_out="unknown"),
        }
        for field, mutate in mutations.items():
            with self.subTest(field=field), TemporaryDirectory() as temp_dir:
                capture = Path(temp_dir) / "capture.json"
                write_capture(capture)
                payload = json.loads(capture.read_text(encoding="utf-8"))
                model = payload["endpoints"]["models_meta"]["body"]["models"]["model-a"]
                mutate(model)
                capture.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaisesRegex(ValueError, "models_meta"):
                    _load_complete_snapshot(capture)

    def test_models_meta_requires_complete_provider_rows(self) -> None:
        valid_provider = {
            "provider": "provider-a",
            "p50_tps": 1.0,
            "p99_tps": None,
            "p50_lat_ms": 2.0,
            "p99_lat_ms": 3.0,
        }
        mutations = {
            "provider name": lambda row: row.update(provider=1),
            "missing metric": lambda row: row.pop("p99_tps"),
            "non-numeric metric": lambda row: row.update(p50_tps="fast"),
            "non-finite metric": lambda row: row.update(p50_lat_ms=float("nan")),
        }
        for field, mutate in mutations.items():
            with self.subTest(field=field), TemporaryDirectory() as temp_dir:
                capture = Path(temp_dir) / "capture.json"
                write_capture(capture)
                payload = json.loads(capture.read_text(encoding="utf-8"))
                provider = valid_provider.copy()
                mutate(provider)
                payload["endpoints"]["models_meta"]["body"]["models"]["model-a"][
                    "providers"
                ] = [provider]
                capture.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaisesRegex(ValueError, "models_meta"):
                    _load_complete_snapshot(capture)

    def test_models_curated_requires_ui_row_shapes_and_finite_values(self) -> None:
        mutations = {
            "benchmark": lambda body: body.update(benchmarks={"benchmark-a": {}}),
            "score row": lambda body: body.update(scores=[None]),
            "score value": lambda body: body["scores"][0].update(v=float("inf")),
        }
        for field, mutate in mutations.items():
            with self.subTest(field=field), TemporaryDirectory() as temp_dir:
                capture = Path(temp_dir) / "capture.json"
                write_capture(capture)
                payload = json.loads(capture.read_text(encoding="utf-8"))
                mutate(payload["endpoints"]["models_curated"]["body"])
                capture.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaisesRegex(ValueError, "models_curated"):
                    _load_complete_snapshot(capture)

    def test_models_curated_requires_complete_extra_model_rows(self) -> None:
        valid_extra = {
            "lab": "Example Lab",
            "display": "Example Model",
            "access": "API",
            "src": "https://example.test/model",
            "released": "2026-09-01",
            "ctx": 1,
            "price_in": None,
            "price_out": 1.0,
        }
        bad_rows = {
            "non-object": None,
            "missing string": {
                key: value for key, value in valid_extra.items() if key != "access"
            },
            "non-string field": {**valid_extra, "src": 1},
            "non-numeric context": {**valid_extra, "ctx": "large"},
            "missing price": {
                key: value for key, value in valid_extra.items() if key != "price_in"
            },
            "non-finite price": {**valid_extra, "price_out": float("inf")},
        }
        for field, extra in bad_rows.items():
            with self.subTest(field=field), TemporaryDirectory() as temp_dir:
                capture = Path(temp_dir) / "capture.json"
                write_capture(capture)
                payload = json.loads(capture.read_text(encoding="utf-8"))
                payload["endpoints"]["models_curated"]["body"]["extra_models"] = {
                    "example/model": extra
                }
                capture.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaisesRegex(ValueError, "models_curated"):
                    _load_complete_snapshot(capture)

    def test_provider_metrics_require_finite_point_tuples(self) -> None:
        mutations = {
            "provider tuple length": lambda metric: metric.update(
                providers={"provider-a": [["2026-09-01", 1.0]]}
            ),
            "provider tuple value": lambda metric: metric["providers"]["provider-a"][
                0
            ].__setitem__(1, float("nan")),
            "market tuple": lambda metric: metric.update(
                market_median=[["2026-09-01", float("inf")]]
            ),
        }
        for field, mutate in mutations.items():
            with self.subTest(field=field), TemporaryDirectory() as temp_dir:
                capture = Path(temp_dir) / "capture.json"
                write_capture(capture)
                payload = json.loads(capture.read_text(encoding="utf-8"))
                metric = payload["endpoints"]["provider_throughput_by_model"]["body"][
                    "by_model"
                ]["model-a"]["metrics"]["tps"]
                mutate(metric)
                capture.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaisesRegex(ValueError, "model-a:tps"):
                    _load_complete_snapshot(capture)

    def test_provider_metrics_require_string_units(self) -> None:
        mutations = {
            "missing": lambda metric: metric.pop("unit"),
            "non-string": lambda metric: metric.update(unit=1),
        }
        for metric_name in ("tps", "latency", "tail"):
            for field, mutate in mutations.items():
                with (
                    self.subTest(metric=metric_name, field=field),
                    TemporaryDirectory() as temp_dir,
                ):
                    capture = Path(temp_dir) / "capture.json"
                    write_capture(capture)
                    payload = json.loads(capture.read_text(encoding="utf-8"))
                    metric = payload["endpoints"]["provider_throughput_by_model"][
                        "body"
                    ]["by_model"]["model-a"]["metrics"][metric_name]
                    mutate(metric)
                    capture.write_text(json.dumps(payload), encoding="utf-8")

                    with self.assertRaisesRegex(ValueError, f"model-a:{metric_name}"):
                        _load_complete_snapshot(capture)

    def test_numeric_ui_series_reject_non_finite_values(self) -> None:
        with TemporaryDirectory() as temp_dir:
            capture = Path(temp_dir) / "capture.json"
            write_capture(capture)
            payload = json.loads(capture.read_text(encoding="utf-8"))
            payload["endpoints"]["openrouter_market"]["body"]["daily_totals"][0][1] = (
                float("nan")
            )
            capture.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "openrouter_market"):
                _load_complete_snapshot(capture)

    def test_openrouter_top_models_require_ui_row_shape(self) -> None:
        bad_rows = {
            "non-object": None,
            "missing slug": {"tokens_b": 1.0},
            "non-string slug": {"slug": 1, "tokens_b": 1.0},
            "non-numeric tokens": {"slug": "model-a", "tokens_b": "many"},
            "non-finite tokens": {"slug": "model-a", "tokens_b": float("inf")},
        }
        for collection in ("top_models_latest", "top_models_7d"):
            for field, bad_row in bad_rows.items():
                with (
                    self.subTest(collection=collection, field=field),
                    TemporaryDirectory() as temp_dir,
                ):
                    capture = Path(temp_dir) / "capture.json"
                    write_capture(capture)
                    payload = json.loads(capture.read_text(encoding="utf-8"))
                    payload["endpoints"]["openrouter_market"]["body"][collection] = [
                        bad_row
                    ]
                    capture.write_text(json.dumps(payload), encoding="utf-8")

                    with self.assertRaisesRegex(ValueError, "openrouter_market"):
                        _load_complete_snapshot(capture)

    def test_refuses_required_ui_endpoints_without_their_minimum_schema(
        self,
    ) -> None:
        endpoints = (
            "arr_nowcast",
            "openrouter_market",
            "openrouter_market_fable",
            "vercel_gateway",
            "vercel_gateway_history",
            "sdk_downloads",
            "codex_wau",
            "curated_signals",
            "models_curated",
        )
        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint), TemporaryDirectory() as temp_dir:
                capture = Path(temp_dir) / "capture.json"
                write_capture(capture)
                payload = json.loads(capture.read_text(encoding="utf-8"))
                payload["endpoints"][endpoint]["body"] = {}
                capture.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaisesRegex(ValueError, endpoint):
                    _load_complete_snapshot(capture)

    def test_vercel_history_days_must_be_strings(self) -> None:
        for metric_name in ("cost", "tokens"):
            with self.subTest(metric=metric_name), TemporaryDirectory() as temp_dir:
                capture = Path(temp_dir) / "capture.json"
                write_capture(capture)
                payload = json.loads(capture.read_text(encoding="utf-8"))
                payload["endpoints"]["vercel_gateway_history"]["body"]["history"][
                    metric_name
                ]["days"] = [1]
                capture.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaisesRegex(ValueError, "vercel_gateway_history"):
                    _load_complete_snapshot(capture)

    def test_codex_events_must_be_string_pairs(self) -> None:
        bad_events = {
            "non-array": "2026-09-01",
            "wrong length": ["2026-09-01"],
            "non-string date": [1, "launch"],
            "non-string label": ["2026-09-01", 1],
        }
        for field, event in bad_events.items():
            with self.subTest(field=field), TemporaryDirectory() as temp_dir:
                capture = Path(temp_dir) / "capture.json"
                write_capture(capture)
                payload = json.loads(capture.read_text(encoding="utf-8"))
                payload["endpoints"]["codex_wau"]["body"]["events"] = [event]
                capture.write_text(json.dumps(payload), encoding="utf-8")

                with self.assertRaisesRegex(ValueError, "codex_wau"):
                    _load_complete_snapshot(capture)

    def test_refuses_semantically_empty_model_scores(self) -> None:
        with TemporaryDirectory() as temp_dir:
            capture = Path(temp_dir) / "capture.json"
            write_capture(capture)
            payload = json.loads(capture.read_text(encoding="utf-8"))
            payload["endpoints"]["models_scores"]["body"]["scores"] = {
                "arena": {},
                "aa": {},
                "vals": {},
            }
            capture.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "models_scores"):
                _load_complete_snapshot(capture)

    def test_refuses_semantically_empty_provider_metric_series(self) -> None:
        with TemporaryDirectory() as temp_dir:
            capture = Path(temp_dir) / "capture.json"
            write_capture(capture)
            payload = json.loads(capture.read_text(encoding="utf-8"))
            metric = payload["endpoints"]["provider_throughput_by_model"]["body"][
                "by_model"
            ]["model-a"]["metrics"]["tps"]
            metric["providers"] = {}
            metric["market_median"] = []
            capture.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "model-a:tps"):
                _load_complete_snapshot(capture)

    def test_new_capture_is_dry_run_published_and_read_back(self) -> None:
        with TemporaryDirectory() as temp_dir:
            capture = Path(temp_dir) / "capture.json"
            write_capture(capture)
            requests: list[tuple[str, dict, dict[str, str]]] = []
            reads: list[str] = []

            def send(url: str, payload: dict, headers: dict[str, str]) -> dict:
                requests.append((url, payload, headers))
                return {"id": len(requests), "slug": "ailab-arr", **payload}

            def read(url: str, _headers: dict[str, str]) -> dict | None:
                reads.append(url)
                if len(requests) < 2:
                    return None
                return {"id": 2, "slug": "ailab-arr", **requests[-1][1]}

            result = publish_snapshot(
                capture,
                api_base="https://api.example.test",
                api_key="admin-key",
                send=send,
                read=read,
            )

        self.assertEqual(
            [request[0] for request in requests],
            [
                "https://api.example.test/v1/plays/ailab-arr/snapshot?dry_run=true",
                "https://api.example.test/v1/plays/ailab-arr/snapshot",
            ],
        )
        self.assertIs(requests[0][1], requests[1][1])
        self.assertEqual(requests[0][2]["Authorization"], "Bearer admin-key")
        self.assertEqual(
            reads,
            [
                "https://api.example.test/v1/plays/ailab-arr/snapshot",
                "https://api.example.test/v1/plays/ailab-arr/snapshot?"
                "version=2026-09-01T06%3A00%3A00%2B00%3A00",
                "https://api.example.test/v1/plays/ailab-arr/snapshot?"
                "version=2026-09-01T06%3A00%3A00%2B00%3A00",
            ],
        )
        self.assertEqual(result["id"], 2)

    def test_same_version_revision_and_content_is_an_idempotent_noop(self) -> None:
        with TemporaryDirectory() as temp_dir:
            capture = Path(temp_dir) / "capture.json"
            write_capture(capture)
            snapshot = json.loads(capture.read_text(encoding="utf-8"))
            existing = {
                "id": 41,
                "slug": "ailab-arr",
                "version": snapshot["captured_at"],
                "rev": 1,
                "status": "published",
                "snapshot": snapshot,
                "created_by": "ai-monetization-tracker",
                "created_at": "2026-09-01T06:01:00+00:00",
            }
            methods: list[str] = []

            class Response:
                def __enter__(self) -> Response:
                    return self

                def __exit__(self, *_args: object) -> None:
                    return None

                def read(self) -> bytes:
                    return json.dumps(existing).encode("utf-8")

            def urlopen(request: object, *, timeout: int) -> Response:
                self.assertEqual(timeout, 60)
                methods.append(request.get_method())  # type: ignore[attr-defined]
                return Response()

            with patch("scripts.publish_snapshot.urlopen", urlopen):
                result = publish_snapshot(
                    capture,
                    api_base="https://api.example.test",
                    api_key="admin-key",
                )

        self.assertEqual(methods, ["GET"])
        self.assertEqual(result["id"], 41)

    def test_fails_when_post_publish_readback_does_not_match(self) -> None:
        with TemporaryDirectory() as temp_dir:
            capture = Path(temp_dir) / "capture.json"
            write_capture(capture)
            read_count = 0

            def read(_url: str, _headers: dict[str, str]) -> dict | None:
                nonlocal read_count
                read_count += 1
                if read_count == 1:
                    return None
                return {
                    "slug": "ailab-arr",
                    "version": "wrong-version",
                    "rev": 1,
                    "status": "published",
                    "snapshot": {},
                }

            with self.assertRaisesRegex(RuntimeError, "did not match API readback"):
                publish_snapshot(
                    capture,
                    api_base="https://api.example.test",
                    api_key="admin-key",
                    read=read,
                    send=lambda _url, payload, _headers: {
                        "id": 42,
                        "slug": "ailab-arr",
                        **payload,
                    },
                )

    def test_dry_run_only_does_not_publish_or_read_back(self) -> None:
        with TemporaryDirectory() as temp_dir:
            capture = Path(temp_dir) / "capture.json"
            write_capture(capture)
            sent: list[str] = []
            reads: list[str] = []

            result = publish_snapshot(
                capture,
                api_base="https://api.example.test",
                api_key="admin-key",
                dry_run_only=True,
                read=lambda url, _headers: reads.append(url) or None,
                send=lambda url, payload, _headers: (
                    sent.append(url) or {"id": 43, "slug": "ailab-arr", **payload}
                ),
            )

        self.assertEqual(len(reads), 2)
        self.assertEqual(
            sent, ["https://api.example.test/v1/plays/ailab-arr/snapshot?dry_run=true"]
        )
        self.assertEqual(result["id"], 43)

    def test_same_coordinate_with_different_content_requires_a_revision_bump(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            capture = Path(temp_dir) / "capture.json"
            write_capture(capture)
            snapshot = json.loads(capture.read_text(encoding="utf-8"))
            existing_snapshot = json.loads(json.dumps(snapshot))
            existing_snapshot["endpoints"]["dashboard"]["body"] = {"old": True}
            existing = {
                "slug": "ailab-arr",
                "version": snapshot["captured_at"],
                "rev": 1,
                "status": "published",
                "snapshot": existing_snapshot,
            }

            with self.assertRaisesRegex(ValueError, "different content"):
                publish_snapshot(
                    capture,
                    api_base="https://api.example.test",
                    api_key="admin-key",
                    read=lambda _url, _headers: existing,
                    send=lambda _url, _payload, _headers: {"id": 99},
                )

    def test_refuses_to_publish_behind_a_newer_revision(self) -> None:
        with TemporaryDirectory() as temp_dir:
            capture = Path(temp_dir) / "capture.json"
            write_capture(capture)
            snapshot = json.loads(capture.read_text(encoding="utf-8"))
            existing = {
                "slug": "ailab-arr",
                "version": snapshot["captured_at"],
                "rev": 3,
                "status": "published",
                "snapshot": snapshot,
            }

            with self.assertRaisesRegex(ValueError, "newer published revision"):
                publish_snapshot(
                    capture,
                    api_base="https://api.example.test",
                    api_key="admin-key",
                    rev=2,
                    read=lambda _url, _headers: existing,
                    send=lambda _url, _payload, _headers: {"id": 99},
                )

    def test_refuses_to_publish_an_older_capture_than_the_latest(self) -> None:
        with TemporaryDirectory() as temp_dir:
            capture = Path(temp_dir) / "capture.json"
            write_capture(capture)
            latest = {
                "slug": "ailab-arr",
                "version": "2026-09-02T06:00:00+00:00",
                "rev": 1,
                "status": "published",
                "snapshot": {},
            }

            with self.assertRaisesRegex(ValueError, "older than latest"):
                publish_snapshot(
                    capture,
                    api_base="https://api.example.test",
                    api_key="admin-key",
                    read=lambda _url, _headers: latest,
                    send=lambda _url, _payload, _headers: {"id": 99},
                )

    def test_refuses_to_publish_when_one_required_endpoint_is_missing(self) -> None:
        with TemporaryDirectory() as temp_dir:
            capture = Path(temp_dir) / "capture.json"
            write_capture(capture)
            payload = json.loads(capture.read_text(encoding="utf-8"))
            del payload["endpoints"]["models_meta"]
            capture.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "models_meta"):
                publish_snapshot(
                    capture,
                    api_base="https://api.example.test",
                    api_key="admin-key",
                )

    def test_refuses_to_publish_a_partial_endpoint_result(self) -> None:
        with TemporaryDirectory() as temp_dir:
            capture = Path(temp_dir) / "capture.json"
            write_capture(capture)
            payload = json.loads(capture.read_text(encoding="utf-8"))
            payload["endpoints"]["models_scores"] = {
                "status": 503,
                "ok": False,
                "body": {},
            }
            capture.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "models_scores"):
                publish_snapshot(
                    capture,
                    api_base="https://api.example.test",
                    api_key="admin-key",
                )

    def test_refuses_to_publish_a_future_dated_capture(self) -> None:
        with TemporaryDirectory() as temp_dir:
            capture = Path(temp_dir) / "capture.json"
            write_capture(capture)
            payload = json.loads(capture.read_text(encoding="utf-8"))
            payload["captured_at"] = "2100-01-01T00:00:00+00:00"
            capture.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "future"):
                publish_snapshot(
                    capture,
                    api_base="https://api.example.test",
                    api_key="admin-key",
                )

    def test_refuses_to_publish_a_null_endpoint_body(self) -> None:
        with TemporaryDirectory() as temp_dir:
            capture = Path(temp_dir) / "capture.json"
            write_capture(capture)
            payload = json.loads(capture.read_text(encoding="utf-8"))
            payload["endpoints"]["models_curated"]["body"] = None
            capture.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "models_curated"):
                publish_snapshot(
                    capture,
                    api_base="https://api.example.test",
                    api_key="admin-key",
                )

    def test_refuses_a_capture_without_the_supported_schema_version(self) -> None:
        with TemporaryDirectory() as temp_dir:
            capture = Path(temp_dir) / "capture.json"
            write_capture(capture)
            payload = json.loads(capture.read_text(encoding="utf-8"))
            del payload["schema_version"]
            capture.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "schema_version"):
                publish_snapshot(
                    capture,
                    api_base="https://api.example.test",
                    api_key="admin-key",
                )

    def test_refuses_a_partial_models_roster(self) -> None:
        with TemporaryDirectory() as temp_dir:
            capture = Path(temp_dir) / "capture.json"
            write_capture(capture)
            payload = json.loads(capture.read_text(encoding="utf-8"))
            body = payload["endpoints"]["provider_throughput_by_model"]["body"]
            body["models"].append("missing-model")
            capture.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "model roster"):
                publish_snapshot(
                    capture,
                    api_base="https://api.example.test",
                    api_key="admin-key",
                )

    def test_refuses_a_model_capture_with_missing_metrics(self) -> None:
        with TemporaryDirectory() as temp_dir:
            capture = Path(temp_dir) / "capture.json"
            write_capture(capture)
            payload = json.loads(capture.read_text(encoding="utf-8"))
            metrics = payload["endpoints"]["provider_throughput_by_model"]["body"][
                "by_model"
            ]["model-a"]["metrics"]
            del metrics["tail"]
            capture.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "model metrics"):
                publish_snapshot(
                    capture,
                    api_base="https://api.example.test",
                    api_key="admin-key",
                )

    def test_refuses_a_capture_built_from_a_stale_anchor_source(self) -> None:
        with TemporaryDirectory() as temp_dir:
            capture = Path(temp_dir) / "capture.json"
            write_capture(capture)
            payload = json.loads(capture.read_text(encoding="utf-8"))
            latest = payload["endpoints"]["predictor_openai"]["body"]["all_arr_known"][
                -1
            ]
            latest["arr_b"] = 40.0
            latest["source"] = "reported:stale process"
            capture.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "anchor manifest"):
                _load_complete_snapshot(capture)

    def test_refuses_a_capture_missing_a_historical_manifest_anchor(self) -> None:
        with TemporaryDirectory() as temp_dir:
            capture = Path(temp_dir) / "capture.json"
            write_capture(capture)
            payload = json.loads(capture.read_text(encoding="utf-8"))
            payload["endpoints"]["predictor_openai"]["body"]["all_arr_known"].pop(0)
            capture.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "anchor manifest"):
                _load_complete_snapshot(capture)

    def test_refuses_an_openai_curve_anchor_without_evidence_classification(
        self,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            capture = Path(temp_dir) / "capture.json"
            write_capture(capture)
            payload = json.loads(capture.read_text(encoding="utf-8"))
            del payload["endpoints"]["arr_curve"]["body"]["companies"]["openai"]["cps"][
                0
            ]["classification"]
            capture.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "classification"):
                publish_snapshot(
                    capture,
                    api_base="https://api.example.test",
                    api_key="admin-key",
                )

    def test_refuses_to_publish_an_oversized_capture(self) -> None:
        with TemporaryDirectory() as temp_dir:
            capture = Path(temp_dir) / "capture.json"
            write_capture(capture)
            payload = json.loads(capture.read_text(encoding="utf-8"))
            payload["endpoints"]["dashboard"]["body"]["oversized"] = (
                "x" * MAX_SNAPSHOT_BYTES
            )
            capture.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "over the"):
                publish_snapshot(
                    capture,
                    api_base="https://api.example.test",
                    api_key="admin-key",
                )
