#!/usr/bin/env python3
"""Validate and publish one immutable AI Lab ARR play snapshot."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anthropic_arr.openai_anchors import load_openai_anchor_records  # noqa: E402

MAX_SNAPSHOT_BYTES = 8 * 1024 * 1024
REQUIRED_ENDPOINTS = frozenset(
    {
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
)
PLAY_PATH = "/v1/plays/ailab-arr/snapshot"
ANCHOR_MANIFEST_PATH = PROJECT_ROOT / "config" / "openai_arr_anchors.json"
SendJson = Callable[[str, dict[str, Any], dict[str, str]], dict[str, Any]]
ReadJson = Callable[[str, dict[str, str]], dict[str, Any] | None]


def _is_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_optional_number(value: object) -> bool:
    return value is None or _is_number(value)


def _has_non_finite_number(value: object) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(_has_non_finite_number(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_non_finite_number(item) for item in value)
    return False


def _is_dated_series(value: object, *, minimum: int = 1) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= minimum
        and all(
            isinstance(point, list)
            and len(point) == 2
            and isinstance(point[0], str)
            and _is_number(point[1])
            for point in value
        )
    )


def _is_share_series(value: object) -> bool:
    return _is_dated_series(value)


def _is_top_model_row(value: object) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("slug"), str)
        and _is_number(value.get("tokens_b"))
    )


def _is_string_pair(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(part, str) for part in value)
    )


def _is_curve_point(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(_is_number(part) for part in value)
    )


def _is_curve_anchor(value: object) -> bool:
    return (
        isinstance(value, dict)
        and _is_number(value.get("t"))
        and _is_number(value.get("v"))
        and isinstance(value.get("src"), str)
    )


def _is_mom_growth_row(value: object) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("from_month"), str)
        and isinstance(value.get("to_month"), str)
        and _is_number(value.get("gap_months"))
        and _is_number(value.get("mom_pct"))
        and _is_optional_number(value.get("annualized_pct"))
        and _is_optional_number(value.get("from_arr_b"))
        and _is_optional_number(value.get("to_arr_b"))
        and isinstance(value.get("source"), str)
    )


def _is_extra_model(value: object) -> bool:
    return (
        isinstance(value, dict)
        and all(
            isinstance(value.get(field), str)
            for field in ("lab", "display", "access", "src", "released")
        )
        and _is_number(value.get("ctx"))
        and "price_in" in value
        and _is_optional_number(value.get("price_in"))
        and "price_out" in value
        and _is_optional_number(value.get("price_out"))
    )


def _is_model_provider(value: object) -> bool:
    metric_fields = ("p50_tps", "p99_tps", "p50_lat_ms", "p99_lat_ms")
    return (
        isinstance(value, dict)
        and isinstance(value.get("provider"), str)
        and all(
            field in value and _is_optional_number(value[field])
            for field in metric_fields
        )
    )


def _is_provider_point(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 5
        and isinstance(value[0], str)
        and all(_is_number(part) for part in value[1:])
    )


def _is_provider_reference_point(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and _is_number(value[1])
    )


def _require_manifest_anchors(
    predictor: dict[str, Any], records: list[dict[str, Any]]
) -> None:
    expected_by_month = {row["date"][:7]: row for row in records}
    known = predictor.get("all_arr_known")
    if not isinstance(known, list):
        raise ValueError("snapshot predictor_openai has no ARR anchors")
    actual_by_month: dict[str, dict[str, Any]] = {}
    for row in known:
        if not isinstance(row, dict) or not isinstance(row.get("month"), str):
            raise ValueError(
                "snapshot OpenAI ARR anchors do not match the reviewed anchor manifest"
            )
        month = row["month"]
        if month in actual_by_month:
            raise ValueError(
                "snapshot OpenAI ARR anchors do not match the reviewed anchor manifest"
            )
        actual_by_month[month] = row
    if set(actual_by_month) != set(expected_by_month):
        raise ValueError(
            "snapshot OpenAI ARR anchors do not match the reviewed anchor manifest"
        )
    for month, expected in expected_by_month.items():
        actual = actual_by_month[month]
        if (
            actual.get("arr_b") != expected["arr_bn"]
            or actual.get("source")
            != f"{expected['classification']}:{expected['source']}"
            or expected["url"] not in str(actual.get("notes") or "")
        ):
            raise ValueError(
                "snapshot OpenAI ARR anchors do not match the reviewed anchor manifest"
            )


def _require_manifest_curve_anchors(
    anchors: object, records: list[dict[str, Any]]
) -> None:
    expected = [
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
    ]
    if anchors != expected:
        raise ValueError(
            "snapshot OpenAI curve anchors do not match the reviewed anchor manifest"
        )


def _parse_capture_time(value: object) -> datetime:
    if not isinstance(value, str):
        raise TypeError("snapshot captured_at must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("snapshot captured_at must be valid ISO-8601") from exc
    if parsed.tzinfo is None:
        raise ValueError("snapshot captured_at must include a timezone")
    return parsed.astimezone(timezone.utc)


def _load_complete_snapshot(path: Path) -> dict[str, Any]:
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(snapshot, dict):
        raise TypeError("snapshot root must be an object")
    if snapshot.get("schema_version") != 1:
        raise ValueError("snapshot schema_version must be 1")

    captured = _parse_capture_time(snapshot.get("captured_at"))
    if captured > datetime.now(timezone.utc) + timedelta(minutes=5):
        raise ValueError("snapshot captured_at is in the future")

    endpoints = snapshot.get("endpoints")
    if not isinstance(endpoints, dict) or not endpoints:
        raise ValueError("snapshot endpoints must be a non-empty object")
    missing = sorted(REQUIRED_ENDPOINTS.difference(endpoints))
    if missing:
        raise ValueError(
            "snapshot is missing required endpoints: " + ", ".join(missing)
        )
    failures = [
        name
        for name, result in endpoints.items()
        if not isinstance(result, dict)
        or result.get("ok") is not True
        or result.get("status") != 200
        or not isinstance(result.get("body"), dict)
    ]
    if failures:
        raise ValueError(
            "refusing to publish a partial capture; failed endpoints: "
            + ", ".join(sorted(failures))
        )

    dashboard = endpoints["dashboard"]["body"]
    public_meta = dashboard.get("public_meta")
    known_arr = dashboard.get("known_arr")
    dashboard_predictor = dashboard.get("predictor")
    mom_growth = (
        dashboard_predictor.get("mom_growth")
        if isinstance(dashboard_predictor, dict)
        else None
    )
    if (
        not isinstance(public_meta, dict)
        or not _is_number(public_meta.get("hero_arr_b"))
        or not _is_number(public_meta.get("hero_ci_low"))
        or not _is_number(public_meta.get("hero_ci_high"))
        or not isinstance(dashboard_predictor, dict)
        or not isinstance(dashboard_predictor.get("target_month"), str)
        or not isinstance(mom_growth, list)
        or not mom_growth
        or any(not _is_mom_growth_row(row) for row in mom_growth)
        or not isinstance(known_arr, list)
        or not known_arr
        or any(
            not isinstance(row, dict)
            or not isinstance(row.get("month"), str)
            or not _is_number(row.get("arr_b"))
            for row in known_arr
        )
    ):
        raise ValueError("snapshot dashboard has an invalid body")
    predictor = endpoints["predictor_openai"]["body"]
    if (
        not isinstance(predictor.get("all_arr_known"), list)
        or not predictor["all_arr_known"]
    ):
        raise ValueError("snapshot predictor_openai has no ARR anchors")
    manifest_records = load_openai_anchor_records(ANCHOR_MANIFEST_PATH)
    _require_manifest_anchors(predictor, manifest_records)

    models_meta = endpoints["models_meta"]["body"]
    model_roster = models_meta.get("models")
    if (
        not isinstance(models_meta.get("as_of"), str)
        or not isinstance(model_roster, dict)
        or not model_roster
        or any(
            not isinstance(model, dict)
            or not isinstance(model.get("lab"), str)
            or not isinstance(model.get("display"), str)
            or not _is_number(model.get("ctx"))
            or not isinstance(model.get("modalities_in"), list)
            or any(not isinstance(modality, str) for modality in model["modalities_in"])
            or "price_in" not in model
            or not _is_optional_number(model.get("price_in"))
            or "price_out" not in model
            or not _is_optional_number(model.get("price_out"))
            or not isinstance(model.get("providers"), list)
            or any(not _is_model_provider(provider) for provider in model["providers"])
            for model in model_roster.values()
        )
    ):
        raise ValueError("snapshot models_meta has an invalid body")
    models_scores = endpoints["models_scores"]["body"]
    scores = models_scores.get("scores")
    if (
        not isinstance(models_scores.get("as_of"), str)
        or not isinstance(scores, dict)
        or any(
            not isinstance(scores.get(name), dict) for name in ("arena", "aa", "vals")
        )
        or not any(scores[name] for name in ("arena", "aa", "vals"))
    ):
        raise ValueError("snapshot models_scores has an invalid body")
    arr_curve = endpoints["arr_curve"]["body"]
    companies = arr_curve.get("companies")
    if not isinstance(companies, dict) or not {"anthropic", "openai"} <= set(companies):
        raise ValueError("snapshot arr_curve has no complete company set")
    for company_name in ("anthropic", "openai"):
        company = companies[company_name]
        if not isinstance(company, dict) or any(
            not isinstance(company.get(key), list)
            for key in ("hist", "ext", "fanLo", "fanHi", "cps")
        ):
            raise ValueError(f"snapshot arr_curve {company_name} series are incomplete")
        if not company["hist"] or not company["cps"]:
            raise ValueError(
                f"snapshot arr_curve {company_name} has no anchors/history"
            )
        if any(
            not _is_curve_point(point)
            for key in ("hist", "ext", "fanLo", "fanHi")
            for point in company[key]
        ) or any(not _is_curve_anchor(anchor) for anchor in company["cps"]):
            raise ValueError(f"snapshot arr_curve {company_name} has invalid points")
        if company_name == "openai" and any(
            not isinstance(anchor, dict)
            or anchor.get("classification") not in {"reported", "third_party_estimate"}
            for anchor in company["cps"]
        ):
            raise ValueError(
                "snapshot OpenAI curve anchor classification is missing or invalid"
            )
    _require_manifest_curve_anchors(companies["openai"]["cps"], manifest_records)
    nowcast = endpoints["arr_nowcast"]["body"]
    nowcast_companies = nowcast.get("companies")
    if (
        not isinstance(nowcast.get("available"), bool)
        or not isinstance(nowcast_companies, dict)
        or any(
            not isinstance(nowcast_companies.get(name), dict)
            or not isinstance(nowcast_companies[name].get("available"), bool)
            for name in ("anthropic", "openai")
        )
    ):
        raise ValueError("snapshot arr_nowcast has an invalid body")

    market = endpoints["openrouter_market"]["body"]
    watchlist = market.get("watchlist")
    if (
        not isinstance(market.get("latest_date"), str)
        or not _is_dated_series(market.get("daily_totals"))
        or not _is_dated_series(market.get("ma7"))
        or not isinstance(watchlist, dict)
        or not watchlist
        or any(not _is_dated_series(series) for series in watchlist.values())
        or any(
            not isinstance(market.get(name), list)
            or not market[name]
            or any(not _is_top_model_row(row) for row in market[name])
            for name in ("top_models_latest", "top_models_7d")
        )
    ):
        raise ValueError("snapshot openrouter_market has an invalid body")

    fable = endpoints["openrouter_market_fable"]["body"]
    if not isinstance(fable.get("prefix"), str) or not _is_dated_series(
        fable.get("series")
    ):
        raise ValueError("snapshot openrouter_market_fable has an invalid body")

    gateway = endpoints["vercel_gateway"]["body"]
    snapshots = gateway.get("snapshots")
    if (
        not isinstance(gateway.get("as_of"), str)
        or not isinstance(snapshots, list)
        or not snapshots
        or any(
            not isinstance(snapshot, dict)
            or not isinstance(snapshot.get("date"), str)
            or not _is_share_series(snapshot.get("token_share"))
            or not _is_share_series(snapshot.get("spend_share"))
            for snapshot in snapshots
        )
    ):
        raise ValueError("snapshot vercel_gateway has an invalid body")

    gateway_history = endpoints["vercel_gateway_history"]["body"].get("history")
    if not isinstance(gateway_history, dict):
        raise ValueError("snapshot vercel_gateway_history has an invalid body")
    for metric_name in ("cost", "tokens"):
        metric = gateway_history.get(metric_name)
        if (
            not isinstance(metric, dict)
            or not isinstance(metric.get("days"), list)
            or not metric["days"]
            or any(not isinstance(day, str) for day in metric["days"])
            or not isinstance(metric.get("series"), dict)
            or not metric["series"]
            or any(
                not isinstance(series, list)
                or len(series) != len(metric["days"])
                or not all(_is_number(value) for value in series)
                for series in metric["series"].values()
            )
        ):
            raise ValueError("snapshot vercel_gateway_history has an invalid body")

    sdk = endpoints["sdk_downloads"]["body"]
    sdk_series = {
        "npm": ("@anthropic-ai/sdk", "openai"),
        "pypi": ("anthropic", "openai"),
    }
    if not isinstance(sdk.get("as_of"), str):
        raise ValueError("snapshot sdk_downloads has an invalid body")
    for registry, packages in sdk_series.items():
        registry_rows = sdk.get(registry)
        if not isinstance(registry_rows, dict) or any(
            not _is_dated_series(registry_rows.get(package), minimum=8)
            for package in packages
        ):
            raise ValueError("snapshot sdk_downloads has an invalid body")

    codex = endpoints["codex_wau"]["body"]
    if (
        not isinstance(codex.get("as_of"), str)
        or not isinstance(codex.get("source"), str)
        or not _is_dated_series(codex.get("series"))
        or not isinstance(codex.get("events"), list)
        or any(not _is_string_pair(event) for event in codex["events"])
    ):
        raise ValueError("snapshot codex_wau has an invalid body")

    curated_signals = endpoints["curated_signals"]["body"]
    if (
        "as_of" not in curated_signals
        or not isinstance(curated_signals.get("kol"), list)
        or not isinstance(curated_signals.get("reports"), list)
    ):
        raise ValueError("snapshot curated_signals has an invalid body")

    models_curated = endpoints["models_curated"]["body"]
    benchmarks = models_curated.get("benchmarks")
    curated_scores = models_curated.get("scores")
    if (
        not isinstance(models_curated.get("as_of"), str)
        or not isinstance(models_curated.get("method"), str)
        or not isinstance(benchmarks, dict)
        or not benchmarks
        or any(
            not isinstance(row, dict)
            or any(
                not isinstance(row.get(field), str)
                for field in ("label", "unit", "cat", "note")
            )
            for row in benchmarks.values()
        )
        or not isinstance(curated_scores, list)
        or not curated_scores
        or any(
            not isinstance(row, dict)
            or any(
                not isinstance(row.get(field), str)
                for field in ("slug", "bench", "cfg", "src", "date", "note")
            )
            or not isinstance(row.get("comparable"), bool)
            or not isinstance(row.get("official"), bool)
            or "v" not in row
            or not _is_optional_number(row.get("v"))
            for row in curated_scores
        )
        or not isinstance(models_curated.get("extra_models"), dict)
        or any(
            not _is_extra_model(row) for row in models_curated["extra_models"].values()
        )
    ):
        raise ValueError("snapshot models_curated has an invalid body")

    by_model = endpoints["provider_throughput_by_model"]["body"]
    roster = by_model.get("models")
    captures = by_model.get("by_model")
    if (
        not isinstance(by_model.get("as_of"), str)
        or not isinstance(by_model.get("granularity"), str)
        or not isinstance(roster, list)
        or not roster
        or not isinstance(captures, dict)
        or set(roster) != set(captures)
    ):
        raise ValueError("snapshot provider model roster does not match by_model")
    for slug, entry in captures.items():
        metrics = entry.get("metrics") if isinstance(entry, dict) else None
        if not isinstance(metrics, dict):
            raise ValueError(f"snapshot provider model metrics are missing for {slug}")
        for metric_name in ("tps", "latency", "tail"):
            metric = metrics.get(metric_name)
            if (
                not isinstance(metric, dict)
                or not isinstance(metric.get("unit"), str)
                or not isinstance(metric.get("providers"), dict)
                or not metric["providers"]
                or any(
                    not isinstance(series, list)
                    or not series
                    or any(not _is_provider_point(point) for point in series)
                    for series in metric["providers"].values()
                )
                or not isinstance(metric.get("market_median"), list)
                or not metric["market_median"]
                or any(
                    not _is_provider_reference_point(point)
                    for point in metric["market_median"]
                )
            ):
                raise ValueError(
                    f"snapshot provider model metrics are incomplete for "
                    f"{slug}:{metric_name}"
                )

    non_finite = [
        name
        for name, result in endpoints.items()
        if _has_non_finite_number(result["body"])
    ]
    if non_finite:
        raise ValueError(
            "snapshot contains non-finite numeric values in endpoints: "
            + ", ".join(sorted(non_finite))
        )

    size = len(json.dumps(snapshot, separators=(",", ":")).encode("utf-8"))
    if size > MAX_SNAPSHOT_BYTES:
        raise ValueError(
            f"snapshot is {size} bytes, over the {MAX_SNAPSHOT_BYTES}-byte API limit"
        )
    return snapshot


def _send_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            body = response.read()
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(
            f"snapshot publish failed: HTTP {exc.code}: {detail}"
        ) from exc
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise TypeError("snapshot publish returned a non-object response")
    return parsed


def _read_json(url: str, headers: dict[str, str]) -> dict[str, Any] | None:
    request = Request(url, headers=headers, method="GET")
    try:
        with urlopen(request, timeout=60) as response:
            body = response.read()
    except HTTPError as exc:
        if exc.code == 404:
            return None
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(
            f"snapshot readback failed: HTTP {exc.code}: {detail}"
        ) from exc
    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise TypeError("snapshot readback returned a non-object response")
    return parsed


def publish_snapshot(
    path: Path,
    *,
    api_base: str,
    api_key: str,
    rev: int = 1,
    dry_run_only: bool = False,
    send: SendJson = _send_json,
    read: ReadJson = _read_json,
) -> dict[str, Any]:
    """Dry-run then publish a complete capture to the play snapshot store."""
    api_base = api_base.strip()
    if not api_base:
        raise ValueError("api_base is required")
    parsed_api_base = urlparse(api_base)
    if parsed_api_base.scheme != "https" or not parsed_api_base.netloc:
        raise ValueError("api_base must be an HTTPS URL")
    if not api_key.strip():
        raise ValueError("api_key is required")
    if rev < 1:
        raise ValueError("rev must be at least 1")

    snapshot = _load_complete_snapshot(path)
    version = str(snapshot["captured_at"])
    payload = {
        "version": version,
        "rev": rev,
        "status": "published",
        "snapshot": snapshot,
        "created_by": "ai-monetization-tracker",
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    endpoint = api_base.rstrip("/") + PLAY_PATH
    version_url = endpoint + "?" + urlencode({"version": version})

    latest = read(endpoint, headers)
    if latest is not None:
        if latest.get("slug") != "ailab-arr" or latest.get("status") != "published":
            raise ValueError("ailab-arr latest read returned an unexpected row")
        latest_version = _parse_capture_time(latest.get("version"))
        if latest_version > _parse_capture_time(version):
            raise ValueError(
                f"ailab-arr capture {version} is older than latest published "
                f"version {latest.get('version')}"
            )
    existing = (
        latest
        if latest and latest.get("version") == version
        else read(version_url, headers)
    )
    if existing is not None:
        existing_rev = existing.get("rev")
        same_coordinate = (
            existing.get("slug") == "ailab-arr"
            and existing.get("version") == version
            and existing_rev == rev
        )
        if same_coordinate and existing.get("snapshot") == snapshot:
            return existing
        if same_coordinate:
            raise ValueError(
                f"ailab-arr {version} r{rev} already exists with different content; "
                "bump --rev instead of overwriting immutable data"
            )
        if isinstance(existing_rev, int) and existing_rev > rev:
            raise ValueError(
                f"ailab-arr {version} already has newer published revision "
                f"r{existing_rev}; requested r{rev}"
            )

    preview = send(endpoint + "?dry_run=true", payload, headers)
    if dry_run_only:
        return preview
    send(endpoint, payload, headers)
    confirmed = read(version_url, headers)
    if (
        confirmed is None
        or confirmed.get("slug") != "ailab-arr"
        or confirmed.get("version") != version
        or confirmed.get("rev") != rev
        or confirmed.get("status") != "published"
        or confirmed.get("snapshot") != snapshot
    ):
        raise RuntimeError(
            f"published ailab-arr {version} r{rev} did not match API readback"
        )
    return confirmed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--api-base", default=os.environ.get("FUNDA_API_BASE_URL", ""))
    parser.add_argument(
        "--rev", type=int, default=int(os.environ.get("SNAPSHOT_REV", "1"))
    )
    parser.add_argument("--dry-run-only", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("FUNDA_ADMIN_API_KEY", "")
    if not args.api_base:
        parser.error("--api-base or FUNDA_API_BASE_URL is required")
    if not api_key:
        parser.error("FUNDA_ADMIN_API_KEY is required")

    result = publish_snapshot(
        args.snapshot,
        api_base=args.api_base,
        api_key=api_key,
        rev=args.rev,
        dry_run_only=args.dry_run_only,
    )
    action = "validated" if args.dry_run_only else "published"
    print(
        f"{action}: ailab-arr {result.get('version')} r{result.get('rev')} "
        f"(id={result.get('id')})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
