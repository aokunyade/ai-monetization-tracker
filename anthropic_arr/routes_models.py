"""Model matrix routes (for /models page).

- GET /api/v1/models_meta      — public: roster + specs + list pricing
                                  (includes cache read/write). Not gated —
                                  tracker also serves basic meta publicly.
- GET /api/v1/models_scores    — session: AA / Arena / vals reruns. Gated
                                  because tracker gates overseas model scores.
- GET /api/v1/models_curated   — public: human-curated config + per-slug
                                  overrides (models_curated_seed.json).
- GET /api/v1/models_arena_history — session: rating-over-time for the Elo curve.
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter

from . import ROOT
from .db import get_db

# Curated seed lives on disk (models_curated_seed.json). We serve it raw so
# the frontend gets castora's exact schema: {as_of, benchmarks, scores,
# extra_models, ...}. The DB `model_curated` table is a legacy path that
# reshapes the seed into {config, per_slug} — kept live as
# /api/v1/models_curated_legacy for existing consumers.
_CURATED_SEED_PATH = ROOT / "data" / "models_curated_seed.json"

router = APIRouter(prefix="/api/v1")


@router.get("/models_meta")
def models_meta() -> dict[str, Any]:
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT slug, payload_json, as_of FROM model_meta ORDER BY slug"
        ).fetchall()
        as_of = None
        models: dict[str, Any] = {}
        for r in rows:
            models[r["slug"]] = json.loads(r["payload_json"])
            as_of = r["as_of"] or as_of
        return {"as_of": as_of, "models": models}
    finally:
        conn.close()


@router.get("/models_scores")
def models_scores() -> dict[str, Any]:
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT slug, source, payload_json, as_of FROM model_scores "
            "ORDER BY slug, source"
        ).fetchall()
        by_source: dict[str, dict[str, Any]] = {"arena": {}, "aa": {}, "vals": {}}
        as_of = None
        for r in rows:
            src = r["source"]
            if src in by_source:
                by_source[src][r["slug"]] = json.loads(r["payload_json"])
                as_of = r["as_of"] or as_of
        return {"as_of": as_of, "scores": by_source}
    finally:
        conn.close()


@router.get("/models_curated")
def models_curated() -> dict[str, Any]:
    """Serve castora-format models_curated_seed.json raw.

    Shape (exact castora schema):
      { as_of, method, benchmarks: {key: {label, unit, cat, note}},
        scores: [{slug, bench, v, cfg, comparable, official, src, date, note}, ...],
        extra_models: {slug: {lab, display, ctx, price_in, price_out, ...}} }

    Falls through to an empty payload if the seed file is missing so the
    frontend degrades gracefully rather than 500-ing.
    """
    if _CURATED_SEED_PATH.exists():
        try:
            return json.loads(_CURATED_SEED_PATH.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            pass
    return {"as_of": None, "benchmarks": {}, "scores": [], "extra_models": {}}


@router.get("/models_curated_legacy")
def models_curated_legacy() -> dict[str, Any]:
    """Legacy DB-backed curated shape: {config, per_slug}. Retained for the
    old dashboard code path that reshaped it — new code should use
    /models_curated (raw castora schema)."""
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT slug, payload_json FROM model_curated ORDER BY slug"
        ).fetchall()
        curated: dict[str, Any] = {}
        config: dict[str, Any] = {}
        for r in rows:
            payload = json.loads(r["payload_json"])
            if r["slug"] == "__config__":
                config = payload
            else:
                curated[r["slug"]] = payload
        return {"config": config, "per_slug": curated}
    finally:
        conn.close()


@router.get("/models_arena_history")
def models_arena_history() -> dict[str, Any]:
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT slug, date, overall, coding, webdev "
            "FROM model_arena_history ORDER BY slug, date"
        ).fetchall()
        history: dict[str, list[list[Any]]] = {}
        for r in rows:
            history.setdefault(r["slug"], []).append(
                [r["date"], r["overall"], r["coding"], r["webdev"]]
            )
        return {"history": history}
    finally:
        conn.close()


def _pctile(xs: list[float], q: float) -> float:
    """Linear-interpolated percentile of already-collected samples (q in [0,1])."""
    xs = sorted(xs)
    if len(xs) == 1:
        return xs[0]
    i = q * (len(xs) - 1)
    lo = int(i)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (i - lo)


# Sub-daily buckets keep the trend readable while history is still days old;
# once there are enough distinct days for a daily line to have shape, switch.
_BUCKET_HOURS = 6
_DAILY_AFTER_DAYS = 7


def _bucket_key(captured_at: str, date: str, daily: bool) -> str:
    """Bucket label for a 30-min sample — the date, or a 6h slot inside it."""
    if daily:
        return date
    hour = int(captured_at[11:13])
    return f"{date} {hour // _BUCKET_HOURS * _BUCKET_HOURS:02d}:00"


@router.get("/provider_throughput_history")
def provider_throughput_history(model: str | None = None) -> dict[str, Any]:
    """Per-provider throughput and TTFT series for the GPU-supply proxy chart.

    `model` restricts to one canonical slug — the useful view, because a
    provider's own models differ far more from each other than the same model
    does across half an hour, so an unfiltered band measures model mix rather
    than supply. Unfiltered, each 30-min window is first collapsed to the median
    across whatever models that provider served in it, which keeps the band a
    time-drift measure in both modes.

    Within a bucket the center is the MEDIAN of the per-window values (robust;
    not a request-weighted mean of medians, which isn't a real percentile) and
    lo/hi are the p10/p90 across those windows — how much the provider's typical
    speed drifted, NOT a request-level percentile. We do not emit a blended
    "market p50": percentiles don't compose and a request-weighted blend
    confounds throughput with provider mix-shift; `market_median` is a plain
    cross-provider median per bucket, offered only as a light reference line.

    Payload:
      {
        "as_of": "<latest bucket>",
        "granularity": "6h" | "1d",
        "model": "anthropic/claude-opus-4.8" | null,
        "models": ["anthropic/claude-opus-4.8", ...],   (everything with data)
        "top_providers": ["Anthropic", ...],            (by latest bucket, top 8)
        "metrics": {
          "tps":     {"unit": "tok/s", "providers": {...}, "market_median": [...]},
          "latency": {"unit": "ms",    "providers": {...}, "market_median": [...]},
        },
      }
    where each provider series is [[bucket, median, p10, p90, n_windows], ...].
    """
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT captured_at, date, provider, canonical_slug, p50_tps, p50_lat_ms, "
            "p99_lat_ms, request_count FROM provider_throughput_history "
            "WHERE p50_tps IS NOT NULL ORDER BY provider, canonical_slug, captured_at"
        ).fetchall()

        models = sorted({r["canonical_slug"] for r in rows})
        if model:
            rows = [r for r in rows if r["canonical_slug"] == model]

        daily = len({r["date"] for r in rows}) >= _DAILY_AFTER_DAYS

        # (provider, bucket, window) -> the models that provider served then.
        windows: dict[tuple[str, str, str], dict[str, list[float]]] = {}
        for r in rows:
            key = (
                r["provider"],
                _bucket_key(r["captured_at"], r["date"], daily),
                r["captured_at"],
            )
            slot = windows.setdefault(key, {"tps": [], "lat": [], "tail": []})
            slot["tps"].append(float(r["p50_tps"]))
            if r["p50_lat_ms"]:
                slot["lat"].append(float(r["p50_lat_ms"]))
                if r["p99_lat_ms"] is not None:
                    slot["tail"].append(float(r["p99_lat_ms"]) / float(r["p50_lat_ms"]))

        # Collapse each window to one value, then regroup by bucket.
        buckets: dict[tuple[str, str], dict[str, list[float]]] = {}
        for (prov, bucket, _ts), slot in windows.items():
            agg = buckets.setdefault((prov, bucket), {"tps": [], "lat": [], "tail": []})
            for metric in ("tps", "lat", "tail"):
                if slot[metric]:
                    agg[metric].append(_pctile(slot[metric], 0.5))

        def series_for(metric: str) -> dict[str, list]:
            out: dict[str, list] = {}
            for (prov, bucket), agg in buckets.items():
                xs = agg[metric]
                if not xs:
                    continue
                out.setdefault(prov, []).append(
                    [
                        bucket,
                        round(_pctile(xs, 0.5), 2),
                        round(_pctile(xs, 0.1), 2),
                        round(_pctile(xs, 0.9), 2),
                        len(xs),
                    ]
                )
            for series in out.values():
                series.sort()
            return out

        def market(series: dict[str, list]) -> list:
            by_bucket: dict[str, list[float]] = {}
            for rows_ in series.values():
                for bucket, med, *_ in rows_:
                    by_bucket.setdefault(bucket, []).append(med)
            return [
                [b, round(_pctile(v, 0.5), 2)] for b, v in sorted(by_bucket.items())
            ]

        tps, lat, tail = series_for("tps"), series_for("lat"), series_for("tail")
        latest = max((s[-1][0] for s in tps.values()), default=None)
        top_providers = [
            name
            for name, series in sorted(
                tps.items(),
                key=lambda kv: -(kv[1][-1][1] if kv[1] else 0),  # latest median
            )
            if series and series[-1][0] == latest
        ][:8]

        return {
            "as_of": latest,
            "granularity": "1d" if daily else f"{_BUCKET_HOURS}h",
            "model": model,
            "models": models,
            "top_providers": top_providers,
            "metrics": {
                "tps": {
                    "unit": "tok/s",
                    "providers": tps,
                    "market_median": market(tps),
                },
                "latency": {
                    "unit": "ms",
                    "providers": lat,
                    "market_median": market(lat),
                },
                # p99÷p50 TTFT. Queueing blows out the tail long before it moves
                # the median, so this reads supply tightness earlier than either
                # throughput or median latency does.
                "tail": {"unit": "×", "providers": tail, "market_median": market(tail)},
            },
        }
    finally:
        conn.close()
