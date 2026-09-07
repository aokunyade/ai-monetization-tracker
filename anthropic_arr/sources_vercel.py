"""Vercel AI Gateway fetcher — documented leaderboard export.

Vercel publishes the data behind /ai-gateway/leaderboards under CC BY 4.0 via a
documented export endpoint (https://vercel.com/docs/ai-gateway/leaderboards):

    GET https://vercel.com/api/ai/leaderboard-export
        ?dataset=labs|models|apps|providers
        &modality=all|text|image|video      # labs + models only
        &format=json|csv

Two rules this module exists to enforce:

1. **Only a lab row is an honest family total.** The `models` dataset ships the
   *current* top-10 backfilled across history — 4 names on 2026-05-18 growing to
   10 today, summing 31.63% -> 78.27%. Summing it per lab measures "share held by
   the models winning today" and climbs from ~0 by construction. Individual model
   shares are true shares of the whole (Claude's models sum to 59.23% under
   anthropic's 61.00%, and never exceed it on any day), so they are kept for
   detail — but a lab total is NEVER rebuilt from them.

2. **Only the documented endpoint.** The undocumented
   /api/ai/v4/gateway-model-leaderboard looks equivalent and is not: it drops
   Fable 5 from anthropic's numerator *and* from the shared denominator, so
   anthropic reads 10-17pp low through July while every other lab reads
   correspondingly high (openai +4-7pp) — inverting anthropic's trend from +1.35%
   to -11.58%. It matches the export across May only, before Fable existed, which
   is why checking it against a mirror that reads the same endpoint (aokunyade's
   data.js) looked clean. The test that catches it is structural: a lab's own
   models must sum to no more than the lab.

The export serves a rolling ~61-day window, so a day is unrecoverable once it
ages out. Every dataset x modality is archived verbatim into
`vercel_leaderboard_raw` on each sync — whether or not anything reads it — so
history accretes past what Vercel serves.
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
from loguru import logger

from .sources import make_client

EXPORT_URL = "https://vercel.com/api/ai/leaderboard-export"
UA = "Mozilla/5.0 (compatible; anthropic_arr-fusion)"

# Dated share series. `all` is what the leaderboard page itself renders and the
# only modality the signals read; the rest are archived for history accrual.
SERIES_EXPORTS: tuple[tuple[str, str], ...] = tuple(
    (ds, mod) for ds in ("labs", "models") for mod in ("all", "text", "image", "video")
)
# Ranked lists — no date dimension, no modality. Stamped with the fetch day.
RANKED_EXPORTS: tuple[str, ...] = ("apps", "providers")

# Lab key in the export -> the display name consumers filter on (arr_curve,
# arr_nowcast.vg_family, sources_derived, dashboard.js).
LAB_DISPLAY: dict[str, str] = {
    "anthropic": "Claude (family)",
    "openai": "OpenAI (family)",
}

# The export names the money metric `spend`; this DB has always called it `cost`
# and the frontend reads `history.cost`. Archive keeps upstream's spelling.
METRIC_TO_DB = {"spend": "cost", "tokens": "tokens", "requests": "requests"}
SERIES_METRICS = ("tokens", "cost")


def _fetch_export(
    client: httpx.Client, dataset: str, modality: str | None = None
) -> list[dict] | None:
    """One export dataset as a row list, or None if the request/shape fails."""
    params = {"dataset": dataset, "format": "json"}
    if modality:
        params["modality"] = modality
    try:
        r = client.get(
            EXPORT_URL, params=params, headers={"User-Agent": UA}, timeout=120
        )
        r.raise_for_status()
        rows = r.json().get("rows")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"vercel export {dataset}/{modality or '-'} failed: {e}")
        return None
    if not isinstance(rows, list) or not rows:
        logger.warning(f"vercel export {dataset}/{modality or '-'}: no rows")
        return None
    return rows


def fetch_vercel_gateway(
    client: httpx.Client | None = None,
) -> dict[str, list[dict]] | None:
    """Fetch every export slice.

    Returns {raw, ranked, history, snapshots} row lists, or None if the
    authoritative labs/all slice failed — in which case sync keeps the rows it
    already has rather than writing partials.
    """
    own_client = client is None
    client = client or make_client()
    try:
        series: dict[tuple[str, str], list[dict]] = {}
        for dataset, modality in SERIES_EXPORTS:
            rows = _fetch_export(client, dataset, modality)
            if rows:
                series[(dataset, modality)] = rows
        ranked_raw = {ds: _fetch_export(client, ds) for ds in RANKED_EXPORTS}
    finally:
        if own_client:
            client.close()

    labs = series.get(("labs", "all"))
    if not labs:
        logger.warning("vercel: labs/all export unavailable — keeping existing data")
        return None

    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # ---- Archive: every slice, verbatim, upstream's own metric names. ----
    raw_rows = [
        {
            "date": r["date"],
            "dataset": dataset,
            "modality": modality,
            "name": r["name"],
            "metric": r["metric"],
            "share_pct": float(r["share_percent"]),
        }
        for (dataset, modality), rows in series.items()
        for r in rows
        if r.get("date") and r.get("name") and r.get("metric")
    ]
    ranked_rows = [
        {
            "date": today_utc,
            "dataset": dataset,
            "rank": int(r["rank"]),
            "ranked_by": r.get("ranked_by") or "",
            "name": r["name"],
            "url": r.get("url") or "",
            "description": r.get("description") or "",
        }
        for dataset, rows in ranked_raw.items()
        if rows
        for r in rows
        if r.get("rank") and r.get("name")
    ]

    def by_day(rows: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
        """{db_metric: {day: {name: share}}} for the metrics this DB tracks."""
        out: dict[str, dict[str, dict[str, float]]] = {
            m: {} for m in METRIC_TO_DB.values()
        }
        for r in rows:
            db_metric = METRIC_TO_DB.get(r["metric"])
            if db_metric:
                out[db_metric].setdefault(r["date"], {})[r["name"]] = float(
                    r["share_percent"]
                )
        return out

    lab_hist = by_day(labs)
    model_hist = by_day(series.get(("models", "all")) or [])

    # ---- History: lab families are authoritative; models are detail only. ----
    # Today is still accumulating and the export is cached 24h, so a partial day
    # is not a data point. Downstream growth math (arr_curve, arr_nowcast) reads
    # these rows straight from the DB and does not drop a partial tail itself.
    history_rows: list[dict] = []
    for metric in SERIES_METRICS:
        for day, vals in sorted(lab_hist[metric].items()):
            if day == today_utc:
                continue
            for lab, display in LAB_DISPLAY.items():
                history_rows.append(
                    {
                        "date": day,
                        "metric": metric,
                        "name": display,
                        "share_pct": round(vals.get(lab, 0.0), 2),
                    }
                )
        m_days = sorted(model_hist[metric])
        if not m_days:
            continue
        latest = model_hist[metric][m_days[-1]]
        tracked = [n for n, _ in sorted(latest.items(), key=lambda kv: -kv[1])[:6]]
        fable = next((n for n in latest if "fable" in n.lower()), None)
        if fable and fable not in tracked:
            tracked.append(fable)
        for day in m_days:
            if day == today_utc:
                continue
            for name in tracked:
                history_rows.append(
                    {
                        "date": day,
                        "metric": metric,
                        "name": name,
                        "share_pct": round(model_hist[metric][day].get(name, 0.0), 2),
                    }
                )

    # ---- Snapshots: current top-10 models + Other. Point-in-time by nature,
    # so today's partial is exactly what belongs here.
    snapshot_rows: list[dict] = []
    for metric in METRIC_TO_DB.values():
        if not model_hist[metric]:
            continue
        last = sorted(model_hist[metric])[-1]
        top = sorted(model_hist[metric][last].items(), key=lambda kv: -kv[1])[:10]
        for rank, (name, v) in enumerate(top):
            snapshot_rows.append(
                {
                    "date": last,
                    "metric": metric,
                    "rank": rank,
                    "name": name,
                    "share_pct": round(v, 1),
                }
            )
        snapshot_rows.append(
            {
                "date": last,
                "metric": metric,
                "rank": len(top),
                "name": "Other",
                "share_pct": round(max(0.0, 100.0 - sum(v for _, v in top)), 1),
            }
        )

    lab_days = sorted(lab_hist["cost"])
    logger.info(
        f"vercel: {len(lab_days)} lab-days ({lab_days[0]} -> {lab_days[-1]}), "
        f"archive={len(raw_rows)} rows over {len(series)} slices, "
        f"ranked={len(ranked_rows)}, history={len(history_rows)}, "
        f"snapshots={len(snapshot_rows)}"
    )
    return {
        "raw": raw_rows,
        "ranked": ranked_rows,
        "history": history_rows,
        "snapshots": snapshot_rows,
    }
