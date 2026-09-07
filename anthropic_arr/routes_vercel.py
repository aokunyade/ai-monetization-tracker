"""Vercel Gateway routes.

- GET /api/v1/vercel_gateway — public: latest snapshot (tokens/spend/requests).
- GET /api/v1/vercel_gateway/history — session: full daily history (contains Fable).

Payload shape matches tracker's vercel_gateway.json so the port to
`web/tracker.js:initVercel` is byte-for-byte compatible.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from .db import get_db

router = APIRouter(prefix="/api/v1/vercel_gateway")


def _latest_snapshot(conn) -> dict[str, Any]:
    latest = conn.execute(
        "SELECT MAX(date) AS d FROM vercel_gateway_snapshots"
    ).fetchone()
    d = latest["d"] if latest else None
    if not d:
        return {"snapshots": [], "as_of": None}
    rows = conn.execute(
        "SELECT metric, rank, name, share_pct "
        "FROM vercel_gateway_snapshots WHERE date = ? "
        "ORDER BY metric, rank",
        (d,),
    ).fetchall()
    snap: dict[str, list[list[Any]]] = {
        "token_share": [],
        "spend_share": [],
        "request_share": [],
    }
    metric_key = {
        "tokens": "token_share",
        "cost": "spend_share",
        "requests": "request_share",
    }
    for r in rows:
        key = metric_key.get(r["metric"])
        if key:
            snap[key].append([r["name"], r["share_pct"]])
    snap["date"] = d
    return {
        "as_of": d,
        "source": (
            "Vercel AI Gateway Leaderboards (vercel.com/ai-gateway/leaderboards/models)"
        ),
        "window_note": "share % per day, parsed from page-embedded daily series",
        "snapshots": [snap],
    }


@router.get("")
def vercel_public() -> dict[str, Any]:
    conn = get_db()
    try:
        return _latest_snapshot(conn)
    finally:
        conn.close()


@router.get("/history")
def vercel_history() -> dict[str, Any]:
    conn = get_db()
    try:
        pub = _latest_snapshot(conn)
        rows = conn.execute(
            "SELECT date, metric, name, share_pct FROM vercel_gateway_history "
            "ORDER BY metric, name, date"
        ).fetchall()
        history: dict[str, dict[str, Any]] = {}
        for r in rows:
            m = history.setdefault(r["metric"], {"days_set": set(), "series": {}})
            m["days_set"].add(r["date"])
            m["series"].setdefault(r["name"], {})[r["date"]] = r["share_pct"]
        for m, node in history.items():
            days = sorted(node.pop("days_set"))
            node["days"] = days
            series_flat: dict[str, list[float]] = {}
            for name, per_day in node["series"].items():
                series_flat[name] = [per_day.get(d, 0.0) for d in days]
            node["series"] = series_flat
        pub["history"] = history
        return pub
    finally:
        conn.close()
