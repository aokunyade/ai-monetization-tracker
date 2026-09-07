"""OpenRouter market routes.

- GET /api/v1/openrouter_market — public: daily totals + 7DMA + lab_share +
  top-15 (latest + 7d) + watchlist *minus Fable* (Fable is gated).
- GET /api/v1/openrouter_market/fable — session: Fable prefix full daily series.

Data layout matches tracker's openrouter_daily.json so the port to
`web/tracker.js:initTokens` is byte-for-byte compatible.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from .db import get_db

router = APIRouter(prefix="/api/v1/openrouter_market")

FABLE_PREFIX = "anthropic/claude-5-fable"


def _load_market(conn, include_fable: bool) -> dict[str, Any]:
    dates = [
        r["date"]
        for r in conn.execute(
            "SELECT date FROM openrouter_daily_totals ORDER BY date"
        ).fetchall()
    ]
    totals = conn.execute(
        "SELECT date, total_tokens_b, ma7_b, as_of "
        "FROM openrouter_daily_totals ORDER BY date"
    ).fetchall()
    lab_rows = conn.execute(
        "SELECT date, lab, tokens_b FROM openrouter_lab_share"
    ).fetchall()
    watch_rows = conn.execute(
        "SELECT date, prefix, tokens_b FROM openrouter_watchlist"
    ).fetchall()
    top_rows = conn.execute(
        "SELECT window, permaslug, tokens_b, as_of "
        "FROM openrouter_top_models ORDER BY window, -tokens_b"
    ).fetchall()

    # tracker shape: watchlist = {prefix: [[date, B], ...]}
    watchlist: dict[str, list[list[Any]]] = {}
    for r in watch_rows:
        if not include_fable and r["prefix"] == FABLE_PREFIX:
            continue
        watchlist.setdefault(r["prefix"], []).append(
            [r["date"], round(r["tokens_b"], 2)]
        )
    for k in watchlist:
        watchlist[k].sort()

    # lab_share = {dates:[...], labs: {lab: [B for each date]}}
    lab_map: dict[str, dict[str, float]] = {}
    for r in lab_rows:
        lab_map.setdefault(r["lab"], {})[r["date"]] = r["tokens_b"]
    # Filter to a rolling 180-day window matching tracker
    lab_dates = dates[-180:] if len(dates) > 180 else dates
    labs_sorted = sorted(
        lab_map.keys(), key=lambda lab: -sum(lab_map[lab].get(d, 0) for d in lab_dates)
    )
    lab_share = {
        "dates": lab_dates,
        "labs": {
            lab: [round(lab_map[lab].get(d, 0), 2) for d in lab_dates]
            for lab in labs_sorted
        },
    }

    top_latest = [
        {"slug": r["permaslug"], "tokens_b": round(r["tokens_b"], 2)}
        for r in top_rows
        if r["window"] == "latest"
    ]
    top_7d = [
        {"slug": r["permaslug"], "tokens_b": round(r["tokens_b"], 2)}
        for r in top_rows
        if r["window"] == "7d"
    ]
    as_of = totals[-1]["as_of"] if totals else None

    return {
        "as_of": as_of,
        "citation": (
            f"Source: OpenRouter (openrouter.ai/rankings), as of {as_of}."
            if as_of
            else None
        ),
        "tokenizer_note": (
            "Token counts use each provider's own tokenizer "
            "— cross-provider comparisons are approximate."
        ),
        "daily_totals": [[r["date"], round(r["total_tokens_b"], 2)] for r in totals],
        "daily_totals_unit": "B tokens/day",
        "ma7": [[r["date"], round(r["ma7_b"] or 0.0, 2)] for r in totals],
        "lab_share": lab_share,
        "watchlist": watchlist,
        "top_models_latest": top_latest,
        "top_models_7d": top_7d,
        "latest_date": dates[-1] if dates else None,
    }


@router.get("")
def market_public() -> dict[str, Any]:
    conn = get_db()
    try:
        return _load_market(conn, include_fable=False)
    finally:
        conn.close()


@router.get("/fable")
def market_fable() -> dict[str, Any]:
    conn = get_db()
    try:
        rows = conn.execute(
            "SELECT date, tokens_b FROM openrouter_watchlist "
            "WHERE prefix = ? ORDER BY date",
            (FABLE_PREFIX,),
        ).fetchall()
        return {
            "prefix": FABLE_PREFIX,
            "series": [[r["date"], round(r["tokens_b"], 2)] for r in rows],
        }
    finally:
        conn.close()
