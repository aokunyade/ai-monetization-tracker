"""OpenRouter rankings-daily fetcher — whole-market Top-50 + `other` row.

Ports ai-monetization-tracker/fetchers/fetch_openrouter.py to a rows-returning
function that sync.py can compose into its single-transaction upsert.

API: GET https://openrouter.ai/api/v1/datasets/rankings-daily
  - Top 50 public models per day by total tokens + aggregated `other` row
  - Auth: any OpenRouter API key (env OPENROUTER_API_KEY)
  - Rate limit: 30/min, 500/day

Regression guard: if the newly-fetched `latest_date` is older than what the DB
already has, we return empty lists — sync will keep the older good data rather
than clobber it. Matches tracker's "keep existing file" fallback.
"""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import date, timedelta

import httpx
from loguru import logger

from .labs import MARKET_LABS
from .sources import make_client

API = "https://openrouter.ai/api/v1/datasets/rankings-daily"
START_DATE = "2025-01-01"

# Known author prefixes; anything else goes to `others`. `other` model rows are
# the platform's own aggregation of long-tail models. Sourced from the shared
# labs.MARKET_LABS registry — includes cloud aggregators (amazon/microsoft/
# nvidia) beyond the frontier-model publishers.
LABS = list(MARKET_LABS.keys())

# Watchlist prefixes. These match against the `model_permaslug` returned by
# rankings-daily, which is date-stamped (e.g. "anthropic/claude-5-fable-20260609").
# Note the "5-fable" vs "fable-5" difference from sources_models_meta's
# auto-roster — they are NOT inconsistent: OpenRouter uses two different slug
# schemes across its endpoints. The `/api/v1/models` catalog canonicalizes as
# "anthropic/claude-fable-5" (fetched by sources_models_meta), while the
# `rankings-daily` dataset uses the dated permaslug
# "anthropic/claude-5-fable-YYYYMMDD". Do not "reconcile" — each is correct
# for its own endpoint.
WATCHLIST = [
    "anthropic/claude-5-fable",
    "anthropic/claude-sonnet",
    "anthropic/claude-opus",
    "openai/gpt-5",
    "google/gemini-3",
    "deepseek/",
    "x-ai/grok",
]

B = 1e9  # store token counts in billions


def _fetch_windowed(client: httpx.Client, key: str) -> list[dict]:
    """Loop over ≤100-day windows since START_DATE; return concatenated rows."""
    rows: list[dict] = []
    cur = date.fromisoformat(START_DATE)
    today = date.today()
    while cur <= today:
        end = min(cur + timedelta(days=99), today)
        try:
            r = client.get(
                API,
                headers={"Authorization": f"Bearer {key}"},
                params={"start_date": cur.isoformat(), "end_date": end.isoformat()},
                timeout=180,
            )
            r.raise_for_status()
            payload = r.json()
            rows.extend(payload.get("data", []) or [])
            logger.info(
                f"or_market: window {cur}..{end} -> "
                f"{len(payload.get('data', []) or [])} rows"
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"or_market window {cur}..{end} failed: {e}")
        cur = end + timedelta(days=1)
    return rows


def fetch_openrouter_market(
    prev_latest: str | None, client: httpx.Client | None = None
) -> dict[str, list[dict]] | None:
    """Return {totals, lab_share, watchlist, top_models} row lists, or None if
    the fetch should not overwrite existing data (missing key / regression /
    empty response).

    `prev_latest` is the max(date) currently in openrouter_daily_totals — pass
    it so we can detect a regression (fetched latest < prev_latest) and refuse
    to clobber. Pass None if the table is empty.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        logger.info("or_market: OPENROUTER_API_KEY not set — skipping")
        return None

    own_client = client is None
    client = client or make_client()
    try:
        raw = _fetch_windowed(client, key)
    finally:
        if own_client:
            client.close()
    if not raw:
        logger.warning("or_market: all windows failed — skipping")
        return None

    # Dedup by (date, permaslug); later windows win on collision.
    dedup: dict[tuple[str, str], dict] = {}
    for row in raw:
        dedup[(row["date"], row["model_permaslug"])] = row
    rows = list(dedup.values())

    dates = sorted({r["date"] for r in rows})
    if not dates:
        return None
    latest = dates[-1]

    if prev_latest and latest < prev_latest:
        logger.warning(
            f"or_market: fetched latest {latest} < existing "
            f"{prev_latest} — refusing to overwrite"
        )
        return None

    # Aggregations
    daily_total: dict[str, int] = defaultdict(int)
    lab_daily: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    watch: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_latest: dict[str, int] = defaultdict(int)
    by_7d: dict[str, int] = defaultdict(int)
    last7 = set(dates[-7:])

    for r in rows:
        d, slug, tok = r["date"], r["model_permaslug"], int(r["total_tokens"])
        daily_total[d] += tok
        if slug == "other":
            lab_daily[d]["long-tail"] += tok
        else:
            lab = slug.split("/")[0]
            lab_daily[d][lab if lab in LABS else "others"] += tok
            for prefix in WATCHLIST:
                if slug.startswith(prefix):
                    watch[prefix][d] += tok
            if d == latest:
                by_latest[slug] += tok
            if d in last7:
                by_7d[slug] += tok

    # 7-day MA on totals
    totals_sorted = [(d, daily_total[d]) for d in dates]
    ma7: dict[str, float] = {}
    for i, (d, _) in enumerate(totals_sorted):
        window = totals_sorted[max(0, i - 6) : i + 1]
        ma7[d] = sum(v for _, v in window) / len(window) / B

    as_of_iso = _now_iso()
    out: dict[str, list[dict]] = {
        "totals": [
            {
                "date": d,
                "total_tokens_b": round(daily_total[d] / B, 4),
                "ma7_b": round(ma7[d], 4),
                "as_of": as_of_iso,
            }
            for d in dates
        ],
        "lab_share": [
            {"date": d, "lab": lab, "tokens_b": round(v / B, 4)}
            for d in dates
            for lab, v in lab_daily[d].items()
        ],
        "watchlist": [
            {"date": d, "prefix": prefix, "tokens_b": round(t / B, 4)}
            for prefix, days in watch.items()
            for d, t in days.items()
        ],
        "top_models": (
            [
                {
                    "window": "latest",
                    "permaslug": s,
                    "tokens_b": round(t / B, 4),
                    "as_of": as_of_iso,
                }
                for s, t in sorted(by_latest.items(), key=lambda kv: -kv[1])[:15]
            ]
            + [
                {
                    "window": "7d",
                    "permaslug": s,
                    "tokens_b": round(t / B, 4),
                    "as_of": as_of_iso,
                }
                for s, t in sorted(by_7d.items(), key=lambda kv: -kv[1])[:15]
            ]
        ),
    }
    logger.info(
        f"or_market: {len(dates)} days, latest={latest}, "
        f"totals={len(out['totals'])}, lab_share={len(out['lab_share'])}, "
        f"watchlist={len(out['watchlist'])}, top_models={len(out['top_models'])}"
    )
    return out


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")
