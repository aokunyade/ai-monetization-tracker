"""Direct data sources — no parent project dependency.

Three classes of fetchers:
  1. npm registry downloads (npmjs.org public API)
  2. PyPI downloads (pypistats.org public API)
  3. OpenRouter Anthropic-only model breakdown (frontend stats API)

Public KNOWN_ARR rows are baked in so the standalone app can seed its own
ground-truth table. Private research anchors are injected at deployment time.
"""

from __future__ import annotations

import json
import math
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from typing import Iterable
from urllib.parse import quote

import httpx
from loguru import logger

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
OPENROUTER_BASE = "https://openrouter.ai"
MODELS_API_URL = f"{OPENROUTER_BASE}/api/v1/models"
# Undocumented frontend API — versioned to /v1 mid-2026. Keep the /v1 prefix
# in sync with sources_models_meta.STATS_API / PERMA_API.
MODEL_ACTIVITY_URL = f"{OPENROUTER_BASE}/api/frontend/v1/stats/model-activity"

NPM_PACKAGES: list[str] = [
    "@anthropic-ai/sdk",
    "@anthropic-ai/claude-code",
    # castora also tracks this — Vercel AI SDK wrapper; downstream framework
    # signal, kept as shadow until MAPE proves out.
    "@ai-sdk/anthropic",
    # OpenAI-side new packages that predict_openai_arr() SHADOW-tracks.
    # `openai`, `@ai-sdk/openai`, `@openai/codex`, and `pypi:openai` are
    # ALREADY fetched by sources_sdk_extra.py (NPM_EXTRA / PYPI_EXTRA) —
    # don't re-add them here or every sync doubles-fetches them and risks
    # tripping npm 429s. Only `gpt-tokenizer` and `@openai/agents` are new;
    # see experiments/openai_arr_forecast_v4.py leave-one-out results.
    "gpt-tokenizer",
    "@openai/agents",
]
PYPI_PACKAGES: list[str] = [
    "anthropic",
    # castora tracks — LangChain integration; enterprise-adjacent framework
    # adoption proxy. Kept as shadow until MAPE proves out.
    "langchain-anthropic",
    # pypi:openai lives in sources_sdk_extra.PYPI_EXTRA — not duplicated
    # here. See NPM_PACKAGES comment above.
]
PYPI_CONTROL_PACKAGES: list[str] = [
    "requests",
    "certifi",
    "urllib3",
    "six",
    "boto3",
]


def make_client(timeout: float = 60) -> httpx.Client:
    """Single shared httpx.Client to amortize TCP+TLS across all fetches."""
    return httpx.Client(timeout=timeout, headers={"User-Agent": USER_AGENT})


# ---------------------------------------------------------------------------
# npm signals
# ---------------------------------------------------------------------------
def fetch_npm_range(client: httpx.Client, pkg: str, start: str, end: str) -> list[dict]:
    """Daily npm download counts. npm caps at 365 days/range; chain calls."""
    out: list[dict] = []
    cur_start = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    while cur_start <= end_dt:
        cur_end = min(cur_start + timedelta(days=364), end_dt)
        url = (
            f"https://api.npmjs.org/downloads/range/"
            f"{cur_start.strftime('%Y-%m-%d')}:{cur_end.strftime('%Y-%m-%d')}/{pkg}"
        )
        try:
            r = client.get(url)
            r.raise_for_status()
            payload = r.json()
        except Exception as e:
            logger.warning(
                f"npm fetch {pkg} {cur_start.date()}-{cur_end.date()} failed: {e}"
            )
            cur_start = cur_end + timedelta(days=1)
            continue
        for item in payload.get("downloads") or []:
            out.append({"date": item["day"], "value": float(item["downloads"])})
        cur_start = cur_end + timedelta(days=1)
        time.sleep(0.3)
    logger.info(f"npm:{pkg}: fetched {len(out)} daily records")
    return out


# ---------------------------------------------------------------------------
# PyPI signals
# ---------------------------------------------------------------------------
def fetch_pypi_overall(client: httpx.Client, pkg: str) -> list[dict]:
    """pypistats /overall returns ~180 days of daily downloads."""
    url = f"https://pypistats.org/api/packages/{pkg}/overall"
    try:
        r = client.get(url, params={"mirrors": "false"})
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        logger.warning(f"pypi {pkg} fetch failed: {e}")
        return []
    out: list[dict] = []
    for x in payload.get("data") or []:
        if x.get("category") != "without_mirrors":
            continue
        out.append({"date": x["date"], "value": float(x["downloads"])})
    logger.info(f"pypi:{pkg}: fetched {len(out)} daily records")
    return out


def fetch_signals(
    client: httpx.Client, start: str = "2023-05-01", end: str | None = None
) -> list[dict]:
    """All npm + PyPI signals as upsert rows for `anthropic_signals`."""
    if end is None:
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    captured_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out: list[dict] = []
    for pkg in NPM_PACKAGES:
        for r in fetch_npm_range(client, pkg, start, end):
            out.append(
                {
                    "date": r["date"],
                    "signal_name": f"npm:{pkg}",
                    "value": r["value"],
                    "captured_at": captured_at,
                }
            )
        time.sleep(0.5)
    for pkg in PYPI_PACKAGES:
        for r in fetch_pypi_overall(client, pkg):
            out.append(
                {
                    "date": r["date"],
                    "signal_name": f"pypi:{pkg}",
                    "value": r["value"],
                    "captured_at": captured_at,
                }
            )
        time.sleep(0.5)
    # Control-only basket for detecting index-wide PyPI shifts; these are not
    # revenue proxies and stay outside the model signal registries.
    for pkg in PYPI_CONTROL_PACKAGES:
        for r in fetch_pypi_overall(client, pkg):
            out.append(
                {
                    "date": r["date"],
                    "signal_name": f"pypictl:{pkg}",
                    "value": r["value"],
                    "captured_at": captured_at,
                }
            )
        time.sleep(1.5)
    return out


# ---------------------------------------------------------------------------
# OpenRouter — Anthropic-only model_breakdown + pricing
# ---------------------------------------------------------------------------
def fetch_openrouter_catalog(client: httpx.Client) -> list[dict]:
    """Fetch OpenRouter's full /api/v1/models catalog once per sync.

    Consumed by fetch_anthropic_pricing (keeps only anthropic/*) and
    fetch_models_meta (auto-selects the frontier roster). Passing the same
    list into both avoids two 500KB round-trips per sync.
    """
    r = client.get(MODELS_API_URL)
    r.raise_for_status()
    return r.json().get("data", []) or []


def fetch_anthropic_pricing(
    client: httpx.Client, catalog: list[dict] | None = None
) -> list[dict]:
    """Pull /api/v1/models (or use pre-fetched `catalog`), keep only Anthropic
    rows. Returns pricing rows."""
    captured_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if catalog is None:
        catalog = fetch_openrouter_catalog(client)
    out: list[dict] = []
    for m in catalog:
        slug = m.get("id") or m.get("canonical_slug")
        if not slug or not slug.startswith("anthropic/"):
            continue
        canonical = m.get("canonical_slug") or slug
        pricing = m.get("pricing") or {}
        out.append(
            {
                "canonical_slug": canonical,
                "model_slug": slug,
                "prompt_price_usd": _to_float(pricing.get("prompt")),
                "completion_price_usd": _to_float(pricing.get("completion")),
                "cache_read_price_usd": _to_float(pricing.get("input_cache_read")),
                "captured_at": captured_at,
            }
        )
    logger.info(f"anthropic pricing: {len(out)} rows")
    return out


def _fetch_model_activity(
    client: httpx.Client, canonical_slugs: Iterable[str], tag: str
) -> list[dict]:
    """Daily per-model token/request rows from OpenRouter's frontend stats API.

    Slugs must be OpenRouter *permaslugs* — the date-stamped form that
    /api/v1/models returns as `canonical_slug`
    (e.g. anthropic/claude-4.8-opus-20260528). The display slug
    (anthropic/claude-opus-4.8) silently matches nothing.

    Today's UTC entry is dropped; it is still accumulating.
    """
    slugs = list(canonical_slugs)
    if not slugs:
        logger.warning(f"{tag} breakdown: no slugs supplied — returning []")
        return []
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out: list[dict] = []
    n_dropped = 0
    n_failed = 0
    for slug in slugs:
        url = f"{MODEL_ACTIVITY_URL}?permaslug={quote(slug, safe='')}&variant=standard"
        try:
            r = client.get(url)
            r.raise_for_status()
            payload = r.json()
        except Exception as e:
            logger.warning(f"model-activity {slug} failed: {e}")
            n_failed += 1
            continue
        analytics = (payload.get("data") or {}).get("analytics") or []
        for entry in analytics:
            date_only = (entry.get("date") or "")[:10]
            if not date_only:
                continue
            if date_only == today_utc:
                n_dropped += 1
                continue
            out.append(
                {
                    "date": date_only,
                    "canonical_slug": slug,
                    "variant": entry.get("variant") or "standard",
                    "prompt_tokens": int(entry.get("total_prompt_tokens") or 0),
                    "completion_tokens": int(entry.get("total_completion_tokens") or 0),
                    "cached_tokens": int(entry.get("total_native_tokens_cached") or 0),
                    "reasoning_tokens": int(
                        entry.get("total_native_tokens_reasoning") or 0
                    ),
                    "request_count": int(entry.get("count") or 0),
                    # Dropped from the payload when the endpoint moved to /v1;
                    # writes NULL until OpenRouter restores it. Nothing reads it.
                    "volume_usd": _to_float(entry.get("volume")),
                }
            )
    # Every slug coming back empty means the contract moved, not that every
    # model went quiet — this endpoint is undocumented and the /v1 prefix
    # appeared once already, zeroing this table for six weeks unnoticed.
    if not out:
        logger.error(
            f"{tag} breakdown: 0 rows from {len(slugs)} slugs "
            f"({n_failed} fetch failures) — endpoint contract may "
            f"have changed: {MODEL_ACTIVITY_URL}"
        )
        return out
    logger.info(
        f"{tag} breakdown: {len(out)} rows from {len(slugs)} slugs "
        f"(dropped {n_dropped} partial-today entries for {today_utc})"
    )
    return out


def fetch_anthropic_breakdown(
    client: httpx.Client, canonical_slugs: Iterable[str]
) -> list[dict]:
    return _fetch_model_activity(client, canonical_slugs, "anthropic")


def fetch_openai_breakdown(
    client: httpx.Client, canonical_slugs: Iterable[str]
) -> list[dict]:
    return _fetch_model_activity(client, canonical_slugs, "openai")


def _to_float(x) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def compute_openrouter_anthropic_signal_rows(
    conn: sqlite3.Connection,
) -> list[dict]:
    """Aggregate `anthropic_breakdown` into a daily proxy signal.

    signal_name = "openrouter:anthropic_tokens"
    value       = SUM(prompt_tokens + completion_tokens) per date
    Skips dates with zero total (partial-today filtering already done upstream).
    """
    captured_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out: list[dict] = []
    for r in conn.execute(
        "SELECT date, SUM(prompt_tokens + completion_tokens) AS tt "
        "FROM anthropic_breakdown GROUP BY date"
    ):
        if r["tt"] and r["tt"] > 0:
            out.append(
                {
                    "date": r["date"],
                    "signal_name": "openrouter:anthropic_tokens",
                    "value": float(r["tt"]),
                    "captured_at": captured_at,
                }
            )
    logger.info(f"openrouter:anthropic_tokens: {len(out)} daily rows from breakdown")
    return out


# Anthropic slugs that have churned off /api/v1/models but still have history
# in OpenRouter's stats. Without these the Spend tab undercounts pre-2025 volume.
HISTORICAL_ANTHROPIC_SLUGS: list[str] = [
    "anthropic/claude-3-haiku",
    "anthropic/claude-3-opus",
    "anthropic/claude-3-sonnet",
    "anthropic/claude-3.5-haiku-20241022",
    "anthropic/claude-3.5-sonnet-20240620",
    "anthropic/claude-3.5-sonnet-20241022",
    "anthropic/claude-3.7-sonnet-20250219",
    "anthropic/claude-4-opus-20250522",
    "anthropic/claude-4-sonnet-20250522",
    "anthropic/claude-4.5-opus-20251124",
    "anthropic/claude-4.5-haiku-20251001",
    "anthropic/claude-4.5-sonnet-20250929",
]

# The OpenAI counterpart. Without it the OpenAI slug set came from the current
# /api/v1/models catalog alone, so every model that has since been delisted was
# absent from *every* past day — history undercounts and growth climbs by
# construction. That's the same survivorship shape that made the Vercel model
# roster unusable for lab totals.
#
# Derived, not hand-guessed: rankings-daily reports whatever actually ranked on
# each historical day (delisted models included), so the churned set is
# {ranked in history} - {in catalog now}. Re-derive by re-running that diff.
#
# Embeddings are deliberately excluded. The diff also surfaces
# openai/text-embedding-3-small (ranked on 75 day-rows) and -3-large, but no
# embedding model is in the catalog today — so counting them historically would
# make the past include a token class the present doesn't and show a decline
# that never happened. They're also near-zero revenue per token, which is the
# wrong thing for an ARR proxy to track.
HISTORICAL_OPENAI_SLUGS: list[str] = [
    "openai/chatgpt-4o-latest",
    "openai/codex-mini",
    "openai/o1-mini",
    "openai/o1-preview",
]


# ---------------------------------------------------------------------------
# Known ARR — public evidence plus deployment-injected private research
# ---------------------------------------------------------------------------
# (month, arr_b, source, notes). Used as ground-truth for the regression.
_PUBLIC_KNOWN_ARR: list[tuple[str, float, str, str]] = [
    (
        "2024-12",
        1.0,
        "reuters_2025-05",
        "Reuters via The Decoder May 2025: 'around $3B in annualized revenue, "
        "up from $1B in December 2024'. Alt $0.875B per InfoTech Lead Jan 2025.",
    ),
    (
        "2025-03",
        2.0,
        "cnbc_2025-05_anthropic_confirmed",
        "CNBC May 16 2025 quoting Anthropic: 'annualized revenue reached "
        "$2 billion in the first quarter'.",
    ),
    (
        "2025-05",
        3.0,
        "reuters_2025-05-30",
        "Reuters May 30 2025: '$3B annualized revenue, up from $1B in December 2024'.",
    ),
    (
        "2025-06",
        4.0,
        "the_information_2025-07-01",
        "The Information July 1 2025: 'Anthropic's annual run rate was $4B..."
        "revenue for the month of June 2025 was around $333M'. Cross-verified by "
        "aibars.net timeline.",
    ),
    (
        "2025-07",
        5.0,
        "the_information_2025-10",
        "The Information Oct 15 2025: 'the figure rose from roughly $5 billion in July'.",
    ),
    (
        "2025-10",
        7.0,
        "reuters_2025-10-15",
        "Reuters Oct 15 2025: 'annualized revenue stood at around "
        "$7 billion as of October'.",
    ),
    (
        "2025-12",
        9.5,
        "funda_research",
        "Funda research (no direct press citation). Sacra reports $9B for end of 2025; kept at $9.5B per research.",
    ),
    ("2026-01", 13.0, "funda_research", "Funda research (no direct press citation)."),
    ("2026-02", 19.0, "funda_research", "Funda research (no direct press citation)."),
    ("2026-03", 31.0, "funda_research", "Funda research (no direct press citation)."),
    (
        "2026-04",
        44.0,
        "funda_research",
        "Funda research. Sacra confirms $43B for April 2026.",
    ),
    (
        "2026-07",
        65.0,
        "reuters_2026-08-17",
        "Reuters Aug 17 2026: annualized run rate 'topped $65 billion by the end "
        "of July'. Same figure and period in Bloomberg, CNBC and TechCrunch that "
        "day.",
    ),
]

PRIVATE_ARR_ENV = "ANTHROPIC_ARR_PRIVATE_ANCHORS_JSON"


def _decode_private_arr_payload(raw: str | None) -> tuple[object | None, str]:
    if not raw:
        return None, "empty"
    try:
        return json.loads(raw), "ok"
    except json.JSONDecodeError:
        return None, "malformed"


def load_private_arr_anchors() -> list[tuple[str, float, str, str]]:
    """Load confidential ARR anchors without storing their values in Git."""
    payload, decode_status = _decode_private_arr_payload(
        os.environ.get(PRIVATE_ARR_ENV)
    )
    if decode_status == "empty":
        return []
    if decode_status == "malformed":
        raise ValueError(f"{PRIVATE_ARR_ENV} must be valid JSON")
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{PRIVATE_ARR_ENV} must be a non-empty JSON array")

    rows: list[tuple[str, float, str, str]] = []
    public_months = {row[0] for row in _PUBLIC_KNOWN_ARR}
    latest_public_month = max(public_months)
    previous_month: str | None = None
    for index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{PRIVATE_ARR_ENV} entry #{index} must be an object")
        month = str(item.get("month") or "")
        try:
            parsed_month = datetime.strptime(month, "%Y-%m").strftime("%Y-%m")
        except ValueError:
            raise ValueError(f"invalid private ARR month in entry #{index}") from None
        if parsed_month != month:
            raise ValueError(f"invalid private ARR month in entry #{index}")
        if previous_month is not None and month <= previous_month:
            raise ValueError(
                f"private ARR anchor entry #{index} must be strictly increasing"
            )
        if month in public_months:
            raise ValueError(
                f"private ARR anchor entry #{index} conflicts with a public anchor"
            )
        if month >= latest_public_month:
            raise ValueError(
                f"private ARR anchor entry #{index} requires a later public anchor"
            )
        value = item.get("arr_b_usd")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"invalid private ARR value in entry #{index}")
        rows.append(
            (
                month,
                float(value),
                "funda_research_private",
                "Private deployment anchor.",
            )
        )
        previous_month = month
    return rows


_PRIVATE_KNOWN_ARR = load_private_arr_anchors()
KNOWN_ARR: list[tuple[str, float, str, str]] = [
    *_PUBLIC_KNOWN_ARR,
    *_PRIVATE_KNOWN_ARR,
]
KNOWN_ARR.sort(key=lambda row: row[0])

# Private anchors feed the local regression like public KNOWN_ARR rows but are
# scrubbed from published snapshots. Only their deployment-injected months are
# retained here; no confidential values live in the repository.
HIDDEN_ARR_MONTHS: frozenset[str] = frozenset(row[0] for row in _PRIVATE_KNOWN_ARR)


def known_arr_rows() -> list[dict]:
    """KNOWN_ARR in `anthropic_arr_known` upsert-row form."""
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return [
        {"month": m, "arr_b_usd": v, "source": s, "notes": n, "added_at": now_iso}
        for m, v, s, n in KNOWN_ARR
    ]
