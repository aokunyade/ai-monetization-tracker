"""Frontier model metadata fetcher — dynamic roster.

Data source: OpenRouter public API (no key needed).
  - GET /api/v1/models              — pricing, ctx, modalities, max output
  - GET /api/frontend/v1/stats/endpoint?permaslug=...&variant=standard
                                     — provider p50/p99 tps + latency (30-min window)

Roster is BUILT on each sync from OpenRouter's live catalog by filtering:
  (a) lab-prefix is whitelisted in LAB_PREFIXES,
  (b) `created` >= FRONTIER_MIN_TS (drops legacy models),
  (c) `context_length` >= MIN_CTX (drops tiny experimental),
  (d) `price_in > 0` (drops free/preview endpoints that skew comparisons),
  (e) top TOP_N_PER_LAB newest per lab (Qwen alone has 49 models; without
      this cap the roster would drown in one lab's variants),
  (f) not matched by SLUG_EXCLUDE_PATTERNS (images, chat-tuned, misc).

Provider stats are best-effort; if the frontend API 500s we reuse whatever
was in `model_meta` from the previous run.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone
from typing import Any

import httpx
from loguru import logger

from .labs import FRONTIER_LABS as LAB_PREFIXES  # single source of truth

MODELS_API = "https://openrouter.ai/api/v1/models"
STATS_API = "https://openrouter.ai/api/frontend/v1/stats/endpoint"
PERMA_API = "https://openrouter.ai/api/frontend/v1/models/find"


# ─── Auto-roster selection ──────────────────────────────────────────────
# LAB_PREFIXES is imported from labs.py — canonical prefix→display map
# shared with sources_openrouter_market and (via payload) the frontend.
# Add labs there, not here.

# Frontier gate: models released before this timestamp are dropped.
# 2025-01-01 UTC — anything older is legacy from a frontier-tracking POV.
FRONTIER_MIN_TS: int = 1_735_689_600

# Context length floor — drops tiny experimental / embedding endpoints.
MIN_CTX: int = 65_536

# Newest N per lab. Prevents one very prolific lab (Qwen: 49 variants,
# OpenAI: 67 variants) from swamping the roster. Sized to accommodate:
#  - a bulk multi-variant launch (OpenAI's gpt-5.6 sol/luna/terra × base/pro
#    ate all 6 slots at TOP_N=6 and evicted every named GPT-5.5 flagship)
#  - separate reasoning/base/mini/code tiers per lab (DeepSeek V4 pro/flash +
#    R1 reasoning tier; Anthropic Opus/Sonnet/Haiku 4.5-5 line)
# Cap the total roster at 10 × 12 labs = 120 in the worst case.
TOP_N_PER_LAB: int = 10

# Slug exclusion patterns. Matched against `tail` — the substring AFTER the
# lab prefix `/`. Kept intentionally small and defensible; every pattern here
# was audited against the live OpenRouter catalog to avoid dropping real
# frontier models. See notes below each.
SLUG_EXCLUDE_PATTERNS: list[re.Pattern] = [
    # Non-text-only modality endpoints
    re.compile(r"-image(-|$)"),
    re.compile(r"-vision(-|$)"),
    re.compile(r"-embed(-|$)"),
    re.compile(r"-tts(-|$)"),
    re.compile(r"-audio(-|$)"),
    re.compile(r"-realtime(-|$)"),
    # Meta Llama-Guard / safety-tune filters. Meta's frontier chat models are
    # named `llama-<version>-<size>-instruct` (see below), so we can't blanket
    # -instruct; `-guard-` narrows to Guard specifically. `^guard-` is unused
    # (no tail starts with "guard-") but retained for defensive symmetry.
    re.compile(r"(^|-)guard(-|$)"),
    # Open-weights lines (Google Gemma / OpenAI oss / Anthropic there is none):
    re.compile(r"^gemma-"),
    re.compile(r"^gpt-oss(-|$)"),
    # Distills (small fine-tunes of a bigger reasoning model)
    re.compile(r"-distill(-|$)"),
    # Free / hidden endpoints
    re.compile(r":free$"),
]
# NOTES on what we deliberately DO NOT filter here:
# * `-fast$`  — historically thought to be Anthropic prompt-cache tier, but
#   xAI's `grok-4-fast` is a frontier reasoning release. Anthropic in fact
#   never ships `-fast` suffixed slugs; the base tier already covers cache
#   pricing via input_cache_read/write pricing fields.
# * `-chat$`  — DeepSeek's flagship is `deepseek/deepseek-chat`; excluding
#   this dropped it. Chat-tuned variants of other labs are fine to keep.
# * `-instruct($|-)` — Meta's `llama-*` frontier slugs ALL end `-instruct`,
#   `-instruct-<size>`. Excluding drops the entire Meta roster silently.
# * `-preview($|-)`  — Google routinely ships flagships as `-preview` for
#   months before GA; Qwen has `-max-preview`; OpenAI shipped `o1-preview`
#   for the whole pre-GA window. Preview status ≠ non-frontier.
# If you need to suppress a specific slug, use SLUG_HARD_DENY below rather
# than a broad regex.


# Hand-maintained hard-deny list for slugs the auto-filter can't otherwise
# rule out. Add here rather than expanding SLUG_EXCLUDE_PATTERNS when the
# exclusion is model-specific (not pattern-based).
SLUG_HARD_DENY: set[str] = set()


def _display_name(catalog_entry: dict, slug: str) -> str:
    """Human-facing name: OpenRouter's `name` sans lab prefix.

    Names come as `"MoonshotAI: Kimi K3"`; split(": ", 1)[-1] returns the
    tail if the colon is present, else the whole string, else "" if the
    field was missing entirely. Falls back to the slug tail titlecased if
    the upstream name is empty.
    """
    name = (catalog_entry.get("name") or "").split(": ", 1)[-1]
    return name or slug.split("/", 1)[-1].replace("-", " ").title()


def _as_int(v) -> int:
    """Coerce numeric-ish upstream values to int; return 0 on any failure."""
    if isinstance(v, bool):
        return 0
    if isinstance(v, (int, float)):
        return int(v) if v == v else 0  # NaN → 0
    if isinstance(v, str):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return 0
    return 0


def _passes_frontier_gate(m: dict) -> tuple[str, str] | None:
    """Return (slug, prefix) if the model looks like a frontier-tier release,
    else None. Returning the parsed slug + prefix saves _select_roster from
    re-splitting the string.
    """
    slug = m.get("id") or m.get("slug", "")
    if not slug or "/" not in slug:
        return None
    if slug in SLUG_HARD_DENY:
        return None
    prefix, tail = slug.split("/", 1)
    if prefix not in LAB_PREFIXES:
        return None
    for pat in SLUG_EXCLUDE_PATTERNS:
        if pat.search(tail):
            return None
    # `created` and `context_length` may be int, float, or (rare) numeric
    # strings depending on upstream serialization. Coerce defensively — a
    # raw `str < int` comparison would abort the whole sync.
    if _as_int(m.get("created")) < FRONTIER_MIN_TS:
        return None
    if _as_int(m.get("context_length")) < MIN_CTX:
        return None
    pricing = m.get("pricing")
    if not isinstance(pricing, dict):
        # OpenRouter has always shipped `pricing` as a dict; guard defensively
        # so a shape change (list, None) doesn't crash `.get()`.
        return None
    try:
        pin = float(pricing.get("prompt", 0) or 0)
    except (TypeError, ValueError):
        return None
    if pin <= 0:
        return None
    return slug, prefix


def _select_roster(catalog: dict[str, dict]) -> list[tuple[str, str, str]]:
    """Return the auto-selected (slug, lab, display) roster.

    Applies the frontier gate + top-N-per-lab cap. Deterministic ordering:
    newest first per lab, labs in `LAB_PREFIXES` insertion order.
    """
    by_lab: dict[str, list[dict]] = {p: [] for p in LAB_PREFIXES}
    for m in catalog.values():
        result = _passes_frontier_gate(m)
        if result is None:
            continue
        _, prefix = result
        by_lab[prefix].append(m)

    roster: list[tuple[str, str, str]] = []
    for prefix, lab_display in LAB_PREFIXES.items():
        entries = by_lab[prefix]
        # Sort by (created desc, slug asc). The slug tie-break makes ordering
        # deterministic across upstream reshuffles when a lab does a bulk
        # same-second launch (e.g. OpenAI's 6 gpt-5.6 sol/luna/terra variants
        # all published 2026-07-09 within 17 seconds).
        entries.sort(
            key=lambda x: (-_as_int(x.get("created")), x.get("id") or x.get("slug", ""))
        )
        for m in entries[:TOP_N_PER_LAB]:
            slug = m.get("id") or m.get("slug", "")
            roster.append((slug, lab_display, _display_name(m, slug)))
    return roster


def _mtok(price_str: Any) -> float | None:
    try:
        return round(float(price_str) * 1e6, 4)
    except (TypeError, ValueError):
        return None


def _permaslug_map(client: httpx.Client) -> dict[str, str] | None:
    try:
        r = client.get(PERMA_API, timeout=120)
        r.raise_for_status()
        return {m["slug"]: m["permaslug"] for m in r.json()["data"]["models"]}
    except Exception as e:  # noqa: BLE001
        logger.warning(f"models_meta permaslug map failed: {e}")
        return None


def _provider_stats(
    client: httpx.Client, slug: str, perma_map: dict[str, str] | None
) -> list[dict]:
    try:
        r = client.get(
            STATS_API,
            params={
                "permaslug": (perma_map or {}).get(slug, slug),
                "variant": "standard",
            },
            timeout=60,
        )
        r.raise_for_status()
        payload = r.json()
    except Exception as e:  # noqa: BLE001
        logger.debug(f"stats {slug} failed: {e}")
        return []
    # `.get('data', [])` returns None if the key exists with value null; iter
    # over None would raise TypeError and abort the whole roster refresh.
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    out = []
    for ep in data:
        if not isinstance(ep, dict):
            continue
        s = ep.get("stats") or {}
        # Store the full stat set upstream exposes, not just p50/p99 — the
        # dashboard renders a subset but the DB keeps everything (percentile
        # spread, TTFT, and the endpoint-level supply signals capacity_tpm /
        # status / quantization / region). "latency" here is TTFT (ms).
        out.append(
            {
                "provider": ep.get("provider_display_name") or ep.get("provider_name"),
                "p50_tps": s.get("p50_throughput"),
                "p75_tps": s.get("p75_throughput"),
                "p90_tps": s.get("p90_throughput"),
                "p95_tps": s.get("p95_throughput"),
                "p99_tps": s.get("p99_throughput"),
                "p50_lat_ms": s.get("p50_latency"),
                "p75_lat_ms": s.get("p75_latency"),
                "p90_lat_ms": s.get("p90_latency"),
                "p95_lat_ms": s.get("p95_latency"),
                "p99_lat_ms": s.get("p99_latency"),
                "n": s.get("request_count"),
                "window_minutes": s.get("window_minutes"),
                "capacity_tpm": ep.get("capacity_tpm"),
                "quantization": ep.get("quantization"),
                "provider_region": ep.get("provider_region"),
                "status": ep.get("status"),
            }
        )
    # Keep highest-volume endpoint per provider.
    best: dict[str, dict] = {}
    for e in out:
        k = e["provider"]
        if k and (k not in best or (e["n"] or 0) > (best[k]["n"] or 0)):
            best[k] = e
    return sorted(best.values(), key=lambda e: -(e["n"] or 0))


def fetch_models_meta(
    client: httpx.Client | None = None,
    prev_meta: dict[str, dict] | None = None,
    catalog_list: list[dict] | None = None,
) -> list[dict] | None:
    """Return list of {slug, payload_json, as_of} rows for upsert, or None if
    the catalog fetch failed.

    `catalog_list` — pre-fetched output of sources.fetch_openrouter_catalog().
    Pass this from sync.py so we don't re-download the ~500KB catalog when
    fetch_anthropic_pricing already loaded it in the same run.
    """
    from .sources import fetch_openrouter_catalog, make_client

    own_client = client is None
    client = client or make_client()
    try:
        if catalog_list is None:
            try:
                catalog_list = fetch_openrouter_catalog(client)
            except Exception as e:  # noqa: BLE001
                logger.warning(f"models_meta catalog fetch failed: {e}")
                return None
        catalog = {m["id"]: m for m in catalog_list if m.get("id")}
        perma_map = _permaslug_map(client)
        today = date.today().isoformat()
        as_of_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        prev_meta = prev_meta or {}

        # Build the roster live from the OpenRouter catalog. Falls back to
        # prev_meta's slug set if the catalog somehow returns 0 matches —
        # avoids wiping the DB on a bad upstream response.
        roster = _select_roster(catalog)
        if not roster:
            logger.warning(
                "models_meta: auto-roster empty — "
                "OpenRouter catalog thinned? Refusing to overwrite."
            )
            return None
        logger.info(
            f"models_meta: auto-roster selected {len(roster)} models "
            f"across {len({lab for _, lab, _ in roster})} labs"
        )

        # Every slug in `roster` came from `catalog.values()` via
        # `_select_roster`, so `catalog[slug]` is guaranteed non-None here.
        # prev_meta is still consulted for the `providers` field when the
        # frontend stats API 500s (a per-request failure the outer catalog
        # fetch can't cover).
        #
        # _provider_stats does one HTTP round-trip per slug. Serial at 60+
        # slugs = ~60s wall time; ThreadPoolExecutor with 8 workers brings
        # that to ~8s. httpx.Client is thread-safe for concurrent requests.
        def _stats_for(slug: str) -> list[dict]:
            return _provider_stats(client, slug, perma_map)

        with ThreadPoolExecutor(max_workers=8) as pool:
            providers_map = dict(
                zip(
                    (slug for slug, _, _ in roster),
                    pool.map(_stats_for, (slug for slug, _, _ in roster)),
                )
            )

        rows = []
        for slug, lab, display in roster:
            m = catalog[slug]
            arch = m.get("architecture") or {}
            top = m.get("top_provider") or {}
            pricing = m.get("pricing") or {}
            providers = providers_map.get(slug) or []
            if not providers and slug in prev_meta:
                providers = prev_meta[slug].get("providers", []) or []
            entry = {
                "slug": slug,
                "lab": lab,
                "display": display,
                "ctx": m.get("context_length"),
                "max_out": top.get("max_completion_tokens"),
                "modalities_in": arch.get("input_modalities") or [],
                "price_in": _mtok(pricing.get("prompt")),
                "price_out": _mtok(pricing.get("completion")),
                "cache_read": _mtok(pricing.get("input_cache_read")),
                "cache_write": _mtok(pricing.get("input_cache_write")),
                "reasoning": bool(m.get("reasoning")),
                "cutoff": m.get("knowledge_cutoff"),
                "created": m.get("created"),
                "providers": providers,
                "as_of": today,
            }
            rows.append(
                {
                    "slug": slug,
                    "payload_json": json.dumps(entry, ensure_ascii=False),
                    "as_of": as_of_iso,
                }
            )
        logger.info(f"models_meta: {len(rows)} models resolved from catalog")
        return rows
    finally:
        if own_client:
            client.close()
