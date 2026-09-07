"""Third-party model scores fetcher (Arena / Artificial Analysis / vals.ai).

Ports fetch_model_scores.py. Three independent sources; each failure is
logged but does not abort the others. Data lands in `model_scores` (one
row per (slug, source)) plus `model_arena_history` for the daily Elo curve.

Sources:
- Arena: HF datasets-server (lmarena-ai/leaderboard-dataset), no auth.
- AA:    artificialanalysis.ai leaderboards page (RSC flight), or /api/v2
         if $AA_API_KEY is set.
- vals:  vals.ai per-benchmark page astro-island props (independent reruns
         with cost_per_test and latency).
"""

from __future__ import annotations

import html
import json
import os
import re
from datetime import date, datetime, timezone
from typing import Any

import httpx
from loguru import logger

UA = {"User-Agent": "Mozilla/5.0 (compatible; anthropic_arr-fusion)"}

# Seed of tracker-legacy slugs that are NOT in OpenRouter's catalog
# (Anthropic Mythos, Meta Muse Spark) plus a handful of common frontier
# slugs. The full tracked set is dynamically extended at fetch time with
# the current auto-roster from sources_models_meta.select_frontier_roster
# — see fetch_models_scores(extra_slugs=...).
TRACKED_SEED: list[str] = [
    # OpenRouter-absent slugs (kept because tracker curates scores for them):
    "anthropic/claude-mythos-5",
    "meta/muse-spark-1.1",
]


def _norm(name: str) -> str:
    n = name.lower().split("/")[-1]
    n = re.sub(r"\(.*?\)", "", n)
    return re.sub(r"[^a-z0-9]", "", n)


# Fuzzy-name aliases for upstream leaderboards that use human names rather
# than OpenRouter slugs. Kept as a static map because aliases don't move
# with the roster (they map arbitrary upstream strings to canonical slugs).
ALIASES = {
    "claude5fable": "anthropic/claude-fable-5",
    "musespark11": "meta/muse-spark-1.1",
    "musespark1p1": "meta/muse-spark-1.1",
    "minimaxm3": "minimax/minimax-m3",
    "qwen37max": "qwen/qwen3.7-max",
    "hy3": "tencent/hy3",
    "hunyuanhy3": "tencent/hy3",
    "glm52": "z-ai/glm-5.2",
    "kimik27code": "moonshotai/kimi-k2.7-code",
    "kimik26": "moonshotai/kimi-k2.6",
}

# Module-level NORM2SLUG + TRACKED are mutable, updated at the top of every
# fetch_models_scores() call with the current auto-roster. Downstream helpers
# (_to_slug, fetch_arena, fetch_aa, fetch_vals) read them at call time, so
# extending here is thread-safe within a single sync run.
#
# TRACKED is a membership set of canonical slugs — used to short-circuit
# name-normalization when upstream already gives us a canonical slug (e.g.
# AA payloads include an `or_id` field).
NORM2SLUG: dict[str, str] = {_norm(s): s for s in TRACKED_SEED}
NORM2SLUG.update(ALIASES)
TRACKED: set[str] = set(TRACKED_SEED)


def _rebuild_norm2slug(extra_slugs: list[str] | None = None) -> None:
    """Rebuild NORM2SLUG + TRACKED from TRACKED_SEED + extra_slugs + ALIASES.

    Called at the top of fetch_models_scores with the current auto-roster's
    slug list so that any model in the live roster (Kimi K3, a new Gemini,
    etc.) matches upstream leaderboard entries without hand-editing this file.
    """
    NORM2SLUG.clear()
    TRACKED.clear()
    for s in TRACKED_SEED:
        NORM2SLUG[_norm(s)] = s
        TRACKED.add(s)
    for s in extra_slugs or []:
        NORM2SLUG[_norm(s)] = s
        TRACKED.add(s)
    NORM2SLUG.update(ALIASES)


_VARIANT = re.compile(
    r"-(thinking|preview|high|medium|low|xhigh|max-effort|latest|"
    r"\d{6,8}|\d+k)$",
    re.I,
)


def _to_slug(name: str) -> str | None:
    n = name.lower().split("/")[-1]
    n = re.sub(r"\(.*?\)", "", n).strip()
    while True:
        hit = NORM2SLUG.get(re.sub(r"[^a-z0-9]", "", n))
        if hit:
            return hit
        stripped = _VARIANT.sub("", n)
        if stripped == n:
            return None
        n = stripped


# ---------------------------------------------------------------------------
# Arena
# ---------------------------------------------------------------------------
def fetch_arena(client: httpx.Client) -> dict[str, Any]:
    """Return {updated, boards:{overall,coding,webdev}} per-slug tables."""
    base = "https://datasets-server.huggingface.co/filter"
    ds = "lmarena-ai/leaderboard-dataset"

    def board(config: str, where: str) -> list[dict]:
        rows, offset = [], 0
        while True:
            r = client.get(
                base,
                params={
                    "dataset": ds,
                    "config": config,
                    "split": "latest",
                    "where": where,
                    "length": 100,
                    "offset": offset,
                },
                headers=UA,
                timeout=60,
            )
            r.raise_for_status()
            j = r.json()
            rows += [x["row"] for x in j.get("rows", [])]
            offset += 100
            if offset >= j.get("num_rows_total", 0):
                return rows

    out: dict[str, dict] = {}
    updated: str | None = None
    for key, config, where in [
        ("overall", "text_style_control", "\"category\" = 'overall'"),
        ("coding", "text_style_control", "\"category\" = 'coding'"),
        ("webdev", "webdev", "\"category\" = 'overall'"),
    ]:
        table: dict[str, dict] = {}
        for row in board(config, where):
            slug = _to_slug(row.get("model_name", ""))
            if not slug:
                continue
            if key == "overall":
                updated = row.get("leaderboard_publish_date") or updated
            cur = table.get(slug)
            if not cur or row["rating"] > cur["rating"]:
                table[slug] = {
                    "rating": round(row["rating"], 1),
                    "rank": row.get("rank"),
                    "votes": row.get("vote_count"),
                }
        out[key] = table
    if len(out["overall"]) < 5:
        raise RuntimeError(
            f"arena matched only {len(out['overall'])} tracked models — schema changed?"
        )
    return {"updated": updated, "boards": out}


# ---------------------------------------------------------------------------
# Artificial Analysis
# ---------------------------------------------------------------------------
def fetch_aa(client: httpx.Client) -> dict[str, Any]:
    key = os.environ.get("AA_API_KEY")
    models = None
    if key:
        try:
            r = client.get(
                "https://artificialanalysis.ai/api/v2/data/llms/models",
                headers={"x-api-key": key, **UA},
                timeout=60,
            )
            r.raise_for_status()
            models = [
                {
                    "slug": m.get("slug"),
                    "or_id": m.get("openrouter_api_id") or m.get("openrouterApiId"),
                    "name": m.get("name"),
                    "ii": (m.get("evaluations") or {}).get(
                        "artificial_analysis_intelligence_index"
                    ),
                    "tps": m.get("median_output_tokens_per_second"),
                    "ttfa_s": m.get("median_time_to_first_answer_token_seconds"),
                    "blended": (m.get("pricing") or {}).get("price_1m_blended_3_to_1"),
                }
                for m in r.json().get("data", [])
            ]
            if not models:
                models = None
        except Exception as e:  # noqa: BLE001
            logger.warning(f"AA official API failed ({e}); falling back to page parse")
            models = None
    if models is None:
        src = client.get(
            "https://artificialanalysis.ai/leaderboards/models", headers=UA, timeout=90
        ).text
        chunks = re.findall(
            r'self\.__next_f\.push\(\[1,\s*"((?:[^"\\]|\\.)*)"\]\)', src
        )
        blob = "".join(json.loads('"' + c + '"') for c in chunks)
        dec, arrays, pos = json.JSONDecoder(), [], 0
        while True:
            i = blob.find('"models":', pos)
            if i < 0:
                break
            try:
                arr, _ = dec.raw_decode(blob[blob.index("[", i) :])
                if isinstance(arr, list) and arr and isinstance(arr[0], dict):
                    arrays.append(arr)
            except Exception:  # noqa: BLE001
                pass
            pos = i + 1
        if not arrays:
            raise RuntimeError("AA: no models array found in flight payload")
        raw = max(arrays, key=lambda a: len(a[0].keys()))
        clean = lambda v: None if v == "$undefined" else v  # noqa: E731
        models = [
            {
                "slug": m.get("slug"),
                "or_id": clean(m.get("openrouterApiId")),
                "name": m.get("name"),
                "ii": clean(m.get("intelligenceIndex")),
                "tps": clean(m.get("medianOutputTokensPerSecond")),
                "ttfa_s": clean(m.get("medianTimeToFirstAnswerTokenSeconds")),
                "blended": clean(m.get("price1mBlended0To3To1")),
            }
            for m in raw
        ]

    table: dict[str, dict] = {}
    for m in models:
        slug = None
        if m.get("or_id"):
            slug = m["or_id"] if m["or_id"] in TRACKED else _to_slug(m["or_id"])
        slug = slug or _to_slug(m.get("slug") or "") or _to_slug(m.get("name") or "")
        if not slug or m.get("ii") is None:
            continue
        cur = table.get(slug)
        if not cur or m["ii"] > cur["ii"]:
            table[slug] = {
                "ii": round(float(m["ii"]), 1),
                "tps": m.get("tps"),
                "ttfa_s": m.get("ttfa_s"),
                "blended": m.get("blended"),
            }
    if len(table) < 5:
        raise RuntimeError(
            f"AA matched only {len(table)} tracked models — schema changed?"
        )
    return {
        "updated": date.today().isoformat(),
        "models": table,
        "attribution": "Source: Artificial Analysis (artificialanalysis.ai)",
    }


# ---------------------------------------------------------------------------
# vals.ai
# ---------------------------------------------------------------------------
VALS_BOARDS = [
    ("swebench", "swebench", "SWE-bench Verified", "core"),
    ("terminal_bench_21", "terminal-bench-2-1", "Terminal-Bench 2.1", "core"),
    ("vals_index", "vals_index", "Vals Index", "core"),
    ("code_migration", "code-migration", "Code Migration", "vertical"),
    ("finance_agent", "fabv2", "Finance Agent v2", "vertical"),
    ("legal_agent", "hlab", "Harvey Legal Agent", "vertical"),
    ("medcode", "medcode", "MedCode", "vertical"),
    ("tax_eval", "tax_eval_v2", "TaxEval v2", "vertical"),
]


def _astro_decode(v: Any) -> Any:
    if isinstance(v, list) and len(v) == 2 and isinstance(v[0], int):
        t, val = v
        if t == 0:
            return (
                {k: _astro_decode(x) for k, x in val.items()}
                if isinstance(val, dict)
                else val
            )
        if t == 1:
            return [_astro_decode(x) for x in val]
    if isinstance(v, dict):
        return {k: _astro_decode(x) for k, x in v.items()}
    return v


def fetch_vals(
    client: httpx.Client, prev: dict[str, Any] | None = None
) -> dict[str, Any]:
    prev = prev or {}
    out: dict[str, Any] = {}
    fresh = 0
    for key, slug_page, label, grp in VALS_BOARDS:
        try:
            raw = client.get(
                f"https://www.vals.ai/benchmarks/{slug_page}", headers=UA, timeout=60
            ).text
            view = None
            for p in re.findall(r'<astro-island[^>]*?props="([^"]*)"', raw):
                d = _astro_decode(json.loads(html.unescape(p)))
                if isinstance(d, dict) and "benchmarkView" in d:
                    v = d["benchmarkView"].get("default", d["benchmarkView"])
                    if "tasks" in v and "metadata" in v:
                        view = v
                        break
            if not view:
                raise ValueError("benchmarkView not found")
            table: dict[str, dict] = {}
            for mid, res in view["tasks"]["overall"].items():
                slug = mid if mid in TRACKED else _to_slug(mid)
                if not slug or not isinstance(res, dict):
                    continue
                acc = res.get("accuracy")
                cur = table.get(slug)
                if acc is not None and (not cur or acc > cur["acc"]):
                    table[slug] = {
                        "acc": round(float(acc), 1),
                        "cost": res.get("cost_per_test"),
                        "lat": res.get("latency"),
                    }
            if len(table) < 3:
                raise ValueError(f"only {len(table)} tracked models matched")
            out[key] = {
                "updated": view["metadata"].get("updated"),
                "label": label,
                "group": grp,
                "models": table,
            }
            fresh += 1
        except Exception as e:  # noqa: BLE001
            logger.warning(f"vals {slug_page}: {e}")
            if key in prev:
                out[key] = prev[key]
    if not fresh and not out:
        raise RuntimeError("all vals boards failed and no previous data")
    return out


# ---------------------------------------------------------------------------
# Public entry: returns rows for model_scores + model_arena_history
# ---------------------------------------------------------------------------
def fetch_models_scores(
    client: httpx.Client | None = None,
    prev_vals: dict[str, Any] | None = None,
    extra_slugs: list[str] | None = None,
) -> dict[str, Any]:
    """Return {
        "scores_rows": [ {slug, source, payload_json, as_of}, ... ],
        "arena_history_rows": [ {slug, date, overall, coding, webdev}, ... ],
        "attributions": {arena, aa, vals},
    }

    `extra_slugs` — inject the current auto-roster slugs into NORM2SLUG so
    that upstream Arena/AA/vals rows match to canonical slugs beyond the
    hardcoded TRACKED_SEED. Caller (sync.py) should pass every slug that
    fetch_models_meta just returned; otherwise new frontier models silently
    get their score rows dropped at _to_slug.

    Each source is caught independently; failures log a warning but the
    other sources still land in the returned rows.
    """
    from .sources import make_client

    _rebuild_norm2slug(extra_slugs)
    own_client = client is None
    client = client or make_client()
    try:
        as_of_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        scores_rows: list[dict] = []
        arena_history_rows: list[dict] = []
        errors: dict[str, str] = {}

        # ---- Arena ----
        try:
            arena = fetch_arena(client)
            for slug in TRACKED:
                per_slug = {
                    "overall": arena["boards"]["overall"].get(slug),
                    "coding": arena["boards"]["coding"].get(slug),
                    "webdev": arena["boards"]["webdev"].get(slug),
                    "updated": arena.get("updated"),
                }
                if any((per_slug["overall"], per_slug["coding"], per_slug["webdev"])):
                    scores_rows.append(
                        {
                            "slug": slug,
                            "source": "arena",
                            "payload_json": json.dumps(per_slug, ensure_ascii=False),
                            "as_of": as_of_iso,
                        }
                    )
                # History row: only when Arena has an overall rating
                if arena.get("updated") and per_slug["overall"]:
                    arena_history_rows.append(
                        {
                            "slug": slug,
                            "date": arena["updated"],
                            "overall": per_slug["overall"].get("rating"),
                            "coding": (per_slug["coding"] or {}).get("rating"),
                            "webdev": (per_slug["webdev"] or {}).get("rating"),
                        }
                    )
            logger.info(
                f"arena: {len(arena['boards']['overall'])} overall, "
                f"{len(arena_history_rows)} history rows"
            )
        except Exception as e:  # noqa: BLE001
            errors["arena"] = str(e)
            logger.warning(f"arena failed: {e}")

        # ---- Artificial Analysis ----
        try:
            aa = fetch_aa(client)
            for slug, m in aa["models"].items():
                scores_rows.append(
                    {
                        "slug": slug,
                        "source": "aa",
                        "payload_json": json.dumps(
                            {
                                **m,
                                "updated": aa.get("updated"),
                                "attribution": aa.get("attribution"),
                            },
                            ensure_ascii=False,
                        ),
                        "as_of": as_of_iso,
                    }
                )
            logger.info(f"aa: {len(aa['models'])} models")
        except Exception as e:  # noqa: BLE001
            errors["aa"] = str(e)
            logger.warning(f"aa failed: {e}")

        # ---- vals.ai (per-benchmark) ----
        try:
            vals = fetch_vals(client, prev=prev_vals)
            # Fan out: emit one row per (slug, source="vals:<board>") with the
            # accuracy + cost + latency for that model on that board.
            per_slug: dict[str, dict[str, dict]] = {}
            for board_key, board in vals.items():
                for slug, entry in board.get("models", {}).items():
                    per_slug.setdefault(slug, {})[board_key] = {
                        **entry,
                        "label": board.get("label"),
                        "group": board.get("group"),
                        "updated": board.get("updated"),
                    }
            for slug, boards in per_slug.items():
                scores_rows.append(
                    {
                        "slug": slug,
                        "source": "vals",
                        "payload_json": json.dumps(boards, ensure_ascii=False),
                        "as_of": as_of_iso,
                    }
                )
            logger.info(f"vals: {len(per_slug)} slugs across {len(vals)} boards")
        except Exception as e:  # noqa: BLE001
            errors["vals"] = str(e)
            logger.warning(f"vals failed: {e}")

        return {
            "scores_rows": scores_rows,
            "arena_history_rows": arena_history_rows,
            "errors": errors,
        }
    finally:
        if own_client:
            client.close()
