"""Build the JSON payload consumed by the standalone frontend at port 8301."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

from .backtest import backtest_summary
from .compute import predict_arr
from .labels import CHANNEL_LABELS, METRIC_LABELS, SIGNAL_LABELS


def _month_label(yyyymm: str) -> str:
    """`2026-05` → `May 2026`."""
    try:
        dt = datetime.strptime(yyyymm + "-01", "%Y-%m-%d")
        return dt.strftime("%B %Y")
    except (ValueError, TypeError):
        return yyyymm or ""


def _tier_from_slug_anthropic(slug: str) -> str:
    s = (slug or "").lower()
    if "opus" in s:
        return "opus"
    if "sonnet" in s:
        return "sonnet"
    if "haiku" in s:
        return "haiku"
    if "fable" in s:
        return "fable"
    return "other"


def _tier_from_slug_openai(slug: str) -> str:
    """OpenAI tiering mirrors castora's mental model:
    - `codex` → reasoning/agent (Codex CLI)
    - `gpt-5.6-pro` / `-pro` suffix → pro tier
    - `gpt-5.6` (any luna/sol/terra variant) → gpt5.6 base
    - `gpt-5.5` and earlier gpt-5.x → gpt5.5
    - everything else → other (nano, chat-latest, etc.)"""
    s = (slug or "").lower()
    if "codex" in s:
        return "codex"
    if "-pro" in s:
        return "gpt5.6-pro"
    if "gpt-5.6" in s:
        return "gpt5.6"
    if "gpt-5.5" in s:
        return "gpt5.5"
    return "other"


def _spend_panel(
    conn: sqlite3.Connection, table: str, tier_fn, tiers: list[str]
) -> dict:
    """Generic monthly-tokens / tier-mix / cache-ratio / top-models panel.

    `table` is the breakdown table name (anthropic_breakdown / openai_breakdown).
    `tier_fn(slug) -> tier_name` maps slug → tier bucket. `tiers` is the
    ordered list of tier keys (used to zero-fill missing months).
    """
    # Monthly aggregate
    monthly = []
    for r in conn.execute(
        f"SELECT substr(date,1,7) AS m, "
        f"SUM(prompt_tokens) AS pt, SUM(completion_tokens) AS ct, "
        f"SUM(request_count) AS reqs "
        f"FROM {table} GROUP BY m ORDER BY m"
    ):
        monthly.append(
            {
                "month": r["m"],
                "prompt_tokens": int(r["pt"] or 0),
                "completion_tokens": int(r["ct"] or 0),
                "request_count": int(r["reqs"] or 0),
            }
        )

    # Per-tier monthly volumes
    tier_monthly: dict[str, dict[str, int]] = {}
    for r in conn.execute(
        f"SELECT substr(date,1,7) AS m, canonical_slug, "
        f"SUM(prompt_tokens + completion_tokens) AS tt "
        f"FROM {table} GROUP BY m, canonical_slug"
    ):
        tier = tier_fn(r["canonical_slug"] or "")
        tier_monthly.setdefault(r["m"], {}).setdefault(tier, 0)
        tier_monthly[r["m"]][tier] += int(r["tt"] or 0)

    months_sorted = sorted(tier_monthly.keys())
    tier_series = {t: [tier_monthly[m].get(t, 0) for m in months_sorted] for t in tiers}

    # Cache ratio over time
    cache_ratio = []
    for r in conn.execute(
        f"SELECT substr(date,1,7) AS m, "
        f"SUM(cached_tokens) AS cache, SUM(prompt_tokens) AS pt "
        f"FROM {table} GROUP BY m ORDER BY m"
    ):
        pt = float(r["pt"] or 0)
        cache = float(r["cache"] or 0)
        cache_ratio.append(
            {
                "month": r["m"],
                "ratio": (cache / pt) if pt > 0 else 0,
            }
        )

    # Top models last 30 days
    cutoff = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
    top_models = []
    for r in conn.execute(
        f"SELECT canonical_slug, SUM(prompt_tokens + completion_tokens) AS tt, "
        f"SUM(cached_tokens) AS cache, SUM(request_count) AS reqs "
        f"FROM {table} WHERE date >= ? "
        f"GROUP BY canonical_slug ORDER BY tt DESC LIMIT 10",
        (cutoff,),
    ):
        top_models.append(
            {
                "canonical_slug": r["canonical_slug"],
                "total_tokens": int(r["tt"] or 0),
                "cached_tokens": int(r["cache"] or 0),
                "request_count": int(r["reqs"] or 0),
            }
        )

    return {
        "monthly": monthly,
        "tier_monthly": {"months": months_sorted, "series": tier_series},
        "cache_ratio": cache_ratio,
        "top_models_30d": top_models,
    }


def spend_and_tokens_panel(conn: sqlite3.Connection) -> dict:
    """Anthropic-only monthly tokens, tier mix, cache ratio, top models."""
    return _spend_panel(
        conn,
        "anthropic_breakdown",
        _tier_from_slug_anthropic,
        ["opus", "sonnet", "haiku", "fable", "other"],
    )


def openai_spend_and_tokens_panel(conn: sqlite3.Connection) -> dict:
    """OpenAI-side mirror of spend_and_tokens_panel — same shape, different
    slug source (openai_breakdown) and tier taxonomy (codex / gpt5.6-pro /
    gpt5.6 / gpt5.5 / other)."""
    return _spend_panel(
        conn,
        "openai_breakdown",
        _tier_from_slug_openai,
        ["codex", "gpt5.6-pro", "gpt5.6", "gpt5.5", "other"],
    )


def prediction_history(conn: sqlite3.Connection) -> list[dict]:
    """Return the last published headline for each forecast month."""
    return [
        {
            "as_of": row["forecast_date"],
            "target_month": row["target_month"],
            "arr_b": row["predicted_arr_b"],
            "ci_low": row["ci_low"],
            "ci_high": row["ci_high"],
        }
        for row in conn.execute(
            "SELECT forecast_date, target_month, predicted_arr_b, ci_low, ci_high "
            "FROM arr_predictions_history p "
            "WHERE method = 'hero' AND signal = 'ensemble' "
            "AND forecast_date = ("
            "  SELECT MAX(forecast_date) FROM arr_predictions_history "
            "  WHERE target_month = p.target_month AND method = 'hero' "
            "  AND signal = 'ensemble') "
            "ORDER BY target_month"
        )
    ]


def build_payload(
    conn: sqlite3.Connection,
    target_month: str | None = None,
    admin_enabled: bool = False,
) -> dict:
    last_sync_row = conn.execute("SELECT MAX(sync_at) AS s FROM sync_log").fetchone()
    last_sync_at = last_sync_row["s"] if last_sync_row and last_sync_row["s"] else None

    predictor = predict_arr(conn, target_month=target_month)

    # Public-facing meta block — hero shows the proxy-signal ensemble as the
    # headline because it blends multiple independent public signals and is
    # robust to a single-signal regime shift. The npm-SDK growth indicator
    # (still the strongest individual signal by walk-forward MAPE) and the
    # growth-rate extrapolation are shown as cross-checks. CI for the
    # headline comes from the ensemble's own inverse-variance-weighted band.
    public_meta: dict = {}
    if predictor:
        ens = predictor.get("ensemble") or {}
        ge = predictor.get("growth_extrapolation") or {}
        ng = predictor.get("npm_growth_indicator") or {}
        if ens.get("predicted_arr_b") is not None:
            public_meta["hero_arr_b"] = ens["predicted_arr_b"]
            public_meta["hero_ci_low"] = ens.get("ci_low")
            public_meta["hero_ci_high"] = ens.get("ci_high")
            public_meta["hero_method"] = "proxy-signal ensemble"
            public_meta["hero_alt_arr_b"] = ng.get("predicted_arr_b")
            public_meta["hero_alt_method"] = "npm-SDK growth indicator"
            public_meta["hero_alt2_arr_b"] = ge.get("predicted_arr_b")
            public_meta["hero_alt2_method"] = "growth-rate extrapolation"
        elif ng.get("predicted_arr_b") is not None:
            public_meta["hero_arr_b"] = ng["predicted_arr_b"]
            public_meta["hero_ci_low"] = ng.get("ci_low")
            public_meta["hero_ci_high"] = ng.get("ci_high")
            public_meta["hero_method"] = "npm-SDK growth indicator"
            public_meta["hero_alt_arr_b"] = ge.get("predicted_arr_b")
            public_meta["hero_alt_method"] = "growth-rate extrapolation"
        elif ge.get("predicted_arr_b") is not None:
            public_meta["hero_arr_b"] = ge["predicted_arr_b"]
            public_meta["hero_ci_low"] = ge.get("ci_low")
            public_meta["hero_ci_high"] = ge.get("ci_high")
            public_meta["hero_method"] = "growth-rate extrapolation"
        public_meta["hero_month_label"] = _month_label(
            predictor.get("target_month") or ""
        )
        public_meta["n_signals_used"] = ens.get("n_models")
        public_meta["n_signals_total"] = ens.get("n_models_total")
    public_meta["last_updated_iso"] = last_sync_at
    if last_sync_at:
        try:
            next_dt = datetime.fromisoformat(last_sync_at) + timedelta(days=7)
            public_meta["next_update_expected_iso"] = next_dt.isoformat(
                timespec="seconds"
            )
        except ValueError:
            public_meta["next_update_expected_iso"] = None
    else:
        public_meta["next_update_expected_iso"] = None

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "last_sync_at": last_sync_at,
        "admin_enabled": bool(admin_enabled),
        "public_meta": public_meta,
        "signal_labels": SIGNAL_LABELS,
        "metric_labels": METRIC_LABELS,
        "channel_labels": CHANNEL_LABELS,
        "predictor": predictor,
        "spend_tokens": spend_and_tokens_panel(conn),
        "spend_tokens_openai": openai_spend_and_tokens_panel(conn),
        "backtest": backtest_summary(conn),
        # arr_curve redraws past months against newly disclosed anchors. Keep
        # the last headline published for each month so the original forecast
        # remains auditable after the curve is refit.
        "predictions_history": prediction_history(conn),
        # known_arr is already inside predictor.all_arr_known but expose
        # explicitly for the dedicated tab.
        "known_arr": [
            {
                "month": r["month"],
                "arr_b": float(r["arr_b_usd"]),
                "source": r["source"],
                "notes": r["notes"],
                "added_at": r["added_at"],
                "is_local": bool(r["is_local"]),
            }
            for r in conn.execute(
                "SELECT month, arr_b_usd, source, notes, added_at, is_local "
                "FROM anthropic_arr_known ORDER BY month"
            )
        ],
    }
