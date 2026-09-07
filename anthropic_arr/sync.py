"""Refresh standalone anthropic_arr.db from public sources.

Idempotent — safe to re-run anytime. Network fetches happen first, then a
single short transaction writes everything. The DB is never write-locked
during HTTP I/O.

Run:
    python -m anthropic_arr.sync          # full refresh
    python -m anthropic_arr.sync --rebuild  # truncate then refresh
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

from loguru import logger

from . import OWN_DB_PATH
from .db import (
    db_ctx,
    init_schema,
    log_sync,
    sync_openai_checkpoints_from_parent,
    upsert_arr_known_from_parent,
    upsert_breakdown,
    upsert_model_arena_history,
    upsert_model_meta,
    upsert_model_scores,
    upsert_openai_breakdown,
    upsert_openrouter_lab_share,
    upsert_openrouter_top_models,
    upsert_openrouter_totals,
    upsert_openrouter_watchlist,
    upsert_pricing,
    upsert_provider_throughput_history,
    upsert_signals,
    upsert_vercel_history,
    upsert_vercel_leaderboard_ranked,
    upsert_vercel_leaderboard_raw,
    upsert_vercel_snapshots,
)
from .openai_anchors import openai_checkpoint_rows
from .sources import (
    HISTORICAL_ANTHROPIC_SLUGS,
    HISTORICAL_OPENAI_SLUGS,
    compute_openrouter_anthropic_signal_rows,
    fetch_anthropic_breakdown,
    fetch_anthropic_pricing,
    fetch_openai_breakdown,
    fetch_openrouter_catalog,
    fetch_signals,
    known_arr_rows,
    make_client,
)
from .sources_bedrock import fetch_bedrock_pypi
from .sources_derived import (
    compute_bedrock_spend_estimate_rows,
    compute_vercel_claude_family_cost_share_rows,
)
from .sources_github import fetch_all_github
from .sources_models_meta import fetch_models_meta
from .sources_models_scores import fetch_models_scores
from .sources_openai_consumer import fetch_openai_consumer_rows
from .sources_openrouter_market import fetch_openrouter_market
from .sources_sdk_extra import fetch_sdk_extra_rows
from .sources_vercel import fetch_vercel_gateway


def _bucket_now() -> tuple[str, str]:
    """UTC now floored to a 30-min bucket → (captured_at_iso, YYYY-MM-DD).

    Matches the upstream rolling window so re-runs inside a window overwrite
    rather than piling up near-duplicate throughput rows.
    """
    now = datetime.now(timezone.utc)
    b = now.replace(minute=(now.minute // 30) * 30, second=0, microsecond=0)
    return b.isoformat(timespec="seconds"), b.strftime("%Y-%m-%d")


def _load_prev_meta(conn) -> dict[str, dict]:
    """Previous providers-per-slug, so a failed live stats call can fall back to
    the last good value instead of wiping the row. Only `providers` is consumed,
    so json_extract it at the SQL layer rather than decoding full payloads."""
    prev_meta: dict[str, dict] = {}
    for r in conn.execute(
        "SELECT slug, json_extract(payload_json, '$.providers') AS providers "
        "FROM model_meta"
    ).fetchall():
        providers_json = r["providers"]
        if not providers_json:
            continue
        try:
            prev_meta[r["slug"]] = {"providers": json.loads(providers_json)}
        except Exception:  # noqa: BLE001
            pass
    return prev_meta


def _throughput_rows(
    model_meta_rows: list[dict], captured_at: str, date_str: str
) -> list[dict]:
    """Flatten model_meta payloads into (provider, model) throughput rows."""
    rows = []
    for m in model_meta_rows:
        slug = m.get("slug") or ""
        try:
            pl = json.loads(m.get("payload_json") or "{}")
        except Exception:  # noqa: BLE001
            continue
        for p in pl.get("providers") or []:
            prov = (p.get("provider") or "").strip()
            tps = p.get("p50_tps")
            if not prov or tps is None:
                continue
            rows.append(
                {
                    "captured_at": captured_at,
                    "date": date_str,
                    "provider": prov,
                    "canonical_slug": slug,
                    "p50_tps": tps,
                    "p75_tps": p.get("p75_tps"),
                    "p90_tps": p.get("p90_tps"),
                    "p95_tps": p.get("p95_tps"),
                    "p99_tps": p.get("p99_tps"),
                    "p50_lat_ms": p.get("p50_lat_ms"),
                    "p75_lat_ms": p.get("p75_lat_ms"),
                    "p90_lat_ms": p.get("p90_lat_ms"),
                    "p95_lat_ms": p.get("p95_lat_ms"),
                    "p99_lat_ms": p.get("p99_lat_ms"),
                    "request_count": p.get("n"),
                    "window_minutes": p.get("window_minutes"),
                    "capacity_tpm": p.get("capacity_tpm"),
                    "quantization": p.get("quantization"),
                    "provider_region": p.get("provider_region"),
                    "status": p.get("status"),
                }
            )
    return rows


# Every window metric upstream reports. A genuinely new window moves at least
# request_count, so comparing the full set makes "unchanged" an exact test.
_THRUPUT_METRICS = (
    "p50_tps",
    "p75_tps",
    "p90_tps",
    "p95_tps",
    "p99_tps",
    "p50_lat_ms",
    "p75_lat_ms",
    "p90_lat_ms",
    "p95_lat_ms",
    "p99_lat_ms",
    "request_count",
)


def _drop_unchanged(conn, new_rows: list[dict]) -> list[dict]:
    """Keep only readings that differ from that (provider, model)'s last stored
    one.

    OpenRouter's rolling window refreshes hourly while this job runs every 30
    min, so roughly half of each run is the same upstream window read a second
    time. Re-storing it under a fresh timestamp would inflate the window count
    the API reports per bucket and pull the p10–p90 band inward, since the
    duplicated values pile up at the centre. Filtering per (provider, model) is
    what makes this work: the fleet is ~1000 pairs, so any all-or-nothing
    comparison is defeated by the one pair that happened to move."""
    prev = {
        (r["provider"], r["canonical_slug"]): tuple(r[m] for m in _THRUPUT_METRICS)
        for r in conn.execute(
            f"SELECT provider, canonical_slug, {', '.join(_THRUPUT_METRICS)} FROM ("
            " SELECT *, ROW_NUMBER() OVER (PARTITION BY provider, canonical_slug"
            "   ORDER BY captured_at DESC) AS rn"
            " FROM provider_throughput_history) WHERE rn = 1"
        ).fetchall()
    }
    return [
        row
        for row in new_rows
        if prev.get((row["provider"], row["canonical_slug"]))
        != tuple(row.get(m) for m in _THRUPUT_METRICS)
    ]


def refresh_models_meta() -> dict[str, int]:
    """Fast path (~30-min cadence): OpenRouter model_meta + provider throughput
    only. ~96 requests (catalog + permaslug + one stats call per rostered model).
    Everything else lives in the daily refresh()."""
    init_schema()

    with make_client() as client:
        try:
            or_catalog = fetch_openrouter_catalog(client)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"openrouter catalog fetch failed: {e}")
            or_catalog = None
        with db_ctx() as _probe:
            prev_meta = _load_prev_meta(_probe)
        model_meta_rows = fetch_models_meta(
            client=client, prev_meta=prev_meta, catalog_list=or_catalog
        )

    counts: dict[str, int] = {}
    if not model_meta_rows:
        logger.warning("refresh_models_meta: fetch failed — nothing written")
        return counts

    captured_at, date_str = _bucket_now()
    thruput_rows = _throughput_rows(model_meta_rows, captured_at, date_str)
    with db_ctx() as conn:
        conn.execute("BEGIN")
        try:
            counts["model_meta"] = upsert_model_meta(conn, model_meta_rows)
            log_sync(conn, "model_meta", counts["model_meta"])
            fresh = _drop_unchanged(conn, thruput_rows) if thruput_rows else []
            skipped = len(thruput_rows) - len(fresh)
            if fresh:
                counts["provider_throughput_history"] = (
                    upsert_provider_throughput_history(conn, fresh)
                )
                log_sync(
                    conn,
                    "provider_throughput_history",
                    counts["provider_throughput_history"],
                    notes=f"{skipped} unchanged rows skipped",
                )
            else:
                log_sync(
                    conn,
                    "provider_throughput_history",
                    0,
                    notes=f"skipped (all {skipped} rows unchanged)",
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return counts


def refresh(rebuild: bool = False) -> dict[str, int]:
    """Refresh all daily sources. model_meta + provider throughput live in the
    faster refresh_models_meta(); this reads the roster it maintains from the
    DB. Returns counts per table."""
    init_schema()

    # Stage 1: fetch everything into memory, no DB locks held.
    with make_client() as client:
        # Fetch OpenRouter's full model catalog once, reuse for pricing +
        # meta. Saves one ~500KB round-trip per sync and prevents the two
        # calls from seeing slightly-different snapshots if OpenRouter
        # blips between requests.
        try:
            or_catalog = fetch_openrouter_catalog(client)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"openrouter catalog fetch failed: {e}; "
                "downstream steps will re-attempt individually"
            )
            or_catalog = None
        pricing_rows = fetch_anthropic_pricing(client, catalog=or_catalog)
        signal_rows = fetch_signals(client)
        slugs = sorted(
            {r["canonical_slug"] for r in pricing_rows}
            | set(HISTORICAL_ANTHROPIC_SLUGS)
        )
        breakdown_rows = fetch_anthropic_breakdown(client, slugs)
        # OpenAI slug set derived from the same catalog. We slice on
        # canonical_slug starting with "openai/" — that's how OpenRouter
        # names them (openai/gpt-5.4-nano, openai/gpt-5.6-sol, ...) — then
        # union the delisted-but-historically-ranked slugs, mirroring what
        # HISTORICAL_ANTHROPIC_SLUGS does for the Anthropic side. The catalog
        # alone is today's roster, so on its own it backdates the present onto
        # the past and inflates growth.
        openai_slugs = sorted(
            {
                (m.get("canonical_slug") or m.get("id") or "")
                for m in (or_catalog or [])
                if (m.get("canonical_slug") or m.get("id") or "").startswith("openai/")
            }
            | set(HISTORICAL_OPENAI_SLUGS)
        )
        openai_breakdown_rows = (
            fetch_openai_breakdown(client, openai_slugs) if openai_slugs else []
        )
        github_rows = fetch_all_github(client)
        bedrock_rows = fetch_bedrock_pypi(client)
        # OpenRouter market (rankings-daily) — needs prev latest_date for
        # regression guard, so probe DB before we start writing.
        with db_ctx() as _probe:
            prev_latest = _probe.execute(
                "SELECT MAX(date) AS d FROM openrouter_daily_totals"
            ).fetchone()["d"]
        or_market = fetch_openrouter_market(prev_latest, client=client)
        vercel_out = fetch_vercel_gateway(client=client)
        sdk_extra_rows = fetch_sdk_extra_rows(client)
        # Feed the meta roster's slugs into scores so upstream Arena/AA/vals rows
        # match by canonical slug. The roster is maintained by the faster
        # refresh_models_meta(); read it from the DB rather than re-fetching.
        # Empty on a cold DB (before the first meta run) — scores still matches
        # by canonical slug, so extra_slugs is a hint, not a requirement.
        with db_ctx() as _probe:
            roster_slugs = [
                r["slug"]
                for r in _probe.execute("SELECT slug FROM model_meta").fetchall()
            ]
        model_scores_out = fetch_models_scores(client=client, extra_slugs=roster_slugs)
    arr_rows = known_arr_rows()

    # Stage 2: one short transaction.
    counts: dict[str, int] = {}
    with db_ctx() as conn:
        conn.execute("BEGIN")
        try:
            if rebuild:
                for tbl in (
                    "anthropic_breakdown",
                    "anthropic_pricing",
                    "anthropic_signals",
                    "anthropic_arr_known",
                ):
                    conn.execute(f"DELETE FROM {tbl}")
            counts["pricing"] = upsert_pricing(conn, pricing_rows)
            log_sync(conn, "anthropic_pricing", counts["pricing"])
            # An empty fetch is logged with notes rather than as a plain 0 —
            # a silent 0 here read as a healthy sync while the table sat
            # frozen for six weeks.
            counts["breakdown"] = upsert_breakdown(conn, breakdown_rows)
            log_sync(
                conn,
                "anthropic_breakdown",
                counts["breakdown"],
                notes=None if breakdown_rows else "fetch returned 0 rows",
            )
            counts["openai_breakdown"] = upsert_openai_breakdown(
                conn, openai_breakdown_rows
            )
            log_sync(
                conn,
                "openai_breakdown",
                counts["openai_breakdown"],
                notes=None if openai_breakdown_rows else "fetch returned 0 rows",
            )
            # Compute OR-token signal AFTER breakdown is upserted.
            or_signal_rows = compute_openrouter_anthropic_signal_rows(conn)
            openai_consumer_rows = fetch_openai_consumer_rows()
            all_signal_rows = (
                signal_rows
                + github_rows
                + bedrock_rows
                + or_signal_rows
                + sdk_extra_rows
                + openai_consumer_rows
            )
            n_first_pass = upsert_signals(conn, all_signal_rows)
            counts["github"] = len(github_rows)
            counts["bedrock"] = len(bedrock_rows)
            counts["openrouter_signal"] = len(or_signal_rows)
            counts["sdk_extra"] = len(sdk_extra_rows)
            counts["openai_consumer"] = len(openai_consumer_rows)
            # Derived signals — built from rows that were just upserted.
            derived_rows = compute_bedrock_spend_estimate_rows(conn)
            n_second_pass = upsert_signals(conn, derived_rows) if derived_rows else 0
            counts["derived_bedrock"] = len(derived_rows)
            # The Vercel derived signal depends on vercel_gateway_history rows
            # being upserted first — that happens after this block, so defer
            # its computation until later in the transaction.
            counts["signals"] = n_first_pass + n_second_pass
            log_sync(conn, "anthropic_signals", counts["signals"])
            counts["arr_known"] = upsert_arr_known_from_parent(conn, arr_rows)
            log_sync(conn, "anthropic_arr_known", counts["arr_known"])
            (
                counts["openai_checkpoints"],
                counts["openai_arr_known"],
            ) = sync_openai_checkpoints_from_parent(conn, openai_checkpoint_rows())
            log_sync(conn, "openai_arr_checkpoints", counts["openai_checkpoints"])
            log_sync(conn, "openai_arr_known", counts["openai_arr_known"])
            # OpenRouter market
            if or_market is not None:
                counts["or_market_totals"] = upsert_openrouter_totals(
                    conn, or_market["totals"]
                )
                counts["or_market_lab_share"] = upsert_openrouter_lab_share(
                    conn, or_market["lab_share"]
                )
                counts["or_market_watchlist"] = upsert_openrouter_watchlist(
                    conn, or_market["watchlist"]
                )
                counts["or_market_top_models"] = upsert_openrouter_top_models(
                    conn, or_market["top_models"]
                )
                log_sync(conn, "openrouter_daily_totals", counts["or_market_totals"])
            else:
                log_sync(
                    conn,
                    "openrouter_daily_totals",
                    0,
                    notes="skipped (no key / regression / empty)",
                )
            # Vercel Gateway
            if vercel_out is not None:
                # Archive first: it is the marker _migrate_vercel_to_export
                # reads to tell export-built history from the older fetchers'.
                counts["vercel_raw_archive"] = upsert_vercel_leaderboard_raw(
                    conn, vercel_out["raw"]
                )
                counts["vercel_ranked"] = upsert_vercel_leaderboard_ranked(
                    conn, vercel_out["ranked"]
                )
                log_sync(conn, "vercel_leaderboard_raw", counts["vercel_raw_archive"])
                counts["vercel_snapshots"] = upsert_vercel_snapshots(
                    conn, vercel_out["snapshots"]
                )
                counts["vercel_history"] = upsert_vercel_history(
                    conn, vercel_out["history"]
                )
                log_sync(conn, "vercel_gateway_snapshots", counts["vercel_snapshots"])
                log_sync(conn, "vercel_gateway_history", counts["vercel_history"])
                # Now that Vercel history is in DB, materialize the
                # Claude-family cost-share derived signal so it lands in
                # anthropic_signals for the ensemble/backtest to see.
                vg_signal_rows = compute_vercel_claude_family_cost_share_rows(conn)
                if vg_signal_rows:
                    n_vg = upsert_signals(conn, vg_signal_rows)
                    counts["derived_vercel_claude_share"] = len(vg_signal_rows)
                    counts["signals"] += n_vg
            else:
                log_sync(
                    conn,
                    "vercel_gateway_snapshots",
                    0,
                    notes="skipped (labs/all export unavailable)",
                )
            # model_meta + provider_throughput_history are written by the faster
            # refresh_models_meta(); intentionally not touched here.
            if model_scores_out.get("scores_rows"):
                counts["model_scores"] = upsert_model_scores(
                    conn, model_scores_out["scores_rows"]
                )
                log_sync(conn, "model_scores", counts["model_scores"])
            if model_scores_out.get("arena_history_rows"):
                counts["model_arena_history"] = upsert_model_arena_history(
                    conn, model_scores_out["arena_history_rows"]
                )
                log_sync(conn, "model_arena_history", counts["model_arena_history"])
            errors = model_scores_out.get("errors") or {}
            if errors:
                log_sync(
                    conn,
                    "model_scores",
                    0,
                    notes=f"partial: {list(errors.keys())} failed",
                )
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
    return counts


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rebuild", action="store_true", help="Truncate tables before refreshing."
    )
    args = p.parse_args()
    counts = refresh(rebuild=args.rebuild)
    print("Refreshed from public sources:")
    for k, v in counts.items():
        print(f"  {k}: {v} rows")
    print(f"DB: {OWN_DB_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
