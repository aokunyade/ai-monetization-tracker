"""Schema + connection factory for anthropic_arr.db."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterable, Iterator

from . import OWN_DB_PATH, ROOT

DATA_DIR = ROOT / "data"

SCHEMA = """
PRAGMA journal_mode=WAL;

-- Anthropic-only daily token/cache breakdown (mirror from parent's model_breakdown
-- WHERE author='anthropic').
CREATE TABLE IF NOT EXISTS anthropic_breakdown (
    date              TEXT NOT NULL,
    canonical_slug    TEXT NOT NULL,
    variant           TEXT NOT NULL,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    cached_tokens     INTEGER,
    reasoning_tokens  INTEGER,
    request_count     INTEGER,
    volume_usd        REAL,
    PRIMARY KEY (date, canonical_slug, variant)
);
CREATE INDEX IF NOT EXISTS idx_ab_date ON anthropic_breakdown(date);
CREATE INDEX IF NOT EXISTS idx_ab_slug ON anthropic_breakdown(canonical_slug);

-- OpenAI equivalent — same schema, kept in a separate table so the Anthropic
-- ARR ensemble stays purely Anthropic-scoped (OpenAI runs on castora-style
-- checkpoint model, not proxy-signal regression). Populated by the same
-- fetch_openrouter_breakdown() function with the OpenAI slug set.
CREATE TABLE IF NOT EXISTS openai_breakdown (
    date              TEXT NOT NULL,
    canonical_slug    TEXT NOT NULL,
    variant           TEXT NOT NULL,
    prompt_tokens     INTEGER,
    completion_tokens INTEGER,
    cached_tokens     INTEGER,
    reasoning_tokens  INTEGER,
    request_count     INTEGER,
    volume_usd        REAL,
    PRIMARY KEY (date, canonical_slug, variant)
);
CREATE INDEX IF NOT EXISTS idx_oab_date ON openai_breakdown(date);
CREATE INDEX IF NOT EXISTS idx_oab_slug ON openai_breakdown(canonical_slug);

-- Anthropic latest pricing (mirror from parent's v_latest_pricing WHERE author='anthropic').
CREATE TABLE IF NOT EXISTS anthropic_pricing (
    canonical_slug         TEXT PRIMARY KEY,
    model_slug             TEXT,
    prompt_price_usd       REAL,
    completion_price_usd   REAL,
    cache_read_price_usd   REAL,
    captured_at            TEXT NOT NULL
);

-- External proxy signals: npm @anthropic-ai/sdk, npm claude-code, PyPI anthropic.
CREATE TABLE IF NOT EXISTS anthropic_signals (
    date         TEXT NOT NULL,
    signal_name  TEXT NOT NULL,
    value        REAL NOT NULL,
    captured_at  TEXT NOT NULL,
    PRIMARY KEY (date, signal_name)
);
CREATE INDEX IF NOT EXISTS idx_as_signal ON anthropic_signals(signal_name, date);

-- Known ARR ground-truth (mirror of parent's anthropic_arr_known + locally-added
-- entries. POST /api/v1/known_arr writes here directly so users can add new
-- press-leak datapoints without rebuilding parent DB. is_local=1 marks rows
-- the user added/edited via POST; sync skips those so they survive re-sync.
-- Use `python -m anthropic_arr.sync --rebuild` to force parent to overwrite.
CREATE TABLE IF NOT EXISTS anthropic_arr_known (
    month       TEXT PRIMARY KEY,
    arr_b_usd   REAL NOT NULL,
    source      TEXT,
    notes       TEXT,
    added_at    TEXT NOT NULL,
    is_local    INTEGER NOT NULL DEFAULT 0
);

-- History of past predictions, so we can chart "what we thought May ARR was
-- at each weekly forecast point" (forecast convergence).
CREATE TABLE IF NOT EXISTS arr_predictions_history (
    forecast_date     TEXT NOT NULL,
    target_month      TEXT NOT NULL,
    method            TEXT NOT NULL,
    signal            TEXT NOT NULL,
    predicted_arr_b   REAL NOT NULL,
    ci_low            REAL,
    ci_high           REAL,
    n_train           INTEGER,
    r2                REAL,
    rmse_loo          REAL,
    PRIMARY KEY (forecast_date, target_month, method, signal)
);
CREATE INDEX IF NOT EXISTS idx_aph_target ON arr_predictions_history(target_month);

-- Sync metadata
CREATE TABLE IF NOT EXISTS sync_log (
    sync_at      TEXT NOT NULL,
    table_name   TEXT NOT NULL,
    rows_synced  INTEGER NOT NULL,
    notes        TEXT
);

-- ============================================================================
-- Fusion additions (2026-07): tracker modules 1/2/5/6 + models + nowcast
-- Schema below intentionally mirrors ai-monetization-tracker functions/_lib.js
-- D1 schema where relevant, so future migration to Cloudflare edge is feasible.
-- ============================================================================

-- OpenRouter market (rankings-daily全市场)
CREATE TABLE IF NOT EXISTS openrouter_daily_totals (
    date              TEXT PRIMARY KEY,
    total_tokens_b    REAL NOT NULL,
    ma7_b             REAL,
    as_of             TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS openrouter_lab_share (
    date              TEXT NOT NULL,
    lab               TEXT NOT NULL,
    tokens_b          REAL NOT NULL,
    PRIMARY KEY (date, lab)
);
CREATE TABLE IF NOT EXISTS openrouter_watchlist (
    date              TEXT NOT NULL,
    prefix            TEXT NOT NULL,
    tokens_b          REAL NOT NULL,
    PRIMARY KEY (date, prefix)
);
CREATE TABLE IF NOT EXISTS openrouter_top_models (
    window            TEXT NOT NULL,
    permaslug         TEXT NOT NULL,
    tokens_b          REAL NOT NULL,
    as_of             TEXT NOT NULL,
    PRIMARY KEY (window, permaslug)
);

-- Vercel AI Gateway
CREATE TABLE IF NOT EXISTS vercel_gateway_snapshots (
    date              TEXT NOT NULL,
    metric            TEXT NOT NULL,       -- 'tokens' | 'cost' | 'requests'
    rank              INTEGER NOT NULL,
    name              TEXT NOT NULL,
    share_pct         REAL NOT NULL,
    PRIMARY KEY (date, metric, rank)
);
CREATE TABLE IF NOT EXISTS vercel_gateway_history (
    date              TEXT NOT NULL,
    metric            TEXT NOT NULL,
    name              TEXT NOT NULL,
    share_pct         REAL NOT NULL,
    PRIMARY KEY (date, metric, name)
);
-- Verbatim archive of every leaderboard-export slice, upstream's own metric
-- names ('spend', not 'cost'). The export only serves a rolling ~61-day window,
-- so each sync upserts that window and older rows persist here past the point
-- Vercel stops serving them. Written whether or not anything reads it.
CREATE TABLE IF NOT EXISTS vercel_leaderboard_raw (
    date              TEXT NOT NULL,
    dataset           TEXT NOT NULL,       -- 'labs' | 'models'
    modality          TEXT NOT NULL,       -- 'all' | 'text' | 'image' | 'video'
    name              TEXT NOT NULL,
    metric            TEXT NOT NULL,       -- 'spend'|'tokens'|'requests'|'imageCount'|'videoCount'
    share_pct         REAL NOT NULL,
    PRIMARY KEY (date, dataset, modality, name, metric)
);
-- apps/providers ranked lists. Undated upstream, so each row is stamped with
-- the fetch day to give the ranking a history.
CREATE TABLE IF NOT EXISTS vercel_leaderboard_ranked (
    date              TEXT NOT NULL,       -- fetch day, UTC
    dataset           TEXT NOT NULL,       -- 'apps' | 'providers'
    ranked_by         TEXT NOT NULL,       -- 'Token Volume' | 'Spend'
    rank              INTEGER NOT NULL,
    name              TEXT NOT NULL,
    url               TEXT NOT NULL DEFAULT '',
    description       TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (date, dataset, ranked_by, rank)
);

-- OpenAI ARR. `checkpoint` rows are reviewed anchors owned by
-- config/openai_arr_anchors.json — the same set the model reads — and sync
-- re-upserts them so the served ledger and the model can't drift apart.
-- `estimate` remains a supported admin-only row kind but is never a published
-- model input. is_local=1 marks POST /api/v1/openai_arr edits.
CREATE TABLE IF NOT EXISTS openai_arr_checkpoints (
    kind              TEXT NOT NULL,       -- 'estimate' | 'checkpoint'
    date              TEXT NOT NULL,
    arr_bn            REAL NOT NULL,
    source            TEXT,
    url               TEXT,
    note              TEXT,
    classification    TEXT NOT NULL DEFAULT 'reported',
    is_local          INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (kind, date)
);
CREATE TABLE IF NOT EXISTS openai_arr_extrapolation (
    as_of             TEXT PRIMARY KEY,
    to_date           TEXT NOT NULL,
    low_bn            REAL NOT NULL,
    high_bn           REAL NOT NULL
);

-- Materialized model input for predict_openai_arr. Rebuilt only from
-- manifest-managed, non-local checkpoint rows.
CREATE TABLE IF NOT EXISTS openai_arr_known (
    month       TEXT PRIMARY KEY,
    arr_b_usd   REAL NOT NULL,
    source      TEXT,
    notes       TEXT,
    added_at    TEXT NOT NULL,
    is_local    INTEGER NOT NULL DEFAULT 0
);

-- Prompt Arena
CREATE TABLE IF NOT EXISTS arena_cases (
    id                TEXT PRIMARY KEY,
    updated           TEXT NOT NULL,
    payload_json      TEXT NOT NULL
);

-- Models capability matrix (tracker fetch_model_meta.py + fetch_model_scores.py)
CREATE TABLE IF NOT EXISTS model_meta (
    slug              TEXT PRIMARY KEY,
    payload_json      TEXT NOT NULL,
    as_of             TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_curated (
    slug              TEXT PRIMARY KEY,
    payload_json      TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_scores (
    slug              TEXT NOT NULL,
    source            TEXT NOT NULL,       -- 'arena' | 'aa' | 'vals'
    payload_json      TEXT NOT NULL,
    as_of             TEXT NOT NULL,
    PRIMARY KEY (slug, source)
);
CREATE TABLE IF NOT EXISTS model_arena_history (
    slug              TEXT NOT NULL,
    date              TEXT NOT NULL,
    overall           REAL,
    coding            REAL,
    webdev            REAL,
    PRIMARY KEY (slug, date)
);

-- Intra-day snapshot of every (provider, model) endpoint's OpenRouter-reported
-- 30-min-window p50/p99 throughput + latency + request volume. One row per
-- (30-min bucket, provider, model). `captured_at` is the UTC sample time floored
-- to a 30-min boundary — matching the upstream rolling window — so a re-run
-- within the same window overwrites rather than piling up near-duplicate rows,
-- while successive windows accumulate an intra-day series.
--
-- Read as a supply-side proxy PER PROVIDER (rising tok/s → GPU pressure easing).
-- Do NOT blend into a single market percentile: percentiles don't compose, and
-- a request-weighted blend confounds throughput with provider mix-shift. `date`
-- is derived from captured_at for daily rollups.
CREATE TABLE IF NOT EXISTS provider_throughput_history (
    captured_at       TEXT NOT NULL,   -- UTC ISO, floored to a 30-min bucket
    date              TEXT NOT NULL,   -- YYYY-MM-DD (derived from captured_at)
    provider          TEXT NOT NULL,
    canonical_slug    TEXT NOT NULL,
    p50_tps           REAL,            -- throughput (generation tok/s)
    p75_tps           REAL,
    p90_tps           REAL,
    p95_tps           REAL,
    p99_tps           REAL,
    p50_lat_ms        REAL,            -- latency = TTFT (time to first token, ms)
    p75_lat_ms        REAL,
    p90_lat_ms        REAL,
    p95_lat_ms        REAL,
    p99_lat_ms        REAL,
    request_count     INTEGER,
    window_minutes    INTEGER,         -- upstream rolling-window size (=30)
    capacity_tpm      REAL,            -- provider capacity (tokens/min) — supply signal
    quantization      TEXT,            -- e.g. fp8 / bf16 — affects throughput
    provider_region   TEXT,
    status            TEXT,            -- endpoint health/status
    PRIMARY KEY (captured_at, provider, canonical_slug)
);
CREATE INDEX IF NOT EXISTS idx_pth_captured ON provider_throughput_history(captured_at);
CREATE INDEX IF NOT EXISTS idx_pth_date ON provider_throughput_history(date);
CREATE INDEX IF NOT EXISTS idx_pth_provider ON provider_throughput_history(provider);
"""


def get_db(path=None) -> sqlite3.Connection:
    p = path or OWN_DB_PATH
    conn = sqlite3.connect(p, timeout=60, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")
    return conn


@contextmanager
def db_ctx(path=None) -> Iterator[sqlite3.Connection]:
    conn = get_db(path)
    try:
        yield conn
    finally:
        conn.close()


def init_schema(conn: sqlite3.Connection | None = None) -> None:
    own = conn is None
    conn = conn or get_db()
    try:
        # Runs before SCHEMA: the new idx_pth_captured index references a column
        # the pre-migration table lacks, so the table must be reshaped first.
        _migrate_provider_throughput_pk(conn)
        conn.executescript(SCHEMA)
        # Widen provider_throughput_history for DBs created before the extra
        # percentiles / supply-signal columns were added (additive, nullable).
        _ensure_throughput_columns(conn)
        _prune_duplicate_throughput_rows(conn)
        cols = {
            r[1]
            for r in conn.execute("PRAGMA table_info(anthropic_arr_known)").fetchall()
        }
        if "is_local" not in cols:
            conn.execute(
                "ALTER TABLE anthropic_arr_known "
                "ADD COLUMN is_local INTEGER NOT NULL DEFAULT 0"
            )
        cols = {
            r[1]
            for r in conn.execute(
                "PRAGMA table_info(openai_arr_checkpoints)"
            ).fetchall()
        }
        if "is_local" not in cols:
            conn.execute(
                "ALTER TABLE openai_arr_checkpoints "
                "ADD COLUMN is_local INTEGER NOT NULL DEFAULT 0"
            )
        if "classification" not in cols:
            conn.execute(
                "ALTER TABLE openai_arr_checkpoints "
                "ADD COLUMN classification TEXT NOT NULL DEFAULT 'reported'"
            )
            conn.execute(
                "UPDATE openai_arr_checkpoints "
                "SET classification = 'third_party_estimate' "
                "WHERE kind = 'estimate'"
            )
        _migrate_model_scores_pk(conn)
        _migrate_vercel_to_export(conn)
        _migrate_arr_known_code_owned(conn)
        owns_anchor_transaction = not conn.in_transaction
        if owns_anchor_transaction:
            conn.execute("BEGIN IMMEDIATE")
        try:
            from .sources import known_arr_rows

            upsert_arr_known_from_parent(conn, known_arr_rows())
            _seed_openai_arr(conn)
            if owns_anchor_transaction:
                conn.execute("COMMIT")
        except Exception:
            if owns_anchor_transaction:
                conn.execute("ROLLBACK")
            raise
        _seed_models_curated(conn)
        # _seed_arena_cases: arena stashed 2026-07-14, see _stash/arena/README.md.
        # Table + seed function retained for a future revival.
    finally:
        if own:
            conn.close()


def _prune_duplicate_throughput_rows(conn: sqlite3.Connection) -> None:
    """Drop stored windows that only repeat the previous reading for the same
    (provider, model).

    Until the writer filtered per row, every 30-min run stored the whole fleet
    even though OpenRouter's window only refreshes hourly — so ~56% of rows were
    the same upstream window a second time, inflating the window count each
    bucket reports and narrowing its p10–p90 band. Idempotent: the writer no
    longer emits these, so after the first pass there is nothing to delete."""
    conn.execute("""
        DELETE FROM provider_throughput_history WHERE rowid IN (
          SELECT rowid FROM (
            SELECT rowid,
                   p50_tps, p50_lat_ms, request_count,
                   LAG(p50_tps)       OVER w AS prev_tps,
                   LAG(p50_lat_ms)    OVER w AS prev_lat,
                   LAG(request_count) OVER w AS prev_n
            FROM provider_throughput_history
            WINDOW w AS (PARTITION BY provider, canonical_slug ORDER BY captured_at)
          )
          WHERE p50_tps IS prev_tps AND p50_lat_ms IS prev_lat
            AND request_count IS prev_n
        )
    """)


def _ensure_throughput_columns(conn: sqlite3.Connection) -> None:
    """Additive, idempotent widening of provider_throughput_history. New metric
    columns are nullable, so ALTER TABLE ADD COLUMN backfills existing rows with
    NULL — no rebuild. Mirrors the is_local pattern above."""
    cols = {
        r[1]
        for r in conn.execute(
            "PRAGMA table_info(provider_throughput_history)"
        ).fetchall()
    }
    extra = {
        "p75_tps": "REAL",
        "p90_tps": "REAL",
        "p95_tps": "REAL",
        "p75_lat_ms": "REAL",
        "p90_lat_ms": "REAL",
        "p95_lat_ms": "REAL",
        "window_minutes": "INTEGER",
        "capacity_tpm": "REAL",
        "quantization": "TEXT",
        "provider_region": "TEXT",
        "status": "TEXT",
    }
    for name, typ in extra.items():
        if name not in cols:
            conn.execute(
                f"ALTER TABLE provider_throughput_history ADD COLUMN {name} {typ}"
            )


def _migrate_provider_throughput_pk(conn: sqlite3.Connection) -> None:
    """The original throughput table was keyed by (date, provider, canonical_slug),
    so it held at most one row per UTC day and 30-min re-samples overwrote each
    other. Migrate to (captured_at, provider, canonical_slug) so an intra-day
    series can accumulate. Existing daily rows are preserved with a synthetic
    midnight captured_at.

    Wrapped in explicit BEGIN/COMMIT so a crash mid-migration doesn't strand
    data in the _old table.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' "
        "AND name='provider_throughput_history'"
    ).fetchone()
    # No table yet (fresh DB) → SCHEMA will create it. Already has captured_at →
    # nothing to do. Both cases are no-ops, so the migration is idempotent.
    if not row or "captured_at" in (row["sql"] or ""):
        return
    # Per-statement execute (not executescript, which force-commits and would
    # break the explicit transaction) so the rename+recreate+copy+drop is atomic.
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute(
            "ALTER TABLE provider_throughput_history "
            "RENAME TO provider_throughput_history_old"
        )
        conn.execute("""
            CREATE TABLE provider_throughput_history (
                captured_at       TEXT NOT NULL,
                date              TEXT NOT NULL,
                provider          TEXT NOT NULL,
                canonical_slug    TEXT NOT NULL,
                p50_tps           REAL,
                p99_tps           REAL,
                p50_lat_ms        REAL,
                p99_lat_ms        REAL,
                request_count     INTEGER,
                PRIMARY KEY (captured_at, provider, canonical_slug)
            )
        """)
        conn.execute("""
            INSERT INTO provider_throughput_history
                (captured_at, date, provider, canonical_slug, p50_tps, p99_tps,
                 p50_lat_ms, p99_lat_ms, request_count)
            SELECT date || 'T00:00:00+00:00', date, provider, canonical_slug,
                   p50_tps, p99_tps, p50_lat_ms, p99_lat_ms, request_count
            FROM provider_throughput_history_old
        """)
        conn.execute("DROP TABLE provider_throughput_history_old")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def _migrate_model_scores_pk(conn: sqlite3.Connection) -> None:
    """Early builds of the fusion schema had model_scores with PK=(slug) which
    collapsed arena/aa/vals into one row per slug. Detect that and migrate to
    PK=(slug, source) preserving the data we have.

    Wrapped in explicit BEGIN/COMMIT (connection is autocommit) so a crash
    mid-migration doesn't leave data stranded in model_scores_old — the whole
    rename+recreate+copy either lands together or not at all.
    """
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='model_scores'"
    ).fetchone()
    if not row or "PRIMARY KEY (slug, source)" in (row["sql"] or ""):
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.executescript("""
            ALTER TABLE model_scores RENAME TO model_scores_old;
            CREATE TABLE model_scores (
                slug              TEXT NOT NULL,
                source            TEXT NOT NULL,
                payload_json      TEXT NOT NULL,
                as_of             TEXT NOT NULL,
                PRIMARY KEY (slug, source)
            );
            INSERT INTO model_scores(slug, source, payload_json, as_of)
            SELECT slug, source, payload_json, as_of FROM model_scores_old;
            DROP TABLE model_scores_old;
        """)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def _migrate_vercel_to_export(conn: sqlite3.Connection) -> None:
    """Drop vercel_gateway_* rows written before the documented-export switch.

    Two generations of fetcher wrote wrong values here and neither can be
    corrected in place:

      - The scrape summed lab families from the leaderboard's model roster,
        which only ships today's top-10, so every past day omitted models that
        have since dropped out (OpenAI read 0.04% for May against 14.70%).
      - The undocumented v4 endpoint drops Fable 5 from anthropic's numerator
        and from the shared denominator, so anthropic read 10-17pp low through
        July and every other lab read high.

    `vercel_leaderboard_raw` is the marker: only the export fetcher writes it,
    and it writes it in the same transaction as the history. So raw-empty +
    history-present means a pre-export fetcher built that history. A fresh DB
    has both empty and skips. Once the next sync populates raw, this never fires
    again.
    """
    raw = conn.execute("SELECT COUNT(*) AS n FROM vercel_leaderboard_raw").fetchone()
    if raw["n"]:
        return
    stale = conn.execute("SELECT COUNT(*) AS n FROM vercel_gateway_history").fetchone()
    if not stale["n"]:
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("DELETE FROM vercel_gateway_history")
        conn.execute("DELETE FROM vercel_gateway_snapshots")
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def _migrate_arr_known_code_owned(conn: sqlite3.Connection) -> None:
    """Clear is_local on anchors that KNOWN_ARR now carries.

    An is_local=1 row that KNOWN_ARR does not define is invisible to code: the
    2025-06 The Information anchor lived only here, so `sync --rebuild` — the
    documented way to let the parent overwrite — deleted it with nothing to
    restore it from. Now that KNOWN_ARR defines it, is_local=1 would instead
    pin the row against future edits to that list. Clearing the flag makes code
    the single owner; genuine local-only anchors keep theirs.
    """
    from .sources import KNOWN_ARR

    months = [m for m, _v, _s, _n in KNOWN_ARR]
    conn.execute(
        "UPDATE anthropic_arr_known SET is_local = 0 "
        f"WHERE is_local = 1 AND month IN ({','.join('?' * len(months))})",
        months,
    )


def _seed_openai_arr(conn: sqlite3.Connection) -> None:
    """Sync the reviewed manifest; no legacy estimate seed is shipped."""
    from .openai_anchors import openai_checkpoint_rows

    conn.execute("DELETE FROM openai_arr_extrapolation")
    sync_openai_checkpoints_from_parent(conn, openai_checkpoint_rows())


def _seed_openai_arr_known(conn: sqlite3.Connection) -> None:
    """Rebuild openai_arr_known only from manifest-managed checkpoints.

    Local/admin and legacy estimate rows are excluded so they cannot bypass the
    reviewed manifest on a published snapshot.
    """
    rows = conn.execute(
        "SELECT kind, date, arr_bn, source, url, note, classification "
        "FROM openai_arr_checkpoints "
        "WHERE kind = 'checkpoint' AND is_local = 0 ORDER BY date"
    ).fetchall()
    conn.execute("DELETE FROM openai_arr_known")
    if not rows:
        return
    # Group by YYYY-MM: checkpoint beats estimate, and within a kind the latest
    # date wins. Rows arrive in date order, so "latest wins" is just letting a
    # same-kind row overwrite.
    #
    # It used to be first-wins among checkpoints, which only held while no month
    # carried two of them. This series is month-end ARR, so the reading closest
    # to month end is the one that represents the month.
    by_month: dict[str, tuple] = {}
    for r in rows:
        mo = r["date"][:7]
        existing = by_month.get(mo)
        if existing is None or r["kind"] == "checkpoint" or existing[0] != "checkpoint":
            by_month[mo] = (
                r["kind"],
                r["arr_bn"],
                r["source"],
                r["url"],
                r["note"],
                r["classification"],
            )
    added_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    prepared = []
    for mo, (_kind, arr_bn, source, url, note, classification) in by_month.items():
        notes_parts = []
        if note:
            notes_parts.append(str(note))
        if url:
            notes_parts.append(str(url))
        prepared.append(
            {
                "month": mo,
                "arr_b_usd": float(arr_bn),
                "source": f"{classification}:{source or 'openai_arr_checkpoints'}",
                "notes": " | ".join(notes_parts) or None,
                "added_at": added_at,
            }
        )
    sql = (
        "INSERT INTO openai_arr_known "
        "  (month, arr_b_usd, source, notes, added_at, is_local) "
        "VALUES (:month, :arr_b_usd, :source, :notes, :added_at, 0)"
    )
    conn.executemany(sql, prepared)


def _seed_models_curated(conn: sqlite3.Connection) -> None:
    """First-run only: load models_curated_seed.json into model_curated.

    Tracker's models_curated.json is a global config blob (benchmarks defs,
    scores overrides list, extra_models/facts dicts) — not a per-slug map.
    We store the whole blob under slug='__config__' plus fan out per-model
    entries from `scores`/`extra_models`/`facts` for per-slug lookups.
    """
    existing = conn.execute("SELECT COUNT(*) FROM model_curated").fetchone()[0]
    if existing:
        return
    seed_path = DATA_DIR / "models_curated_seed.json"
    if not seed_path.exists():
        return
    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    rows: list[tuple] = []
    rows.append(("__config__", json.dumps(payload, ensure_ascii=False)))
    for entry in payload.get("scores", []) or []:
        if isinstance(entry, dict) and entry.get("slug"):
            rows.append((entry["slug"], json.dumps(entry, ensure_ascii=False)))
    for slug, meta in (payload.get("extra_models") or {}).items():
        if isinstance(meta, dict):
            rows.append((slug, json.dumps({"slug": slug, **meta}, ensure_ascii=False)))
    for slug, fact in (payload.get("facts") or {}).items():
        # facts entries can overwrite an earlier scores/extra row — that's fine,
        # facts are the most curated layer.
        payload_merged = (
            {"slug": slug, "fact": fact}
            if not isinstance(fact, dict)
            else {"slug": slug, **fact}
        )
        rows.append((slug, json.dumps(payload_merged, ensure_ascii=False)))
    if rows:
        # Later rows for same slug overwrite earlier (facts > extra > scores > config).
        deduped: dict[str, str] = {}
        for slug, j in rows:
            deduped[slug] = j
        conn.executemany(
            "INSERT OR IGNORE INTO model_curated (slug, payload_json) VALUES (?,?)",
            list(deduped.items()),
        )


def _seed_arena_cases(conn: sqlite3.Connection) -> None:
    """Seed arena_cases from data/arena/cases.json — first-run only per case.

    We use INSERT OR IGNORE so any future admin edit of an existing case
    (via a future PUT /arena/cases/<id> route) is not clobbered on restart.
    New cases added to cases.json on disk still get picked up because their
    id doesn't collide with existing rows. Case files themselves live on disk
    under data/arena/<id>/* and are always read from there (routes_arena),
    so this table is currently a manifest cache, not authoritative storage.
    """
    seed_path = DATA_DIR / "arena" / "cases.json"
    if not seed_path.exists():
        return
    payload = json.loads(seed_path.read_text(encoding="utf-8"))
    updated = payload.get("updated") or datetime.now(timezone.utc).date().isoformat()
    rows = []
    for case in payload.get("cases", []):
        cid = case.get("id")
        if cid:
            rows.append((cid, updated, json.dumps(case, ensure_ascii=False)))
    if rows:
        conn.executemany(
            "INSERT OR IGNORE INTO arena_cases (id, updated, payload_json) "
            "VALUES (?,?,?)",
            rows,
        )


def upsert_arr_known(conn: sqlite3.Connection, rows: Iterable[dict]) -> int:
    """Upsert ARR rows. Each row may set `is_local` (default 0)."""
    sql = (
        "INSERT INTO anthropic_arr_known "
        "  (month, arr_b_usd, source, notes, added_at, is_local) "
        "VALUES (:month, :arr_b_usd, :source, :notes, :added_at, :is_local) "
        "ON CONFLICT(month) DO UPDATE SET "
        "  arr_b_usd=excluded.arr_b_usd, source=excluded.source, "
        "  notes=excluded.notes, added_at=excluded.added_at, "
        "  is_local=excluded.is_local"
    )
    prepared = [{"is_local": 0, **r} for r in rows]
    conn.executemany(sql, prepared)
    return len(prepared)


def rebuild_openai_arr_known(conn: sqlite3.Connection) -> int:
    """Sync-time rebuild of openai_arr_known from openai_arr_checkpoints.

    Same semantics as _seed_openai_arr_known but exposed publicly so the sync
    step can force a refresh right after ingesting any new checkpoint rows.
    Returns the row count in openai_arr_known after the rebuild."""
    _seed_openai_arr_known(conn)
    return conn.execute("SELECT COUNT(*) FROM openai_arr_known").fetchone()[0]


def upsert_arr_known_from_parent(conn: sqlite3.Connection, rows: Iterable[dict]) -> int:
    """Replace code-owned rows while preserving is_local=1 user edits.

    Returns the number of rows actually inserted/updated, not attempted.
    """
    rows = list(rows)
    months = [row["month"] for row in rows]
    if months:
        placeholders = ",".join("?" for _ in months)
        conn.execute(
            "DELETE FROM anthropic_arr_known WHERE is_local = 0 "
            f"AND month NOT IN ({placeholders})",
            months,
        )
    else:
        conn.execute("DELETE FROM anthropic_arr_known WHERE is_local = 0")
    sql = (
        "INSERT INTO anthropic_arr_known "
        "  (month, arr_b_usd, source, notes, added_at, is_local) "
        "VALUES (:month, :arr_b_usd, :source, :notes, :added_at, 0) "
        "ON CONFLICT(month) DO UPDATE SET "
        "  arr_b_usd=excluded.arr_b_usd, source=excluded.source, "
        "  notes=excluded.notes, added_at=excluded.added_at "
        "WHERE anthropic_arr_known.is_local = 0"
    )
    applied = 0
    for row in rows:
        cur = conn.execute(sql, row)
        applied += cur.rowcount
    return applied


def upsert_breakdown(conn: sqlite3.Connection, rows: Iterable[dict]) -> int:
    sql = (
        "INSERT OR REPLACE INTO anthropic_breakdown "
        "(date, canonical_slug, variant, prompt_tokens, completion_tokens, "
        " cached_tokens, reasoning_tokens, request_count, volume_usd) "
        "VALUES (:date, :canonical_slug, :variant, :prompt_tokens, "
        ":completion_tokens, :cached_tokens, :reasoning_tokens, :request_count, "
        ":volume_usd)"
    )
    rows = list(rows)
    conn.executemany(sql, rows)
    return len(rows)


def upsert_openai_breakdown(conn: sqlite3.Connection, rows: Iterable[dict]) -> int:
    """Same schema as upsert_breakdown, different table (openai_breakdown)."""
    sql = (
        "INSERT OR REPLACE INTO openai_breakdown "
        "(date, canonical_slug, variant, prompt_tokens, completion_tokens, "
        " cached_tokens, reasoning_tokens, request_count, volume_usd) "
        "VALUES (:date, :canonical_slug, :variant, :prompt_tokens, "
        ":completion_tokens, :cached_tokens, :reasoning_tokens, :request_count, "
        ":volume_usd)"
    )
    rows = list(rows)
    conn.executemany(sql, rows)
    return len(rows)


def upsert_pricing(conn: sqlite3.Connection, rows: Iterable[dict]) -> int:
    sql = (
        "INSERT OR REPLACE INTO anthropic_pricing "
        "(canonical_slug, model_slug, prompt_price_usd, completion_price_usd, "
        " cache_read_price_usd, captured_at) "
        "VALUES (:canonical_slug, :model_slug, :prompt_price_usd, "
        ":completion_price_usd, :cache_read_price_usd, :captured_at)"
    )
    rows = list(rows)
    conn.executemany(sql, rows)
    return len(rows)


def upsert_signals(conn: sqlite3.Connection, rows: Iterable[dict]) -> int:
    sql = (
        "INSERT OR REPLACE INTO anthropic_signals "
        "(date, signal_name, value, captured_at) "
        "VALUES (:date, :signal_name, :value, :captured_at)"
    )
    rows = list(rows)
    conn.executemany(sql, rows)
    return len(rows)


def record_prediction(conn: sqlite3.Connection, rows: Iterable[dict]) -> int:
    sql = (
        "INSERT OR REPLACE INTO arr_predictions_history "
        "(forecast_date, target_month, method, signal, predicted_arr_b, "
        " ci_low, ci_high, n_train, r2, rmse_loo) "
        "VALUES (:forecast_date, :target_month, :method, :signal, "
        ":predicted_arr_b, :ci_low, :ci_high, :n_train, :r2, :rmse_loo)"
    )
    rows = list(rows)
    conn.executemany(sql, rows)
    return len(rows)


def log_sync(
    conn: sqlite3.Connection, table_name: str, rows: int, notes: str | None = None
) -> None:
    conn.execute(
        "INSERT INTO sync_log (sync_at, table_name, rows_synced, notes) "
        "VALUES (?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            table_name,
            rows,
            notes,
        ),
    )


# ============================================================================
# Fusion upsert helpers
# ============================================================================


def upsert_openrouter_totals(conn: sqlite3.Connection, rows: Iterable[dict]) -> int:
    sql = (
        "INSERT INTO openrouter_daily_totals "
        "(date, total_tokens_b, ma7_b, as_of) VALUES "
        "(:date, :total_tokens_b, :ma7_b, :as_of) "
        "ON CONFLICT(date) DO UPDATE SET "
        "total_tokens_b=excluded.total_tokens_b, ma7_b=excluded.ma7_b, "
        "as_of=excluded.as_of"
    )
    rows = list(rows)
    conn.executemany(sql, rows)
    return len(rows)


def upsert_openrouter_lab_share(conn: sqlite3.Connection, rows: Iterable[dict]) -> int:
    sql = (
        "INSERT OR REPLACE INTO openrouter_lab_share "
        "(date, lab, tokens_b) VALUES (:date, :lab, :tokens_b)"
    )
    rows = list(rows)
    conn.executemany(sql, rows)
    return len(rows)


def upsert_openrouter_watchlist(conn: sqlite3.Connection, rows: Iterable[dict]) -> int:
    sql = (
        "INSERT OR REPLACE INTO openrouter_watchlist "
        "(date, prefix, tokens_b) VALUES (:date, :prefix, :tokens_b)"
    )
    rows = list(rows)
    conn.executemany(sql, rows)
    return len(rows)


def upsert_openrouter_top_models(conn: sqlite3.Connection, rows: Iterable[dict]) -> int:
    # A fresh top-15 replaces the prior window snapshot; delete-then-insert.
    windows = {r["window"] for r in rows}
    for w in windows:
        conn.execute("DELETE FROM openrouter_top_models WHERE window = ?", (w,))
    sql = (
        "INSERT OR REPLACE INTO openrouter_top_models "
        "(window, permaslug, tokens_b, as_of) VALUES "
        "(:window, :permaslug, :tokens_b, :as_of)"
    )
    rows = list(rows)
    conn.executemany(sql, rows)
    return len(rows)


def upsert_vercel_snapshots(conn: sqlite3.Connection, rows: Iterable[dict]) -> int:
    sql = (
        "INSERT OR REPLACE INTO vercel_gateway_snapshots "
        "(date, metric, rank, name, share_pct) VALUES "
        "(:date, :metric, :rank, :name, :share_pct)"
    )
    rows = list(rows)
    conn.executemany(sql, rows)
    return len(rows)


def upsert_vercel_leaderboard_raw(
    conn: sqlite3.Connection, rows: Iterable[dict]
) -> int:
    sql = (
        "INSERT OR REPLACE INTO vercel_leaderboard_raw "
        "(date, dataset, modality, name, metric, share_pct) VALUES "
        "(:date, :dataset, :modality, :name, :metric, :share_pct)"
    )
    rows = list(rows)
    conn.executemany(sql, rows)
    return len(rows)


def upsert_vercel_leaderboard_ranked(
    conn: sqlite3.Connection, rows: Iterable[dict]
) -> int:
    sql = (
        "INSERT OR REPLACE INTO vercel_leaderboard_ranked "
        "(date, dataset, ranked_by, rank, name, url, description) VALUES "
        "(:date, :dataset, :ranked_by, :rank, :name, :url, :description)"
    )
    rows = list(rows)
    conn.executemany(sql, rows)
    return len(rows)


def upsert_vercel_history(conn: sqlite3.Connection, rows: Iterable[dict]) -> int:
    sql = (
        "INSERT OR REPLACE INTO vercel_gateway_history "
        "(date, metric, name, share_pct) VALUES "
        "(:date, :metric, :name, :share_pct)"
    )
    rows = list(rows)
    conn.executemany(sql, rows)
    return len(rows)


def upsert_openai_arr_checkpoint(
    conn: sqlite3.Connection,
    kind: str,
    date: str,
    arr_bn: float,
    source: str | None = None,
    url: str | None = None,
    note: str | None = None,
) -> int:
    """Admin POST path — marks the row is_local so sync leaves it alone."""
    classification = "reported" if kind == "checkpoint" else "third_party_estimate"
    conn.execute(
        "INSERT INTO openai_arr_checkpoints "
        "(kind, date, arr_bn, source, url, note, classification, is_local) "
        "VALUES (?,?,?,?,?,?,?,1) "
        "ON CONFLICT(kind, date) DO UPDATE SET "
        "arr_bn=excluded.arr_bn, source=excluded.source, "
        "url=excluded.url, note=excluded.note, "
        "classification=excluded.classification, is_local=1",
        (kind, date, arr_bn, source, url, note, classification),
    )
    return 1


def upsert_openai_checkpoints_from_parent(
    conn: sqlite3.Connection, rows: Iterable[dict]
) -> int:
    """Sync the authoritative reviewed OpenAI anchor manifest.

    A local row at an exact manifest coordinate is reclaimed by the manifest;
    non-colliding local rows remain preserved. Returns the number of rows
    inserted, updated, or pruned.
    """
    sql = (
        "INSERT INTO openai_arr_checkpoints "
        "  (kind, date, arr_bn, source, url, note, classification, is_local) "
        "VALUES (:kind, :date, :arr_bn, :source, :url, :note, :classification, 0) "
        "ON CONFLICT(kind, date) DO UPDATE SET "
        "  arr_bn=excluded.arr_bn, source=excluded.source, "
        "  url=excluded.url, note=excluded.note, "
        "  classification=excluded.classification, is_local=0"
    )
    rows = list(rows)
    applied = 0
    for row in rows:
        cur = conn.execute(sql, row)
        applied += cur.rowcount
    if rows:
        dates = [row["date"] for row in rows]
        placeholders = ",".join("?" for _ in dates)
        cur = conn.execute(
            "DELETE FROM openai_arr_checkpoints WHERE is_local = 0 "
            f"AND (kind != 'checkpoint' OR date NOT IN ({placeholders}))",  # noqa: S608
            dates,
        )
        applied += cur.rowcount
    return applied


def sync_openai_checkpoints_from_parent(
    conn: sqlite3.Connection,
    rows: Iterable[dict],
) -> tuple[int, int]:
    """Atomically sync checkpoints and rebuild their materialized series."""
    owns_transaction = not conn.in_transaction
    if owns_transaction:
        conn.execute("BEGIN IMMEDIATE")
    try:
        changed = upsert_openai_checkpoints_from_parent(conn, rows)
        known = rebuild_openai_arr_known(conn)
        if owns_transaction:
            conn.execute("COMMIT")
        return changed, known
    except Exception:
        if owns_transaction:
            conn.execute("ROLLBACK")
        raise


def upsert_model_meta(conn: sqlite3.Connection, rows: Iterable[dict]) -> int:
    """Upsert AND prune. The `rows` argument is the AUTHORITATIVE set for
    this sync — any slug in the DB that's not in `rows` is deleted.

    The auto-roster in sources_models_meta selects a dynamic set from the
    OpenRouter catalog; without a prune step, models that fall off the
    top-N-per-lab would linger in the DB (and the /api/v1/models_meta
    response) indefinitely, showing decommissioned models to users.
    """
    sql = (
        "INSERT OR REPLACE INTO model_meta (slug, payload_json, as_of) "
        "VALUES (:slug, :payload_json, :as_of)"
    )
    rows = list(rows)
    current_slugs = [r["slug"] for r in rows]
    conn.executemany(sql, rows)
    if current_slugs:
        # Chunk the DELETE to stay under SQLite's default parameter limit
        # (999 host params per statement) — we never approach it (~120 max)
        # but the pattern is safer than a single unbounded IN clause.
        placeholders = ",".join(["?"] * len(current_slugs))
        conn.execute(
            f"DELETE FROM model_meta WHERE slug NOT IN ({placeholders})",
            current_slugs,
        )
    return len(rows)


def upsert_model_scores(conn: sqlite3.Connection, rows: Iterable[dict]) -> int:
    sql = (
        "INSERT OR REPLACE INTO model_scores "
        "(slug, source, payload_json, as_of) VALUES "
        "(:slug, :source, :payload_json, :as_of)"
    )
    rows = list(rows)
    conn.executemany(sql, rows)
    return len(rows)


def upsert_model_arena_history(conn: sqlite3.Connection, rows: Iterable[dict]) -> int:
    sql = (
        "INSERT OR REPLACE INTO model_arena_history "
        "(slug, date, overall, coding, webdev) VALUES "
        "(:slug, :date, :overall, :coding, :webdev)"
    )
    rows = list(rows)
    conn.executemany(sql, rows)
    return len(rows)


def upsert_provider_throughput_history(
    conn: sqlite3.Connection, rows: Iterable[dict]
) -> int:
    """Idempotent per (captured_at, provider, model). captured_at is floored to
    a 30-min bucket upstream, so a re-run inside the same window overwrites while
    each new window appends — accumulating an intra-day series."""
    sql = (
        "INSERT OR REPLACE INTO provider_throughput_history "
        "(captured_at, date, provider, canonical_slug, "
        " p50_tps, p75_tps, p90_tps, p95_tps, p99_tps, "
        " p50_lat_ms, p75_lat_ms, p90_lat_ms, p95_lat_ms, p99_lat_ms, "
        " request_count, window_minutes, capacity_tpm, quantization, "
        " provider_region, status) VALUES "
        "(:captured_at, :date, :provider, :canonical_slug, "
        " :p50_tps, :p75_tps, :p90_tps, :p95_tps, :p99_tps, "
        " :p50_lat_ms, :p75_lat_ms, :p90_lat_ms, :p95_lat_ms, :p99_lat_ms, "
        " :request_count, :window_minutes, :capacity_tpm, :quantization, "
        " :provider_region, :status)"
    )
    rows = list(rows)
    conn.executemany(sql, rows)
    return len(rows)
