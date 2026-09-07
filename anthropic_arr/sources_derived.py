"""Derived signals — computed from existing rows in anthropic_signals.

Two derived signals live here:

1. `derived:bedrock_spend_estimate` — 2-variable linear combination of
   pypi:mypy-boto3-bedrock-runtime and npm:@anthropic-ai/sdk that approximates
   the Yipit panel-scale "Bedrock Anthropic spend" series. Calibrated offline
   against Yipit's bedrock_spend column (Nov 2025 – Apr 2026 overlap, n=6,
   R²=0.977). Series is in PANEL-SCALE millions of dollars.

2. `derived:vercel_claude_family_cost_share` — Vercel AI Gateway daily $
   spend share for Claude models (family aggregate). Read directly from
   vercel_gateway_history where metric='cost' and name='Claude (family)'.
   Downstream paid-usage proxy — mirrors castora's vercel_spend_share signal.

Both are daily-value signals so the SUM of any month equals a monthly total
useful for ensemble regression.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from loguru import logger

# Calibrated against Yipit:bedrock_spend (Nov 2025 - Apr 2026, n=6, R²=0.977)
# Bedrock_spend($M) = INTERCEPT + B_PYPI * pypi_monthly_dl + B_NPM * npm_monthly_dl
INTERCEPT_M_USD = 0.10
B_PYPI = 8.0e-7
B_NPM = 2.3e-7

PYPI_SIGNAL = "pypi:mypy-boto3-bedrock-runtime"
NPM_SIGNAL = "npm:@anthropic-ai/sdk"
DERIVED_SIGNAL = "derived:bedrock_spend_estimate"


def compute_bedrock_spend_estimate_rows(conn: sqlite3.Connection) -> list[dict]:
    """Daily prorated estimates such that monthly sum reproduces the 2-var fit.

    Joins pypi:bedrock and npm:sdk on date — drops days where either is missing.
    Intercept is amortized across the days observed in each month so the monthly
    sum equals INTERCEPT + B_PYPI * pypi_month_total + B_NPM * npm_month_total.
    """
    captured_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    rows = conn.execute(
        """SELECT p.date AS d, p.value AS pv, n.value AS nv
           FROM anthropic_signals p
           JOIN anthropic_signals n ON p.date = n.date
           WHERE p.signal_name = ? AND n.signal_name = ?
             AND p.value > 0 AND n.value > 0
           ORDER BY p.date""",
        (PYPI_SIGNAL, NPM_SIGNAL),
    ).fetchall()
    if not rows:
        logger.warning(f"{DERIVED_SIGNAL}: no overlapping pypi+npm rows; skipping")
        return []

    # Bucket dates by month for intercept proration
    by_month: dict[str, list] = {}
    for r in rows:
        m = r["d"][:7]
        by_month.setdefault(m, []).append(r)

    out: list[dict] = []
    for month, mrows in by_month.items():
        days_obs = len(mrows)
        intercept_per_day = INTERCEPT_M_USD / days_obs
        for r in mrows:
            value_m = (
                intercept_per_day + B_PYPI * float(r["pv"]) + B_NPM * float(r["nv"])
            )
            if value_m <= 0:
                continue
            out.append(
                {
                    "date": r["d"],
                    "signal_name": DERIVED_SIGNAL,
                    "value": value_m,  # $M, panel-scale
                    "captured_at": captured_at,
                }
            )

    logger.info(
        f"{DERIVED_SIGNAL}: produced {len(out)} daily rows "
        f"across {len(by_month)} months (panel-scale $M)"
    )
    return out


VERCEL_CLAUDE_SIGNAL = "derived:vercel_claude_family_cost_share"


def compute_vercel_claude_family_cost_share_rows(
    conn: sqlite3.Connection,
) -> list[dict]:
    """Daily Claude-family $ share from vercel_gateway_history.

    Reads rows where metric='cost' and name='Claude (family)' — the
    aggregate row that sources_vercel.py already writes. Emits one signal
    row per day so the standard monthly-sum aggregation in compute.py
    yields the "sum of daily Claude spend share %" for that month, which
    is monotonic in Claude's downstream paid usage.
    """
    captured_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = conn.execute(
        """SELECT date, share_pct
             FROM vercel_gateway_history
            WHERE metric = 'cost' AND name = 'Claude (family)'
              AND share_pct > 0
            ORDER BY date"""
    ).fetchall()
    if not rows:
        logger.info(f"{VERCEL_CLAUDE_SIGNAL}: no Claude (family) cost rows; skipping")
        return []
    out = [
        {
            "date": r["date"],
            "signal_name": VERCEL_CLAUDE_SIGNAL,
            "value": float(r["share_pct"]),
            "captured_at": captured_at,
        }
        for r in rows
    ]
    logger.info(f"{VERCEL_CLAUDE_SIGNAL}: produced {len(out)} daily rows")
    return out
