"""SDK download routes for the tracker page.

- GET /api/v1/sdk_downloads — public: 4 series
  npm:@anthropic-ai/sdk, npm:openai, pypi:anthropic, pypi:openai

7DMA is computed client-side (tracker.js dma7()).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from .db import get_db

router = APIRouter(prefix="/api/v1/sdk_downloads")

SIGNALS_NPM = [
    ("@anthropic-ai/sdk", "npm:@anthropic-ai/sdk"),
    ("openai", "npm:openai"),
]
SIGNALS_PYPI = [
    ("anthropic", "pypi:anthropic"),
    ("openai", "pypi:openai"),
]


def _load(conn, signal_name: str) -> list[list[Any]]:
    rows = conn.execute(
        "SELECT date, value FROM anthropic_signals WHERE signal_name = ? ORDER BY date",
        (signal_name,),
    ).fetchall()
    # tracker stores downloads in M/day; here we keep raw daily count and let
    # the frontend divide by 1e6 (dma7 already lives client-side).
    return [[r["date"], float(r["value"])] for r in rows]


@router.get("")
def sdk_downloads() -> dict[str, Any]:
    conn = get_db()
    try:
        return {
            "as_of": (
                conn.execute(
                    "SELECT MAX(captured_at) AS c FROM anthropic_signals "
                    "WHERE signal_name LIKE 'npm:%' OR signal_name LIKE 'pypi:%'"
                ).fetchone()["c"]
            ),
            "source": "npm registry + pypistats",
            "unit": "downloads/day (raw); tracker.js applies 7DMA + /1e6",
            "npm": {pkg: _load(conn, signal) for pkg, signal in SIGNALS_NPM},
            "pypi": {pkg: _load(conn, signal) for pkg, signal in SIGNALS_PYPI},
        }
    finally:
        conn.close()
