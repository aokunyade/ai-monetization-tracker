"""ARR curve + nowcast routes.

- GET /api/v1/arr_curve   — public: Anthropic + OpenAI curve payload
                            (hist/ext/fanLo/fanHi/counter/cps/yoyDen)
- GET /api/v1/arr_nowcast — public: bounded-blend nowcast + breakdown table
- POST /api/v1/openai_arr — admin-token: insert/edit an OpenAI checkpoint
                            (reuses X-Admin-Token, same pattern as /known_arr)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException

from .arr_curve import build_arr_curve, build_openai_signal_breakdown
from .arr_nowcast import build_arr_nowcast
from .config import require_admin_token
from .db import get_db, rebuild_openai_arr_known, upsert_openai_arr_checkpoint

router = APIRouter(prefix="/api/v1")

# Shared admin gate — see config.require_admin_token.
_require_admin = require_admin_token


@router.get("/arr_curve")
def arr_curve() -> dict[str, Any]:
    conn = get_db()
    try:
        return build_arr_curve(conn)
    finally:
        conn.close()


@router.get("/arr_nowcast")
def arr_nowcast() -> dict[str, Any]:
    conn = get_db()
    try:
        return build_arr_nowcast(conn)
    finally:
        conn.close()


@router.get("/openai_signals")
def openai_signals() -> dict[str, Any]:
    """OpenAI triangulation breakdown — 4 signals + g_cp + composite math.

    Displayed on the Signals tab so users can see what drives the OpenAI
    ext rate, parallel to Anthropic's ensemble-regression table.
    """
    conn = get_db()
    try:
        return build_openai_signal_breakdown(conn)
    finally:
        conn.close()


@router.get("/openai_arr")
def get_openai_arr() -> dict[str, Any]:
    """Full OpenAI ARR ledger (estimates + checkpoints + latest extrapolation).
    Public read; the POST counterpart is admin-gated."""
    conn = get_db()
    try:
        est = conn.execute(
            "SELECT date, arr_bn, note, classification "
            "FROM openai_arr_checkpoints "
            "WHERE kind = 'estimate' ORDER BY date"
        ).fetchall()
        cps = conn.execute(
            "SELECT date, arr_bn, source, url, note, classification "
            "FROM openai_arr_checkpoints "
            "WHERE kind = 'checkpoint' ORDER BY date"
        ).fetchall()
        ext = conn.execute(
            "SELECT as_of, to_date, low_bn, high_bn "
            "FROM openai_arr_extrapolation ORDER BY as_of DESC LIMIT 1"
        ).fetchone()
        return {
            "estimates": [dict(r) for r in est],
            "checkpoints": [dict(r) for r in cps],
            "extrapolation": (dict(ext) if ext else None),
        }
    finally:
        conn.close()


@router.post("/openai_arr", dependencies=[Depends(_require_admin)])
def post_openai_arr(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    kind = str(payload.get("kind") or "").strip()
    if kind not in ("estimate", "checkpoint"):
        raise HTTPException(400, "kind must be 'estimate' or 'checkpoint'")
    d = str(payload.get("date") or "").strip()
    if not d or len(d) != 10 or d[4] != "-" or d[7] != "-":
        raise HTTPException(400, "date must be YYYY-MM-DD")
    try:
        arr_bn = float(payload.get("arr_bn"))
    except (TypeError, ValueError):
        raise HTTPException(400, "arr_bn must be numeric") from None
    if arr_bn <= 0:
        raise HTTPException(400, "arr_bn must be > 0")
    source = payload.get("source") or None
    url = payload.get("url") or None
    note = payload.get("note") or None
    conn = get_db()
    try:
        upsert_openai_arr_checkpoint(conn, kind, d, arr_bn, source, url, note)
        # Rematerialize openai_arr_known so predict_openai_arr() sees the
        # new checkpoint without a server restart. Without this, /predictor
        # would keep returning the pre-POST prediction until init_schema
        # re-runs at next process start.
        n_known = rebuild_openai_arr_known(conn)
        return {
            "ok": True,
            "kind": kind,
            "date": d,
            "arr_bn": arr_bn,
            "openai_arr_known_rows": n_known,
        }
    finally:
        conn.close()
