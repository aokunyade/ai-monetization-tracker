"""FastAPI server for the standalone Anthropic ARR app on port 8301.

Run:
    python -m uvicorn anthropic_arr.api_server:app --port 8301
or:
    python -m anthropic_arr.api_server
"""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import Body, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import OWN_DB_PATH, PORT, WEB_DIR
from .backtest import backtest_summary
from .compute import predict_arr, predict_openai_arr
from .config import ALLOWED_ORIGINS, admin_enabled, require_admin_token
from .db import db_ctx, get_db, init_schema, upsert_arr_known
from .payload import build_payload, spend_and_tokens_panel
from .routes_arr import router as arr_router
from .routes_models import router as models_router
from .routes_openrouter_market import router as openrouter_market_router
from .routes_sdk import router as sdk_router
from .routes_signals import router as signals_router
from .routes_vercel import router as vercel_router
from .sync import refresh

# Shared admin gate — protects the mutating admin-token endpoints (POST
# /known_arr, POST /openai_arr, POST /sync). Session-based auth (users,
# invites, sessions) was removed in July 2026; this X-Admin-Token gate is
# the last remaining auth surface.
_require_admin = require_admin_token


# Single mutex shared by startup background refresh and POST /api/v1/sync.
# Prevents two concurrent refreshes from hammering npm/PyPI/GitHub in parallel.
_REFRESH_LOCK = threading.Lock()


def _safe_refresh(rebuild: bool = False) -> tuple[bool, dict | str]:
    """Run refresh under the mutex. Returns (acquired, counts_or_msg).

    If another refresh is already in progress, returns (False, "...") immediately.
    """
    if not _REFRESH_LOCK.acquire(blocking=False):
        return False, "refresh already in progress"
    try:
        return True, refresh(rebuild=rebuild)
    finally:
        _REFRESH_LOCK.release()


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    init_schema()
    yield


app = FastAPI(
    title="Anthropic ARR Predictor",
    description="Standalone Anthropic ARR forecaster + Anthropic-only spend & tokens.",
    version="1.0.0",
    lifespan=_lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(openrouter_market_router)
app.include_router(vercel_router)
app.include_router(sdk_router)
app.include_router(models_router)
app.include_router(arr_router)
app.include_router(signals_router)


def _conn() -> sqlite3.Connection:
    return get_db()


@app.get("/api/v1/config")
def get_config() -> dict[str, Any]:
    from .labs import to_config_payload

    return {
        "admin_enabled": admin_enabled(),
        "labs": to_config_payload(),
    }


@app.get("/api/v1/dashboard")
def get_dashboard(target_month: str | None = Query(default=None)) -> dict[str, Any]:
    conn = _conn()
    try:
        return build_payload(
            conn, target_month=target_month, admin_enabled=admin_enabled()
        )
    finally:
        conn.close()


@app.get("/api/v1/predictor")
def get_predictor(
    target_month: str | None = Query(default=None),
    company: str = Query(default="anthropic"),
) -> dict[str, Any]:
    """Ensemble ARR prediction.

    company=anthropic (default) → predict_arr against anthropic_arr_known
    company=openai              → predict_openai_arr against openai_arr_known

    Payload shape is identical across companies so the frontend can just
    swap the query param. See compute.predict_openai_arr for the OpenAI
    signal registry and its walk-forward calibration."""
    company_norm = (company or "anthropic").strip().lower()
    if company_norm not in ("anthropic", "openai"):
        raise HTTPException(400, "company must be 'anthropic' or 'openai'")
    conn = _conn()
    try:
        if company_norm == "openai":
            result = predict_openai_arr(conn, target_month=target_month)
        else:
            result = predict_arr(conn, target_month=target_month)
        if not result:
            raise HTTPException(404, "Insufficient training data")
        return result
    finally:
        conn.close()


@app.get("/api/v1/spend")
def get_spend() -> dict[str, Any]:
    conn = _conn()
    try:
        return spend_and_tokens_panel(conn)
    finally:
        conn.close()


@app.get("/api/v1/backtest")
def get_backtest() -> dict[str, Any]:
    conn = _conn()
    try:
        return backtest_summary(conn)
    finally:
        conn.close()


@app.get("/api/v1/known_arr")
def get_known_arr() -> dict[str, Any]:
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT month, arr_b_usd, source, notes, added_at, is_local "
            "FROM anthropic_arr_known ORDER BY month"
        ).fetchall()
        return {"rows": [dict(r) for r in rows]}
    finally:
        conn.close()


def _str_field(payload: dict[str, Any], key: str) -> str:
    val = payload.get(key)
    if val is None:
        return ""
    if not isinstance(val, str):
        raise HTTPException(400, f"{key} must be a string")
    return val.strip()


@app.post("/api/v1/known_arr", dependencies=[Depends(_require_admin)])
def post_known_arr(payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
    month = _str_field(payload, "month")
    source = _str_field(payload, "source") or None
    notes = _str_field(payload, "notes") or None
    if (
        len(month) != 7
        or month[4] != "-"
        or not month[:4].isdigit()
        or not month[5:].isdigit()
    ):
        raise HTTPException(400, "month must be YYYY-MM")
    yyyy, mm = int(month[:4]), int(month[5:])
    if yyyy < 2020 or yyyy > 2100 or mm < 1 or mm > 12:
        raise HTTPException(400, "month out of range (year 2020–2100, month 1–12)")
    try:
        arr_b_val = float(payload.get("arr_b_usd"))
    except (TypeError, ValueError):
        raise HTTPException(400, "arr_b_usd must be a number") from None
    if arr_b_val <= 0:
        raise HTTPException(400, "arr_b_usd must be > 0")
    if not source:
        raise HTTPException(400, "source is required")
    now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with db_ctx() as conn:
        upsert_arr_known(
            conn,
            [
                {
                    "month": month,
                    "arr_b_usd": arr_b_val,
                    "source": source,
                    "notes": notes,
                    "added_at": now_iso,
                    "is_local": 1,
                }
            ],
        )
    return {
        "ok": True,
        "month": month,
        "arr_b_usd": arr_b_val,
        "source": source,
        "notes": notes,
        "added_at": now_iso,
        "is_local": True,
    }


@app.post("/api/v1/sync", dependencies=[Depends(_require_admin)])
def post_sync(rebuild: bool = Query(default=False)) -> dict[str, Any]:
    acquired, result = _safe_refresh(rebuild=rebuild)
    if not acquired:
        raise HTTPException(409, str(result))
    return {"ok": True, "counts": result, "db": str(OWN_DB_PATH)}


# ---- static frontend ----
# 2026-07-16: single-page dashboard.html merges the former /, /tracker, /models
# into one SPA with sidebar sections. /tracker and /models kept as 302
# redirects for ~90 days to preserve external bookmarks.
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")

    # `no-store` (in addition to no-cache + must-revalidate) tells the
    # browser NEVER to keep a copy on disk, so refreshing after we ship
    # dashboard.js edits always fetches fresh — otherwise Chrome's back-
    # forward cache and disk cache combine to serve a stale bundle even
    # when Cache-Control says "revalidate" (revalidation is optional
    # when the cached response is still fresh per heuristic policy).
    _NO_CACHE = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0",
    }

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(WEB_DIR / "dashboard.html"), headers=_NO_CACHE)

    @app.get("/dashboard.js")
    def dashboard_js() -> FileResponse:
        return FileResponse(
            str(WEB_DIR / "dashboard.js"),
            media_type="application/javascript",
            headers=_NO_CACHE,
        )

    @app.get("/dashboard.css")
    def dashboard_css() -> FileResponse:
        return FileResponse(
            str(WEB_DIR / "dashboard.css"), media_type="text/css", headers=_NO_CACHE
        )

    # Legacy redirects to preserve external bookmarks. Delete after 90 days.
    # Query string carried through so /tracker?target_month=2025-03 lands
    # on /?target_month=2025-03#live with the picker honoring the month.
    def _redirect_with_query(request: Request, target_hash: str) -> RedirectResponse:
        qs = request.url.query
        url = ("/?" + qs + target_hash) if qs else ("/" + target_hash)
        return RedirectResponse(url=url, status_code=302)

    @app.get("/tracker")
    def tracker_redirect(request: Request) -> RedirectResponse:
        return _redirect_with_query(request, "#live")

    @app.get("/models")
    def models_redirect(request: Request) -> RedirectResponse:
        return _redirect_with_query(request, "#models")

    # /simple retained — separate short-form share view, different audience.
    @app.get("/simple")
    def simple_index() -> FileResponse:
        return FileResponse(str(WEB_DIR / "simple.html"), headers=_NO_CACHE)

    @app.get("/simple.js")
    def simple_js() -> FileResponse:
        return FileResponse(
            str(WEB_DIR / "simple.js"),
            media_type="application/javascript",
            headers=_NO_CACHE,
        )

    @app.get("/simple.css")
    def simple_css() -> FileResponse:
        return FileResponse(
            str(WEB_DIR / "simple.css"), media_type="text/css", headers=_NO_CACHE
        )

    # Shared helpers used by both /simple and /dashboard.
    @app.get("/helpers.js")
    def helpers_js() -> FileResponse:
        return FileResponse(
            str(WEB_DIR / "helpers.js"),
            media_type="application/javascript",
            headers=_NO_CACHE,
        )

    # Shared /api/v1/arr_curve payload transforms, used by dashboard.js.
    @app.get("/arr_curve_shared.js")
    def arr_curve_shared_js() -> FileResponse:
        return FileResponse(
            str(WEB_DIR / "arr_curve_shared.js"),
            media_type="application/javascript",
            headers=_NO_CACHE,
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "anthropic_arr.api_server:app",
        host="127.0.0.1",
        port=int(os.environ.get("ANTHROPIC_ARR_PORT", str(PORT))),
        reload=False,
    )
