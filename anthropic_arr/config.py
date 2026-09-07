"""Server-level config read once at process import.

Both `api_server.py` and `payload.py` import from here so the public payload's
`admin_enabled` flag and the FastAPI admin-token dependency cannot drift
out of sync at runtime.

Auth model (July 2026): session/invite-based auth was removed. The only
remaining auth surface is the X-Admin-Token header on POST endpoints
(/known_arr, /openai_arr, /sync). All read endpoints are public.
"""

from __future__ import annotations

import os

from fastapi import Header, HTTPException

ADMIN_TOKEN: str | None = os.environ.get("ANTHROPIC_ARR_ADMIN_TOKEN") or None

# CORS: read endpoints are all public + no credentials/cookies in play, so
# a permissive wildcard is safe. Override via ANTHROPIC_ARR_ALLOWED_ORIGINS
# (comma-separated) if you want to lock this down for a specific deployment.
ALLOWED_ORIGINS: list[str] = [
    o.strip()
    for o in os.environ.get("ANTHROPIC_ARR_ALLOWED_ORIGINS", "*").split(",")
    if o.strip()
] or ["*"]


def admin_enabled() -> bool:
    return bool(ADMIN_TOKEN)


def require_admin_token(
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
) -> None:
    """FastAPI dependency: validate the X-Admin-Token header against
    ADMIN_TOKEN. Used by api_server.py (POST /known_arr, POST /sync) and
    routes_arr.py (POST /openai_arr)."""
    if not ADMIN_TOKEN:
        raise HTTPException(503, "Admin endpoints disabled (no token configured)")
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(401, "Invalid admin token")
