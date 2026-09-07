"""Curated signals (KOL / podcast takes) route.

Data source is a hand-maintained JSON file on disk (data/curated_signals_seed.json)
— matches the tracker's convention where these are event-driven curated takes,
not scraped. Public read; edits happen via file + restart (no admin API yet).
"""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter
from loguru import logger

from . import ROOT

router = APIRouter(prefix="/api/v1")

_PATH = ROOT / "data" / "curated_signals_seed.json"
_CODEX_PATH = ROOT / "data" / "codex_wau_seed.json"


@router.get("/curated_signals")
def curated_signals() -> dict[str, Any]:
    if not _PATH.exists():
        return {"as_of": None, "kol": [], "reports": []}
    try:
        return json.loads(_PATH.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"curated_signals: parse failed: {e}")
        return {"as_of": None, "kol": [], "reports": []}


@router.get("/codex_wau")
def codex_wau() -> dict[str, Any]:
    """OpenAI Codex weekly-active-user milestone series (curated).

    Milestone-style data (not continuous) — see the seed's `note` field.
    Same shape as castora's `data.codex`: {series: [[date, M_wau], ...],
    events: [[date, label], ...]}.
    """
    if not _CODEX_PATH.exists():
        return {"as_of": None, "series": [], "events": []}
    try:
        return json.loads(_CODEX_PATH.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        logger.warning(f"codex_wau: parse failed: {e}")
        return {"as_of": None, "series": [], "events": []}
