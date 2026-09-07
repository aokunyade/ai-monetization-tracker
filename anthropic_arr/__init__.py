"""Standalone Anthropic ARR predictor app.

Self-contained: own SQLite (`anthropic_arr.db`) and FastAPI process on port
8301. Pulls data directly from npm registry, pypistats, and OpenRouter's public
APIs, with no parent-project dependency.

Run `python -m anthropic_arr.sync` (or POST /api/v1/sync) to refresh.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

OWN_DB_PATH = PROJECT_ROOT / "anthropic_arr.db"
WEB_DIR = ROOT / "web"
PORT = 8301
