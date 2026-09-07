"""OpenAI SDK download proxies (npm + PyPI), for the /tracker SDK panel.

Kept separate from `sources.py` because that file's intent is Anthropic-only
signals in the ensemble. These new signals feed the tracker view and
_openai_signals in arr_curve.py (castora arr.triangulation).

Signals produced (upserts into `anthropic_signals`):
- npm:openai                  daily downloads
- npm:@ai-sdk/openai          daily downloads
- npm:@openai/codex           daily downloads
- pypi:openai                 daily downloads
"""

from __future__ import annotations

import time
from datetime import datetime, timezone

import httpx
from loguru import logger

from .sources import fetch_npm_range, fetch_pypi_overall

# Matches castora tracker_config.json arr.companies.openai.signals.npm and
# arr.companies.openai.signals.pypi — every package here contributes to the
# OpenAI SDK-download triangulation term.
NPM_EXTRA = ["openai", "@ai-sdk/openai", "@openai/codex"]
PYPI_EXTRA = ["openai"]


def fetch_sdk_extra_rows(
    client: httpx.Client, start: str = "2023-05-01", end: str | None = None
) -> list[dict]:
    if end is None:
        end = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    captured_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out: list[dict] = []
    for pkg in NPM_EXTRA:
        for r in fetch_npm_range(client, pkg, start, end):
            out.append(
                {
                    "date": r["date"],
                    "signal_name": f"npm:{pkg}",
                    "value": r["value"],
                    "captured_at": captured_at,
                }
            )
        time.sleep(0.5)
    for pkg in PYPI_EXTRA:
        for r in fetch_pypi_overall(client, pkg):
            out.append(
                {
                    "date": r["date"],
                    "signal_name": f"pypi:{pkg}",
                    "value": r["value"],
                    "captured_at": captured_at,
                }
            )
        time.sleep(0.5)
    logger.info(f"sdk_extra: {len(out)} rows")
    return out
