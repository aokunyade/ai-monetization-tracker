"""AWS Bedrock SDK adoption proxy via PyPI download stats.

`mypy-boto3-bedrock-runtime` is the type-stub package developers add when
calling Bedrock from Python — closely tracks Bedrock-Anthropic adoption.
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx
from loguru import logger


def fetch_bedrock_pypi(client: httpx.Client) -> list[dict]:
    """Daily download counts for mypy-boto3-bedrock-runtime (180-day window)."""
    pkg = "mypy-boto3-bedrock-runtime"
    url = f"https://pypistats.org/api/packages/{pkg}/overall"
    captured_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        r = client.get(url, params={"mirrors": "false"})
        r.raise_for_status()
        payload = r.json()
    except Exception as e:
        logger.warning(f"bedrock pypi fetch failed: {e}")
        return []
    out: list[dict] = []
    for x in payload.get("data") or []:
        if x.get("category") != "without_mirrors":
            continue
        out.append(
            {
                "date": x["date"],
                "signal_name": f"pypi:{pkg}",
                "value": float(x["downloads"]),
                "captured_at": captured_at,
            }
        )
    logger.info(f"pypi:{pkg}: fetched {len(out)} daily records")
    return out
