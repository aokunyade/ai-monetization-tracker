"""GitHub public-API fetchers for Anthropic-related signals.

Three fetcher categories:
  1. Daily commit counts on Anthropic-owned repos
  2. Daily star-count snapshots (velocity computed at predict time)
  3. Daily repo-search totals for enterprise-channel proxy queries

Unauthenticated GitHub allows 60 req/hr; set GITHUB_TOKEN env to lift to 5000/hr.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import httpx
from loguru import logger

GITHUB_API = "https://api.github.com"
GITHUB_REPOS: list[str] = [
    "anthropics/anthropic-sdk-python",
    "anthropics/anthropic-sdk-typescript",
    "anthropics/claude-code",
    "anthropics/anthropic-cookbook",
]
SEARCH_QUERIES: list[tuple[str, str]] = [
    ("aws bedrock claude", "bedrock_claude"),
    ("vertex anthropic", "vertex_anthropic"),
    # Anthropic code-ref proxies (matches castora's github_queries for
    # Anthropic: every public repo mentioning these strings likely calls
    # Anthropic's API).
    ("ANTHROPIC_API_KEY", "anthropic_api_key"),
    ("api.anthropic.com", "api_anthropic_com"),
]


def _gh_headers() -> dict[str, str]:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def _gh_get(client: httpx.Client, url: str, **kwargs):
    """Wrapper that follows GitHub's 301 repo-rename redirects (default httpx
    Client doesn't follow redirects)."""
    return client.get(url, follow_redirects=True, headers=_gh_headers(), **kwargs)


def _sleep_for_rate_limit() -> None:
    """1.1s per call without a token (60 req/hr); negligible with token."""
    if not os.environ.get("GITHUB_TOKEN"):
        time.sleep(1.1)


def fetch_github_commits(
    client: httpx.Client, repo: str, since: str = "2023-05-01"
) -> list[dict]:
    """Return daily commit counts for `repo` since `since` (YYYY-MM-DD).

    Walks GitHub commits API page-by-page; aggregates by date.
    signal_name = f"github:commits:{repo}".
    """
    captured_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    counts: dict[str, int] = {}
    page = 1
    since_iso = f"{since}T00:00:00Z"
    while True:
        try:
            r = _gh_get(
                client,
                f"{GITHUB_API}/repos/{repo}/commits",
                params={"since": since_iso, "per_page": 100, "page": page},
            )
            if r.status_code in (403, 429):
                logger.warning(f"github commits {repo}: rate-limited at page {page}")
                break
            r.raise_for_status()
            commits = r.json()
        except Exception as e:
            logger.warning(f"github commits {repo} page {page} failed: {e}")
            break
        if not commits:
            break
        for c in commits:
            date_str = ((c.get("commit") or {}).get("author") or {}).get("date") or ""
            day = date_str[:10]
            if day:
                counts[day] = counts.get(day, 0) + 1
        if len(commits) < 100:
            break
        page += 1
        _sleep_for_rate_limit()
        if page > 50:  # safety cap (5000 commits)
            break

    out = [
        {
            "date": d,
            "signal_name": f"github:commits:{repo}",
            "value": float(n),
            "captured_at": captured_at,
        }
        for d, n in counts.items()
    ]
    logger.info(f"github:commits:{repo}: {len(out)} daily records")
    return out


def fetch_github_star_snapshot(client: httpx.Client, repo: str) -> list[dict]:
    """One-row daily snapshot of stargazers_count.

    Velocity (eom - som) is computed at predict time. signal_name = f"github:stars:{repo}".
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    captured_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        r = _gh_get(client, f"{GITHUB_API}/repos/{repo}")
        r.raise_for_status()
        body = r.json()
    except Exception as e:
        logger.warning(f"github stars {repo} failed: {e}")
        return []
    stars = body.get("stargazers_count")
    if stars is None:
        return []
    logger.info(f"github:stars:{repo}: {stars} stars (today)")
    return [
        {
            "date": today,
            "signal_name": f"github:stars:{repo}",
            "value": float(stars),
            "captured_at": captured_at,
        }
    ]


def fetch_github_search_count(
    client: httpx.Client, query: str, label: str
) -> list[dict]:
    """Daily search-result-count snapshot.

    signal_name = f"github:search:{label}".
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    captured_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        r = _gh_get(
            client,
            f"{GITHUB_API}/search/repositories",
            params={"q": query, "per_page": 1},
        )
        r.raise_for_status()
        body = r.json()
    except Exception as e:
        logger.warning(f"github search '{query}' failed: {e}")
        return []
    total = body.get("total_count")
    if total is None:
        return []
    logger.info(f"github:search:{label}: {total} repos for '{query}'")
    return [
        {
            "date": today,
            "signal_name": f"github:search:{label}",
            "value": float(total),
            "captured_at": captured_at,
        }
    ]


def fetch_all_github(client: httpx.Client) -> list[dict]:
    """Run all GitHub fetchers; return upsert rows for anthropic_signals."""
    out: list[dict] = []
    for repo in GITHUB_REPOS:
        out.extend(fetch_github_commits(client, repo))
        _sleep_for_rate_limit()
        out.extend(fetch_github_star_snapshot(client, repo))
        _sleep_for_rate_limit()
    for query, label in SEARCH_QUERIES:
        out.extend(fetch_github_search_count(client, query, label))
        _sleep_for_rate_limit()
    return out
