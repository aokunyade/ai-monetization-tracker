"""Canonical lab registry — one source of truth for prefix → display / color.

Previously three separate files kept their own lab lists:
  - sources_models_meta.LAB_PREFIXES  (12 labs, roster whitelist)
  - sources_openrouter_market.LABS    (14 labs, share buckets)
  - web/dashboard.js LAB_COLOR         (12 labs, substring rules)

They drift silently when a new provider is added. This module consolidates:
  - FRONTIER_LABS: labs we track for the frontier-model roster
  - MARKET_LABS:   FRONTIER_LABS + cloud-provider aggregators (amazon,
                    microsoft, nvidia) that show up in OpenRouter's daily
                    rankings but aren't frontier model publishers
  - LAB_ALIASES:   substring rules for slug fragments that don't match the
                    canonical prefix (kimi/moonshot, glm/z-ai, gemini/google,
                    claude/anthropic, gpt/openai, grok/x-ai, hy3/tencent,
                    llama/meta-llama)

Consumers (`sources_models_meta.py`, `sources_openrouter_market.py`,
`payload.py`, and via payload → `dashboard.js`) import from here.
"""

from __future__ import annotations

# prefix → display name, ordered for stable iteration.
FRONTIER_LABS: dict[str, str] = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "google": "Google",
    "x-ai": "xAI",
    "deepseek": "DeepSeek",
    "z-ai": "Z.ai",
    "moonshotai": "Kimi",
    "minimax": "MiniMax",
    "qwen": "Alibaba",
    "tencent": "Tencent",
    "meta-llama": "Meta",
    "mistralai": "Mistral",
}

# Additional prefixes that appear on OpenRouter rankings but aren't frontier
# model publishers (they're cloud aggregators or fine-tune shops). Included
# in market-share computations, excluded from the frontier roster.
MARKET_ONLY_LABS: dict[str, str] = {
    "amazon": "Amazon",
    "microsoft": "Microsoft",
    "nvidia": "NVIDIA",
}

# Full set for market rankings.
MARKET_LABS: dict[str, str] = {**FRONTIER_LABS, **MARKET_ONLY_LABS}

# Brand colors — matched to the frontend --lab-* CSS custom properties.
# Kept here so the Python side can ship them to the client, avoiding a
# separate hardcoded list in dashboard.js.
LAB_COLORS: dict[str, str] = {
    "anthropic": "#e07a4c",
    "openai": "#10a37f",
    "google": "#6ea8ff",
    "x-ai": "#b5c0cc",
    "deepseek": "#8b7dff",
    "z-ai": "#c084fc",
    "moonshotai": "#3fb6a3",
    "minimax": "#3fb6c4",
    "qwen": "#f0a24d",
    "tencent": "#4d9de0",
    "meta-llama": "#3f6ce0",
    "mistralai": "#ff7000",
    "amazon": "#ff9900",
    "microsoft": "#5390d9",
    "nvidia": "#76b900",
    "other": "#8d7b68",
}

# Substring → canonical prefix. Catches slug fragments that don't include
# the lab name (Anthropic's `claude-*`, Google's `gemini-*`, Moonshot's
# `kimi-*`, etc.). Frontend uses this via the /api/v1/config payload to
# color chart elements without hardcoding the same rules.
LAB_ALIASES: dict[str, str] = {
    "claude": "anthropic",
    "gpt": "openai",
    "gemini": "google",
    "grok": "x-ai",
    "kimi": "moonshotai",
    "glm": "z-ai",
    "alibaba": "qwen",
    "hy3": "tencent",
    "hunyuan": "tencent",
    "llama": "meta-llama",
}


def color_for_slug(slug: str) -> str:
    """Return the brand color for a slug, matching JS LAB_COLOR semantics."""
    lc = (slug or "").lower()
    prefix = lc.split("/", 1)[0]
    if prefix in LAB_COLORS:
        return LAB_COLORS[prefix]
    for substring, canonical in LAB_ALIASES.items():
        if substring in lc:
            return LAB_COLORS.get(canonical, LAB_COLORS["other"])
    return LAB_COLORS["other"]


def to_config_payload() -> dict:
    """Shape used by payload.py to ship the lab registry to the frontend."""
    return {
        "frontier": FRONTIER_LABS,
        "market": MARKET_LABS,
        "colors": LAB_COLORS,
        "aliases": LAB_ALIASES,
    }
