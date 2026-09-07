"""Single source of truth for human-facing labels.

Used by `payload.build_payload()` so the frontend never has to know
internal signal-name conventions.
"""

from __future__ import annotations

SIGNAL_LABELS: dict[str, str] = {
    # Dev — npm/PyPI
    "npm:@anthropic-ai/sdk": "Anthropic SDK (npm) downloads",
    "npm:@anthropic-ai/claude-code": "Claude Code CLI (npm) downloads",
    "pypi:anthropic": "Anthropic Python SDK (PyPI) downloads",
    # Dev — GitHub
    "github:commits:anthropics/anthropic-sdk-python": "GitHub commits — Python SDK",
    "github:commits:anthropics/anthropic-sdk-typescript": "GitHub commits — TypeScript SDK",
    "github:commits:anthropics/claude-code": "GitHub commits — Claude Code",
    "github:commits:anthropics/anthropic-cookbook": "GitHub commits — Cookbook",
    "github:stars:anthropics/anthropic-sdk-python": "GitHub star velocity — Python SDK",
    "github:stars:anthropics/anthropic-sdk-typescript": "GitHub star velocity — TS SDK",
    "github:stars:anthropics/claude-code": "GitHub star velocity — Claude Code",
    # OpenRouter token volume
    "openrouter:anthropic_tokens": "OpenRouter Anthropic token volume",
    # Enterprise proxy
    "pypi:mypy-boto3-bedrock-runtime": "AWS Bedrock SDK stubs (PyPI) — enterprise proxy",
    "derived:bedrock_spend_estimate": "Bedrock-Anthropic spend estimate (panel-scale $M)",
    "github:search:bedrock_claude": 'GitHub repo searches — "aws bedrock claude"',
    "github:search:vertex_anthropic": 'GitHub repo searches — "vertex anthropic"',
    # Naive baseline (backtest only)
    "ARR_trend_last_growth": "Naive growth-rate baseline",
}

METRIC_LABELS: dict[str, str] = {
    "rmse_loo": "typical error ($B)",
    "r2": "fit quality (R²)",
    "n_train": "months trained",
    "ci_low": "lower bound (95%)",
    "ci_high": "upper bound (95%)",
    "mape": "average % error",
    "mae": "average $B error",
    "rmse": "root-mean-squared error ($B)",
}

CHANNEL_LABELS: dict[str, str] = {
    "dev": "Developer-tool signal",
    "enterprise": "Enterprise-proxy signal",
    "other": "Other",
}
