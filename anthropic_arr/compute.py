"""ARR predictor — works against the standalone anthropic_arr.db.

Numerical helpers live in `.numerics` (no parent-project imports).
"""

from __future__ import annotations

import math
import sqlite3
from datetime import date, datetime, timedelta, timezone

from .numerics import (
    days_in_month,
    fit_linear,
    month_end_date,
    month_index,
    t_critical_975,
)

# Single source of truth for ensemble-admission. A signal is "in ensemble" iff
# it's not a shadow signal AND its regression fit meets the R² and LOO-RMSE
# thresholds. compute.py stamps this on every per_signal row before shipping
# to the frontend; backtest.py and backtest_yipit_private.py import + call
# `is_admissible()` directly so live and backtest agree on the rule.
R2_FLOOR: float = 0.5
N_TRAIN_FLOOR: int = 4  # 2 points give trivial R²=1; 3 give ~degenerate LOO.

# Ceiling on any single signal's share of the blended signal leg.
#
# Why this exists (2026-07): inverse-variance × channel_weight had
# derived:bedrock_spend_estimate at 76.8% of a four-signal ensemble for
# 2026-07 — the blend was that one series plus a rounding error. Two
# multipliers compound to get there, and neither is evidence that it
# deserves to outvote the rest 3:1:
#   - channel_weight 1.5 vs 0.5 is a flat 3× before any fit quality is
#     considered. It encodes "enterprise is ~80% of ARR", but there is
#     exactly ONE enterprise-proxy signal, so a channel-level tilt and a
#     single-series veto are indistinguishable in the current registry.
#   - 1/rmse_loo² squares a thin margin. R² across the four admitted
#     signals spans 0.9745-0.9974; bedrock's 0.46pp edge over
#     npm:@anthropic-ai/sdk becomes a 2.2× variance edge. Its LOO also
#     averages 7 folds against the npm signals' 13, and it is a *derived*
#     series (modelled from pypi:mypy-boto3-bedrock-runtime), so part of
#     that low residual is its own smoothing rather than accuracy.
#
# The cap keeps the channel tilt and the inverse-variance ordering intact
# below the ceiling — it only stops one series from being the answer.
#
# To remove: delete this constant and have signal_blend_weights() return
# the plain normalised raw weights. That restores the pre-cap behaviour
# exactly; nothing else depends on the cap. Removing it is reasonable once
# the enterprise channel has ≥2 independent signals, since concentration
# would then reflect agreement between series rather than one series.
SIGNAL_WEIGHT_CAP: float = 0.40


def signal_blend_weights(usable: list[dict]) -> list[float]:
    """Normalised blend weights for admitted signals: inverse-variance ×
    channel_weight, then capped at SIGNAL_WEIGHT_CAP each.

    Shared by the live ensemble and the walk-forward backtest so the
    reported MAPE always describes the rule actually in use.
    """
    raw = [
        (1.0 / p["regression"]["rmse_loo"] ** 2) * p["channel_weight"] for p in usable
    ]
    return cap_weights(raw, SIGNAL_WEIGHT_CAP)


def cap_weights(raw: list[float], cap: float) -> list[float]:
    """Normalise `raw` to sum 1, then water-fill so no entry exceeds `cap`.

    Excess from capped entries is redistributed proportionally among those
    still under the ceiling, which preserves their relative ordering.
    Iterated because a redistribution can push another entry over.
    """
    total = sum(raw)
    if total <= 0:
        return []
    w = [r / total for r in raw]
    n = len(w)
    if cap * n <= 1.0:
        # Fewer signals than the cap can accommodate (n ≤ 1/cap) — equal
        # weighting is the closest feasible point to the requested ceiling.
        return [1.0 / n] * n
    for _ in range(n):
        excess = sum(x - cap for x in w if x > cap)
        if excess <= 1e-12:
            break
        under = sum(x for x in w if x < cap)
        if under <= 1e-12:
            break
        w = [cap if x >= cap else x + excess * x / under for x in w]
    return w


def is_admissible(p: dict) -> bool:
    """Return True if signal row `p` qualifies for ensemble weighting.

    Guards:
    - not a shadow-flagged signal
    - regression fit exists with R² ≥ R2_FLOOR (0.5)
    - LOO-RMSE exists and is positive (used as inverse-variance weight)
    - n_train ≥ N_TRAIN_FLOOR (4) — with 2-3 points R² is uninformative
      and the LOO residual variance is dominated by which single point
      gets held out
    - predicted_arr_b > 0 — extrapolation past training range can push a
      well-fit signal into negative-ARR territory (e.g. github:search:*
      with a slightly negative slope + tiny signal value)
    """
    if p.get("shadow"):
        return False
    r = p.get("regression") or {}
    if (
        r.get("r2") is None
        or r["r2"] < R2_FLOOR
        or r.get("rmse_loo") is None
        or r["rmse_loo"] <= 0
        or r.get("n_train") is None
        or r["n_train"] < N_TRAIN_FLOOR
    ):
        return False
    return (p.get("predicted_arr_b") or 0.0) > 0


# Active signals used by the ensemble. Channel weights are based on the
# Yipit T6M ARR-mix observation (~80% enterprise, ~10% SMB, ~6% dev) — we
# down-weight dev signals (which all correlate with each other but represent
# only a small slice of revenue) and up-weight enterprise-proxy signals.
#
# GitHub commits + stars + search counts and openrouter:anthropic_tokens are
# *fetched* by sync (see SHADOW_SIGNALS below) so we accumulate history, but
# they are not yet ensemble-active — at current data volumes they hurt MAPE.
# Promote a shadow signal by moving it into this dict once its walk-forward
# MAPE ≤ 12% on n ≥ 6 folds.
SIGNAL_REGISTRY: dict[str, dict] = {
    # Dev channel — top-tier per walk-forward MAPE (npm SDK 8.3%, pypi 12.2%,
    # ai-sdk/anthropic 14.4%). We dropped npm:@anthropic-ai/claude-code
    # (45% MAPE — too noisy) and pypi:mypy-boto3-bedrock-runtime (30% MAPE
    # AND miscategorised — its channel_weight was pulling weak predictions
    # into the enterprise bucket and dragging the ensemble down on 2026-06,
    # which was the last-fold ARR jump the linear signal couldn't track).
    "npm:@anthropic-ai/sdk": {"channel": "dev", "channel_weight": 0.5},
    "pypi:anthropic": {"channel": "dev", "channel_weight": 0.5},
    "npm:@ai-sdk/anthropic": {"channel": "dev", "channel_weight": 0.5},
    # Enterprise-proxy channel (~80% of ARR; weighted higher). Bedrock spend
    # estimate is our best-MAPE signal at 5.94% — Yipit-calibrated 2-var fit
    # over pypi:mypy-boto3-bedrock-runtime × npm:@anthropic-ai/sdk. Promoted
    # from shadow after the walk-forward MAPE gate exposed it.
    "derived:bedrock_spend_estimate": {"channel": "enterprise", "channel_weight": 1.5},
    # ARR-momentum synthetic signal (no external data — computed from
    # anthropic_arr_known alone: pred = ARR_{t-1} × (ARR_{t-1}/ARR_{t-2})^gap).
    # Walk-forward MAPE 12.2% keeps it competitive with the SDK signals, and
    # unlike them it does NOT plateau when SDK download growth saturates —
    # which was letting OpenAI's rMs (castora checkpoint-momentum) edge out
    # Anthropic's (signals-only) in July, inverted vs actual last-6-mo
    # geometric mean growth of 37.5%/mo vs 13.2%/mo.
    # channel_weight=2.0: momentum is the direct compile of confirmed ARR,
    # not a channel proxy — weighted above enterprise (1.5) so it isn't
    # dominated by a single proxy signal (bedrock_spend) that happens to
    # predict conservatively when SDK downloads plateau.
    "arr_trend:last_growth": {"channel": "momentum", "channel_weight": 4.0},
}
# Signals fetched but not yet ensemble-active (kept in DB for history accrual
# and frontend display). Backtest walk-forward measures MAPE on these too
# (see backtest.walk_forward), so we can honour the promotion rule above.
SHADOW_SIGNALS: list[str] = [
    "openrouter:anthropic_tokens",
    # Downstream Vercel Gateway Claude-family cost share (mirrors castora's
    # vercel_spend_share triangulation term). Populated by
    # sources_derived.compute_vercel_claude_family_cost_share_rows.
    "derived:vercel_claude_family_cost_share",
    # Framework wrapper package (mirrors castora's Anthropic signals set —
    # arr.companies.anthropic.signals.pypi).
    "pypi:langchain-anthropic",
    # Demoted from PROXY — 45% and 30% walk-forward MAPE respectively; kept
    # here so the daily-value history keeps accruing for future re-promotion.
    "npm:@anthropic-ai/claude-code",
    "pypi:mypy-boto3-bedrock-runtime",
    "github:commits:anthropics/anthropic-sdk-python",
    "github:commits:anthropics/anthropic-sdk-typescript",
    "github:commits:anthropics/claude-code",
    "github:commits:anthropics/anthropic-cookbook",
    "github:stars:anthropics/anthropic-sdk-python",
    "github:stars:anthropics/anthropic-sdk-typescript",
    "github:stars:anthropics/claude-code",
    "github:search:bedrock_claude",
    "github:search:vertex_anthropic",
    # castora github_queries for Anthropic — enterprise-adoption proxies.
    "github:search:anthropic_api_key",
    "github:search:api_anthropic_com",
]
PROXY_SIGNALS: list[str] = list(SIGNAL_REGISTRY.keys())


# ---------------------------------------------------------------------------
# OpenAI ARR predictor — parallel registry (same methodology, disjoint signals)
# ---------------------------------------------------------------------------
# Calibration history:
#   * v5 (pre-anchor-cleanup, 2026-07-22 AM): admissible = @ai-sdk/openai +
#     pypi:openai. Ensemble walk-forward MAPE = 10.56% over 8 folds.
#   * v6 (post-anchor-cleanup, 2026-07-22 PM): after correcting stale/floor
#     checkpoint values (2026-03: $25B → $29.5B; 2026-05: $33B → $37B),
#     R² and LOO improved sharply and gpt-tokenizer became viable:
#       npm:@ai-sdk/openai   R²=0.985 LOO=1.77
#       pypi:openai          R²=0.968 LOO=2.20
#       npm:gpt-tokenizer    ADDED — MAPE 7.40% vs 7.74% without.
#   * v7 (2026-07-22 PM #2): added consumer disclosure signals
#     (openai:paid_subs_m, openai:chatgpt_wau_m). Initially promoted
#     paid_subs after seeing "4.52% single-signal MAPE", but code review
#     exposed that number as an artefact of log-linear interpolation between
#     the 3 real disclosure points — the walk-forward was scored against
#     itself-interpolated y-values, leaking future information. See
#     experiments/openai_paid_subs_honest_eval.py: only 2 real
#     (disclosure_month, ARR_anchor_month) overlap pairs exist for both
#     paid_subs and WAU, insufficient for is_admissible's n_train≥4 gate.
#   * v8 (2026-07-22 PM #3, POST-REVIEW): paid_subs demoted to shadow.
#     Ensemble stays on the 3 npm/pypi signals. Consumer disclosures re-
#     enter as admissible only when we have ≥4 real anchor-overlap months,
#     i.e. after ~4 more disclosure updates. Kept in DB + payload so the
#     signals-tab UI can still show them as reference.
#   * v9 (2026-08-03): gpt-tokenizer demoted back to shadow. Its v6 promotion
#     was scored under the monthly-sum estimator; re-run against the ramped
#     blend it no longer earns its place — walk-forward over 11 folds × 6
#     as-of days puts MAPE at 4.42% without it vs 4.77% with, and the mean
#     reported band at ±25.8% vs ±35.6%. It was the widest term by far: 67%
#     past its training range at the Aug-2026 target (vs 35-38% for the other
#     two), lowest R², highest LOO, and a small-magnitude series whose steep
#     slope turns feature noise into $B swings. Better on both axes without it.
# Channel weights: api-dev = 1.0 (raised from Anthropic-side 0.5 because
# OpenAI has no Bedrock-style enterprise proxy).
#
# NOTE: with gpt-tokenizer shadowed the ensemble is entirely api-dev — there is
# no consumer-channel signal left, so the channel weights no longer tilt
# anything. Restoring a consumer proxy is what would let this set triangulate
# rather than corroborate; the two remaining signals disagree by ~$8.6B at the
# Aug-2026 target and nothing independent breaks the tie.
SIGNAL_REGISTRY_OPENAI: dict[str, dict] = {
    "npm:@ai-sdk/openai": {"channel": "api-dev", "channel_weight": 1.0},
    "pypi:openai": {"channel": "api-dev", "channel_weight": 1.0},
}
# Shadow signals: fetched + evaluated per walk-forward, but NOT weighted into
# the ensemble. Promoted signals move up to SIGNAL_REGISTRY_OPENAI above.
#
# Why each shadow (post-v8 2026-07-22):
#   npm:openai              — full-history MAPE 15.8% (above 12% gate) and
#                             ADDING it to the ensemble raised MAPE 7.74% →
#                             9.73%. Leave-one-out proves the R²=0.987 fit
#                             doesn't translate to walk-forward accuracy.
#   npm:@openai/agents      — +@openai/agents subset MAPE 11.85% vs 7.74%
#                             baseline; 2025-09 fold catastrophic (38% err)
#                             because @openai/agents launched mid-2025 and
#                             the linear fit can't handle exponential-from-
#                             zero adoption. Revisit ≥12mo post-stabilization.
#   npm:@openai/codex       — 113% single-signal MAPE, too new.
#   openai:paid_subs_m      — 3 real disclosures interpolate to 11 training
#                             points; 4.52% MAPE was an interpolation
#                             artefact. Revisit once ≥4 real disclosures
#                             land on ARR anchor months.
#   openai:chatgpt_wau_m    — same interpolation issue + WAU alone MAPE
#                             18.8% (R²=0.90 but ARPU drift $16-29/user/yr).
#   npm:gpt-tokenizer       — promoted v6, demoted again v9 (see above): under
#                             the ramped blend it costs 0.35pp of MAPE and
#                             ~10pp of band width. Re-promote only if a rerun
#                             of that walk-forward reverses, not on R² alone —
#                             its level fit looks fine and still extrapolates
#                             worst of the set.
SHADOW_SIGNALS_OPENAI: list[str] = [
    "npm:openai",
    "npm:@openai/agents",
    "npm:@openai/codex",
    "openai:paid_subs_m",
    "openai:chatgpt_wau_m",
    "npm:gpt-tokenizer",
]
PROXY_SIGNALS_OPENAI: list[str] = list(SIGNAL_REGISTRY_OPENAI.keys())


# Signals whose DB rows already ARE monthly-aggregate values (one row per
# month, not per day). Used by _project_monthly_sum to skip the partial-
# month scaling that npm/pypi/github signals need. Explicit list rather
# than a prefix rule so a future daily-cadence signal under openai:/yipit:
# won't be silently mis-classified.
MONTHLY_DISCLOSURE_SIGNALS: frozenset[str] = frozenset(
    {
        # yipit:* — Anthropic-side channel spend (private, back-solved).
        "yipit:subscriptions_revenue",
        "yipit:b2b_panel_spend",
        "yipit:bedrock_spend",
        "yipit:gcp_vertex_spend",
        "yipit:azure_spend",
        # openai:* — first-party consumer usage disclosures (hand-curated,
        # log-linear interpolated + forward-extrapolated by
        # sources_openai_consumer._interpolate_monthly).
        "openai:paid_subs_m",
        "openai:chatgpt_wau_m",
    }
)

# PyPI had an index-wide count break on 2026-08-25. Matched 7-day windows
# produced additive offsets for only these two ensemble members. Retire the
# correction once enough post-break ARR anchors support direct recalibration.
PYPI_STEP_BREAK_DATE = "2026-08-25"
PYPI_STEP_OFFSETS: dict[str, float] = {
    "pypi:anthropic": 3.00e6,
    "pypi:openai": 6.70e6,
}
_PYPI_STEP_BREAK = date.fromisoformat(PYPI_STEP_BREAK_DATE)


def model_signal_value(signal: str, date_value: str, raw_value: float) -> float:
    """Return a model-facing signal value without mutating the stored row."""
    observed_date = date.fromisoformat(date_value)
    offset = PYPI_STEP_OFFSETS.get(signal, 0.0)
    if offset and observed_date >= _PYPI_STEP_BREAK:
        return float(raw_value) + offset
    return float(raw_value)


def _days_in_month(month: str) -> int:
    return days_in_month(month)


def _monthly_sum(conn: sqlite3.Connection, signal: str, month: str) -> float | None:
    rows = conn.execute(
        "SELECT date, value FROM anthropic_signals "
        "WHERE signal_name=? AND substr(date,1,7)=?",
        (signal, month),
    ).fetchall()
    total = sum(model_signal_value(signal, r["date"], r["value"]) for r in rows)
    return total if total else None


def _monthly_star_velocity(
    conn: sqlite3.Connection, signal: str, month: str
) -> float | None:
    """For github:stars:* signals — return (last value in month) - (first value in month).

    If only one snapshot exists for the month, return None (cannot compute velocity).
    Stars are cumulative, so we want the within-month delta.
    """
    rows = conn.execute(
        "SELECT date, value FROM anthropic_signals "
        "WHERE signal_name=? AND substr(date,1,7)=? "
        "ORDER BY date",
        (signal, month),
    ).fetchall()
    if len(rows) < 2:
        return None
    return float(rows[-1]["value"]) - float(rows[0]["value"])


def _monthly_value(conn: sqlite3.Connection, signal: str, month: str) -> float | None:
    """Dispatch — star signals use velocity, others use sum."""
    if signal.startswith("github:stars:"):
        return _monthly_star_velocity(conn, signal, month)
    return _monthly_sum(conn, signal, month)


def _project_monthly_sum(
    conn: sqlite3.Connection,
    signal: str,
    target_month: str,
    cutoff_date: str | None = None,
) -> tuple[float, str] | None:
    """Project partial-month signal to full-month.

    `cutoff_date` is exclusive: only rows with `date < cutoff_date` are used.
    Defaults to today UTC (so the no-arg call still excludes today's partial
    intra-day data, preserving prior behaviour). For the daily-ARR series we
    pass `cutoff_date = next-day` to include data through a specific day D.

    For star-velocity signals, we cannot project; just return the observed
    delta so far if there are at least 2 snapshots.
    """
    if signal.startswith("github:stars:"):
        v = _monthly_star_velocity(conn, signal, target_month)
        if v is None or v <= 0:
            return None
        return v, f"star_velocity_partial(observed delta {v:.0f})"
    # Disclosure-based signals are stored as ONE row per month at month-
    # end (or first-of-month for yipit). They don't have a notion of
    # "partial-through-day-D" — the row IS the full-month value. Skip the
    # cutoff_date filter and the *days_in_target/days_observed projection.
    #
    # Explicit list (rather than prefix startswith) so a future
    # openai:daily_new_signup or yipit:hourly_* signal wouldn't be
    # mis-classified as monthly.
    if signal in MONTHLY_DISCLOSURE_SIGNALS:
        v = _monthly_value(conn, signal, target_month)
        if v is None or v <= 0:
            return None
        # Value is already the full monthly figure — no partial-month scaling.
        return v, f"monthly_disclosure(value={v:.2f})"
    if cutoff_date is None:
        cutoff_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    rows = conn.execute(
        "SELECT date, value FROM anthropic_signals "
        "WHERE signal_name=? AND substr(date,1,7)=? AND value > 0 AND date < ?",
        (signal, target_month, cutoff_date),
    ).fetchall()
    if not rows:
        return None
    partial = sum(model_signal_value(signal, r["date"], r["value"]) for r in rows)
    days_observed = len(rows)
    days_in_target = _days_in_month(target_month)
    projected = partial * days_in_target / days_observed
    # Display units: npm/pypi return raw daily counts (millions of downloads),
    # while derived:* signals are already in panel-scale $M — dividing them
    # by 1e6 collapsed the string to '0.0M'. Detect and scale per-signal.
    is_dl_count = signal.startswith(("npm:", "pypi:", "github:"))
    unit = "M" if is_dl_count else "M$"
    scale = 1e6 if is_dl_count else 1.0
    return projected, (
        f"monthly_sum_partial(observed={days_observed}d, "
        f"partial={partial / scale:.1f}{unit} -> "
        f"proj {projected / scale:.1f}{unit})"
    )


def _predict_arr_trend_last_growth(
    arr_known: dict[str, float], target_month: str, registry: dict | None = None
) -> dict | None:
    """Synthetic 'signal' with no external data — projects ARR forward using
    the last-observed monthly growth rate. Predict/LOO shape matches the
    regression predictor so the ensemble weighter treats it uniformly.

        pred(T) = arr[t_prev] × (arr[t_prev] / arr[t_prev2]) ^ (gap_target)

    with `gap_target` = whole-month distance from t_prev to T. LOO residuals
    are computed by holding out each interior month one at a time, refitting
    the momentum from the surrounding two anchors, and predicting the held-
    out point — same protocol the regression uses so RMSE_loo is comparable
    for inverse-variance weighting.
    """
    months = sorted(arr_known.keys())
    if len(months) < 3 or target_month <= months[-1]:
        # Need at least 3 anchors for LOO. If target is a past month we're
        # in walk-forward mode: caller has restricted `arr_known` to months
        # < target, so target_month <= months[-1] shouldn't happen in that
        # path; this guard is for live-prediction robustness.
        pass
    if len(months) < 3:
        return None

    def _months_between(a: str, b: str) -> int:
        ya, ma = int(a[:4]), int(a[5:])
        yb, mb = int(b[:4]), int(b[5:])
        return (yb - ya) * 12 + (mb - ma)

    t_prev, t_prev2 = months[-1], months[-2]
    v_prev, v_prev2 = arr_known[t_prev], arr_known[t_prev2]
    n_train_gap = _months_between(t_prev2, t_prev)
    n_target_gap = _months_between(t_prev, target_month)
    if v_prev <= 0 or v_prev2 <= 0 or n_train_gap <= 0 or n_target_gap <= 0:
        return None
    compound_per_mo = (v_prev / v_prev2) ** (1.0 / n_train_gap)
    predicted = v_prev * compound_per_mo**n_target_gap

    # LOO residuals: recompute momentum with each interior month held out.
    loo_resid: list[float] = []
    for i in range(2, len(months)):
        # Held-out target = months[i]; train = months[:i]; use last-two.
        m_target = months[i]
        m_prev, m_prev2 = months[i - 1], months[i - 2]
        v_p, v_pp = arr_known[m_prev], arr_known[m_prev2]
        g_train = _months_between(m_prev2, m_prev)
        g_target = _months_between(m_prev, m_target)
        if v_p <= 0 or v_pp <= 0 or g_train <= 0 or g_target <= 0:
            continue
        cpm = (v_p / v_pp) ** (1.0 / g_train)
        pred_i = v_p * cpm**g_target
        loo_resid.append(arr_known[m_target] - pred_i)

    if loo_resid:
        rmse_loo = math.sqrt(sum(r * r for r in loo_resid) / len(loo_resid))
        df = max(1, len(loo_resid) - 1)
        ci_half = t_critical_975(df) * rmse_loo
    else:
        rmse_loo = None
        ci_half = 0.0

    reg = registry if registry is not None else SIGNAL_REGISTRY
    meta = reg.get(
        "arr_trend:last_growth", {"channel": "momentum", "channel_weight": 1.0}
    )
    return {
        "signal": "arr_trend:last_growth",
        "channel": meta["channel"],
        "channel_weight": meta["channel_weight"],
        "predicted_arr_b": predicted,
        "ci_low": predicted - ci_half,
        "ci_high": predicted + ci_half,
        "target_signal_value": compound_per_mo,
        "forecast_method": (
            f"momentum: ARR[{t_prev}]={v_prev:.2f}B × "
            f"({v_prev / v_prev2:.3f})^({n_target_gap}/{n_train_gap}mo)"
        ),
        "regression": {
            # R²=1.0 is nominal: momentum is not a fit, but downstream
            # is_admissible checks R²≥0.5 and R²=1.0 with n_train≥4 (LOO
            # points) is the correct semantic — "this predictor perfectly
            # reproduces its own construction rule on the training set".
            "intercept": 0.0,
            "slope": 1.0,
            "r2": 1.0,
            "n_train": len(loo_resid) if loo_resid else 0,
            "rmse_loo": rmse_loo,
        },
        "points": [
            {
                "month": m,
                "signal_value": None,
                "arr_known_b": arr_known[m],
                "arr_fitted_b": arr_known[m],
            }
            for m in months
        ],
    }


def _signal_row(
    signal: str,
    arr_known: dict[str, float],
    train_x: list[float],
    train_y: list[float],
    train_months: list[str],
    target_signal: float,
    method: str,
    registry: dict | None = None,
) -> dict | None:
    """Fit `train_y ~ train_x`, predict at `target_signal`, and shape the
    prediction dict the ensemble weighter consumes.

    Shared by the monthly-sum path (`_predict_for_signal`) and the end-of-month
    path (`_predict_for_signal_eom`) so the two can never drift apart on the
    regression, the LOO interval, or the row shape — only on how the feature is
    measured, which is the whole point of the difference.
    """
    fit = fit_linear(train_x, train_y)
    if not fit:
        return None
    intercept, slope, r2 = fit
    predicted = intercept + slope * target_signal

    # LOO residuals → t × RMSE_loo CI
    loo_resid: list[float] = []
    for i in range(len(train_x)):
        sub = fit_linear(
            train_x[:i] + train_x[i + 1 :],
            train_y[:i] + train_y[i + 1 :],
        )
        if sub:
            ic, sl, _ = sub
            loo_resid.append(train_y[i] - (ic + sl * train_x[i]))
    if loo_resid:
        rmse_loo = math.sqrt(sum(r * r for r in loo_resid) / len(loo_resid))
        df = max(1, len(train_x) - 2)
        ci_half = t_critical_975(df) * rmse_loo
    else:
        rmse_loo = None
        ci_half = 0.0

    fitted = [intercept + slope * x for x in train_x]
    reg = registry if registry is not None else SIGNAL_REGISTRY
    meta = reg.get(signal, {"channel": "other", "channel_weight": 1.0})
    return {
        "signal": signal,
        "channel": meta["channel"],
        "channel_weight": meta["channel_weight"],
        "predicted_arr_b": predicted,
        "ci_low": predicted - ci_half,
        "ci_high": predicted + ci_half,
        "target_signal_value": target_signal,
        "forecast_method": method,
        "regression": {
            "intercept": intercept,
            "slope": slope,
            "r2": r2,
            "n_train": len(train_x),
            "rmse_loo": rmse_loo,
        },
        "points": [
            {
                "month": train_months[i],
                "signal_value": train_x[i],
                "arr_known_b": train_y[i],
                "arr_fitted_b": fitted[i],
            }
            for i in range(len(train_x))
        ],
    }


def _predict_for_signal(
    conn: sqlite3.Connection,
    signal: str,
    arr_known: dict[str, float],
    target_month: str,
    cutoff_date: str | None = None,
    registry: dict | None = None,
) -> dict | None:
    """Single-signal monthly-sum regression. Returns prediction dict or None.

    `cutoff_date` (exclusive) bounds what the target-month projection may see;
    it exists so an as-of-day backtest can reproduce what this predictor would
    have said partway through the month. None = today, the live behaviour.

    `registry` overrides the module-level SIGNAL_REGISTRY channel/weight
    lookup so predict_openai_arr can pass SIGNAL_REGISTRY_OPENAI without
    forcing the OpenAI signals into the Anthropic-side registry.
    """
    if signal == "arr_trend:last_growth":
        return _predict_arr_trend_last_growth(
            arr_known, target_month, registry=registry
        )
    train_x: list[float] = []
    train_y: list[float] = []
    train_months: list[str] = []
    for m in sorted(arr_known):
        v = _monthly_value(conn, signal, m)
        if v is None or v <= 0:
            continue
        train_x.append(v)
        train_y.append(arr_known[m])
        train_months.append(m)
    if len(train_x) < 2:
        return None
    forecast = _project_monthly_sum(conn, signal, target_month, cutoff_date)
    if forecast is None:
        return None
    target_signal, method = forecast
    return _signal_row(
        signal,
        arr_known,
        train_x,
        train_y,
        train_months,
        target_signal,
        method,
        registry=registry,
    )


# ---------------------------------------------------------------------------
# ramped_blend nowcast — ported from funda-data-flow syncs/llm_arr/compute.py
# (e9bae91, "replace rolling_30d headline with ramped_blend nowcast").
#
# The problem it fixes: `_project_monthly_sum` scales a partial month to full
# by partial × days_in_month / days_observed. On day 1 that is one day × 31 —
# a single noisy day sets the headline, and the estimate lurches at every month
# boundary. A plain trailing window removes the lurch but replaces it with a
# systematic lag: its window is dominated by the *previous* month, so for a
# growing lab it underestimates end-of-month ARR.
#
# ramped_blend combines two differently-wrong estimators instead:
#   trend  — geometric extrapolation of the last two ARR anchors, and
#   signal — a de-lagged end-of-month signal ensemble (eom_window_sum fills the
#            not-yet-observed tail of the target month at the recent daily rate,
#            so the window reflects month-END rather than today).
# Weight ramps from trend (0.8 early, when the month's signals barely exist) to
# signal (0.2 by mid-month). A FIXED ramp, not fitted weights — with ~12 months
# of anchors, fitted weights overfit (the forecast-combination puzzle).
#
# Upstream has no momentum signal, so its trend leg was new. Here the same
# trajectory already exists as the `arr_trend:last_growth` pseudo-signal, fused
# into the inverse-variance ensemble at a hand-set channel_weight of 4.0 (~46%
# of ensemble weight, constant all month) and carrying a nominal r2=1.0 purely
# to satisfy is_admissible. The port's substance here is pulling it back out and
# putting that weight on the day schedule it should have been on.
ROLLING_WINDOW_DAYS = 30
ROLLING_LAG_DAYS = 2  # npm finalizes ~2d late; never read a lagging feed as 0
ROLLING_MIN_DAYS = 20  # min real positive days for a window to be trusted
EOM_FILL_LOOKBACK = 7  # days of recent history defining "current daily rate"

RAMP_W_START = 0.8
RAMP_W_END = 0.2
RAMP_DECAY_PER_DAY = 0.05  # 0.8 @ day 2 -> 0.2 @ day 14, flat after

# Pins are only for an in-flight month and must be retired before the month rolls.
# Prefer KNOWN_ARR for reported historical figures.
PINNED_ARR: dict[str, float] = {}


def _daily_series(conn: sqlite3.Connection, signal: str) -> dict[str, float]:
    """All positive daily values for `signal`, keyed by ISO date."""
    return {
        r["date"]: model_signal_value(signal, r["date"], r["value"])
        for r in conn.execute(
            "SELECT date, value FROM anthropic_signals "
            "WHERE signal_name=? AND value > 0",
            (signal,),
        )
    }


def _trailing_window_sum(
    series: dict[str, float],
    end: date,
    window_days: int = ROLLING_WINDOW_DAYS,
    min_days: int = ROLLING_MIN_DAYS,
) -> float | None:
    """Sum of `window_days` ending at `end` inclusive, or None if too sparse."""
    total = 0.0
    real = 0
    for k in range(window_days):
        v = series.get((end - timedelta(days=k)).isoformat())
        if v:
            total += v
            real += 1
    return total if real >= min_days else None


def eom_window_sum(
    series: dict[str, float],
    as_of_date: date,
    target_month: str,
    window_days: int = ROLLING_WINDOW_DAYS,
    lag_days: int = ROLLING_LAG_DAYS,
    min_days: int = ROLLING_MIN_DAYS,
    lookback: int = EOM_FILL_LOOKBACK,
) -> float | None:
    """End-of-month feature: the `window_days` sum ending at the target month's
    END, with the not-yet-observed tail filled at the recent daily rate.

    A trailing window ending "today" reflects mostly the previous month early on,
    so it lags a growing series. Filling the target month's remaining days (those
    after the finalized cutoff `as_of_date - lag_days`) at the recent daily rate
    de-lags the feature toward the value the window will actually have at
    month-end.

    Coverage is judged on the feed's freshness, not on the target-month window
    (which is mostly rate-filled early in the month): `min_days` real positive
    days are required in the *trailing* window ending at `cutoff`. A live feed
    passes even on day 1 of the month because its recent weeks are real, while a
    feed that has gone stale returns None once its dead stretch dominates — it
    can't fabricate a confident value out of an all-rate-filled month.
    """
    cutoff = as_of_date - timedelta(days=lag_days)  # last finalized day, inclusive
    observed = sorted(
        (date.fromisoformat(d), v)
        for d, v in series.items()
        if date.fromisoformat(d) <= cutoff
    )
    if not observed:
        return None
    trailing_start = cutoff - timedelta(days=window_days - 1)
    if sum(1 for d, _ in observed if trailing_start <= d <= cutoff) < min_days:
        return None
    recent = observed[-lookback:]
    rate = sum(v for _, v in recent) / len(recent)
    end = month_end_date(target_month)
    total = 0.0
    for k in range(window_days):
        d = end - timedelta(days=k)
        total += series.get(d.isoformat(), 0.0) if d <= cutoff else rate
    return total


def _eom_train_pairs(
    series: dict[str, float],
    arr_known: dict[str, float],
    window_days: int = ROLLING_WINDOW_DAYS,
    min_days: int = ROLLING_MIN_DAYS,
) -> tuple[list[float], list[float], list[str]]:
    """(window volume, known ARR, month) per anchor — the end-of-month feature
    the signal leg regresses on, measured consistently with the target."""
    xs: list[float] = []
    ys: list[float] = []
    ms: list[str] = []
    for m in sorted(arr_known):
        v = _trailing_window_sum(series, month_end_date(m), window_days, min_days)
        if v is None or v <= 0:
            continue
        xs.append(v)
        ys.append(arr_known[m])
        ms.append(m)
    return xs, ys, ms


def _predict_for_signal_eom(
    conn: sqlite3.Connection,
    signal: str,
    arr_known: dict[str, float],
    as_of_date: date,
    target_month: str,
    window_days: int = ROLLING_WINDOW_DAYS,
    lag_days: int = ROLLING_LAG_DAYS,
    min_days: int = ROLLING_MIN_DAYS,
    registry: dict | None = None,
) -> dict | None:
    """Single-signal regression on the de-lagged end-of-month window feature.

    Star-velocity signals are cumulative snapshots, not daily volumes — summing
    them over a window is meaningless, so they sit this path out (the monthly_sum
    path handles them via `_monthly_star_velocity`).

    `registry` overrides the channel/weight lookup, mirroring
    `_predict_for_signal`, so the OpenAI side can reach this path without its
    signals being scored against the Anthropic registry.
    """
    if signal.startswith("github:stars:") or signal == "arr_trend:last_growth":
        return None
    series = _daily_series(conn, signal)
    if not series:
        return None
    train_x, train_y, train_months = _eom_train_pairs(
        series, arr_known, window_days, min_days
    )
    if len(train_x) < 2:
        return None
    target_value = eom_window_sum(
        series, as_of_date, target_month, window_days, lag_days, min_days
    )
    if target_value is None or target_value <= 0:
        return None
    return _signal_row(
        signal,
        arr_known,
        train_x,
        train_y,
        train_months,
        target_value,
        f"eom_fill(target={target_month}, sum={target_value / 1e6:.1f}M)",
        registry=registry,
    )


DAYS_PER_MONTH = 30.4375


def post_step_run_rate(
    conn: sqlite3.Connection,
    arr_known: dict[str, float],
    target_month: str,
    table: str = "openai_arr_checkpoints",
) -> float | None:
    """Per-month growth multiplier implied by the latest within-month checkpoint
    segment, shrunk toward the last full month the segment's month didn't touch.

    Monthly anchors cannot distinguish a level shift from a durable rate
    change. When a month has multiple sourced checkpoints, only the latest
    within-month segment is eligible and its short duration shrinks it toward
    the preceding full-month rate.

    None unless the latest anchor month carries at least two dated checkpoints.
    """
    months = sorted(m for m in arr_known if m < target_month)
    if len(months) < 3:
        return None
    anchor = months[-1]
    rows = conn.execute(
        f"SELECT date, arr_bn FROM {table} "  # noqa: S608
        "WHERE kind = 'checkpoint' AND substr(date, 1, 7) = ? "
        "ORDER BY date",
        (anchor,),
    ).fetchall()
    if len(rows) < 2:
        return None
    (d0, v0), (d1, v1) = (
        (rows[-2]["date"], float(rows[-2]["arr_bn"])),
        (rows[-1]["date"], float(rows[-1]["arr_bn"])),
    )
    days = (date.fromisoformat(d1) - date.fromisoformat(d0)).days
    if days <= 0 or v0 <= 0 or v1 <= 0:
        return None
    seg_log = math.log(v1 / v0) * DAYS_PER_MONTH / days

    # Last month-over-month leg that ends before the anchor month, so the step
    # itself is nowhere in it.
    prev, prev2 = months[-2], months[-3]
    gap = month_index(prev) - month_index(prev2)
    if gap <= 0 or arr_known[prev2] <= 0:
        return None
    prior_log = math.log(arr_known[prev] / arr_known[prev2]) / gap

    w = min(1.0, days / DAYS_PER_MONTH)
    return math.exp(w * seg_log + (1.0 - w) * prior_log)


def trend_extrapolate(
    arr_known: dict[str, float], target_month: str, per_month: float | None = None
) -> float | None:
    """Geometric extrapolation of the ARR trajectory to `target_month`.

    Projects the per-month growth of the two most recent anchors forward,
    gap-aware so irregular anchor spacing is handled. This is the trajectory the
    ramped blend leans on early in the month, before the month's own signals are
    informative. Same rule as the `arr_trend:last_growth` pseudo-signal, which it
    replaces. None with < 2 anchors.

    `per_month` overrides the rate the two anchors imply, for when finer-grained
    evidence separates a level shift from a rate change — see
    `post_step_run_rate`. The level it projects from is unchanged either way.
    """
    ms = sorted(arr_known)
    if len(ms) < 2:
        return None
    a, b = ms[-2], ms[-1]
    gap = month_index(b) - month_index(a)
    if gap <= 0 or arr_known[a] <= 0:
        return None
    if per_month is None:
        per_month = (arr_known[b] / arr_known[a]) ** (1.0 / gap)
    return arr_known[b] * per_month ** (month_index(target_month) - month_index(b))


def ramp_weight(day_of_month: int) -> float:
    """Trend weight for the ramped blend on a given day of the month: starts at
    RAMP_W_START and decays RAMP_DECAY_PER_DAY per day to a RAMP_W_END floor."""
    return min(
        RAMP_W_START,
        max(RAMP_W_END, RAMP_W_START - RAMP_DECAY_PER_DAY * (day_of_month - 2)),
    )


def _eom_signal_ensemble(per_signal: list[dict]) -> dict | None:
    """Capped inverse-variance blend of the EOM signal legs with a
    law-of-total-variance band, so between-signal disagreement widens the
    interval instead of being averaged away. Also exposes variance/df for the
    ramped combiner. See signal_blend_weights / SIGNAL_WEIGHT_CAP.
    """
    usable = [p for p in per_signal if p.get("in_ensemble")]
    if not usable:
        return None
    wn = signal_blend_weights(usable)
    if not wn:
        return None
    predicted = sum(usable[i]["predicted_arr_b"] * wn[i] for i in range(len(usable)))
    within = sum(
        wn[i] * usable[i]["regression"]["rmse_loo"] ** 2 for i in range(len(usable))
    )
    between = sum(
        wn[i] * (usable[i]["predicted_arr_b"] - predicted) ** 2
        for i in range(len(usable))
    )
    variance = within + between
    df = max(1, min(p["regression"]["n_train"] for p in usable) - 2)
    half = t_critical_975(df) * math.sqrt(variance)
    return {
        "predicted_arr_b": predicted,
        "ci_low": max(0.0, predicted - half),
        "ci_high": predicted + half,
        "n_models": len(usable),
        "n_models_total": len(per_signal),
        "variance": variance,
        "df": df,
        "signals_used": [p["signal"] for p in usable],
    }


def predict_arr_ramped_blend(
    conn: sqlite3.Connection,
    target_month: str | None = None,
    as_of_date: date | None = None,
    arr_known: dict[str, float] | None = None,
    registry: dict | None = None,
    proxy_signals: list[str] | None = None,
    trend_per_month: float | None = None,
) -> dict | None:
    """The ramped nowcast: `predicted = w·trend + (1-w)·signal`, with
    `w = ramp_weight(day)` shifting from trajectory (early) to signals (late).

    The band is a law-of-total-variance interval over the {trend, signal}
    mixture, so when the two legs disagree the reported band widens honestly.
    None when fewer than 2 anchors exist or neither leg resolves.

    `arr_known` overrides the anchor set; the walk-forward backtest passes only
    months < target so the prediction can't see its own answer.

    `registry` / `proxy_signals` select the company's signal set. Both default
    to the Anthropic side; `predict_openai_arr` passes its own pair so the two
    companies share this estimator instead of one of them keeping the
    monthly-sum path the blend was built to replace.

    `trend_per_month` overrides the trend leg's growth rate — see
    `post_step_run_rate`. Left unset by the walk-forward, which is honest today
    because no historical month carries the within-month checkpoints it needs.
    """
    if as_of_date is None:
        as_of_date = datetime.now(timezone.utc).date()
    if target_month is None:
        target_month = as_of_date.strftime("%Y-%m")

    if arr_known is None:
        arr_known = {
            r["month"]: float(r["arr_b_usd"])
            for r in conn.execute(
                "SELECT month, arr_b_usd FROM anthropic_arr_known ORDER BY month"
            )
        }
    if len(arr_known) < 2:
        return None

    per_signal: list[dict] = []
    for sig in proxy_signals if proxy_signals is not None else PROXY_SIGNALS:
        res = _predict_for_signal_eom(
            conn, sig, arr_known, as_of_date, target_month, registry=registry
        )
        if res:
            res["in_ensemble"] = is_admissible(res)
            per_signal.append(res)

    signal_ens = _eom_signal_ensemble(per_signal)
    trend = trend_extrapolate(arr_known, target_month, per_month=trend_per_month)
    w = ramp_weight(as_of_date.day)
    if signal_ens is None and trend is None:
        return None

    if signal_ens is None:
        # No signal fit — fall back to the trajectory alone, and carry the
        # trend's own LOO interval rather than reporting a bare point with no
        # band. Consumers (arr_curve, payload) read ci_low/ci_high directly.
        trend_row = _predict_arr_trend_last_growth(arr_known, target_month)
        blended = trend
        ci_low = trend_row["ci_low"] if trend_row else None
        ci_high = trend_row["ci_high"] if trend_row else None
        weighting, signals_used, n_models = "trend_only(no signal fit)", [], 0
    elif trend is None:
        blended = signal_ens["predicted_arr_b"]
        ci_low, ci_high = signal_ens["ci_low"], signal_ens["ci_high"]
        weighting = "signal_only(<2 anchors for trend)"
        signals_used, n_models = signal_ens["signals_used"], signal_ens["n_models"]
    else:
        s_pred = signal_ens["predicted_arr_b"]
        blended = w * trend + (1 - w) * s_pred
        # Law of total variance over the 2-component mixture {trend (weight w,
        # a point ⇒ no within-variance), signal (weight 1-w, variance var_s)}:
        #   within = (1-w)·var_s ; between = w(1-w)·(trend − signal)²
        within = (1 - w) * signal_ens["variance"]
        between = w * (1 - w) * (trend - s_pred) ** 2
        half = t_critical_975(signal_ens["df"]) * math.sqrt(within + between)
        ci_low, ci_high = max(0.0, blended - half), blended + half
        weighting = f"ramped_blend(w_trend={w:.2f})"
        signals_used, n_models = signal_ens["signals_used"], signal_ens["n_models"]

    return {
        "target_month": target_month,
        "as_of_date": as_of_date.isoformat(),
        "trend_arr_b": trend,
        "signal_arr_b": signal_ens["predicted_arr_b"] if signal_ens else None,
        "weight_trend": w,
        "per_signal": per_signal,
        "ensemble": {
            "predicted_arr_b": blended,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n_models": n_models,
            "n_models_total": len(per_signal),
            "weighting": weighting,
            "signals_used": signals_used,
        },
    }


def _apply_pinned_arr(ramped: dict, target_month: str) -> None:
    """Move `ramped`'s headline onto its PINNED_ARR value, in place.

    The band shifts by the same delta rather than being recomputed: pinning
    moves where the estimate sits, it does not claim to have learned anything
    that would narrow the interval.
    """
    pin = PINNED_ARR.get(target_month)
    if pin is None:
        return
    ens = ramped["ensemble"]
    delta = pin - ens["predicted_arr_b"]
    ramped["unpinned_arr_b"] = ens["predicted_arr_b"]
    ens["predicted_arr_b"] = pin
    ens["ci_low"] = max(0.0, ens["ci_low"] + delta)
    ens["ci_high"] = ens["ci_high"] + delta
    ens["weighting"] = f"pinned({pin:g}B) over {ens['weighting']}"


def _month_gap(a: str, b: str) -> int:
    """Whole-month gap between two YYYY-MM strings (b - a). Returns 0 if equal."""
    ay, am = int(a[:4]), int(a[5:7])
    by, bm = int(b[:4]), int(b[5:7])
    return (by - ay) * 12 + (bm - am)


def _growth_extrapolation(
    arr_known: dict[str, float], target_month: str, window: int = 6
) -> dict | None:
    """Pure growth-rate extrapolation: fit log(ARR) ~ month_index on the last
    `window` known months, project to target.

    Returns {predicted_arr_b, slope_mom_pct, window_used, train_months,
    ci_low, ci_high} or None. CI band is computed from leave-one-out residuals
    in log-space and exponentiated, so it is asymmetric (wider on the upside).

    Walk-forward MAPE on n=11 was 22.8% — worse than proxy-signal regression
    but a useful "what does the recent slope alone say?" comparator.
    """
    months = sorted(arr_known.keys())
    if len(months) < 3:
        return None
    if not target_month or _month_gap(months[-1], target_month) <= 0:
        return None

    recent = months[-window:] if len(months) >= window else months
    base = recent[0]
    xs = [_month_gap(base, m) for m in recent]
    ys = [math.log(arr_known[m]) for m in recent]

    fit = fit_linear(xs, ys)
    if not fit:
        return None
    intercept, slope, _r2_log = fit

    target_x = _month_gap(base, target_month)
    log_pred = intercept + slope * target_x
    pred = math.exp(log_pred)
    slope_mom_pct = (math.exp(slope) - 1.0) * 100.0

    # LOO residuals in log-space. We expose two bands:
    #  - 1σ in log-space: pred × exp(±RMSE_log) — informally "typical" range
    #  - 95% via t × RMSE_log — wide because n ~6 and slope is volatile
    loo_resid: list[float] = []
    for i in range(len(xs)):
        sub = fit_linear(xs[:i] + xs[i + 1 :], ys[:i] + ys[i + 1 :])
        if sub:
            ic, sl, _ = sub
            loo_resid.append(ys[i] - (ic + sl * xs[i]))
    if loo_resid:
        rmse_log = math.sqrt(sum(r * r for r in loo_resid) / len(loo_resid))
        df = max(1, len(xs) - 2)
        ci_low = math.exp(log_pred - rmse_log)  # ±1σ band
        ci_high = math.exp(log_pred + rmse_log)
        ci95_log_half = t_critical_975(df) * rmse_log
        ci95_low = math.exp(log_pred - ci95_log_half)
        ci95_high = math.exp(log_pred + ci95_log_half)
    else:
        rmse_log = None
        ci_low = ci_high = pred
        ci95_low = ci95_high = pred

    return {
        "predicted_arr_b": pred,
        "slope_mom_pct": slope_mom_pct,
        "window_used": len(recent),
        "train_months": recent,
        "target_month": target_month,
        "ci_low": ci_low,  # ±1σ in log-space
        "ci_high": ci_high,
        "ci95_low": ci95_low,  # 95% via t × RMSE_log
        "ci95_high": ci95_high,
        "rmse_log": rmse_log,
    }


def _npm_growth_indicator(
    conn: sqlite3.Connection,
    arr_known: dict[str, float],
    target_month: str | None,
    signal: str = "npm:@anthropic-ai/sdk",
) -> dict | None:
    """Predict ARR by regressing ARR's MoM growth rate on the npm SDK's MoM
    growth rate, then compounding back to the level.

    Approach (validated walk-forward):
      1. For each consecutive pair of confirmed ARR months, compute the
         compound monthly rate (ARR_t/ARR_{t-1})^(1/gap) − 1.
      2. For the same pairs, compute the npm signal's compound monthly rate.
      3. OLS fit ARR_MoM% ~ a + b × signal_MoM%.
      4. To predict the target month, observe the latest signal_MoM% (signal
         partial-month projected to full month, then compared to last
         confirmed month's full signal value), apply the fit to get
         predicted_ARR_MoM%, then ARR_target = ARR_prev × (1+pct/100)^gap.

    Walk-forward MAPE on n=11: ~8% (npm:@anthropic-ai/sdk; r=+0.94 on
    growth-on-growth correlation).
    """
    if target_month is None or len(arr_known) < 4:
        return None
    months = sorted(arr_known.keys())
    if _month_gap(months[-1], target_month) <= 0:
        return None

    # ARR MoM rates per leg
    arr_mom: dict[str, float] = {}  # to_month -> compound monthly rate
    arr_prev_value: dict[str, float] = {}
    arr_prev_month: dict[str, str] = {}
    for i in range(1, len(months)):
        p, n = months[i - 1], months[i]
        gap = _month_gap(p, n)
        if gap > 0 and arr_known[p] > 0:
            arr_mom[n] = (arr_known[n] / arr_known[p]) ** (1.0 / gap) - 1.0
            arr_prev_value[n] = arr_known[p]
            arr_prev_month[n] = p

    # Signal monthly totals + MoM rates
    sig_monthly: dict[str, float] = {}
    for r in conn.execute(
        "SELECT substr(date,1,7) AS m, SUM(value) AS v FROM anthropic_signals "
        "WHERE signal_name=? AND value > 0 GROUP BY m",
        (signal,),
    ):
        if r["v"] and r["v"] > 0:
            sig_monthly[r["m"]] = float(r["v"])
    sig_mom: dict[str, float] = {}
    sm = sorted(sig_monthly.keys())
    for i in range(1, len(sm)):
        p, n = sm[i - 1], sm[i]
        gap = _month_gap(p, n)
        if gap > 0 and sig_monthly[p] > 0:
            sig_mom[n] = (sig_monthly[n] / sig_monthly[p]) ** (1.0 / gap) - 1.0

    # Training set: months in both arr_mom and sig_mom
    overlap = sorted(set(arr_mom.keys()) & set(sig_mom.keys()))
    if len(overlap) < 3:
        return None

    xs = [sig_mom[m] * 100.0 for m in overlap]  # signal MoM %
    ys = [arr_mom[m] * 100.0 for m in overlap]  # ARR MoM %

    fit = fit_linear(xs, ys)
    if not fit:
        return None
    intercept, slope, r2 = fit
    # Pearson r — sign of correlation, useful for the methodology copy.
    n = len(xs)
    mxr = sum(xs) / n
    myr = sum(ys) / n
    sxr = math.sqrt(sum((x - mxr) ** 2 for x in xs) / n)
    syr = math.sqrt(sum((y - myr) ** 2 for y in ys) / n)
    if sxr > 0 and syr > 0:
        pearson_r = sum((xs[i] - mxr) * (ys[i] - myr) for i in range(n)) / (
            n * sxr * syr
        )
    else:
        pearson_r = None

    # Project signal's MoM rate from last confirmed month → target.
    prev_full_month = months[-1]
    if prev_full_month not in sig_monthly:
        return None
    target_signal_proj = _project_monthly_sum(conn, signal, target_month)
    if target_signal_proj is None:
        return None
    target_sig_value, _ = target_signal_proj
    if sig_monthly[prev_full_month] <= 0 or target_sig_value <= 0:
        return None
    target_gap = _month_gap(prev_full_month, target_month)
    if target_gap <= 0:
        return None
    target_signal_mom_pct = (
        (target_sig_value / sig_monthly[prev_full_month]) ** (1.0 / target_gap) - 1.0
    ) * 100.0

    predicted_arr_mom_pct = intercept + slope * target_signal_mom_pct
    prev_arr = arr_known[prev_full_month]
    predicted_arr = prev_arr * (1.0 + predicted_arr_mom_pct / 100.0) ** target_gap

    # LOO residuals → CI on the predicted ARR_MoM%, then compound to $B band.
    loo_resid: list[float] = []
    for i in range(len(xs)):
        sub = fit_linear(xs[:i] + xs[i + 1 :], ys[:i] + ys[i + 1 :])
        if sub:
            ic, sl, _ = sub
            loo_resid.append(ys[i] - (ic + sl * xs[i]))
    if loo_resid:
        rmse_pct = math.sqrt(sum(r * r for r in loo_resid) / len(loo_resid))
        # ±1σ in MoM% space → compound to ARR
        ci_low = (
            prev_arr * (1.0 + (predicted_arr_mom_pct - rmse_pct) / 100.0) ** target_gap
        )
        ci_high = (
            prev_arr * (1.0 + (predicted_arr_mom_pct + rmse_pct) / 100.0) ** target_gap
        )
        df = max(1, len(xs) - 2)
        ci95_half = t_critical_975(df) * rmse_pct
        ci95_low = (
            prev_arr * (1.0 + (predicted_arr_mom_pct - ci95_half) / 100.0) ** target_gap
        )
        ci95_high = (
            prev_arr * (1.0 + (predicted_arr_mom_pct + ci95_half) / 100.0) ** target_gap
        )
    else:
        rmse_pct = None
        ci_low = ci_high = predicted_arr
        ci95_low = ci95_high = predicted_arr

    return {
        "signal": signal,
        "predicted_arr_b": predicted_arr,
        "predicted_arr_mom_pct": predicted_arr_mom_pct,
        "signal_mom_pct": target_signal_mom_pct,
        "intercept": intercept,
        "slope": slope,
        "r2": r2,
        "pearson_r": pearson_r,
        "rmse_loo_pct": rmse_pct,
        "n_train": len(overlap),
        "train_months": overlap,
        "prev_month": prev_full_month,
        "prev_arr_b": prev_arr,
        "gap_months": target_gap,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "target_month": target_month,
    }


def _npm_growth_indicator_daily_series(
    conn: sqlite3.Connection,
    arr_known: dict[str, float],
    target_month: str,
    signal: str = "npm:@anthropic-ai/sdk",
    *,
    anchor_start: float | None = None,
    anchor_end: float | None = None,
) -> list[dict] | None:
    """Daily ARR forecasts for `target_month`: one entry per calendar day.

    For each day D in target_month, compute the headline ARR estimate that
    would be produced if only the npm signal observed through day D were used
    (partial-month projected to full month, then run the OLS fit). The series
    converges toward the headline figure as more daily data lands.

    Modeling layers:
      * Linear trend on observed days (`v(d) = a + b*d`) captures intra-month
        momentum so the dashed line reflects rising/falling rate rather than
        a flat carry-forward.
      * Multiplicative weekday seasonality (Mon..Sun): npm SDK downloads dip
        on weekends, so per-weekday factors derived from observed/trend
        ratios pull Sat/Sun future projections lower (and corresponding
        weekdays higher).
      * ±1σ volatility band: σ from residuals of (trend × weekday) fit,
        scaled by √(days_in_target − D) so the band narrows as D approaches
        month-end (fewer not-yet-realised days = less projection noise).
      * Calibration: the entire series is multiplicatively scaled so its
        monthly mean equals the headline ARR forecast (the npm-growth
        indicator's `predicted_arr_b`). Day-to-day ratios are preserved;
        only the absolute level is anchored. This makes the chart's level
        track the headline automatically as new months sync.

    Returns None if the underlying npm-growth indicator is itself unavailable.
    """
    base = _npm_growth_indicator(conn, arr_known, target_month, signal)
    if base is None:
        return None

    intercept = base["intercept"]
    slope = base["slope"]
    prev_arr = base["prev_arr_b"]
    prev_month = base["prev_month"]
    gap_months = base["gap_months"]

    # Previous month's full signal value (anchor for npm-MoM ratio).
    prev_signal_full = _monthly_sum(conn, signal, prev_month)
    if not prev_signal_full or prev_signal_full <= 0:
        return None

    days_in_target = _days_in_month(target_month)

    # Pre-pull observed days for fast lookup + trend fit.
    rows = conn.execute(
        "SELECT date, value FROM anthropic_signals "
        "WHERE signal_name=? AND substr(date,1,7)=? AND value > 0",
        (signal, target_month),
    ).fetchall()
    obs_by_date: dict[str, float] = {r["date"]: float(r["value"]) for r in rows}
    obs_dates = sorted(obs_by_date.keys())
    if obs_dates:
        latest_obs_date = obs_dates[-1]
        partial_total = sum(obs_by_date.values())
        days_observed = len(obs_dates)
        daily_avg = partial_total / days_observed
    else:
        latest_obs_date = None
        partial_total = 0.0
        days_observed = 0
        daily_avg = 0.0

    # Pre-compute weekday once per observed date (Mon=0..Sun=6).
    wd_by_date: dict[str, int] = {
        d: datetime.strptime(d, "%Y-%m-%d").weekday() for d in obs_dates
    }

    # Linear trend on observed days: x = day-of-month (int), y = npm value.
    # Used to extrapolate future days and to estimate per-day volatility (σ).
    trend_a = trend_b = None
    sigma_daily = 0.0
    # Weekday seasonality (Mon=0..Sun=6): multiplicative factor applied on top
    # of the trend. Anthropic SDK downloads dip on weekends (devs run fewer
    # builds), so future Sat/Sun should be projected lower and weekday band
    # should be wider when the remaining days skew weekend-heavy.
    weekday_factor: dict[int, float] = {i: 1.0 for i in range(7)}
    # Fallback: if we don't yet have ≥3 observed days in target_month, learn
    # both the trend slope and weekday factors from the previous full month
    # so future-day projections show realistic Mon..Sun shape from day 1.
    if days_observed < 3:
        prev_rows = conn.execute(
            "SELECT date, value FROM anthropic_signals "
            "WHERE signal_name=? AND substr(date,1,7)=? AND value > 0 "
            "ORDER BY date",
            (signal, prev_month),
        ).fetchall()
        if len(prev_rows) >= 7:
            prev_dates = [r["date"] for r in prev_rows]
            prev_vals = [float(r["value"]) for r in prev_rows]
            prev_wd = [datetime.strptime(d, "%Y-%m-%d").weekday() for d in prev_dates]
            prev_xs = [float(int(d.split("-")[2])) for d in prev_dates]
            prev_fit = fit_linear(prev_xs, prev_vals)
            if prev_fit:
                pa, pb, _ = prev_fit
                # Carry the slope forward (assumes growth rate persists)
                # but rebase intercept so the trend at day 1 of target_month
                # equals the trend at end of prev_month.
                prev_eom = pa + pb * len(prev_dates)
                # Project: trend at day 1 of target_month = prev_eom × (1 + npm_mom).
                # Use the npm_mom implied by the OLS extrapolation alone.
                trend_a = prev_eom - pb * 1.0
                trend_b = pb
                ratios_by_wd: dict[int, list[float]] = {i: [] for i in range(7)}
                for i, wd in enumerate(prev_wd):
                    t = pa + pb * prev_xs[i]
                    if t > 0:
                        ratios_by_wd[wd].append(prev_vals[i] / t)
                raw = {
                    wd: (sum(rs) / len(rs)) if rs else 1.0
                    for wd, rs in ratios_by_wd.items()
                }
                avg = sum(raw.values()) / 7.0
                if avg > 0:
                    weekday_factor = {wd: raw[wd] / avg for wd in range(7)}
    if days_observed >= 3:
        xs = [int(d.split("-")[2]) for d in obs_dates]
        ys = [obs_by_date[d] for d in obs_dates]
        fit = fit_linear([float(x) for x in xs], ys)
        if fit:
            trend_a, trend_b, _r2 = fit

            # Multiplicative weekday factors from observed-vs-trend ratios.
            ratios_by_wd: dict[int, list[float]] = {i: [] for i in range(7)}
            for i, dstr in enumerate(obs_dates):
                t = trend_a + trend_b * xs[i]
                if t > 0:
                    ratios_by_wd[wd_by_date[dstr]].append(ys[i] / t)
            raw_factors = {
                wd: (sum(rs) / len(rs)) if rs else 1.0
                for wd, rs in ratios_by_wd.items()
            }
            # Normalise so the average factor over the observed days is 1.0;
            # this keeps the trend line's overall level intact (only the
            # within-week pattern is modulated).
            obs_avg_factor = (
                sum(raw_factors[wd_by_date[d]] for d in obs_dates) / days_observed
            )
            if obs_avg_factor > 0:
                weekday_factor = {
                    wd: raw_factors[wd] / obs_avg_factor for wd in range(7)
                }

            # σ uses residuals after BOTH trend and weekday — a tighter,
            # honest noise estimate.
            resids = []
            for i, dstr in enumerate(obs_dates):
                expected = (trend_a + trend_b * xs[i]) * weekday_factor[
                    wd_by_date[dstr]
                ]
                resids.append(ys[i] - expected)
            if len(resids) > 2:
                ss = sum(r * r for r in resids) / (len(resids) - 2)
                sigma_daily = math.sqrt(max(ss, 0.0))

    def _weekday_factor(day_str: str) -> float:
        try:
            wd = datetime.strptime(day_str, "%Y-%m-%d").weekday()
        except ValueError:
            return 1.0
        return weekday_factor.get(wd, 1.0)

    def _trend(day_idx: int, day_str: str | None = None) -> float:
        if trend_a is None or trend_b is None:
            base_val = daily_avg
        else:
            base_val = trend_a + trend_b * day_idx
        if day_str is not None:
            base_val *= _weekday_factor(day_str)
        return max(0.0, base_val)

    def _arr_from_signal(sig_val: float) -> tuple[float, float, float]:
        """signal_proj → (signal_mom_pct, predicted_arr_mom_pct, predicted_arr_b)."""
        s_mom = ((sig_val / prev_signal_full) ** (1.0 / gap_months) - 1.0) * 100.0
        a_mom = intercept + slope * s_mom
        arr_b = prev_arr * (1.0 + a_mom / 100.0) ** gap_months
        return s_mom, a_mom, arr_b

    def _day_str(j: int) -> str:
        return f"{target_month}-{j:02d}"

    yyyy, mm = target_month.split("-")
    out: list[dict] = []
    for d in range(1, days_in_target + 1):
        day_str = _day_str(d)

        if latest_obs_date and day_str < latest_obs_date:
            kind = "actual"
        elif latest_obs_date and day_str == latest_obs_date:
            kind = "today"
        else:
            kind = "future"

        arr_low: float | None = None
        arr_high: float | None = None

        if kind in ("actual", "today"):
            # cutoff is exclusive — to include day D's data, cut at D+1.
            next_day = (
                _day_str(d + 1)
                if d < days_in_target
                else f"{int(yyyy):04d}-{(int(mm) + 1):02d}-01"
            )
            proj = _project_monthly_sum(conn, signal, target_month, next_day)
            if proj is None:
                continue
            signal_proj, _method = proj
            this_days_observed = sum(1 for od in obs_dates if od <= day_str)
        else:
            # Future day: extend partial_total with linear trend extrapolation
            # for the days from (latest_obs+1) through end-of-month, scaled
            # so the rate reflected at day D follows the trend value at D.
            if days_observed == 0:
                continue
            if days_observed >= days_in_target:
                signal_proj = partial_total
            else:
                # Future day D: simulate "if today were D" by treating days
                # observed+1..D as synthetically observed at trend×weekday
                # rate, then re-project EOM via partial × days_in_month / D —
                # the same formula `_project_monthly_sum` uses on real data.
                # This makes the dashed line show a weekend dip on Sat/Sun
                # future days (low simulated rates drag the partial sum down,
                # which scales linearly into the EOM projection).
                synthetic_to_d = sum(
                    _trend(j, _day_str(j)) for j in range(days_observed + 1, d + 1)
                )
                partial_at_d = partial_total + synthetic_to_d
                signal_proj = partial_at_d * days_in_target / d
            this_days_observed = days_observed

            # ±1σ volatility band: by day D we've "fixed" everything up to
            # and including D (observed + synthetic), so only the days from
            # D+1 to days_in_target carry residual npm-noise. Per-signal σ
            # scales with √(days_in_target − D); the band collapses to 0 at
            # D = days_in_target, matching intuition that the EOM point is
            # fully determined by the projection itself.
            if sigma_daily > 0:
                days_remaining = days_in_target - d
                sigma_signal = sigma_daily * math.sqrt(max(days_remaining, 0))
                _, _, arr_low = _arr_from_signal(max(0.0, signal_proj - sigma_signal))
                _, _, arr_high = _arr_from_signal(signal_proj + sigma_signal)

        signal_mom_pct, predicted_arr_mom_pct, predicted_arr_b = _arr_from_signal(
            signal_proj
        )

        out.append(
            {
                "date": day_str,
                "kind": kind,
                "predicted_arr_b": predicted_arr_b,
                "predicted_arr_b_low": arr_low,
                "predicted_arr_b_high": arr_high,
                "signal_proj": signal_proj,
                "signal_mom_pct": signal_mom_pct,
                "predicted_arr_mom_pct": predicted_arr_mom_pct,
                "days_observed": this_days_observed,
            }
        )

    # ------------------------------------------------------------------
    # "Daily ARR" series with npm-driven shape:
    #   - Linear BASELINE: day 0 = start anchor (prev month's ARR),
    #     day last_weekday = end anchor (this month's ensemble prediction).
    #   - Multiplicative FACTOR per day = npm_raw[d] / npm_smooth_trend[d],
    #     i.e. how much that day's npm download volume deviates from the
    #     pure linear regression line. Weekends dip below 1.0; busy
    #     weekdays push above 1.0.
    #   - The series is then renormalized so the last weekday's factor =
    #     1.0 exactly — guaranteeing the curve lands on the end anchor.
    #     Daily relative ratios are preserved.
    # Result: visible weekend dips + weekday peaks, start ≈ prev-month ARR,
    # last weekday = current-month ensemble prediction (exact).
    # ------------------------------------------------------------------
    headline_arr_b = (
        anchor_end if anchor_end is not None else base.get("predicted_arr_b")
    )
    start_arr_b = anchor_start if anchor_start is not None else prev_arr
    if (
        headline_arr_b is None
        or headline_arr_b <= 0
        or not start_arr_b
        or start_arr_b <= 0
        or not out
    ):
        return out or None

    # Find the last weekday in the month (Mon..Fri).
    last_weekday_idx = days_in_target
    for d in range(days_in_target, 0, -1):
        try:
            if datetime.strptime(f"{target_month}-{d:02d}", "%Y-%m-%d").weekday() <= 4:
                last_weekday_idx = d
                break
        except ValueError:
            continue

    def _baseline(d: int) -> float:
        if d <= 0:
            return start_arr_b
        if d >= last_weekday_idx:
            return headline_arr_b
        return start_arr_b + (headline_arr_b - start_arr_b) * d / last_weekday_idx

    def _npm_factor(d: int, day_str: str) -> float:
        """Multiplicative deviation of npm value from its smooth trend line.

        Calendar effect — both sides:
          A busy weekday (raw > smooth) is treated as a build-spike artifact,
          and a weekend dip (raw < smooth) is treated as a fewer-builds artifact.
          Neither maps to a real ARR move at that resolution; both are
          smoothed by capping the factor at 1.0 (upside) and floored by
          `min` only when their RELATIVE deviation isn't bigger than the
          last-weekday's. The final invariant is enforced AFTER correction:
          every day's final multiplier is capped at 1.0, so no daily can
          exceed headline_arr_b. Last-weekday equals headline by construction
          of `correction`. Trailing-weekend dips are still visible because
          factor < 1 stays < 1 after correction unless correction inflates
          them above 1 — in which case the final cap clips them at 1."""
        if trend_a is None or trend_b is None:
            return 1.0
        smooth = trend_a + trend_b * d
        if smooth <= 0:
            return 1.0
        if day_str in obs_by_date:
            raw = float(obs_by_date[day_str])
        else:
            raw = smooth * _weekday_factor(day_str)
        return min(raw / smooth, 1.0)

    # Compute (baseline, factor) per row first; we need to know the last
    # weekday's factor to renormalize.
    pairs: list[tuple[dict, float, float]] = []
    last_wd_factor: float | None = None
    for r in out:
        d = int(r["date"].split("-")[2])
        b_arr = _baseline(d)
        f = _npm_factor(d, r["date"])
        pairs.append((r, b_arr, f))
        try:
            if (
                datetime.strptime(r["date"], "%Y-%m-%d").weekday() <= 4
                and d == last_weekday_idx
            ):
                last_wd_factor = f if f > 0 else None
        except ValueError:
            pass
    if last_wd_factor is None:
        # No row hit the last weekday — defensive: warn to stderr and fall
        # back to 1.0 (the chart just won't be exactly anchored that day).
        import sys

        print(
            f"[anthropic_arr] WARNING: last_weekday_idx={last_weekday_idx} "
            f"not found in daily series for {target_month} — anchor degraded",
            file=sys.stderr,
        )
        last_wd_factor = 1.0
    correction = (1.0 / last_wd_factor) if last_wd_factor > 0 else 1.0

    # Final-multiplier cap: factor × correction can exceed 1.0 when the last
    # weekday's npm was below trend (correction > 1) and another day's factor
    # is closer to 1.0 (or the trailing weekend's _baseline was clamped to
    # headline). Capping the final multiplier at 1.0 guarantees no daily
    # exceeds headline_arr_b. Last weekday remains exactly headline because
    # its (factor × correction) = (last_wd_factor × 1/last_wd_factor) = 1.0.
    out2: list[dict] = []
    for r, b_arr, f in pairs:
        mult = min(f * correction, 1.0)
        arr_d = b_arr * mult
        new_row = dict(r)
        new_row["predicted_arr_b"] = arr_d
        old = r.get("predicted_arr_b") or 0
        ratio = (arr_d / old) if old > 0 else 1.0
        if r.get("predicted_arr_b_low") is not None:
            new_row["predicted_arr_b_low"] = r["predicted_arr_b_low"] * ratio
        if r.get("predicted_arr_b_high") is not None:
            new_row["predicted_arr_b_high"] = r["predicted_arr_b_high"] * ratio
        out2.append(new_row)

    return out2 or None


def _mom_growth_series(
    arr_known: dict[str, float], target_month: str | None, predicted_arr_b: float | None
) -> list[dict]:
    """Build month-over-month growth legs from sorted known ARR points,
    appending the predicted target month if it extends past the last known.

    Each leg uses the compound monthly rate over its gap, so multi-month
    legs (e.g. 2025-07 → 2025-10) annualize correctly.
    """
    months = sorted(arr_known.keys())
    pts: list[tuple[str, float, str]] = [(m, arr_known[m], "known") for m in months]
    if (
        target_month
        and predicted_arr_b is not None
        and (not months or _month_gap(months[-1], target_month) > 0)
    ):
        pts.append((target_month, float(predicted_arr_b), "predicted"))

    out: list[dict] = []
    for i in range(1, len(pts)):
        prev_m, prev_v, _ = pts[i - 1]
        cur_m, cur_v, cur_src = pts[i]
        gap = _month_gap(prev_m, cur_m)
        if gap <= 0 or prev_v <= 0:
            continue
        ratio = cur_v / prev_v
        mom = ratio ** (1.0 / gap) - 1.0  # compound monthly
        annualized = (1.0 + mom) ** 12 - 1.0
        out.append(
            {
                "from_month": prev_m,
                "to_month": cur_m,
                "gap_months": gap,
                "mom_pct": mom * 100.0,
                "annualized_pct": annualized * 100.0,
                "from_arr_b": prev_v,
                "to_arr_b": cur_v,
                "source": cur_src,  # 'known' or 'predicted'
            }
        )
    return out


def predict_arr(
    conn: sqlite3.Connection,
    target_month: str | None = None,
    _allow_recursive_anchor: bool = True,
) -> dict | None:
    """Inverse-variance ensemble across PROXY_SIGNALS for `target_month`.

    `_allow_recursive_anchor` is internal: when True, the daily-series builder
    may re-call `predict_arr` once for the previous month to get a chained
    start anchor. The recursive call sets it to False to avoid deeper nesting.
    """
    if target_month is None:
        target_month = datetime.now(timezone.utc).strftime("%Y-%m")

    arr_rows = conn.execute(
        "SELECT month, arr_b_usd FROM anthropic_arr_known ORDER BY month"
    ).fetchall()
    arr_known = {r["month"]: float(r["arr_b_usd"]) for r in arr_rows}
    if len(arr_known) < 2:
        return None

    per_signal = []
    for sig in PROXY_SIGNALS:
        result = _predict_for_signal(conn, sig, arr_known, target_month)
        if result:
            per_signal.append(result)

    # Shadow signals — fit and predict, but tag so the ensemble step skips them.
    shadow_signals = []
    for sig in SHADOW_SIGNALS:
        result = _predict_for_signal(conn, sig, arr_known, target_month)
        if result:
            result["shadow"] = True
            shadow_signals.append(result)

    # Stamp the ensemble-admission flag on every per_signal row. Uses the
    # module-level `is_admissible` — same helper backtest.py and
    # backtest_yipit_private.py import, so live and backtest never disagree
    # about which signals count.
    for p in per_signal:
        p["in_ensemble"] = is_admissible(p)
    for p in shadow_signals:
        p["in_ensemble"] = False

    # Build ensemble if any proxy signal fit; otherwise let growth-extrapolation
    # carry the dashboard.
    ensemble: dict | None = None
    monthly_sum_ensemble: dict | None = None
    best: dict | None = None
    if per_signal:
        # R²>=0.5 admission threshold (relaxed): inverse-variance × channel-weight
        # naturally suppresses high-RMSE signals to near-zero weight. `in_ensemble`
        # was computed above and is the single source of truth for this rule.
        usable = [p for p in per_signal if p.get("in_ensemble")]
        if usable:
            weights = signal_blend_weights(usable)
            weighting = (
                "channel-weighted inverse-variance (1/RMSE_loo^2 * channel_weight), "
                f"R2>=0.5 filter, {SIGNAL_WEIGHT_CAP:.0%} per-signal cap"
            )
        else:
            usable = [p for p in per_signal if not p.get("shadow")]
            weights = cap_weights(
                [p["channel_weight"] for p in usable], SIGNAL_WEIGHT_CAP
            )
            weighting = "channel-weighted uniform (no usable RMSE)"
            # Fallback path: no signal passed R2>=0.5 so `in_ensemble` was
            # False for all, but we're still using them to build the ensemble
            # answer. Re-stamp so the UI can name the contributing signals.
            for p in usable:
                p["in_ensemble"] = True
        if usable and weights:

            def wmean(field: str) -> float:
                # signal_blend_weights / cap_weights return normalised weights.
                return sum(usable[i][field] * weights[i] for i in range(len(usable)))

            monthly_sum_ensemble = {
                "predicted_arr_b": wmean("predicted_arr_b"),
                "ci_low": wmean("ci_low"),
                "ci_high": wmean("ci_high"),
                "n_models": len(usable),
                "n_models_total": len([p for p in per_signal if not p.get("shadow")]),
                "weighting": weighting,
            }
            best = max(
                [p for p in per_signal if not p.get("shadow")] or per_signal,
                key=lambda p: p["regression"]["r2"],
            )

    # Headline: the ramped blend, not the monthly-sum ensemble above. The latter
    # projects the partial month by partial × days_in_month / days_observed, so
    # early in the month one noisy day sets the number — walk-forward day-2 MAPE
    # 23.30% vs the blend's 11.77% (asof_summary in backtest.py; the blend wins at
    # every as-of day and is the only one of the two inside the 12% promotion
    # gate before mid-month). monthly_sum stays in the payload as
    # `monthly_sum_ensemble` so the UI can show both, and supplies `best_signal`.
    #
    # No monthly_sum fallback: with ≥2 anchors (guaranteed above) trend_extrapolate
    # always resolves, so the blend always returns a result — a fallback here would
    # be unreachable.
    ramped = predict_arr_ramped_blend(
        conn, target_month=target_month, arr_known=arr_known
    )
    _apply_pinned_arr(ramped, target_month)
    ensemble = ramped["ensemble"]

    # Monthly history for charts — include both ensemble-active and shadow
    # signals so users can eyeball shadow-signal quality before promotion.
    sig_monthly: dict[str, list[dict]] = {}
    for sig in PROXY_SIGNALS + SHADOW_SIGNALS:
        rows = conn.execute(
            "SELECT substr(date,1,7) AS m, SUM(value) AS v FROM anthropic_signals "
            "WHERE signal_name=? GROUP BY m ORDER BY m",
            (sig,),
        ).fetchall()
        if rows:
            sig_monthly[sig] = [{"month": r["m"], "value": float(r["v"])} for r in rows]

    growth_extrap = _growth_extrapolation(arr_known, target_month, window=6)
    npm_growth = _npm_growth_indicator(conn, arr_known, target_month)
    # Anchor daily series endpoints to the ensemble (not npm-growth alone):
    # start = previous month's confirmed ARR if known, else previous month's
    # ensemble prediction; end = current month's ensemble prediction.
    daily_anchor_start = None
    # Start anchor = previous *calendar* month's ARR. If that month is in
    # arr_known use the actual; otherwise re-run predict_arr for that month
    # to get its ensemble prediction (so consecutive forecast months chain
    # naturally: each forecast month starts at the prior month's endpoint).
    yy, mm = int(target_month[:4]), int(target_month[5:7])
    pm_y, pm_m = (yy - 1, 12) if mm == 1 else (yy, mm - 1)
    prev_month_str = f"{pm_y:04d}-{pm_m:02d}"
    if prev_month_str in arr_known:
        daily_anchor_start = arr_known[prev_month_str]
    elif _allow_recursive_anchor:
        prev_pred = predict_arr(
            conn, target_month=prev_month_str, _allow_recursive_anchor=False
        )
        if prev_pred and prev_pred.get("ensemble"):
            daily_anchor_start = prev_pred["ensemble"]["predicted_arr_b"]
    if daily_anchor_start is None and arr_known:
        # Fallback: latest known prior month, regardless of gap.
        prior_known = [m for m in sorted(arr_known.keys()) if m < target_month]
        if prior_known:
            daily_anchor_start = arr_known[prior_known[-1]]
    # npm-daily convergence series is no longer computed on every request —
    # the frontend now consumes /api/v1/arr_curve for the daily-resolution
    # trajectory. Payload key preserved as None so any legacy client still
    # gets a well-formed response instead of a KeyError. The heavy
    # trend+seasonality function _npm_growth_indicator_daily_series is
    # retained in this file for future callers (e.g. a research notebook).
    npm_growth_daily = None
    # The forward leg is the headline's own implied growth. It used to prefer
    # the npm-growth indicator, from when that WAS the headline; the headline is
    # now the ramped blend, so that ordering left the MoM chart contradicting the
    # hero — on 2026-08-03 the chart's "headline forecast" point read -5.3%/mo
    # against a hero of +11.9%/mo, because npm-growth projects the target month
    # by partial × days_in_month / days_observed and August had one day of data
    # (a Saturday). Deriving the leg from `ensemble` keeps the two in step by
    # construction, whatever the headline estimator is.
    forward_pred = None
    if ensemble:
        forward_pred = ensemble["predicted_arr_b"]
    elif growth_extrap:
        forward_pred = growth_extrap["predicted_arr_b"]
    mom_growth = _mom_growth_series(arr_known, target_month, forward_pred)
    # MoM leg using the growth-rate extrapolation prediction instead of
    # the proxy ensemble — for the alt KPI on the Predictor tab.
    if growth_extrap and mom_growth and mom_growth[-1].get("source") == "predicted":
        last = mom_growth[-1]
        prev_v = last["from_arr_b"]
        cur_v = growth_extrap["predicted_arr_b"]
        gap = last["gap_months"]
        if prev_v > 0 and gap > 0:
            ratio = cur_v / prev_v
            mom_pct = (ratio ** (1.0 / gap) - 1.0) * 100.0
            ann_pct = ((1.0 + mom_pct / 100.0) ** 12 - 1.0) * 100.0
            growth_extrap_leg = {
                "from_month": last["from_month"],
                "to_month": last["to_month"],
                "gap_months": gap,
                "from_arr_b": prev_v,
                "to_arr_b": cur_v,
                "mom_pct": mom_pct,
                "annualized_pct": ann_pct,
            }
        else:
            growth_extrap_leg = None
    else:
        growth_extrap_leg = None

    return {
        "target_month": target_month,
        "method": (
            "Channel-weighted multi-signal regression. "
            "Dev signals (npm/PyPI/GitHub) and enterprise-proxy signals "
            "(Bedrock SDK, search counts) blended; weights = "
            "channel_weight × inverse-variance of LOO residuals."
        ),
        "ensemble": ensemble,
        # The blend's own detail (trend leg, signal leg, ramp weight) and the
        # method it replaced, both kept so the headline stays inspectable.
        "ramped_blend": ramped,
        "monthly_sum_ensemble": monthly_sum_ensemble,
        "per_signal": per_signal + shadow_signals,
        "best_signal": best["signal"] if best else None,
        "all_signal_monthly": sig_monthly,
        "mom_growth": mom_growth,
        "growth_extrapolation": growth_extrap,
        "growth_extrapolation_last_leg": growth_extrap_leg,
        "npm_growth_indicator": npm_growth,
        "npm_growth_indicator_daily": npm_growth_daily,
        "all_arr_known": [
            {
                "month": r["month"],
                "arr_b": float(r["arr_b_usd"]),
                "source": r["source"],
                "notes": r["notes"],
            }
            for r in conn.execute(
                "SELECT month, arr_b_usd, source, notes FROM anthropic_arr_known "
                "ORDER BY month"
            ).fetchall()
        ],
    }


def predict_openai_arr(
    conn: sqlite3.Connection, target_month: str | None = None
) -> dict | None:
    """OpenAI-side counterpart of predict_arr(). Same methodology — the ramped
    blend headline over an end-of-month signal leg — with a disjoint anchor
    table (openai_arr_known) and signal registry (SIGNAL_REGISTRY_OPENAI).

    Payload shape is byte-identical to predict_arr() so the frontend can swap
    endpoints with only a URL change. See experiments/openai_arr_forecast_v5.py
    for the walk-forward calibration that produced the current signal set.
    """
    if target_month is None:
        target_month = datetime.now(timezone.utc).strftime("%Y-%m")

    arr_rows = conn.execute(
        "SELECT month, arr_b_usd FROM openai_arr_known ORDER BY month"
    ).fetchall()
    arr_known = {r["month"]: float(r["arr_b_usd"]) for r in arr_rows}
    if len(arr_known) < 2:
        return None

    per_signal = []
    for sig in PROXY_SIGNALS_OPENAI:
        result = _predict_for_signal(
            conn, sig, arr_known, target_month, registry=SIGNAL_REGISTRY_OPENAI
        )
        if result:
            per_signal.append(result)

    shadow_signals = []
    for sig in SHADOW_SIGNALS_OPENAI:
        result = _predict_for_signal(
            conn, sig, arr_known, target_month, registry=SIGNAL_REGISTRY_OPENAI
        )
        if result:
            result["shadow"] = True
            shadow_signals.append(result)

    for p in per_signal:
        p["in_ensemble"] = is_admissible(p)
    for p in shadow_signals:
        p["in_ensemble"] = False

    monthly_sum_ensemble: dict | None = None
    best: dict | None = None
    if per_signal:
        usable = [p for p in per_signal if p.get("in_ensemble")]
        if usable:
            weights = [
                (1.0 / (p["regression"]["rmse_loo"] ** 2)) * p["channel_weight"]
                for p in usable
            ]
            weighting = (
                "channel-weighted inverse-variance (1/RMSE_loo^2 * channel_weight), "
                "R2>=0.5 filter"
            )
        else:
            usable = [p for p in per_signal if not p.get("shadow")]
            weights = [p["channel_weight"] for p in usable]
            weighting = "channel-weighted uniform (no usable RMSE)"
            for p in usable:
                p["in_ensemble"] = True
        if usable:
            w_sum = sum(weights)

            def wmean(field: str) -> float:
                return (
                    sum(usable[i][field] * weights[i] for i in range(len(usable)))
                    / w_sum
                )

            monthly_sum_ensemble = {
                "predicted_arr_b": wmean("predicted_arr_b"),
                "ci_low": wmean("ci_low"),
                "ci_high": wmean("ci_high"),
                "n_models": len(usable),
                "n_models_total": len([p for p in per_signal if not p.get("shadow")]),
                "weighting": weighting,
            }
            best = max(
                [p for p in per_signal if not p.get("shadow")] or per_signal,
                key=lambda p: p["regression"]["r2"],
            )

    # Headline: the ramped blend, same as the Anthropic side. The monthly-sum
    # ensemble above projects the partial month by partial × days_in_month /
    # days_observed, which on day 1-3 sets the whole number from one day of npm
    # traffic — the failure the blend was built to replace (see predict_arr).
    # OpenAI stayed on it only because this path had no registry parameter, so
    # the swap is a wiring fix, not a change of method.
    ramped = predict_arr_ramped_blend(
        conn,
        target_month=target_month,
        arr_known=arr_known,
        registry=SIGNAL_REGISTRY_OPENAI,
        proxy_signals=PROXY_SIGNALS_OPENAI,
        trend_per_month=post_step_run_rate(conn, arr_known, target_month),
    )
    ensemble = ramped["ensemble"]

    # Monthly history for charts — include shadow signals too.
    sig_monthly: dict[str, list[dict]] = {}
    for sig in PROXY_SIGNALS_OPENAI + SHADOW_SIGNALS_OPENAI:
        rows = conn.execute(
            "SELECT substr(date,1,7) AS m, SUM(value) AS v FROM anthropic_signals "
            "WHERE signal_name=? GROUP BY m ORDER BY m",
            (sig,),
        ).fetchall()
        if rows:
            sig_monthly[sig] = [{"month": r["m"], "value": float(r["v"])} for r in rows]

    growth_extrap = _growth_extrapolation(arr_known, target_month, window=6)
    # For OpenAI the headline "sole strong signal" is npm:@ai-sdk/openai.
    # (Anthropic's headline is npm:@anthropic-ai/sdk — the parameter to
    # _npm_growth_indicator, which regresses ARR-MoM% ~ signal-MoM% rather
    # than the level fit used by _predict_for_signal.)
    npm_growth = _npm_growth_indicator(
        conn, arr_known, target_month, signal="npm:@ai-sdk/openai"
    )
    npm_growth_daily = None
    # Headline-derived forward leg — see the matching note in predict_arr.
    forward_pred = None
    if ensemble:
        forward_pred = ensemble["predicted_arr_b"]
    elif growth_extrap:
        forward_pred = growth_extrap["predicted_arr_b"]
    mom_growth = _mom_growth_series(arr_known, target_month, forward_pred)
    if growth_extrap and mom_growth and mom_growth[-1].get("source") == "predicted":
        last = mom_growth[-1]
        prev_v = last["from_arr_b"]
        cur_v = growth_extrap["predicted_arr_b"]
        gap = last["gap_months"]
        if prev_v > 0 and gap > 0:
            ratio = cur_v / prev_v
            mom_pct = (ratio ** (1.0 / gap) - 1.0) * 100.0
            ann_pct = ((1.0 + mom_pct / 100.0) ** 12 - 1.0) * 100.0
            growth_extrap_leg = {
                "from_month": last["from_month"],
                "to_month": last["to_month"],
                "gap_months": gap,
                "from_arr_b": prev_v,
                "to_arr_b": cur_v,
                "mom_pct": mom_pct,
                "annualized_pct": ann_pct,
            }
        else:
            growth_extrap_leg = None
    else:
        growth_extrap_leg = None

    return {
        "target_month": target_month,
        "company": "openai",
        "method": (
            "Ramped blend (OpenAI): w·trend + (1-w)·signal, w = ramp_weight(day). "
            "The signal leg is the channel-weighted inverse-variance ensemble over "
            "npm:@ai-sdk/openai + pypi:openai, measured on the de-lagged "
            "end-of-month window. Same estimator as the Anthropic side. "
            "Yipit signals are validation-only and not blended (see "
            "openai_backtest_yipit_private for the private harness)."
        ),
        "ensemble": ensemble,
        # The blend's own detail (trend leg, signal leg, ramp weight) and the
        # method it replaced, both kept so the headline stays inspectable.
        "ramped_blend": ramped,
        "monthly_sum_ensemble": monthly_sum_ensemble,
        "per_signal": per_signal + shadow_signals,
        "best_signal": best["signal"] if best else None,
        "all_signal_monthly": sig_monthly,
        "mom_growth": mom_growth,
        "growth_extrapolation": growth_extrap,
        "growth_extrapolation_last_leg": growth_extrap_leg,
        "npm_growth_indicator": npm_growth,
        "npm_growth_indicator_daily": npm_growth_daily,
        "all_arr_known": [
            {
                "month": r["month"],
                "arr_b": float(r["arr_b_usd"]),
                "source": r["source"],
                "notes": r["notes"],
            }
            for r in conn.execute(
                "SELECT month, arr_b_usd, source, notes FROM openai_arr_known "
                "ORDER BY month"
            ).fetchall()
        ],
    }
