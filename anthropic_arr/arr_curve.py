"""ARR curve preparation — tracker-style visuals on ensemble + OpenAI checkpoints.

Ports `build_bundle.py:prep_arr` (tracker source lines 214-278). Produces the
render-ready payload consumed by tracker.js:initARR:

    {
      "updated": "YYYY-MM-DD",
      "companies": {
        "anthropic": {
          "label": "Anthropic", "color": "#d97757",
          "hist":  [[ms_epoch, arr_b], ...],   # solid line before last anchor
          "ext":   [[ms_epoch, arr_b], ...],   # dashed line after last anchor
          "fanLo": [[ms_epoch, arr_b], ...],   # lower uncertainty band
          "fanHi": [[ms_epoch, arr_b], ...],   # upper uncertainty band
          "counter": {"tLast": ms, "vLast": B, "rMs": rate_per_ms},
          "cps":     [{"t": ms, "v": B, "src": "..."}, ...],   # ◆ scatters
          "yoyDen":  B          # ARR value one year ago (for YoY pill)
        },
        "openai": { ... same shape ... }
      }
    }

Anthropic side reads ensemble forecasts + KNOWN_ARR (ground-truth) as anchors
and uses the ensemble 95% CI at the extrapolation horizon as low/high.
OpenAI side reads manifest-managed checkpoints and derives its forward path
from the same proxy-signal estimator exposed by the predictor endpoint.
"""

from __future__ import annotations

import math
import sqlite3
import threading
from datetime import date, datetime, timedelta, timezone
from typing import Any

from .compute import (
    model_signal_value,
)
from .compute import (
    predict_arr as _predict_arr_uncached,
)
from .compute import (
    predict_openai_arr as _predict_openai_uncached,
)
from .openai_anchors import OPENAI_ANCHOR_RECORDS, OPENAI_CHECKPOINTS

# Cache predict_arr() calls within a short window so /arr_curve and
# /arr_nowcast on the same page load don't run the ensemble twice. Keyed by
# (target_month, latest sync_log timestamp) so the cache invalidates any
# time new signal data lands.
_PREDICT_CACHE: dict[tuple, tuple[float, dict | None]] = {}
_PREDICT_CACHE_LOCK = threading.Lock()
_PREDICT_CACHE_TTL_SEC = 15.0


def _cached_predict_arr(conn, target_month=None):
    """Thin cache in front of compute.predict_arr. Reads MAX(sync_at) so the
    cache expires whenever sync writes new signal rows."""
    import time as _time

    row = conn.execute("SELECT MAX(sync_at) AS s FROM sync_log").fetchone()
    sync_key = row["s"] if row else None
    key = (target_month, sync_key)
    now = _time.time()
    with _PREDICT_CACHE_LOCK:
        entry = _PREDICT_CACHE.get(key)
        if entry and (now - entry[0]) < _PREDICT_CACHE_TTL_SEC:
            return entry[1]
    result = _predict_arr_uncached(conn, target_month=target_month)
    with _PREDICT_CACHE_LOCK:
        _PREDICT_CACHE[key] = (now, result)
        # Bound memory — evict oldest if the map grows beyond ~50 entries.
        if len(_PREDICT_CACHE) > 50:
            oldest = min(_PREDICT_CACHE.items(), key=lambda kv: kv[1][0])
            _PREDICT_CACHE.pop(oldest[0], None)
    return result


# Everything else in this module calls predict_arr as if imported from compute;
# alias for backward-compat within this file only.
predict_arr = _cached_predict_arr  # noqa: F811


# Parallel cache for the OpenAI predictor. Separate map (distinct call
# signature + underlying data), same TTL + invalidation semantics.
_PREDICT_CACHE_OPENAI: dict[tuple, tuple[float, dict | None]] = {}


def _cached_predict_openai(conn, target_month=None):
    import time as _time

    row = conn.execute("SELECT MAX(sync_at) AS s FROM sync_log").fetchone()
    sync_key = row["s"] if row else None
    key = (target_month, sync_key)
    now = _time.time()
    with _PREDICT_CACHE_LOCK:
        entry = _PREDICT_CACHE_OPENAI.get(key)
        if entry and (now - entry[0]) < _PREDICT_CACHE_TTL_SEC:
            return entry[1]
    result = _predict_openai_uncached(conn, target_month=target_month)
    with _PREDICT_CACHE_LOCK:
        _PREDICT_CACHE_OPENAI[key] = (now, result)
        if len(_PREDICT_CACHE_OPENAI) > 50:
            oldest = min(_PREDICT_CACHE_OPENAI.items(), key=lambda kv: kv[1][0])
            _PREDICT_CACHE_OPENAI.pop(oldest[0], None)
    return result


DAY = 86_400_000  # ms
MONTH_MS = DAY * 30.4375


def _ms(iso_date: str) -> int:
    """Return epoch millis at UTC midnight for a YYYY-MM-DD date."""
    d = datetime.strptime(iso_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(d.timestamp() * 1000)


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _pchip_slopes(xs: list[float], ys: list[float]) -> list[float]:
    """Fritsch-Carlson slopes for monotone cubic (PCHIP) interpolation.

    Returns dy/dx at each knot chosen so the resulting Hermite cubic passes
    through every knot without overshooting between them.
    """
    n = len(xs)
    h = [xs[i + 1] - xs[i] for i in range(n - 1)]
    delta = [(ys[i + 1] - ys[i]) / h[i] for i in range(n - 1)]
    m = [0.0] * n
    m[0], m[-1] = delta[0], delta[-1]
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] <= 0:
            m[i] = 0.0  # local extremum — a flat knot keeps it monotone
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
    return m


def _pchip_eval(xs: list[float], ys: list[float], ms: list[float], x: float) -> float:
    """Hermite cubic evaluated on the segment containing x."""
    for i in range(len(xs) - 1):
        if x <= xs[i + 1]:
            h = xs[i + 1] - xs[i]
            s = (x - xs[i]) / h
            s2, s3 = s * s, s * s * s
            return (
                (2 * s3 - 3 * s2 + 1) * ys[i]
                + (s3 - 2 * s2 + s) * h * ms[i]
                + (-2 * s3 + 3 * s2) * ys[i + 1]
                + (s3 - s2) * h * ms[i + 1]
            )
    return ys[-1]


def _month_end(month: str) -> str:
    """month YYYY-MM → last weekday of month YYYY-MM-DD.

    Anchored ARR figures published against a month (e.g. "$65B in June") are
    typically snapshot on the month's last business day, not on Saturday/
    Sunday when finance systems don't book. We roll back to the nearest
    Mon-Fri so anchor dates line up with the actual reporting cadence.
    """
    y, m = int(month[:4]), int(month[5:])
    if m == 12:
        next_month = date(y + 1, 1, 1)
    else:
        next_month = date(y, m + 1, 1)
    last = next_month - timedelta(days=1)
    # weekday(): Mon=0 … Sun=6 — pull back to Friday if we land on weekend.
    while last.weekday() >= 5:
        last -= timedelta(days=1)
    return last.isoformat()


def _prep_series(
    anchors: list[tuple[int, float]],
    pj_to_ms: int,
    pj_low: float,
    pj_high: float,
    now_ms: int,
    checkpoints: list[dict],
    shape_fn=None,
    ext_decay_tau_months: float | None = None,
) -> dict[str, Any]:
    """Given monthly anchor points, extrapolation horizon, and low/high band,
    produce daily-resolution hist/ext/fanLo/fanHi/counter/yoyDen.

    Sampling is 1 point per day (v2). Callers that want a "textured" current
    month can pass `shape_fn(t_ms) -> float` returning a multiplicative factor
    (clamped ±3% by us) that gets applied to the base log-linear interp on
    hist points; ext/fan are shape-free (future is smooth).

    Domain guards: log-linear math requires all anchors > 0 and pj_low/pj_high
    > 0. Bad seed rows or admin edits with zero/negative values would raise
    math domain errors — we return {} so the caller can skip that company
    rather than 500 the whole /arr_curve endpoint.
    """
    # Drop non-positive anchors; if fewer than 2 usable anchors remain, bail.
    anchors = sorted((t, v) for t, v in anchors if v is not None and v > 0)
    if len(anchors) < 2 or pj_low <= 0 or pj_high <= 0:
        return {}
    t_last, v_last = anchors[-1]
    # Extrapolation horizon must be strictly after the last anchor. If a user
    # hand-inserts a future-dated anchor past year-end, refuse to produce a
    # reversed ext series (which would then sign-flip nowcast slope math).
    if pj_to_ms <= t_last:
        return {}
    center = math.sqrt(pj_low * pj_high)
    denom = pj_to_ms - t_last
    r_c = math.log(center / v_last) / denom
    r_l = math.log(pj_low / v_last) / denom
    r_h = math.log(pj_high / v_last) / denom

    # How the horizon's total log-growth is distributed across the window.
    # Linear (the default) compounds at one constant rate the whole way, which
    # never plateaus. With `ext_decay_tau_months` the same total is laid onto a
    # damped path — instantaneous rate ∝ e^(-s/τ) — so growth is front-loaded
    # at the seam and tapers with horizon. The horizon endpoint is identical
    # under both, so pj_low/pj_high stay the authoritative band.
    tau = ext_decay_tau_months
    span_months = denom / MONTH_MS
    if tau and tau > 0 and span_months > 0:
        _norm = 1.0 - math.exp(-span_months / tau)

        def ext_shape(dt_ms: float) -> float:
            return (1.0 - math.exp(-(dt_ms / MONTH_MS) / tau)) / _norm
    else:

        def ext_shape(dt_ms: float) -> float:
            return dt_ms / denom

    def interp(t: int) -> float:
        if t <= anchors[0][0]:
            return anchors[0][1]
        for i in range(1, len(anchors)):
            if t <= anchors[i][0]:
                (ta, va), (tb, vb) = anchors[i - 1], anchors[i]
                # log-linear interpolation; anchors filtered > 0 above.
                return va * (vb / va) ** ((t - ta) / (tb - ta))
        return v_last

    def est(t: int, r: float = r_c) -> float:
        if t <= t_last:
            return interp(t)
        return v_last * math.exp(r * denom * ext_shape(t - t_last))

    def with_shape(t: int, v: float) -> float:
        """Apply npm-daily shape overlay to current-month segment only."""
        if shape_fn is None:
            return v
        factor = shape_fn(t)
        if factor is None:
            return v
        # Clamp ±3% to prevent weekend/holiday troughs from distorting the trend.
        factor = max(0.97, min(1.03, float(factor)))
        return v * factor

    # Daily hist sampling from first anchor to last known anchor.
    hist: list[list[Any]] = []
    t = anchors[0][0]
    while t < t_last:
        hist.append([int(t), round(with_shape(t, est(t)), 2)])
        t += DAY
    hist.append([int(t_last), round(v_last, 2)])

    # Daily ext + fan sampling from last anchor to year-end horizon. Ext
    # starts strictly *after* t_last (no seam duplication with hist[-1]).
    # Renderers that want visual continuity between hist and ext should draw
    # the two series with the same color/style; a duplicated seam point
    # confuses timestamp-keyed transforms (e.g. app.js alignedFrom).
    ext: list[list[Any]] = []
    lo: list[list[Any]] = []
    hi: list[list[Any]] = []
    t = t_last + DAY
    while t <= pj_to_ms:
        ext.append([int(t), round(est(t), 2)])
        lo.append([int(t), round(est(t, r_l), 2)])
        hi.append([int(t), round(est(t, r_h), 2)])
        t += DAY
    # Ensure the horizon endpoint is exact (guard against off-by-one from step size).
    if not ext or ext[-1][0] != pj_to_ms:
        ext.append([int(pj_to_ms), round(center, 2)])
        lo.append([int(pj_to_ms), round(pj_low, 2)])
        hi.append([int(pj_to_ms), round(pj_high, 2)])

    # counter uses the segment containing "now": if now is inside two anchors,
    # use that segment's slope; else use the extrapolation slope r_c.
    counter = {
        "tLast": int(t_last),
        "vLast": round(v_last, 1),
        "rMs": float(f"{r_c:.6e}"),
    }
    if now_ms < t_last:
        for (ta, va), (tb, vb) in zip(anchors, anchors[1:]):
            if ta <= now_ms < tb:
                counter = {
                    "tLast": int(ta),
                    "vLast": round(va, 1),
                    "rMs": float(f"{math.log(vb / va) / max(tb - ta, 1):.6e}"),
                }
                break

    return {
        "hist": hist,
        "ext": ext,
        "fanLo": lo,
        "fanHi": hi,
        "counter": counter,
        "cps": checkpoints,
        "yoyDen": round(est(now_ms - 365 * DAY), 1),
    }


# ---------------------------------------------------------------------------
# Shape overlay — npm daily texture for current-month segment
# ---------------------------------------------------------------------------
def _npm_shape_fn(conn: sqlite3.Connection):
    """Return a callable `shape(t_ms) -> factor` that maps a daily timestamp
    to a normalized npm-download deviation for the current UTC month. Used
    only for the current-month segment of the ARR curve, so users can see
    day-to-day movement layered on top of the smooth ensemble anchor path.

    - Uses `npm:@anthropic-ai/sdk` daily counts for the current month
    - Normalizes each day by the month's mean → factor around 1.0
    - Applies simple 3-day centered smoothing to dampen weekend noise
    - Returns None outside the current month (caller uses raw base value)
    """
    this_month = datetime.now(timezone.utc).strftime("%Y-%m")
    rows = conn.execute(
        "SELECT date, value FROM anthropic_signals "
        "WHERE signal_name = 'npm:@anthropic-ai/sdk' AND substr(date,1,7) = ? "
        "ORDER BY date",
        (this_month,),
    ).fetchall()
    if not rows or len(rows) < 3:
        return None
    # Filter out zero-value days (pending-report / partial-day rows that npm
    # returns as 0 before the ETL completes). If the whole month is zeros,
    # bail out — there's no signal.
    vals = [
        (r["date"], float(r["value"])) for r in rows if r["value"] and r["value"] > 0
    ]
    if len(vals) < 3:
        return None
    mean_v = sum(v for _, v in vals) / len(vals)
    if mean_v <= 0:
        return None
    # 3-day centered smoothing over the non-zero rows.
    smoothed: dict[str, float] = {}
    for i, (d, _) in enumerate(vals):
        window = vals[max(0, i - 1) : i + 2]
        smoothed[d] = sum(v for _, v in window) / len(window)
    # date-string → factor lookup
    factor_by_date = {d: (smoothed[d] / mean_v) for d, _ in vals}

    def shape(t_ms: int):
        d = datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc).date().isoformat()
        return factor_by_date.get(d)

    return shape


# ---------------------------------------------------------------------------
# Anthropic — ensemble-driven anchors
# ---------------------------------------------------------------------------
def _anthropic_curve(conn: sqlite3.Connection) -> dict[str, Any]:
    """Build Anthropic curve from KNOWN_ARR anchors + ensemble forecast for
    current month + ensemble 95% CI at horizon (r_low / r_high band)."""
    known_rows = conn.execute(
        "SELECT month, arr_b_usd, source FROM anthropic_arr_known ORDER BY month"
    ).fetchall()
    if not known_rows:
        return {}
    # Anchors: last day of each known month → arr_b_usd
    anchors: list[tuple[int, float]] = []
    for r in known_rows:
        anchors.append((_ms(_month_end(r["month"])), float(r["arr_b_usd"])))

    # Prepend an ensemble forecast for the current month as an additional
    # (soft) anchor, so the curve doesn't flatline between last known and
    # extrapolation start.
    this_month = datetime.now(timezone.utc).strftime("%Y-%m")
    forecast = predict_arr(conn, target_month=this_month)
    forecast_v: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    if forecast and forecast.get("ensemble"):
        e = forecast["ensemble"]
        forecast_v = float(e["predicted_arr_b"])
        # The ramped blend's trend_only leg can leave the band unset when no
        # signal fits and there are too few anchors for the trend's own LOO.
        ci_low = None if e["ci_low"] is None else float(e["ci_low"])
        ci_high = None if e["ci_high"] is None else float(e["ci_high"])
        this_month_end_ms = _ms(_month_end(this_month))
        # Only add if it's ahead of the last known anchor (avoid duplicating).
        if this_month_end_ms > anchors[-1][0]:
            anchors.append((this_month_end_ms, forecast_v))

    # Extrapolation horizon: end of this year (Dec 31, current year).
    now = datetime.now(timezone.utc)
    horizon_iso = f"{now.year}-12-31"
    pj_to_ms = _ms(horizon_iso)

    t_last, v_last = anchors[-1]
    # Determine forward growth rate at horizon.
    #
    # Previous behaviour: `base_slope` = full-history geometric mean growth
    # rate → compound at that rate all the way to Dec-31. With 18 months of
    # Anthropic data averaging ~24%/mo (and the last 4 months alone
    # accelerating to ~40%/mo), this produced a ~$230B year-end target
    # while castora sits at $155B — a 50% gap that read as "we're way too
    # bullish" even to the user driving the assumptions.
    #
    # New behaviour: mirror castora — damped-decay from the last-leg growth
    # rate with tau=2mo. Ext then averages ~half the initial rate over 6mo
    # instead of compounding forever. Signal composite is folded in the same
    # way as OpenAI, but Anthropic keeps its ensemble-driven current-month
    # anchor (that's what makes the ARR line pass through the ensemble hero
    # value rather than a checkpoint-only guess).
    # Departure rate = the leg the history arrives on, so the forecast leaves
    # the last anchor at the rate it reached it and there's no step at the seam.
    # This used to be a log-linear fit over the trailing 4 anchors, which folded
    # the much hotter spring legs into the departure rate: on 2026-08-03 that
    # left August at 11.04%/mo against the month's own 8.22%, a visible kink up
    # exactly where the line stops being history. A trailing fit also silently
    # fights any move in the current-month anchor — pinning August lowered the
    # last leg but not the fit, which is what surfaced the defect.
    if len(anchors) >= 2:
        t_prev, v_prev = anchors[-2]
        leg_months = (t_last - t_prev) / MONTH_MS
        g0 = math.log(v_last / v_prev) / leg_months if leg_months > 0 else 0.0
    else:
        g0 = 0.0

    # Growth decays with horizon rather than compounding flat to year-end.
    # The old code evaluated the decay at `dt_now = max(0, now - t_last)`, but
    # t_last is the *current month's* end — always in the future — so dt_now was
    # 0 every day but the last of the month and exp(-dt/τ) was always exactly 1.
    # Nothing decayed anywhere: the "damped" extrapolation compounded the
    # departure rate unchanged for four months. Decay against horizon instead,
    # which is the variable it was always meant to run on.
    tau = _EXT_DECAY_TAU_MONTHS
    horizon_months = (pj_to_ms - t_last) / MONTH_MS
    # ∫₀ʰ g0·e^(-s/τ) ds — total log-growth to the horizon under the damped path.
    total_log_growth = g0 * tau * (1.0 - math.exp(-horizon_months / tau))
    center_at_horizon = v_last * math.exp(total_log_growth)
    # Window-average rate, for consumers that want one number for the ext leg.
    r_avg = total_log_growth / horizon_months if horizon_months > 0 else 0.0

    if ci_low is not None and ci_high is not None and forecast_v:
        # Ensemble CI at the current-month point defines the band width;
        # apply the same relative width at horizon.
        band_ratio = max(ci_high, forecast_v * 1.05) / forecast_v
        pj_low = center_at_horizon / band_ratio
        pj_high = center_at_horizon * band_ratio
    else:
        # No ensemble; fall back to ±35% at horizon.
        pj_low = center_at_horizon * 0.75
        pj_high = center_at_horizon * 1.35

    checkpoints = [
        {
            "t": _ms(_month_end(r["month"])),
            "v": float(r["arr_b_usd"]),
            "src": r["source"] or "",
        }
        for r in known_rows
    ]
    # No shape_fn: castora draws a pure log-linear curve between checkpoints
    # and it reads much cleaner than our ±3% npm-daily texture overlay, which
    # was adding weekday/weekend noise that showed up on the trajectory chart
    # as jagged 5-7% day-over-day swings. The npm data is still visible in
    # the Signals tab; here we want trend, not texture.
    now_ms = _now_ms()
    series = _prep_series(
        anchors,
        pj_to_ms,
        pj_low,
        pj_high,
        now_ms,
        checkpoints,
        shape_fn=None,
        ext_decay_tau_months=tau,
    )
    if not series:
        return {}  # domain-guard bailed; don't ship a naked label.

    # Anchor the live counter to the *interpolated hist value at now* so the
    # ticker never disagrees with the chart at t=now. Slope points at the
    # month-end forecast (matches dashboard hero) if we have one; otherwise
    # it uses the ext-segment slope from _prep_series.
    #
    # Previously we anchored vLast = forecast_v (the month-end value), which
    # caused a 5-8% offset vs the hist line at t=now — user-visible drift.
    hist_pts = series.get("hist") or []
    ext_pts = series.get("ext") or []

    def _interp_ts(pts, target_ts):
        """Piecewise-linear interpolation on [t, v] pairs at target_ts (ms)."""
        if not pts:
            return None
        if target_ts <= pts[0][0]:
            return pts[0][1]
        for i in range(1, len(pts)):
            ta, va = pts[i - 1]
            tb, vb = pts[i]
            if ta <= target_ts <= tb:
                if tb == ta:
                    return vb
                frac = (target_ts - ta) / (tb - ta)
                return va + (vb - va) * frac
        return pts[-1][1]

    # Value at now — prefer hist (has npm shape overlay), fall back to ext.
    now_v = None
    if hist_pts and hist_pts[-1][0] >= now_ms:
        now_v = _interp_ts(hist_pts, now_ms)
    elif ext_pts and ext_pts[0][0] <= now_ms <= ext_pts[-1][0]:
        now_v = _interp_ts(ext_pts, now_ms)
    elif hist_pts:
        now_v = hist_pts[-1][1]

    if now_v is not None and now_v > 0:
        # `rMs` is the curve's slope at now, so the ticker and the chart agree
        # under the cursor. Now sits inside the current month, on the leg the
        # ext departs at, so that slope is g0. The two fields used to carry the
        # same number because the extrapolation ran at one constant rate; under
        # the damped path they genuinely differ, and rMsExt is the one the
        # nowcast wants — its baseline is the whole ext leg, not this instant.
        series["counter"] = {
            "tLast": now_ms,
            "vLast": round(now_v, 2),
            "rMs": float(f"{g0 / MONTH_MS:.6e}"),
            "rMsExt": float(f"{r_avg / MONTH_MS:.6e}"),
        }
    return {"label": "Anthropic", "color": "#d97757", **series}


# ---------------------------------------------------------------------------
# OpenAI — castora-style: fixed press-cited checkpoints + damped-decay extrap
# ---------------------------------------------------------------------------
# Anchor set + damping constants copied from
# https://github.com/aokunyade/ai-monetization-tracker
# (config/tracker_config.json + scripts/update_data.py build_arr) so our
# OpenAI curve reproduces theirs.
#
# Anthropic is intentionally NOT switched to this model — we keep our
# ensemble-driven anchor path there because we have signal-derived data
# and denser press anchors that castora doesn't.
# Horizon decay τ, shared by both companies: the forecast leaves the last
# anchor at the rate the history arrived on and its instantaneous rate falls as
# e^(-s/τ) from there. Both curves used to compound a constant rate to the
# horizon, which meant neither ever plateaued — Anthropic's ran 11%/mo through
# December, OpenAI's 12.3%/mo through February.
#
# τ picks how hard the taper bites. Off the 2026-08 anchors (Anthropic $79B,
# 8.22%/mo departure) the year-end center lands at:
#
#     τ = 2mo  → 1.1%/mo by Dec, $90.6B    (near-halt; reads as a hard stop)
#     τ = 4mo  → 3.0%/mo by Dec, $96.5B    ← chosen
#     τ = 6mo  → 4.1%/mo by Dec, $99.5B
#     none     → 8.2%/mo flat,   $108.4B   (previous behaviour, modulo the fit)
#
# 4 months tapers visibly without implying growth stops by year-end.
_EXT_DECAY_TAU_MONTHS = 4.0

# Reviewed reported or third-party estimate anchors used by the OpenAI curve.
# Internal-research points do not belong in this published series.
_OPENAI_CHECKPOINTS = OPENAI_CHECKPOINTS

# Only unpublished research anchors belong here. The sourced July figure is
# visible in the published capture.
HIDDEN_OPENAI_ARR_MONTHS: frozenset[str] = frozenset()
# castora tracker_config.json arr.* — damped-decay extrapolation parameters.
# Damping lowered from castora's 0.8 to 0.7 so the OpenAI ext line is
# consistent with the historical 6-mo geometric mean (Dec-25→Jun-26 was
# only ~13%/mo for OpenAI vs ~37%/mo for Anthropic): castora's damping
# banks on the 6/28→7/15 sprint continuing, which then compounds fast
# enough to give an rMs (14.7%/mo) that nudges past Anthropic's ensemble.
# Trimming to 0.7 recovers Anthropic > OpenAI ordering the priors would
# expect while staying within castora's fan-band.
_OPENAI_GROWTH_DAMPING = 0.7
_OPENAI_GROWTH_DECAY_MONTHS = 2.0  # exp-decay τ (months)
_OPENAI_GROWTH_FIT_N = 4  # log-linear fit over trailing N anchors
_OPENAI_EXTRAP_MONTHS = 6  # how far past last checkpoint to draw
_OPENAI_FAN_PCT_PER_MONTH = 0.025  # ±2.5% widening per month
# The hand-off out of the last observed leg now runs on _EXT_DECAY_TAU_MONTHS,
# shared with Anthropic. It had its own τ=1.0 back when the asymptote was solved
# to hold the window endpoint fixed, so τ only chose how growth was distributed
# and a short one kept the seam tight. Under the decay the same τ sets both the
# seam and the taper, and a 1-month constant would have the forecast flat within
# a quarter.
#
# How much of the final anchor-to-anchor leg's growth rate carries into the
# forecast. At 1.0 the forecast inherits that leg outright, which on the
# 2026-07 anchor set meant projecting ~31%/mo against Anthropic's ~13% on the
# strength of short-spaced estimates. Shrinking toward r_const keeps the
# projected rate from inheriting one unusually fast interval outright.
_OPENAI_SEAM_LEG_WEIGHT = 0.2

# Triangulation (castora arr.triangulation) — blend live-signal growth into
# the damped-checkpoint growth. Weight is 0.3 of the ext slope.
_OPENAI_SIG_WEIGHT = 0.3
_OPENAI_SIG_LOOKBACK_DAYS = 30
_OPENAI_SIG_MIN_POINTS = 14
_OPENAI_SIG_CLAMP = (-0.10, 0.30)  # monthly growth range
# Per-signal contribution weights inside the composite — castora
# tracker_config.json arr.triangulation.signal_weights for OpenAI, minus
# castora's github_code_refs (0.1). That term needs 14 points inside a 30-day
# window; GitHub-search rows only land when sync is run by hand, so it never
# resolved and the divisor below simply redistributed its weight across these
# three. Restore it alongside a real sync cadence, not before.
# Values stay on castora's scale rather than being rescaled to sum to 1.0 —
# `tw` normalises whatever is present, so the two are equivalent.
_OPENAI_SIG_WEIGHTS = {
    "openrouter_tokens": 0.4,
    "vercel_spend_share": 0.2,
    "sdk_downloads": 0.3,
}
# npm/pypi packages that contribute to the OpenAI SDK-downloads signal.
# Kept in sync with sources_sdk_extra.NPM_EXTRA / PYPI_EXTRA.
_OPENAI_SDK_SIGNALS = [
    "npm:openai",
    "npm:@ai-sdk/openai",
    "npm:@openai/codex",
    "pypi:openai",
]


def _series_monthly_growth(
    pts: list[tuple[str, float]], lookback_days: int, min_points: int
) -> float | None:
    """Log-monthly growth from a 7-day-smoothed daily series, mirroring
    castora scripts/update_data.py:_series_growth. Returns None if the
    series is too short / degenerate. `pts` must be sorted ascending by date.
    """
    if not pts or len(pts) < min_points:
        return None
    ma: list[float] = []
    win: list[float] = []
    for _, v in pts:
        win.append(v)
        win = win[-7:]
        ma.append(sum(win) / len(win))
    from datetime import datetime as _dt
    from datetime import timedelta as _td

    d_last = _dt.strptime(pts[-1][0], "%Y-%m-%d")
    target = d_last - _td(days=lookback_days)
    i0 = 0
    for i, p in enumerate(pts):
        if _dt.strptime(p[0], "%Y-%m-%d") <= target:
            i0 = i
    if i0 >= len(pts) - 3 or ma[i0] <= 0 or ma[-1] <= 0:
        return None
    span_days = (d_last - _dt.strptime(pts[i0][0], "%Y-%m-%d")).days or 1
    return math.log(ma[-1] / ma[i0]) * (30.4375 / span_days)


def _openai_signals(conn: sqlite3.Connection) -> dict[str, float]:
    """Live per-signal monthly growth rates for OpenAI.

    Returns {} if no signals cleared their min-points guard. Each signal's
    growth rate is clamped inside the caller (composite scope), not here.
    """
    found: dict[str, float] = {}

    # (1) OpenRouter openai-lab token volume — daily rows.
    try:
        rows = conn.execute(
            "SELECT date, tokens_b FROM openrouter_lab_share "
            "WHERE lab = 'openai' ORDER BY date"
        ).fetchall()
        pts = [
            (r["date"], float(r["tokens_b"])) for r in rows if r["tokens_b"] is not None
        ]
        g = _series_monthly_growth(
            pts, _OPENAI_SIG_LOOKBACK_DAYS, _OPENAI_SIG_MIN_POINTS
        )
        if g is not None:
            found["openrouter_tokens"] = g
    except Exception:  # noqa: BLE001
        pass

    # (2) Vercel Gateway 'OpenAI (family)' cost-share daily series.
    #     Source is the aggregated row emitted by sources_vercel.py (mirrors
    #     the Claude (family) row); castora reads history.cost.labs.openai.
    try:
        rows = conn.execute(
            "SELECT date, share_pct FROM vercel_gateway_history "
            "WHERE metric = 'cost' AND name = 'OpenAI (family)' "
            "ORDER BY date"
        ).fetchall()
        pts = [
            (r["date"], float(r["share_pct"]))
            for r in rows
            if r["share_pct"] is not None and r["share_pct"] > 0
        ]
        g = _series_monthly_growth(
            pts, _OPENAI_SIG_LOOKBACK_DAYS, _OPENAI_SIG_MIN_POINTS
        )
        if g is not None:
            found["vercel_spend_share"] = g
    except Exception:  # noqa: BLE001
        pass

    # (3) SDK downloads — each package's growth measured on its own series,
    # then blended by recent volume.
    #
    # The packages have different lifespans (@openai/codex only exists from
    # 2025-04) and different retention (pypistats serves ~180 days vs npm's
    # full history), so a summed series steps whenever a package enters or a
    # fetch drops a day — and a per-day zero in one package hides inside a
    # positive total. Measuring growth per package sidesteps both: coverage
    # gaps only affect their own series, and a package that goes quiet drops
    # out of the blend instead of dragging it.
    try:
        rates: list[tuple[float, float]] = []  # (growth, weight)
        for signal in _OPENAI_SDK_SIGNALS:
            rows = conn.execute(
                "SELECT date, value FROM anthropic_signals "
                "WHERE signal_name = ? AND value > 0 ORDER BY date",
                (signal,),
            ).fetchall()
            pts = [
                (
                    r["date"],
                    model_signal_value(signal, r["date"], r["value"]),
                )
                for r in rows
            ]
            # drop last row (often partial upstream, per the reference impl)
            if len(pts) > 1:
                pts = pts[:-1]
            g = _series_monthly_growth(
                pts, _OPENAI_SIG_LOOKBACK_DAYS, _OPENAI_SIG_MIN_POINTS
            )
            if g is None:
                continue
            # Weight by mean daily volume over the lookback, so the blend keeps
            # tracking total SDK adoption rather than averaging a 1M/day
            # package against a 12M/day one.
            recent = [v for _, v in pts[-_OPENAI_SIG_LOOKBACK_DAYS:]]
            rates.append((g, sum(recent) / len(recent)))
        tw = sum(w for _, w in rates)
        if tw > 0:
            found["sdk_downloads"] = sum(g * w for g, w in rates) / tw
    except Exception:  # noqa: BLE001
        pass

    return found


def _openai_forward_rate(
    conn: sqlite3.Connection, pts: list[tuple[int, float]], now_ms: int
) -> dict[str, Any]:
    """Everything behind the OpenAI forward rate, in one place.

    `_openai_curve` draws with it and `build_openai_signal_breakdown` explains
    it. They used to derive it separately, and the breakdown only ever computed
    the composite fallback — so the Signals tab spent months explaining an
    8.35%/mo rate while the curve ran on the ensemble's 11.75%/mo. Two
    implementations of one number is the bug; this is the fix.
    """
    dt_now_months = (now_ms - pts[-1][0]) / MONTH_MS
    out: dict[str, Any] = {
        "dt_now_months": dt_now_months,
        "ens_ci_span": None,
        "w_sig": 0.0,
        "g_signal_raw": None,
        "g_signal": None,
        "ens_pred": None,
        "ens_gap_months": None,
        "g_month": None,
        "g0": None,
        "g_cp_at_now": None,
    }

    # ── ensemble-derived monthly rate (primary path) ────────────────────
    # r_const = log(ensemble_pred_target / last_anchor) / gap_months, i.e. the
    # constant rate carrying the last anchor to the dashboard hero.
    target_month = datetime.now(timezone.utc).strftime("%Y-%m")
    ens = (_cached_predict_openai(conn, target_month=target_month) or {}).get(
        "ensemble"
    ) or {}
    ens_pred = ens.get("predicted_arr_b")
    r_const: float | None = None
    if ens_pred and ens_pred > 0 and pts[-1][1] > 0:
        gap_months = max(0.5, (_ms(_month_end(target_month)) - pts[-1][0]) / MONTH_MS)
        try:
            r_const = math.log(ens_pred / pts[-1][1]) / gap_months
            out["ens_gap_months"] = gap_months
        except (ValueError, ZeroDivisionError):
            r_const = None
        lo, hi = ens.get("ci_low"), ens.get("ci_high")
        if lo is not None and hi is not None:
            out["ens_ci_span"] = max(0.0, (hi - lo) / ens_pred / 2.0)
        out["ens_pred"] = ens_pred

    # ── castora composite fallback (only if the ensemble is unavailable) ──
    if r_const is None:
        tail = (
            pts[-_OPENAI_GROWTH_FIT_N:]
            if len(pts) >= _OPENAI_GROWTH_FIT_N
            else pts[-2:]
        )
        t0 = tail[0][0]
        xs = [(t - t0) / MONTH_MS for t, _ in tail]
        ys = [math.log(v) for _, v in tail]
        n = len(xs)
        sx = sum(xs)
        sy = sum(ys)
        sxx = sum(x * x for x in xs)
        sxy = sum(x * y for x, y in zip(xs, ys))
        denom = n * sxx - sx * sx
        g_month = (n * sxy - sx * sy) / denom if denom > 0 else 0.0
        g0 = g_month * _OPENAI_GROWTH_DAMPING
        g_cp_at_now = g0 * math.exp(-dt_now_months / _OPENAI_GROWTH_DECAY_MONTHS)
        sigs = _openai_signals(conn)
        if sigs:
            weights = _OPENAI_SIG_WEIGHTS
            tw = sum(weights.get(k, 0.1) for k in sigs) or 1.0
            raw = sum(g * weights.get(k, 0.1) for k, g in sigs.items()) / tw
            clamp_lo, clamp_hi = _OPENAI_SIG_CLAMP
            out["g_signal_raw"] = raw
            out["g_signal"] = max(clamp_lo, min(clamp_hi, raw))
        out["w_sig"] = _OPENAI_SIG_WEIGHT if out["g_signal"] is not None else 0.0
        out.update(g_month=g_month, g0=g0, g_cp_at_now=g_cp_at_now)
        r_const = (1 - out["w_sig"]) * g_cp_at_now + out["w_sig"] * (
            out["g_signal"] or 0.0
        )

    # ── departure rate at the seam ──────────────────────────────────────
    tau = _EXT_DECAY_TAU_MONTHS
    observed_leg = _pchip_slopes(
        [t / MONTH_MS for t, _ in pts], [math.log(v) for _, v in pts]
    )[-1]
    gap = out["ens_gap_months"]
    if gap:
        # r_const is the CONSTANT rate that reaches the hero. Under a decaying
        # rate the same journey has to start hotter to cover the same ground, so
        # solve the departure rate that still lands on it:
        #     g·τ·(1 − e^(−gap/τ)) = r_const·gap
        # Skipping this solve is what a naive swap to decay does, and it quietly
        # walks the curve off the headline endpoint.
        area = tau * (1.0 - math.exp(-gap / tau))
        g_end = r_const * gap / area if area > 0 else r_const
    else:
        # No target point to hit, so the departure rate is the last leg shrunk
        # toward the composite rate.
        g_end = r_const + _OPENAI_SEAM_LEG_WEIGHT * (observed_leg - r_const)

    out.update(
        source="ensemble" if gap else "composite",
        r_const=r_const,
        observed_leg=observed_leg,
        g_end=g_end,
        tau_months=tau,
        rate_at_now=g_end * math.exp(-dt_now_months / tau),
    )
    return out


def _openai_curve(conn: sqlite3.Connection) -> dict[str, Any]:
    """OpenAI ARR curve — ensemble-regression driven.

    Since 2026-07 the OpenAI ext rate is derived from the same proxy-signal
    ensemble that drives Anthropic (see compute.predict_openai_arr) rather
    than the castora 4-signal composite. History still uses the fixed
    checkpoint set (openai_arr_checkpoints table via _OPENAI_CHECKPOINTS
    constant). Fan bands derive from the ensemble's inverse-variance CI
    scaled by horizon.

    Fallback: if the ensemble is unavailable (insufficient training data,
    fresh DB), we fall back to the original castora damped-decay + signal
    composite so the chart still renders during first-boot.
    """
    # ── anchors ─────────────────────────────────────────────────────────
    # Read from openai_arr_checkpoints (kind='checkpoint') so the curve
    # tracks the same ground truth the ensemble uses. Falls back to the
    # module-level _OPENAI_CHECKPOINTS constant if the DB has no rows
    # (fresh init, or table not seeded yet).
    db_cps = list(
        conn.execute(
            "SELECT date, arr_bn, source, classification "
            "FROM openai_arr_checkpoints "
            "WHERE kind = 'checkpoint' AND is_local = 0 ORDER BY date"
        )
    )
    if db_cps:
        pts: list[tuple[int, float]] = [
            (_ms(r["date"]), float(r["arr_bn"])) for r in db_cps
        ]
    else:
        pts = [(_ms(d), v) for d, v, _s, _u in _OPENAI_CHECKPOINTS]
    pts.sort()
    if len(pts) < 2:
        return {}

    now_ms = _now_ms()
    rate = _openai_forward_rate(conn, pts, now_ms)
    dt_now_months = rate["dt_now_months"]
    ens_ci_span = rate["ens_ci_span"]
    w_sig = rate["w_sig"]

    # ── smooth path through the anchors ─────────────────────────────────
    # The anchors are unevenly spaced (16mo down to 0.6mo) and each leg has
    # run hotter than the last, so linear-in-log interpolation put a slope
    # discontinuity at every checkpoint — +12pp at 2026-05-31, +5pp at
    # 2026-06-28. Monotone cubic in log space is C1: it still passes exactly
    # through every anchor and can't overshoot between them, but the implied
    # growth rate transitions continuously instead of stepping.
    xs = [t / MONTH_MS for t, _ in pts]
    ys = [math.log(v) for _, v in pts]
    slopes = _pchip_slopes(xs, ys)
    tau = _EXT_DECAY_TAU_MONTHS
    g_end = rate["g_end"]
    # Approach the last anchor at the same rate we leave it. Otherwise the
    # curve arrives at the observed leg's rate and departs at the shrunk one —
    # a corner at the final anchor, which is the exact defect this shape
    # exists to remove. With the slopes matched the deceleration happens
    # inside the last anchor-to-anchor segment instead, behind "Today", and
    # the seam stays smooth. The anchors are still hit exactly either way.
    slopes[-1] = g_end

    # The hand-off used to relax to a solved asymptote chosen so the window's
    # total growth stayed exactly what a constant r_const would have produced —
    # τ redistributed growth without ever reducing it, so the curve compounded
    # 12.3%/mo all the way to the far horizon and never plateaued. Decay to zero
    # instead, over the shared horizon τ: the rate still leaves the seam at the
    # leg the history ran, but it now tapers with distance out. Same shape
    # Anthropic's ext uses, so the two labs' forecasts are read the same way.

    def tail_rate(dt_months: float) -> float:
        """Instantaneous monthly growth dt months past the last anchor."""
        return g_end * math.exp(-dt_months / tau)

    def ext_v(dt_months: float) -> float:
        """Value dt months past the last anchor — ∫ of `tail_rate`, so it leaves
        the anchor at the rate that leg actually ran at and tapers from there."""
        return math.exp(ys[-1] + g_end * tau * (1.0 - math.exp(-dt_months / tau)))

    def value_at(t: int) -> float:
        """Monotone cubic on anchors; damped-hand-off ext past the last one."""
        x = t / MONTH_MS
        if x <= xs[0]:
            return pts[0][1]
        if x >= xs[-1]:
            return ext_v(x - xs[-1])
        return math.exp(_pchip_eval(xs, ys, slopes, x))

    # ── build daily hist through today ──────────────────────────────────
    hist: list[list[Any]] = []
    t = pts[0][0]
    while t < now_ms:
        hist.append([int(t), round(value_at(t), 2)])
        t += DAY
    v_now = value_at(now_ms)
    hist.append([int(now_ms), round(v_now, 2)])

    # ── build daily ext + fan for _EXTRAP_MONTHS past now ───────────────
    ext: list[list[Any]] = []
    fan_lo: list[list[Any]] = []
    fan_hi: list[list[Any]] = []
    # Fan-band width policy:
    #  * When the ensemble is the r_const source, ens_ci_span is the ±half-
    #    width at the *ensemble target month* (typically ≈1 month out). We
    #    use that as an anchor and let uncertainty widen with sqrt(months)
    #    beyond it — the standard Brownian-drift growth for forecast noise.
    #    A previous version used `per_month × mth × fan_scale` linearly,
    #    which double-counted the CI span (16% × 6mo = 96% band; not real).
    #  * When the castora composite fallback is used, keep the original
    #    "signals add info but also noise → widen fan with w_sig" rule.
    if ens_ci_span is not None:
        # Anchor: 1-month-out band ≈ ens_ci_span. From there the band grows
        # as sqrt(mth) — the discrete-Brownian analogue of √t in random-walk
        # forecast intervals. Floor keeps a small training set with near-zero
        # LOO residuals from producing an unrealistic tight fan.
        ci_anchor = max(_OPENAI_FAN_PCT_PER_MONTH, ens_ci_span)
        # Cap the band at 45% at the far horizon to prevent nonsensical
        # negative-ARR fan_lo readings; this reflects the fact that our
        # linear-fit model is not honest about horizons past the training
        # window regardless of what LOO says.
        ci_cap = 0.45

        def band_fn(mth: float) -> float:
            return min(ci_cap, ci_anchor * math.sqrt(max(1.0, mth)))

    else:
        # castora fallback: signals add info but also noise → widen w/ w_sig.
        per_month = _OPENAI_FAN_PCT_PER_MONTH
        fan_scale = 1.0 + 0.3 * w_sig

        def band_fn(mth: float) -> float:
            return per_month * mth * fan_scale

    # 1 point per day.
    n_days = int(_OPENAI_EXTRAP_MONTHS * 30.4375)
    for i in range(1, n_days + 1):
        mth = _OPENAI_EXTRAP_MONTHS * i / n_days
        te = now_ms + int(mth * 30.4375 * DAY)
        v = ext_v(dt_now_months + mth)
        band = band_fn(mth)
        ext.append([int(te), round(v, 2)])
        fan_lo.append([int(te), round(v * (1 - band), 2)])
        fan_hi.append([int(te), round(v * (1 + band), 2)])

    # ── counter — `rMs` is the curve's slope at now, so the ticker and the
    # chart agree at t=now. Since the seam hand-off decays, that is hotter
    # than the far tail for the first ~τ months past the last anchor. ──
    rMs = tail_rate(dt_now_months) / MONTH_MS
    # arr_nowcast reads counter.rMsExt as the "baseline monthly growth" for its
    # breakdown table: the average rate over the ext leg as drawn, not the slope
    # at now. This used to be r_const, which the solved asymptote made exactly
    # equal to that average; with growth decaying the two part company, so
    # integrate `tail_rate` over the drawn window and divide.
    _w = float(_OPENAI_EXTRAP_MONTHS)
    r_ext_avg = (
        g_end
        * tau
        * (math.exp(-dt_now_months / tau) - math.exp(-(dt_now_months + _w) / tau))
        / _w
    )
    counter = {
        "tLast": int(now_ms),
        "vLast": round(v_now, 3),
        "rMs": float(f"{rMs:.6e}"),
        "rMsExt": float(f"{r_ext_avg / MONTH_MS:.6e}"),
    }

    if db_cps:
        checkpoints = [
            {
                "t": _ms(r["date"]),
                "v": float(r["arr_bn"]),
                "src": r["source"] or "",
                "classification": r["classification"],
            }
            for r in db_cps
        ]
    else:
        checkpoints = [
            {
                "t": _ms(row["date"]),
                "v": row["arr_bn"],
                "src": row["source"],
                "classification": row["classification"],
            }
            for row in OPENAI_ANCHOR_RECORDS
        ]
    yoy_den = round(value_at(now_ms - 365 * DAY), 2)

    return {
        "label": "OpenAI",
        "color": "#10a37f",
        "hist": hist,
        "ext": ext,
        "fanLo": fan_lo,
        "fanHi": fan_hi,
        "cps": checkpoints,
        "counter": counter,
        "yoyDen": yoy_den,
    }


# ---------------------------------------------------------------------------
# Public: OpenAI signal breakdown for the Signals tab. Mirrors the internal
# state of `_openai_curve` so users can see WHY the OpenAI ext rate is what
# it is — the four triangulation signals, their per-signal weights, the
# clamp, and the final r_const that drives the ticker + Dec-31 endpoint.
# ---------------------------------------------------------------------------
def build_openai_signal_breakdown(conn: sqlite3.Connection) -> dict[str, Any]:
    """Explain the OpenAI forward rate, off the same derivation the curve draws.

    Reads the anchors the way `_openai_curve` does — DB checkpoints first, the
    module constant only as a fresh-boot fallback. It used to read the constant
    unconditionally and recompute the composite rate from scratch, so a curve
    running on the ensemble path got explained by numbers it never used.
    """
    db_cps = list(
        conn.execute(
            "SELECT date, arr_bn FROM openai_arr_checkpoints "
            "WHERE kind = 'checkpoint' AND is_local = 0 ORDER BY date"
        )
    )
    if db_cps:
        pts: list[tuple[int, float]] = [
            (_ms(r["date"]), float(r["arr_bn"])) for r in db_cps
        ]
    else:
        pts = [(_ms(d), v) for d, v, _s, _u in _OPENAI_CHECKPOINTS]
    pts.sort()
    if len(pts) < 2:
        return {"available": False, "reason": "no OpenAI anchors"}

    r = _openai_forward_rate(conn, pts, _now_ms())
    sigs = _openai_signals(conn)
    ensemble_path = r["source"] == "ensemble"

    return {
        "available": True,
        "as_of": date.today().isoformat(),
        "rate_source": r["source"],
        "method": (
            "Ensemble-anchored. r_const is the constant monthly rate carrying "
            "the last checkpoint to the ensemble's target-month estimate, so "
            "the curve passes through the dashboard hero. History runs on a "
            "monotone cubic through the anchors; the forecast leaves the last "
            "one at the rate that lands on the hero under decay, and from "
            "there the instantaneous rate decays as e^(-s/τ) with horizon "
            "rather than compounding flat. The signal composite below is "
            "reported for reference — it only sets the rate on the fallback "
            "path, when the ensemble is unavailable."
            if ensemble_path
            else "Composite fallback — the ensemble was unavailable, so rate = "
            "(1 - w_sig) × g_cp + w_sig × g_signal, where g_cp is the "
            "trailing-4-anchor log-linear slope × damping, decayed by "
            "exp(-dt_now/τ), and g_signal is the clamped weighted mean of the "
            "signal growth rates."
        ),
        "ensemble": {
            "predicted_arr_b": r["ens_pred"],
            "gap_months": r["ens_gap_months"],
            "ci_span": r["ens_ci_span"],
        },
        # Populated on the fallback path only; null when the ensemble wins, so
        # the tab can't print a number the curve didn't use.
        "last_leg_fit": {
            "n_anchors_used": _OPENAI_GROWTH_FIT_N,
            "g_month_raw": r["g_month"],
            "damping": _OPENAI_GROWTH_DAMPING,
            "g0_damped": r["g0"],
            "tau_months": _OPENAI_GROWTH_DECAY_MONTHS,
            "dt_now_months": r["dt_now_months"],
            "g_cp_at_now": r["g_cp_at_now"],
        },
        "composite": {
            "in_use": not ensemble_path,
            "signal_weights": _OPENAI_SIG_WEIGHTS,
            "clamp": {"lo": _OPENAI_SIG_CLAMP[0], "hi": _OPENAI_SIG_CLAMP[1]},
            "g_signal_raw": r["g_signal_raw"],
            "g_signal_clamped": r["g_signal"],
            "w_sig": r["w_sig"],
        },
        "final_rate_per_month": r["r_const"],
        "seam": {
            "interpolation": "monotone cubic (PCHIP) in log space",
            "observed_last_leg": r["observed_leg"],
            "leg_weight": None if ensemble_path else _OPENAI_SEAM_LEG_WEIGHT,
            "g_end_last_leg": r["g_end"],
            "tau_months": r["tau_months"],
            "rate_at_now": r["rate_at_now"],
        },
        "signals": [
            {
                "signal": key,
                "monthly_growth": sigs.get(key),
                "weight": _OPENAI_SIG_WEIGHTS.get(key),
                "in_composite": sigs.get(key) is not None and not ensemble_path,
                "reason": (
                    None
                    if sigs.get(key) is not None
                    else "insufficient history (need \u226514 days)"
                ),
            }
            for key in ("openrouter_tokens", "vercel_spend_share", "sdk_downloads")
        ],
    }


# ---------------------------------------------------------------------------
# Public: build the full payload
# ---------------------------------------------------------------------------
def build_arr_curve(conn: sqlite3.Connection) -> dict[str, Any]:
    return {
        "render": True,
        "updated": date.today().isoformat(),
        "companies": {
            "anthropic": _anthropic_curve(conn),
            "openai": _openai_curve(conn),
        },
    }
