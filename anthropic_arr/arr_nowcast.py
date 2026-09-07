"""ARR short-window nowcast — bounded blend of leading indicators.

Ports `preview_nowcast.html:run()` (lines 79-155). Reads:
  - `openrouter_daily_totals` for market-wide token momentum (rMarket).
    This is total-market and is company-agnostic — both Anthropic and
    OpenAI nowcasts consume the same rMarket signal.
  - `vercel_gateway_history` (cost + tokens for the company's family row
    — 'Claude (family)' for Anthropic, 'OpenAI (family)' for OpenAI) for
    spend-share momentum (rShare) and high-price mix bonus (mixBonus).

Combines them into a bounded effective monthly growth rate:
    rProxy  = rMarket + rShare + mixBonus
    mmEff   = clamp(0.4·rProxy + 0.6·baseline, 0.7·baseline, 1.5·baseline)

Baseline is the current segment slope of the ARR curve (r_c from
build_arr_curve for the extrapolation segment, converted to per-month).

Returns baseline vs nowcast values + a breakdown table + a plain-language
verdict, matching the tracker's data shape so the frontend renderer can
be lifted verbatim.
"""

from __future__ import annotations

import math
import sqlite3
from typing import Any

from .arr_curve import _now_ms, build_arr_curve

MS_MO = 30.4375 * 86400 * 1000


def _clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x to [min(lo,hi), max(lo,hi)] so caller can pass 0.7*mm0 / 1.5*mm0
    regardless of mm0's sign — needed because ARR ext-segment slope can go
    negative on any decelerating anchor pair, which would otherwise silently
    invert the bounds and pin nowcast to the lower bound."""
    if lo > hi:
        lo, hi = hi, lo
    return max(lo, min(hi, x))


def _avg(a: list[float]) -> float:
    return (sum(a) / len(a)) if a else 0.0


def _monthly_growth(from_v: float, to_v: float, days: float) -> float:
    if from_v <= 0 or days <= 0:
        return 0.0
    return math.pow(to_v / from_v, 30.4375 / days) - 1


def _openrouter_totals(conn: sqlite3.Connection) -> list[float]:
    rows = conn.execute(
        "SELECT total_tokens_b FROM openrouter_daily_totals ORDER BY date"
    ).fetchall()
    return [float(r["total_tokens_b"]) for r in rows]


def _vercel_series(conn: sqlite3.Connection, metric: str, name: str) -> list[float]:
    rows = conn.execute(
        "SELECT share_pct FROM vercel_gateway_history "
        "WHERE metric = ? AND name = ? ORDER BY date",
        (metric, name),
    ).fetchall()
    return [float(r["share_pct"]) for r in rows]


# Per-company configuration: which VG family row to read and how to label
# things in the breakdown/verdict text. Add a new company by extending this
# dict — the algorithm itself is company-agnostic.
_COMPANY_CFG: dict[str, dict[str, str]] = {
    "anthropic": {
        "label": "Anthropic",
        "vg_family": "Claude (family)",
        "share_label_prefix": "Claude",
    },
    "openai": {
        "label": "OpenAI",
        "vg_family": "OpenAI (family)",
        "share_label_prefix": "OpenAI",
    },
}


def _build_one(conn: sqlite3.Connection, company: str) -> dict[str, Any]:
    """Compute a nowcast breakdown for one company.

    company: 'anthropic' | 'openai'. See _COMPANY_CFG for the family-row
    name mapping. Both companies share the same rMarket (OpenRouter total-
    market momentum) since it's an ecosystem-wide signal.
    """
    cfg = _COMPANY_CFG.get(company)
    if cfg is None:
        return {"available": False, "reason": f"unknown company '{company}'"}
    curve = build_arr_curve(conn).get("companies", {}).get(company) or {}
    if not curve.get("ext") or not curve.get("counter"):
        return {"available": False, "reason": f"no {company} curve available"}

    ext = curve["ext"]
    e0_t, e0_v = ext[0][0], ext[0][1]
    e1_t, e1_v = ext[-1][0], ext[-1][1]
    if e0_v <= 0 or (e1_t - e0_t) <= 0:
        return {"available": False, "reason": "extrapolation segment invalid"}
    # Prefer the analytic ext-segment slope carried through counter.rMsExt
    # (set by _anthropic_curve). Fall back to recomputing from ext endpoints
    # only if the field is missing — that path suffers rounding loss because
    # ext values are stored round(v, 2).
    counter = curve.get("counter") or {}
    r_ms_baseline = counter.get("rMsExt")
    if r_ms_baseline is None:
        r_ms_baseline = math.log(e1_v / e0_v) / (e1_t - e0_t)
    mm0 = math.exp(r_ms_baseline * MS_MO) - 1
    pj_t = e1_t
    now = _now_ms()

    # ---- Signal 1: OpenRouter market token momentum ----
    tot = _openrouter_totals(conn)
    g7 = (
        _monthly_growth(_avg(tot[-14:-7]), _avg(tot[-7:]), 7) if len(tot) >= 14 else 0.0
    )
    g14 = (
        _monthly_growth(_avg(tot[-28:-14]), _avg(tot[-14:]), 14)
        if len(tot) >= 28
        else 0.0
    )
    r_market = _clamp((g7 + g14) / 2, 0.0, 0.12)

    # ---- Signal 2: Vercel {family} $-spend share momentum + high-price mix bonus ----
    cs = _vercel_series(conn, "cost", cfg["vg_family"])
    ts = _vercel_series(conn, "tokens", cfg["vg_family"])
    if len(cs) >= 30 and len(ts) >= 30:
        cs_recent, cs_prev = _avg(cs[-15:]), _avg(cs[-30:-15])
        r_share = _clamp((cs_recent - cs_prev) / (cs_prev or 1.0), -0.04, 0.06)
        gap_now = cs_recent - _avg(ts[-15:])
        gap_prev = cs_prev - _avg(ts[-30:-15])
        mix_bonus = _clamp((gap_now - gap_prev) / (abs(gap_prev) or 1.0), -0.02, 0.03)
    else:
        cs_recent, cs_prev, r_share, mix_bonus = 0.0, 0.0, 0.0, 0.0

    # ---- Composite + bounded blend ----
    r_proxy = r_market + r_share + mix_bonus
    mm_eff = _clamp(0.4 * r_proxy + 0.6 * mm0, 0.7 * mm0, 1.5 * mm0)
    r_ms_eff = math.log(1 + mm_eff) / MS_MO

    def val(rate: float, t: int) -> float:
        return e0_v * math.exp(rate * (t - e0_t))

    b_now = val(r_ms_baseline, now) * 1e9
    n_now = val(r_ms_eff, now) * 1e9
    b_ye = val(r_ms_baseline, pj_t)
    n_ye = val(r_ms_eff, pj_t)

    # Sensitivity — aggressive stale-slope + over-annualised
    g30a = (
        _monthly_growth(_avg(tot[-60:-30]), _avg(tot[-30:]), 30)
        if len(tot) >= 60
        else 0.0
    )
    r_market_a = _clamp((g14 + g30a) / 2, 0.0, 0.30)
    r_share_a = 0.0
    if len(cs) >= 28:
        r_share_a = _clamp(
            ((_avg(cs[-14:]) - _avg(cs[-28:-14])) / (_avg(cs[-28:-14]) or 1.0))
            * (30.4375 / 14),
            -0.05,
            0.10,
        )
    mm_eff_a = _clamp(
        0.4 * (r_market_a + r_share_a + 0.04) + 0.6 * mm0, 0.6 * mm0, 1.8 * mm0
    )
    r_ms_eff_a = math.log(1 + mm_eff_a) / MS_MO
    n_ye_a = val(r_ms_eff_a, pj_t)

    diff_ppt = (mm_eff - mm0) * 100
    if abs(diff_ppt) < 0.4:
        verdict_line = (
            "Under the disciplined blend, nowcast monthly growth is "
            "essentially flat vs baseline. Leading indicators "
            "neither confirm acceleration nor deceleration."
        )
    elif diff_ppt > 0:
        verdict_line = (
            f"Nowcast raises monthly growth by +{diff_ppt:.1f} ppt. "
            "Leading indicators (OpenRouter market momentum + Vercel "
            f"{cfg['share_label_prefix']} spend-share) are running hotter "
            "than the baseline path implies."
        )
    else:
        verdict_line = (
            f"Nowcast lowers monthly growth by {diff_ppt:.1f} ppt. "
            "This does NOT mean ARR is falling — it means the last 15–30 days "
            "of leading indicators are running below the baseline extrapolation "
            "path. The 0.4·proxy + 0.6·baseline blend caps overreaction; "
            "if you disagree, adjust extrapolation.low/high rather than the "
            "high-frequency signal."
        )

    prefix = cfg["share_label_prefix"]
    breakdown = [
        ("baseline monthly growth (anchor extrapolation)", f"{mm0 * 100:.1f}%"),
        ("(1) market 7d / 14d token momentum", f"{g7 * 100:.1f}% / {g14 * 100:.1f}%"),
        (
            "(1) rMarket used (current momentum, capped 12%)",
            f"{r_market * 100:.1f}% / mo",
        ),
        (
            f"(2) {prefix} $-spend share (prev 30-15d → recent 15d)",
            f"{cs_prev:.1f}% → {cs_recent:.1f}%",
        ),
        ("(2) rShare (share relative growth, capped 6%)", f"{r_share * 100:.1f}% / mo"),
        ("(2) high-price mix bonus (capped 3%)", f"{mix_bonus * 100:.1f}% / mo"),
        ("composite rProxy", f"{r_proxy * 100:.1f}% / mo"),
        (
            "bounded blend mmEff = 0.4·rProxy + 0.6·baseline",
            f"{mm_eff * 100:.1f}% / mo",
        ),
        (
            "— sensitivity: aggressive stale-slope + over-annualised",
            f"{mm_eff_a * 100:.1f}% / mo · YE ${n_ye_a:.0f}B (caps saturated)",
        ),
    ]

    return {
        "available": True,
        "company": company,
        "label": cfg["label"],
        "baseline_source": (
            "Baseline slope = ext-segment log-linear rate from the ARR curve "
            "(last known anchor → year-end center). This is a full-horizon "
            "path, so its per-month rate can look aggressive relative to any "
            "single 15-day window."
        ),
        "nowcast_note": (
            "Nowcast is a bounded microadjustment: 0.4·(rMarket + rShare + "
            "mixBonus) + 0.6·baseline, capped 0.7×–1.5× of baseline. Treat "
            "it as a monitor for divergence, not an accelerator."
        ),
        "baseline": {
            "now_b": round(b_now / 1e9, 2),
            "mm": mm0,
            "ye_b": round(b_ye, 1),
            "rMs": r_ms_baseline,
        },
        "nowcast": {
            "now_b": round(n_now / 1e9, 2),
            "mm": mm_eff,
            "ye_b": round(n_ye, 1),
            "rMs": r_ms_eff,
            "delta_ye_b": round(n_ye - b_ye, 1),
        },
        "signals": {
            "r_market": r_market,
            "r_share": r_share,
            "mix_bonus": mix_bonus,
            "g7": g7,
            "g14": g14,
            "cs_recent": cs_recent,
            "cs_prev": cs_prev,
        },
        "sensitivity": {"mm": mm_eff_a, "ye_b": round(n_ye_a, 1)},
        "breakdown": [{"label": k, "value": v} for k, v in breakdown],
        "verdict": verdict_line,
    }


def build_arr_nowcast(conn: sqlite3.Connection) -> dict[str, Any]:
    """Compute nowcasts for every configured company + a legacy top-level
    view that mirrors the Anthropic block for callers built before the
    OpenAI card existed. Shape:

        {
          "available": bool,                        # any company available
          "companies": {"anthropic": {...}, "openai": {...}},
          # legacy Anthropic fields at top level (unchanged from v1):
          "baseline_source": ..., "nowcast_note": ..., "baseline": {...},
          "nowcast": {...}, "signals": {...}, "sensitivity": {...},
          "breakdown": [...], "verdict": ...
        }

    Frontend renderers pick companies.{key} when they want side-by-side;
    the flat fields keep the existing renderNowcastCard call site working
    until it's swapped for the new grid.
    """
    companies = {c: _build_one(conn, c) for c in _COMPANY_CFG}
    any_avail = any(v.get("available") for v in companies.values())
    anth = companies.get("anthropic") or {}
    top = {"available": any_avail, "companies": companies}
    if anth.get("available"):
        for k in (
            "baseline_source",
            "nowcast_note",
            "baseline",
            "nowcast",
            "signals",
            "sensitivity",
            "breakdown",
            "verdict",
        ):
            top[k] = anth.get(k)
    else:
        top["reason"] = anth.get("reason", "no companies available")
    return top
