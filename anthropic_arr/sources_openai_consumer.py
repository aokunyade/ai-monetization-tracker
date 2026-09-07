"""OpenAI consumer usage disclosures — first-party numbers ingested by sync.

Distinct from sources.py (which fetches live public data endpoints) because
these are HAND-CURATED disclosure points from Sam Altman tweets, OpenAI
blog posts, keynotes, and press citations. Each addition here should have
a source URL. Deep-research 2026-07-22 populated the initial set.

Signals produced (upserts into `anthropic_signals`, prefix openai:):
  openai:chatgpt_wau_m    — ChatGPT weekly active users (millions)
  openai:paid_subs_m      — ChatGPT Plus/Pro/Team paying subscribers (millions)

Data conventions:
  * Stored dates are the MONTH-END of the snapshot month (e.g. 2024-12-31),
    NOT month-start like yipit's 2024-12-01. Both work under
    `substr(date,1,7)` grouping.
  * `_interpolate_monthly` produces log-linear fills between real
    disclosures + forward-extrapolates `extrap_months` months past the
    last one, so `_predict_for_signal`'s target-month lookup can succeed
    for a target beyond the last disclosure. Rows are tagged
    'interp:...' / 'extrap:...' in source_tag but the DB row is a plain
    (date, value) pair; the tag is not currently persisted.
  * Because interpolation and extrapolation are used, single-signal R²
    and MAPE numbers are INFLATED for these signals — the fit sees a
    ~75%-synthetic training set. See code-review 2026-07-22 finding
    'inflated-interp-mape' and experiments/openai_paid_subs_honest_eval.py.
    Both signals are consequently SHADOW in compute.py, not admissible.
  * Staleness: `_check_freshness` logs a WARNING if the most recent real
    disclosure for either signal is older than STALE_THRESHOLD_DAYS at
    sync time. Fix by updating the *_DISCLOSURES constants in this file.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

from loguru import logger

STALE_THRESHOLD_DAYS = 90


# --- Snapshot points --------------------------------------------------
# Format: (snapshot_month_end_iso_date, value_millions, source_tag, source_url)
# snapshot_date is stored as YYYY-MM-DD so it aggregates cleanly into the
# right month via substr(date, 1, 7).

CHATGPT_WAU_DISCLOSURES: list[tuple[str, float, str, str]] = [
    (
        "2023-11-30",
        100.0,
        "altman-devday-2023",
        "https://en.wikipedia.org/wiki/ChatGPT",
    ),
    (
        "2024-08-31",
        200.0,
        "openai-reuters-2024-08-29",
        "https://www.thestar.com.my/tech/tech-news/2024/08/30/"
        "openai-says-chatgpt039s-weekly-users-have-grown-to-200-million",
    ),
    (
        "2024-12-31",
        300.0,
        "altman-dealbook-2024-12-04",
        "https://sg.finance.yahoo.com/news/"
        "chatgpt-doubled-weekly-active-users-210444714.html",
    ),
    (
        "2025-02-28",
        400.0,
        "openai-cnbc-2025-02-20",
        "https://sg.finance.yahoo.com/news/"
        "chatgpt-doubled-weekly-active-users-210444714.html",
    ),
    (
        "2025-07-31",
        700.0,
        "nber-w34255",
        "https://sg.news.yahoo.com/3-years-old-800-million-090107647.html",
    ),
    (
        "2026-02-28",
        900.0,
        "openai-110b-funding-2026-02-27",
        "https://www.newsweek.com/"
        "openai-hits-900-million-weekly-users-raises-110b-in-fresh-funding-11596631",
    ),
]

CHATGPT_PAID_SUBS_DISCLOSURES: list[tuple[str, float, str, str]] = [
    (
        "2024-12-31",
        15.5,
        "the-information-2024-12",
        "https://finance.yahoo.com/news/chatgpt-crosses-20m-paid-users-140519166.html",
    ),
    (
        "2025-04-30",
        20.0,
        "the-information-2025-04-02",
        "https://finance.yahoo.com/news/chatgpt-crosses-20m-paid-users-140519166.html",
    ),
    (
        "2026-02-28",
        50.0,
        "openai-110b-funding-2026-02-27",
        "https://www.newsweek.com/"
        "openai-hits-900-million-weekly-users-raises-110b-in-fresh-funding-11596631",
    ),
]


def _month_gap(a: str, b: str) -> int:
    """Whole-month gap between YYYY-MM (or YYYY-MM-DD) strings."""
    return (int(b[:4]) - int(a[:4])) * 12 + (int(b[5:7]) - int(a[5:7]))


def _add_months(start: str, k: int) -> str:
    """start = YYYY-MM-DD, returns YYYY-MM-DD k months later, always at
    month-end (last day of the resulting month, e.g. 28/29/30/31)."""
    import calendar

    y, m = int(start[:4]), int(start[5:7])
    total = y * 12 + (m - 1) + k
    ny, nm = total // 12, (total % 12) + 1
    _, last = calendar.monthrange(ny, nm)
    return f"{ny:04d}-{nm:02d}-{last:02d}"


def _interpolate_monthly(
    disclosures: list[tuple[str, float, str, str]], extrap_months: int = 6
) -> list[tuple[str, float, str]]:
    """Log-linear interpolate between disclosure snapshots to produce one
    monthly point per month between the first and last disclosure, then
    forward-extrapolate `extrap_months` months past the last disclosure
    using the slope of the trailing 3 disclosures.

    Returns list of (date_YYYY_MM_DD, value, source_tag). The source_tag
    on interpolated rows is 'interp:<from_tag>→<to_tag>' and extrapolated
    rows are 'extrap:from<last_tag>' so it's obvious which points are
    ground truth vs derived. Anchor points themselves keep their own tag.

    Forward extrapolation is what lets predict_openai_arr's target-month
    lookup succeed for a target beyond the last disclosure. Without it,
    _project_monthly_sum returns None and the signal is dropped from the
    ensemble entirely (silent, since is_admissible then rejects it)."""
    if len(disclosures) < 2:
        return [(d, v, tag) for d, v, tag, _ in disclosures]
    pts = sorted(disclosures, key=lambda x: x[0])
    out: list[tuple[str, float, str]] = []
    for i in range(len(pts) - 1):
        d1, v1, tag1, _ = pts[i]
        d2, v2, tag2, _ = pts[i + 1]
        gap = _month_gap(d1, d2)
        if gap <= 0 or v1 <= 0 or v2 <= 0:
            continue
        log_slope = (math.log(v2) - math.log(v1)) / gap
        # Anchor point d1 keeps its own tag.
        out.append((d1, v1, tag1))
        for k in range(1, gap):
            interp_date = _add_months(d1, k)
            interp_v = math.exp(math.log(v1) + log_slope * k)
            out.append((interp_date, interp_v, f"interp:{tag1}→{tag2}"))
    # Final anchor.
    out.append((pts[-1][0], pts[-1][1], pts[-1][2]))

    # Forward extrapolate. Use trailing-3 slope in log-space if available,
    # else fall back to the last-leg slope. This is conservative — a WAU
    # signal that grew log-linearly for 2 years is more likely to keep that
    # slope than one estimated from just the last pair.
    if extrap_months > 0 and len(pts) >= 2:
        window = pts[-3:] if len(pts) >= 3 else pts[-2:]
        d_base = window[0][0]
        xs = [_month_gap(d_base, d) for d, _, _, _ in window]
        ys = [math.log(v) for _, v, _, _ in window]
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        log_slope = (
            sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 0 else 0.0
        )
        d_last, v_last, tag_last, _ = pts[-1]
        for k in range(1, extrap_months + 1):
            date = _add_months(d_last, k)
            value = math.exp(math.log(v_last) + log_slope * k)
            out.append((date, value, f"extrap:from-{tag_last}"))
    return out


def _check_freshness(name: str, disclosures: list[tuple[str, float, str, str]]) -> None:
    """Log a WARNING if the most recent disclosure is older than
    STALE_THRESHOLD_DAYS. Extrapolated rows continue to be produced but
    they no longer represent recent ground truth; the operator should
    update the *_DISCLOSURES constants when this warning fires."""
    if not disclosures:
        logger.warning(f"openai_consumer:{name}: no disclosures configured")
        return
    latest_iso = max(d for d, _v, _t, _u in disclosures)
    try:
        latest = datetime.strptime(latest_iso, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        logger.warning(f"openai_consumer:{name}: unparsable date '{latest_iso}'")
        return
    age_days = (datetime.now(timezone.utc) - latest).days
    if age_days > STALE_THRESHOLD_DAYS:
        logger.warning(
            f"openai_consumer:{name}: latest real disclosure is "
            f"{age_days}d old ({latest_iso}). "
            f"Extrapolation past this point is compounding old assumptions. "
            f"Update *_DISCLOSURES in anthropic_arr/sources_openai_consumer.py."
        )
    else:
        logger.info(
            f"openai_consumer:{name}: latest disclosure {latest_iso} "
            f"({age_days}d ago, fresh)"
        )


def fetch_openai_consumer_rows() -> list[dict]:
    """Return upsert rows for anthropic_signals table.

    Each disclosure is one monthly point per signal (log-linear
    interpolated between actual disclosures, forward-extrapolated past
    the last). Also logs staleness warnings if the latest hand-curated
    disclosure is stale — extrapolation continues silently otherwise
    and the predictor would compound old assumptions.
    """
    _check_freshness("chatgpt_wau_m", CHATGPT_WAU_DISCLOSURES)
    _check_freshness("paid_subs_m", CHATGPT_PAID_SUBS_DISCLOSURES)

    captured_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out: list[dict] = []
    for date, value, _tag in _interpolate_monthly(CHATGPT_WAU_DISCLOSURES):
        out.append(
            {
                "date": date,
                "signal_name": "openai:chatgpt_wau_m",
                "value": float(value),
                "captured_at": captured_at,
            }
        )
    for date, value, _tag in _interpolate_monthly(CHATGPT_PAID_SUBS_DISCLOSURES):
        out.append(
            {
                "date": date,
                "signal_name": "openai:paid_subs_m",
                "value": float(value),
                "captured_at": captured_at,
            }
        )
    return out
