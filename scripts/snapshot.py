"""Snapshot every API fetch dashboard.html makes on boot.

Endpoints mirror the Promise.all in web/dashboard.js:199-212 (API = '/api/v1').
The two admin fetches (POST /known_arr, POST /sync) are writes, not reads, and
are deliberately excluded.

Bodies are captured verbatim, dumped compact — this is machine-read, and
indenting it costs ~2.5x the bytes for the same data. To read it by eye:
    python -m json.tool dashboard_api_snapshot.json | less

Usage:
    python scripts/snapshot.py              # writes <UTC date>.json
    python scripts/snapshot.py other.json   # explicit path
"""

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anthropic_arr.arr_curve import (  # noqa: E402
    HIDDEN_OPENAI_ARR_MONTHS,
    MONTH_MS,
    _month_end,
    _ms,
)
from anthropic_arr.sources import HIDDEN_ARR_MONTHS  # noqa: E402

BASE = os.environ.get("TRACKER_API_BASE_URL", "http://127.0.0.1:8301/api/v1").rstrip(
    "/"
)
SNAPSHOT_SCHEMA_VERSION = 1

# (key, path) in the same order dashboard.js requests them.
ENDPOINTS = [
    ("dashboard", "/dashboard"),
    ("arr_curve", "/arr_curve"),
    ("arr_nowcast", "/arr_nowcast"),
    ("openrouter_market", "/openrouter_market"),
    ("openrouter_market_fable", "/openrouter_market/fable"),
    ("vercel_gateway", "/vercel_gateway"),
    ("vercel_gateway_history", "/vercel_gateway/history"),
    ("sdk_downloads", "/sdk_downloads"),
    ("curated_signals", "/curated_signals"),
    ("codex_wau", "/codex_wau"),
    ("openai_signals", "/openai_signals"),
    ("models_meta", "/models_meta"),
    ("models_scores", "/models_scores"),
    ("models_curated", "/models_curated"),
    ("provider_throughput_history", "/provider_throughput_history"),
    ("predictor_openai", "/predictor?company=openai"),
]

# The pooled /provider_throughput_history (model=None) collapses each window to
# the median across whatever models a provider served, so its band measures
# model mix rather than supply — useless for a "provider trend BY model" view.
# The dashboard's chart re-fetches ?model=<slug> per selection; a static capture
# cannot, so we pre-expand every model that has data into one provider×model
# capture the play slices server-side.
PTH_BY_MODEL_KEY = "provider_throughput_by_model"


def capture_by_model(
    client: httpx.Client,
    pooled_body: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """One ?model= fetch per model with data, keyed by canonical slug.

    Each per-model payload keeps the pooled shape (metrics.<m> =
    {unit, providers, market_median}) so the renderer reuses a single path; the
    redundant `models`/`model` fields are dropped from the inner entries.
    """
    models = pooled_body.get("models") or []
    by_model: dict[str, Any] = {}
    failures: list[str] = []
    if not models:
        failures.append("provider throughput returned no models")
    for slug in models:
        try:
            r = client.get(
                BASE + "/provider_throughput_history", params={"model": slug}
            )
            r.raise_for_status()
            b = r.json()
            by_model[slug] = {
                "top_providers": b.get("top_providers"),
                "metrics": b.get("metrics"),
            }
        except Exception as e:  # noqa: BLE001
            failures.append(f"{slug} -> {e!r}")
    if failures:
        print(
            f"  by_model: {len(failures)} model(s) failed:\n    "
            + "\n    ".join(failures),
            file=sys.stderr,
        )
    return (
        {
            "as_of": pooled_body.get("as_of"),
            "granularity": pooled_body.get("granularity"),
            "models": models,
            "by_model": by_model,
        },
        failures,
    )


# Per-month deceleration used to redraw the published `hist` span after hidden
# checkpoints are removed. The same path restates affected predictor MoM legs,
# keeping the curve and growth table consistent without exposing private knots.
# Zero produces a constant log-space monthly rate; larger values front-load it.
_HIDDEN_SPAN_DECAY_PER_MONTH = 0.371
_REDACTED_FORECAST_METHOD = "Forecast uses redacted private ARR anchors."


def _hidden_span_model(company: dict[str, Any], hidden: frozenset[str]):
    """Return `(t_pub, t_end, value_at, bounded)` for the span bracketing the
    hidden months, or None when there is nothing to model.

    `value_at(t)` is a decelerating log-space path pinned to the last public
    checkpoint on either side of the withheld run. Both endpoints are published
    anyway, so pinning them costs nothing and keeps the chart's shape.

    What it buys is that the months between have no interior knot to read a
    withheld anchor off. What it cannot buy is secrecy: two public endpoints four
    months apart fix the average rate by arithmetic, so any plausible path
    between them estimates the withheld months to within a few percent. This
    defeats exact quotation, not inference.
    """
    hist = company.get("hist")
    cps = company.get("cps")
    if not hist or not cps or not hidden:
        return None
    first_hidden = _ms(_month_end(min(hidden)))
    last_hidden = _ms(_month_end(max(hidden)))
    before = [cp for cp in cps if cp["t"] <= first_hidden]
    if not before:
        return None
    t_pub, v_pub = before[-1]["t"], before[-1]["v"]
    after = [cp for cp in cps if cp["t"] > last_hidden]
    bounded = bool(after)
    t_end, v_end = (after[0]["t"], after[0]["v"]) if bounded else hist[-1]
    if t_end <= t_pub or v_pub <= 0 or v_end <= 0:
        return None
    return t_pub, t_end, _decay_path(t_pub, v_pub, t_end, v_end), bounded


def _decay_path(t_pub: float, v_pub: float, t_end: float, v_end: float):
    """Log-space path from `(t_pub, v_pub)` to `(t_end, v_end)`, decelerating by
    `_HIDDEN_SPAN_DECAY_PER_MONTH`. Both endpoints are hit exactly."""
    span = t_end - t_pub
    total = math.log(v_end / v_pub)
    lam = _HIDDEN_SPAN_DECAY_PER_MONTH * (span / MONTH_MS)
    denom = (1.0 - math.exp(-lam)) if lam > 0 else 1.0

    def value_at(t: float) -> float:
        s = min(1.0, max(0.0, (t - t_pub) / span))
        shape = s if lam <= 0 else (1.0 - math.exp(-lam * s)) / denom
        return v_pub * math.exp(total * shape)

    return value_at


def _mom_span_model(span_model, legs: list[dict[str, Any]]):
    """Extend an unbounded modelled span to the forecast endpoint.

    A later public checkpoint bounds an interior hidden span, so that span must
    not be stretched to a forecast that postdates it.

    The `hist` span ends where the series does. Anthropic's runs to the target
    month-end, which is also the last MoM leg's endpoint, so both legs are
    already shaped by it. OpenAI's stops at today, a month short — leaving its
    forecast leg priced straight off the anchors and the published rates reading
    13.5% > 22.2% > 18.2%: a spike, when what the anchors describe is a July
    inflection that decays out of August. Re-solving to the forecast endpoint
    puts both legs back on one path. No-op wherever the two already coincide.
    """
    if span_model is None or not legs:
        return span_model
    t_pub, t_end, value_at, bounded = span_model
    if bounded:
        return span_model
    last = legs[-1]
    to_month, v_to = last.get("to_month"), last.get("to_arr_b")
    if not to_month or not v_to or v_to <= 0:
        return span_model
    t_to = _ms(_month_end(to_month))
    if t_to <= t_end:
        return span_model
    return t_pub, t_to, _decay_path(t_pub, value_at(t_pub), t_to, v_to), bounded


def _smooth_hist_past_public(company: dict[str, Any], hidden: frozenset[str]):
    """Redraw the `hist` span between public checkpoints around hidden months.

    `hist` is sampled daily off a curve that interpolates between every anchor,
    hidden ones included. Dropping only the `cps` entries would hide the markers,
    not the corresponding interior knots in the line.

    Call after `cps` has been filtered.
    """
    model = _hidden_span_model(company, hidden)
    if model is None:
        return None
    t_pub, t_end, value_at, _bounded = model
    company["hist"] = [
        [t, round(value_at(t), 2)] if t_pub < t < t_end else [t, v]
        for t, v in company["hist"]
    ]
    return model


def _restate_hidden_mom(
    legs: list[dict[str, Any]], hidden: frozenset[str], span_model
) -> None:
    """Null the $ endpoints of legs touching hidden months, and restate the rate
    of any leg whose span the modelled path covers.

    A leg is left alone when both its months sit before the last public
    checkpoint — those are reported figures and stay exact.
    """
    for leg in legs:
        if leg.get("from_month") in hidden:
            leg["from_arr_b"] = None
        if leg.get("to_month") in hidden:
            leg["to_arr_b"] = None
            leg["annualized_pct"] = None

    if span_model is None:
        return
    t_pub, t_end, value_at, _bounded = span_model
    for leg in legs:
        frm, to = leg.get("from_month"), leg.get("to_month")
        gap = leg.get("gap_months") or 0
        if not frm or not to or gap <= 0:
            continue
        t_from, t_to = _ms(_month_end(frm)), _ms(_month_end(to))
        if t_from < t_pub:
            continue  # reported leg — publish it as measured
        # An endpoint past the modelled span has to come from the leg itself:
        # `value_at` clamps there, so reading it would price the leg against the
        # span's last day instead of its real endpoint. Anthropic never hits
        # this — its series runs to the target month-end — but OpenAI's stops at
        # today, so its forecast leg lands outside and read 0.95%/mo against an
        # actual 19.49%. Those endpoints are the published hero anyway.
        v_from = value_at(t_from) if t_from <= t_end else leg.get("from_arr_b")
        v_to = value_at(t_to) if t_to <= t_end else leg.get("to_arr_b")
        if not v_from or not v_to or v_from <= 0:
            continue
        mom = ((v_to / v_from) ** (1.0 / gap) - 1.0) * 100.0
        leg["mom_pct"] = mom
        if leg.get("annualized_pct") is not None:
            leg["annualized_pct"] = ((1.0 + mom / 100.0) ** 12 - 1.0) * 100.0


def _scrub_curve_company(company: dict[str, Any], hidden: frozenset[str]):
    """Drop hidden checkpoint dots, then redraw their span in `hist`.

    Returns the modelled span so the MoM legs can be restated off it.

    Checkpoints carry a timestamp rather than a month — Anthropic's land on
    month ends, OpenAI's wherever the citation does — so resolve to month before
    testing membership.
    """
    if company.get("cps"):
        company["cps"] = [
            cp
            for cp in company["cps"]
            if datetime.fromtimestamp(cp["t"] / 1000, tz=timezone.utc).strftime("%Y-%m")
            not in hidden
        ]
    return _smooth_hist_past_public(company, hidden)


def _scrub_predictor(pred: dict[str, Any], hidden: frozenset[str], span_model) -> None:
    """Redact hidden-month anchor values from one predictor payload.

    Both companies serve the same payload shape from the same estimator, so the
    leak surface is identical: the anchor rows, the regression training points
    that quote them, the MoM legs they bound, and the forecast-method strings
    that print them inline.
    """
    if not pred:
        return
    pred["all_arr_known"] = [
        r for r in pred.get("all_arr_known") or [] if r["month"] not in hidden
    ]

    # MoM legs keep their months so the chart's shape is unchanged; nulled $
    # endpoints (the hidden anchors themselves) both redact the values and mark
    # the point for the play to render without a dot or tooltip. Annualized goes
    # with them — it is just the hidden growth restated.
    #
    # The rates go too, and have to: a leg's rate against a published endpoint is
    # the other endpoint. The last leg's true 11.9%/mo over the hero's $78.34B
    # gives July to the cent, and from there each preceding rate unwinds the one
    # before it back to the last public anchor. Restating them off the same
    # modelled path the curve now uses keeps the two charts telling one story.
    legs = pred.get("mom_growth") or []
    _restate_hidden_mom(legs, hidden, _mom_span_model(span_model, legs))
    leg = pred.get("growth_extrapolation_last_leg")
    if leg and leg.get("from_month") in hidden:
        leg["from_arr_b"] = None

    npm = pred.get("npm_growth_indicator") or {}
    if npm.get("prev_month") in hidden:
        npm["prev_arr_b"] = None

    per_signal = (pred.get("per_signal") or []) + (
        (pred.get("ramped_blend") or {}).get("per_signal") or []
    )
    for sig in per_signal:
        points = sig.get("points") or []
        uses_hidden_anchor = any(p.get("month") in hidden for p in points)
        if points:
            sig["points"] = [p for p in points if p.get("month") not in hidden]
        if uses_hidden_anchor and sig.get("forecast_method"):
            sig["forecast_method"] = _REDACTED_FORECAST_METHOD


def scrub_hidden_anchors(out: dict[str, Any]) -> None:
    """Strip HIDDEN_ARR_MONTHS anchor *values* from the published snapshot.

    The API server and every fit keep the full anchor set; only the published
    capture is rewritten. Hidden anchor rows, curve checkpoint dots, regression
    training points, backtest actuals, and values quoted by predictor metadata
    are removed. The affected daily `hist` span is redrawn between public
    endpoints, and its MoM legs are restated from that same path. Those legs stay
    in place with hidden $ endpoints nulled so the play preserves the timeline
    without rendering private dots or tooltips.

    See `_smooth_hist_past_public` and `_restate_hidden_mom` for those two
    transformations.
    """
    eps = out["endpoints"]
    companies = ((eps.get("arr_curve") or {}).get("body") or {}).get("companies") or {}
    dash = (eps.get("dashboard") or {}).get("body") or {}

    # ── Anthropic ───────────────────────────────────────────────────────────
    span = _scrub_curve_company(companies.get("anthropic") or {}, HIDDEN_ARR_MONTHS)
    if dash.get("known_arr"):
        dash["known_arr"] = [
            r for r in dash["known_arr"] if r["month"] not in HIDDEN_ARR_MONTHS
        ]
    bt = dash.get("backtest") or {}
    for k in ("rows", "ensemble_rows"):
        if bt.get(k):
            bt[k] = [r for r in bt[k] if r.get("target_month") not in HIDDEN_ARR_MONTHS]
    _scrub_predictor(dash.get("predictor") or {}, HIDDEN_ARR_MONTHS, span)

    # ── OpenAI ──────────────────────────────────────────────────────────────
    span_openai = _scrub_curve_company(
        companies.get("openai") or {}, HIDDEN_OPENAI_ARR_MONTHS
    )
    _scrub_predictor(
        (eps.get("predictor_openai") or {}).get("body") or {},
        HIDDEN_OPENAI_ARR_MONTHS,
        span_openai,
    )


def drop_kol_as_of(out: dict[str, Any]) -> None:
    """Null the curated_signals as_of in the published snapshot.

    KOL is hand-curated and only refreshed on manual edits, so its as_of
    lags the sync-driven signals and reads as stale on the dashboard. The
    dashboard guards the "KOL: <as_of>" freshness stamp with `if
    (kol?.as_of)`, so nulling it drops the stamp while the KOL takes and
    reports stay intact.
    """
    body = (out["endpoints"].get("curated_signals") or {}).get("body")
    if isinstance(body, dict):
        body["as_of"] = None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dest", nargs="?", help="output path (default: <UTC date>.json)")
    args = ap.parse_args()

    captured = datetime.now(timezone.utc)
    dest = args.dest or f"{captured:%Y-%m-%d}.json"

    out = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "captured_at": captured.isoformat(),
        "base_url": BASE,
        "source": "web/dashboard.js boot() Promise.all",
        "endpoints": {},
    }

    failures = []
    with httpx.Client(timeout=60.0) as client:
        for key, path in ENDPOINTS:
            try:
                r = client.get(BASE + path)
                ct = r.headers.get("content-type", "")
                out["endpoints"][key] = {
                    "path": "/api/v1" + path,
                    "status": r.status_code,
                    "ok": r.is_success,
                    "body": r.json() if ct.startswith("application/json") else None,
                }
                if not r.is_success:
                    failures.append(f"{path} -> HTTP {r.status_code}")
                print(f"  {r.status_code}  {path}", file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                out["endpoints"][key] = {
                    "path": "/api/v1" + path,
                    "status": None,
                    "ok": False,
                    "error": repr(e),
                    "body": None,
                }
                failures.append(f"{path} -> {e!r}")
                print(f"  ERR  {path}: {e!r}", file=sys.stderr)

        pooled = out["endpoints"].get("provider_throughput_history") or {}
        if pooled.get("ok") and isinstance(pooled.get("body"), dict):
            body, model_failures = capture_by_model(client, pooled["body"])
            complete = not model_failures
            out["endpoints"][PTH_BY_MODEL_KEY] = {
                "path": "/api/v1/provider_throughput_history?model=<each>",
                "status": 200 if complete else 502,
                "ok": complete,
                "body": body,
            }
            if model_failures:
                out["endpoints"][PTH_BY_MODEL_KEY]["error"] = "; ".join(model_failures)
                failures.extend(
                    f"{PTH_BY_MODEL_KEY} -> {failure}" for failure in model_failures
                )
            print(
                f"  {200 if complete else 502}  {PTH_BY_MODEL_KEY} "
                f"({len(body['by_model'])}/"
                f"{len(body['models'])} models)",
                file=sys.stderr,
            )
        else:
            out["endpoints"][PTH_BY_MODEL_KEY] = {
                "path": "/api/v1/provider_throughput_history?model=<each>",
                "status": None,
                "ok": False,
                "error": "pooled provider_throughput_history unavailable",
                "body": None,
            }
            failures.append(f"{PTH_BY_MODEL_KEY} -> pooled capture unavailable")
            print(
                f"  SKIP {PTH_BY_MODEL_KEY}: pooled capture unavailable",
                file=sys.stderr,
            )

    scrub_hidden_anchors(out)
    print(
        "  scrubbed hidden ARR anchors: "
        f"anthropic={len(HIDDEN_ARR_MONTHS)}, "
        f"openai={len(HIDDEN_OPENAI_ARR_MONTHS)}",
        file=sys.stderr,
    )
    drop_kol_as_of(out)
    print("  dropped curated_signals (KOL) as_of", file=sys.stderr)

    with open(dest, "w") as f:
        json.dump(out, f, separators=(",", ":"))

    print(f"\nwrote {dest}", file=sys.stderr)
    if failures:
        print("FAILURES:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
