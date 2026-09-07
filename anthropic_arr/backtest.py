"""Walk-forward backtest: for each known ARR month T, train on months < T
and predict T using only data available before T. Reports MAPE per method.

Every entry point takes `company`, defaulting to "anthropic" so the dashboard's
call site reads the same. The OpenAI side used to have no harness here at all —
its estimator was validated from a throwaway script, which is how a per-company
signal set and a per-company trend rule got shipped without either being
measurable by the thing named "backtest".
"""

from __future__ import annotations

import math
import sqlite3
from datetime import date

from .compute import (
    PROXY_SIGNALS,
    PROXY_SIGNALS_OPENAI,
    SHADOW_SIGNALS,
    SHADOW_SIGNALS_OPENAI,
    SIGNAL_REGISTRY,
    SIGNAL_REGISTRY_OPENAI,
    SIGNAL_WEIGHT_CAP,
    _monthly_value,
    _predict_for_signal,
    cap_weights,
    is_admissible,
    post_step_run_rate,
    predict_arr_ramped_blend,
    signal_blend_weights,
)
from .numerics import days_in_month, fit_linear

# Everything that differs between the two labs' estimators, in one place, so a
# harness function only has to name the company.
#
# `trend_rate` is the hook for a company whose dated checkpoints can resolve a
# level shift out of a monthly rate (see compute.post_step_run_rate). It returns
# None for any month without within-month checkpoints, so wiring it in leaves
# every historical fold untouched — which is the point: the rule has to be
# measurable here even while it has nothing to measure.
COMPANIES: dict[str, dict] = {
    "anthropic": {
        "table": "anthropic_arr_known",
        "registry": SIGNAL_REGISTRY,
        "proxies": PROXY_SIGNALS,
        "shadows": SHADOW_SIGNALS,
        "trend_rate": None,
    },
    "openai": {
        "table": "openai_arr_known",
        "registry": SIGNAL_REGISTRY_OPENAI,
        "proxies": PROXY_SIGNALS_OPENAI,
        "shadows": SHADOW_SIGNALS_OPENAI,
        "trend_rate": post_step_run_rate,
    },
}


def _spec(company: str) -> dict:
    try:
        return COMPANIES[company]
    except KeyError:
        raise ValueError(
            f"unknown company {company!r}; expected one of {sorted(COMPANIES)}"
        ) from None


def _arr_known(
    conn: sqlite3.Connection, company: str = "anthropic"
) -> dict[str, float]:
    table = _spec(company)["table"]
    return {
        r["month"]: float(r["arr_b_usd"])
        for r in conn.execute(
            f"SELECT month, arr_b_usd FROM {table} ORDER BY month"  # noqa: S608
        )
    }


def walk_forward(
    conn: sqlite3.Connection, method: str = "monthly_sum", company: str = "anthropic"
) -> list[dict]:
    """For each target month with ARR + signal data, predict using prior months.

    Returns rows of {target_month, signal, actual, predicted, abs_err, pct_err}.
    """
    spec = _spec(company)
    proxies, shadows = spec["proxies"], spec["shadows"]
    arr = _arr_known(conn, company)
    months = sorted(arr.keys())
    out: list[dict] = []

    # Include SHADOW signals in the walk-forward so their MAPE gets measured
    # too — that's the very number the promotion rule (documented in
    # compute.SHADOW_SIGNALS) checks. Without this, shadow signals could
    # never satisfy the "MAPE ≤ 12% on n ≥ 6 folds" gate that would let
    # them graduate into PROXY_SIGNALS.
    for i in range(2, len(months)):  # need ≥2 train points
        target = months[i]
        train = months[:i]
        actual = arr[target]

        for sig in proxies + shadows:
            train_x = []
            train_y = []
            for m in train:
                v = _monthly_value(conn, sig, m)
                if v and v > 0:
                    train_x.append(v)
                    train_y.append(arr[m])
            if len(train_x) < 2:
                continue
            target_x = _monthly_value(conn, sig, target)
            if not target_x or target_x <= 0:
                continue
            fit = fit_linear(train_x, train_y)
            if not fit:
                continue
            ic, sl, _ = fit
            pred = ic + sl * target_x
            err = pred - actual
            out.append(
                {
                    "target_month": target,
                    "signal": sig,
                    "actual_arr_b": actual,
                    "predicted_arr_b": pred,
                    "abs_err_b": abs(err),
                    "pct_err": abs(err) / actual * 100,
                    "signed_err_b": err,
                    "n_train": len(train_x),
                    "shadow": sig in shadows,
                }
            )

    # Add naive baseline: ARR_t = ARR_{t-1} × (ARR_{t-1}/ARR_{t-2})
    from datetime import datetime as _dt

    for i in range(2, len(months)):
        target = months[i]
        train = months[:i]
        d_prev = _dt.strptime(train[-1], "%Y-%m")
        d_prev2 = _dt.strptime(train[-2], "%Y-%m")
        d_target = _dt.strptime(target, "%Y-%m")
        n_train = (d_prev.year - d_prev2.year) * 12 + (d_prev.month - d_prev2.month)
        n_target = (d_target.year - d_prev.year) * 12 + (d_target.month - d_prev.month)
        if n_train > 0 and n_target > 0:
            compound = (arr[train[-1]] / arr[train[-2]]) ** (1 / n_train)
            pred = arr[train[-1]] * compound**n_target
            err = pred - arr[target]
            out.append(
                {
                    "target_month": target,
                    "signal": "ARR_trend_last_growth",
                    "actual_arr_b": arr[target],
                    "predicted_arr_b": pred,
                    "abs_err_b": abs(err),
                    "pct_err": abs(err) / arr[target] * 100,
                    "signed_err_b": err,
                    "n_train": len(train),
                }
            )

    return out


def walk_forward_ensemble(
    conn: sqlite3.Connection, company: str = "anthropic"
) -> list[dict]:
    """Per-month walk-forward ENSEMBLE prediction.

    Mirrors `compute.predict_arr` ensemble logic exactly: for each target month T,
    fit each PROXY_SIGNAL on `arr_known` restricted to months < T, then blend by
    R²≥0.5 admission + inverse-variance × channel_weight (same rule as live
    ensemble). Returns one row per T with both ensemble prediction and the
    actual confirmed ARR — usable as "what would the live system have said
    on the day month T closed".
    """
    spec = _spec(company)
    arr = _arr_known(conn, company)
    months = sorted(arr.keys())
    out: list[dict] = []

    for i in range(2, len(months)):
        target = months[i]
        train_arr = {m: arr[m] for m in months[:i]}
        if len(train_arr) < 2:
            continue

        per_signal: list[dict] = []
        for sig in spec["proxies"]:
            res = _predict_for_signal(
                conn, sig, train_arr, target, registry=spec["registry"]
            )
            if res:
                per_signal.append(res)
        if not per_signal:
            continue

        # Same admission + weighting rule as the live ensemble — imported,
        # not re-implemented, so the cap can't apply to one and not the other.
        usable = [p for p in per_signal if is_admissible(p)]
        if usable:
            weights = signal_blend_weights(usable)
        else:
            usable = per_signal
            weights = cap_weights(
                [p["channel_weight"] for p in usable], SIGNAL_WEIGHT_CAP
            )
        if not weights:
            continue
        pred = sum(
            usable[k]["predicted_arr_b"] * weights[k] for k in range(len(usable))
        )
        ci_low = sum(usable[k]["ci_low"] * weights[k] for k in range(len(usable)))
        ci_high = sum(usable[k]["ci_high"] * weights[k] for k in range(len(usable)))
        actual = arr[target]
        err = pred - actual
        out.append(
            {
                "target_month": target,
                "actual_arr_b": actual,
                "predicted_arr_b": pred,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "abs_err_b": abs(err),
                "pct_err": abs(err) / actual * 100,
                "signed_err_b": err,
                "n_models": len(usable),
                "n_train_months": len(train_arr),
                "signals_used": [p["signal"] for p in usable],
            }
        )

    return out


def _ensemble_of(per_signal: list[dict]) -> float | None:
    """Admission + capped inverse-variance × channel_weight, the live rule."""
    usable = [p for p in per_signal if is_admissible(p)]
    if usable:
        weights = signal_blend_weights(usable)
    else:
        usable = per_signal
        weights = cap_weights([p["channel_weight"] for p in usable], SIGNAL_WEIGHT_CAP)
    if not usable or not weights:
        return None
    return sum(usable[k]["predicted_arr_b"] * weights[k] for k in range(len(usable)))


def walk_forward_asof(
    conn: sqlite3.Connection,
    as_of_days: tuple[int, ...] = (2, 5, 10, 15, 20, 25, 31),
    company: str = "anthropic",
) -> list[dict]:
    """As-of-day walk-forward: predict month T from *partway through T*.

    `walk_forward_ensemble` only ever scores a target month whose data is
    complete, which is exactly the case where monthly_sum and ramped_blend agree
    — the projection has a full month to scale from and ramp_weight has decayed
    to its floor. The partial-month behaviour the ramped blend exists to fix is
    invisible to it. This harness re-asks the question at day 2, 5, 10 ... of the
    target month, feeding each method only what it could have known on that day.

    Training anchors are always restricted to months < T.
    """
    spec = _spec(company)
    arr = _arr_known(conn, company)
    months = sorted(arr.keys())
    out: list[dict] = []

    for i in range(2, len(months)):
        target = months[i]
        train_arr = {m: arr[m] for m in months[:i]}
        actual = arr[target]
        dim = days_in_month(target)
        y, mo = int(target[:4]), int(target[5:7])

        for d in as_of_days:
            day = min(d, dim)
            as_of = date(y, mo, day)
            # Exclusive cutoff: monthly_sum may see days strictly before as_of,
            # matching the live path (which excludes today's partial).
            cutoff = as_of.isoformat()

            per_signal = [
                r
                for sig in spec["proxies"]
                if (
                    r := _predict_for_signal(
                        conn, sig, train_arr, target, cutoff, registry=spec["registry"]
                    )
                )
            ]
            ms_pred = _ensemble_of(per_signal) if per_signal else None

            trend_rate = spec["trend_rate"]
            rb = predict_arr_ramped_blend(
                conn,
                target_month=target,
                as_of_date=as_of,
                arr_known=train_arr,
                registry=spec["registry"],
                proxy_signals=spec["proxies"],
                trend_per_month=(
                    trend_rate(conn, train_arr, target) if trend_rate else None
                ),
            )
            rb_pred = rb["ensemble"]["predicted_arr_b"] if rb else None

            row = {"target_month": target, "as_of_day": day, "actual_arr_b": actual}
            for name, pred in (("monthly_sum", ms_pred), ("ramped_blend", rb_pred)):
                row[name] = pred
                row[f"{name}_pct_err"] = (
                    abs(pred - actual) / actual * 100 if pred else None
                )
            if rb:
                row["w_trend"] = rb["weight_trend"]
                row["trend_arr_b"] = rb["trend_arr_b"]
                row["signal_arr_b"] = rb["signal_arr_b"]
            out.append(row)
    return out


def asof_summary(conn: sqlite3.Connection, company: str = "anthropic") -> list[dict]:
    """Mean |%err| per method per as-of day — the table that decides whether
    ramped_blend earns the headline under the project's own promotion gate
    (MAPE ≤ 12% on n ≥ 6 folds)."""
    rows = walk_forward_asof(conn, company=company)
    by_day: dict[int, list[dict]] = {}
    for r in rows:
        by_day.setdefault(r["as_of_day"], []).append(r)
    out = []
    for day in sorted(by_day):
        rs = by_day[day]
        entry: dict = {"as_of_day": day}
        for m in ("monthly_sum", "ramped_blend"):
            errs = [r[f"{m}_pct_err"] for r in rs if r[f"{m}_pct_err"] is not None]
            entry[f"{m}_mape"] = sum(errs) / len(errs) if errs else None
            entry[f"{m}_n"] = len(errs)
        out.append(entry)
    return out


def backtest_summary(conn: sqlite3.Connection, company: str = "anthropic") -> dict:
    """Produce per-signal MAPE/MAE summary plus the row-level table."""
    rows = walk_forward(conn, company=company)
    by_signal: dict[str, list[dict]] = {}
    for r in rows:
        by_signal.setdefault(r["signal"], []).append(r)

    summary = []
    for sig, rs in by_signal.items():
        mape = sum(r["pct_err"] for r in rs) / len(rs)
        mae = sum(r["abs_err_b"] for r in rs) / len(rs)
        rmse = math.sqrt(sum(r["abs_err_b"] ** 2 for r in rs) / len(rs))
        summary.append(
            {
                "signal": sig,
                "n": len(rs),
                "mape": mape,
                "mae": mae,
                "rmse": rmse,
                "shadow": bool(rs[0].get("shadow", False)),
            }
        )

    ensemble_rows = walk_forward_ensemble(conn, company=company)
    if ensemble_rows:
        mape = sum(r["pct_err"] for r in ensemble_rows) / len(ensemble_rows)
        mae = sum(r["abs_err_b"] for r in ensemble_rows) / len(ensemble_rows)
        rmse = math.sqrt(
            sum(r["abs_err_b"] ** 2 for r in ensemble_rows) / len(ensemble_rows)
        )
        summary.append(
            {
                "signal": "ENSEMBLE (walk-forward)",
                "n": len(ensemble_rows),
                "mape": mape,
                "mae": mae,
                "rmse": rmse,
            }
        )

    summary.sort(key=lambda x: x["mape"])

    return {
        "summary": summary,
        "rows": rows,
        "ensemble_rows": ensemble_rows,
    }
