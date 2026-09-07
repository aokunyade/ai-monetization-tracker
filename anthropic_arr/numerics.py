"""Numerical helpers — copied from parent project to keep this app standalone.

Originally lived in the parent `compute.py`. Inlined here so this package has
no parent imports.
"""

from __future__ import annotations

from datetime import date, datetime


def days_in_month(month: str) -> int:
    """Days in YYYY-MM."""
    yyyy, mm = month.split("-")
    if mm == "12":
        nxt = (int(yyyy) + 1, 1)
    else:
        nxt = (int(yyyy), int(mm) + 1)
    return (datetime(nxt[0], nxt[1], 1) - datetime(int(yyyy), int(mm), 1)).days


def month_end_date(month: str) -> date:
    """Last calendar day of YYYY-MM."""
    return date(int(month[:4]), int(month[5:7]), days_in_month(month))


def month_index(month: str) -> int:
    """Absolute month number for gap-aware arithmetic across year boundaries."""
    return int(month[:4]) * 12 + int(month[5:7])


def fit_linear(xs: list[float], ys: list[float]) -> tuple[float, float, float] | None:
    """Linear regression. Returns (intercept, slope, r2)."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den == 0:
        return None
    slope = sum((xs[i] - mx) * (ys[i] - my) for i in range(n)) / den
    intercept = my - slope * mx
    fitted = [intercept + slope * x for x in xs]
    ss_res = sum((ys[i] - fitted[i]) ** 2 for i in range(n))
    ss_tot = sum((y - my) ** 2 for y in ys)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return intercept, slope, r2


# Two-sided 95% t critical values. For df not in the table, picks the nearest
# larger df (conservative). For df ≥ 120 returns the normal z=1.96.
_T975_TABLE: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    12: 2.179,
    15: 2.131,
    20: 2.086,
    25: 2.060,
    30: 2.042,
    40: 2.021,
    60: 2.000,
    120: 1.980,
}


def t_critical_975(df: int) -> float:
    if df < 1:
        return 12.706
    if df >= 120:
        return 1.96
    for k in sorted(_T975_TABLE.keys()):
        if df <= k:
            return _T975_TABLE[k]
    return 1.96
