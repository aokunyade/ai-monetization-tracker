#!/usr/bin/env python3
"""Refresh every data family required by a publishable play snapshot."""

from __future__ import annotations

from collections.abc import Callable

from anthropic_arr.sync import refresh, refresh_models_meta


def refresh_snapshot_data(
    refresh_models: Callable[[], dict[str, int]] = refresh_models_meta,
    refresh_daily: Callable[[], dict[str, int]] = refresh,
) -> dict[str, dict[str, int]]:
    model_counts = refresh_models()
    if model_counts.get("model_meta", 0) <= 0:
        raise RuntimeError("model metadata refresh produced no publishable models")
    return {"models": model_counts, "daily": refresh_daily()}


def main() -> int:
    counts = refresh_snapshot_data()
    print("Refreshed snapshot inputs:")
    for group, values in counts.items():
        print(f"  {group}:")
        for name, count in values.items():
            print(f"    {name}: {count} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
