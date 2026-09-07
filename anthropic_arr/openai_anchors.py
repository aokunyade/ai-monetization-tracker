"""Load the reviewed OpenAI ARR anchor manifest."""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path
from urllib.parse import urlparse

from . import PROJECT_ROOT

OPENAI_ANCHORS_PATH = PROJECT_ROOT / "config" / "openai_arr_anchors.json"


def load_openai_anchor_records(path: Path | None = None) -> list[dict]:
    manifest_path = OPENAI_ANCHORS_PATH if path is None else path
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("OpenAI ARR anchor manifest schema_version must be 2")
    rows = payload.get("anchors") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("OpenAI ARR anchor manifest must contain anchors")
    anchors: list[dict] = []
    seen_dates: set[str] = set()
    previous_date: date | None = None
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("OpenAI ARR anchors must be objects")
        date_value = row.get("date")
        if not isinstance(date_value, str):
            raise TypeError("OpenAI ARR anchor date must be a string")
        parsed_date = date.fromisoformat(date_value)
        if parsed_date > date.today():
            raise ValueError(f"future OpenAI ARR anchor: {date_value}")
        if date_value in seen_dates:
            raise ValueError(f"duplicate OpenAI ARR anchor date: {date_value}")
        if previous_date is not None and parsed_date <= previous_date:
            raise ValueError("OpenAI ARR anchor dates must be strictly increasing")
        seen_dates.add(date_value)
        previous_date = parsed_date
        arr_value = row.get("arr_bn")
        source = row.get("source")
        url = row.get("url")
        currency = row.get("currency")
        unit = row.get("unit")
        source_title = row.get("source_title")
        classification = row.get("evidence_type")
        if (
            isinstance(arr_value, bool)
            or not isinstance(arr_value, (int, float))
            or not math.isfinite(arr_value)
            or arr_value <= 0
        ):
            raise ValueError(
                f"OpenAI ARR anchor must have a positive finite value: {date_value}"
            )
        arr_bn = float(arr_value)
        if not isinstance(source, str) or not source:
            raise ValueError(f"incomplete OpenAI ARR anchor: {date_value}")
        parsed_url = urlparse(url) if isinstance(url, str) else None
        if (
            parsed_url is None
            or parsed_url.scheme != "https"
            or not parsed_url.hostname
        ):
            raise ValueError(f"OpenAI ARR anchor URL must use HTTPS: {date_value}")
        if (
            currency != "USD"
            or unit != "billions"
            or not isinstance(source_title, str)
            or not source_title
        ):
            raise ValueError(f"invalid OpenAI ARR units/source title: {date_value}")
        if classification not in {"reported", "third_party_estimate"}:
            raise ValueError(f"invalid OpenAI ARR classification: {date_value}")
        anchors.append(
            {
                "date": date_value,
                "arr_bn": arr_bn,
                "currency": currency,
                "unit": unit,
                "source_title": source_title,
                "classification": classification,
                "source": source,
                "url": url,
                "note": row.get("note"),
            }
        )
    return anchors


OPENAI_ANCHOR_RECORDS = load_openai_anchor_records()
OPENAI_CHECKPOINTS: list[tuple[str, float, str, str]] = [
    (row["date"], row["arr_bn"], row["source"], row["url"])
    for row in OPENAI_ANCHOR_RECORDS
]


def openai_checkpoint_rows() -> list[dict]:
    return [
        {
            "kind": "checkpoint",
            "date": row["date"],
            "arr_bn": row["arr_bn"],
            "classification": row["classification"],
            "source": row["source"],
            "url": row["url"],
            "note": row["note"],
        }
        for row in OPENAI_ANCHOR_RECORDS
    ]
