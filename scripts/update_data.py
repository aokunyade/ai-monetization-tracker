#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Monetization Tracker — data pipeline.

Fetches every data source and writes data/data.js (window.__DATA__ = {...}).
Each section fails independently: on error the previous value of that section
is kept (or a clearly-flagged SAMPLE placeholder is generated on first run).

Sources:
  - OpenRouter datasets API (needs OPENROUTER_API_KEY, free tier is fine)
  - Ornn Compute Price Index (public)
  - Epoch AI "AI Data Centers" CSVs (CC-BY 4.0)
  - Vercel AI Gateway leaderboards (public page + public API)
  - npm downloads API + pypistats (public)
  - Google News RSS (public)
  - ARR model: computed locally from config/tracker_config.json checkpoints
"""

import csv
import io
import json
import math
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timedelta, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_JS = os.path.join(ROOT, "data", "data.js")
CFG = json.load(open(os.path.join(ROOT, "config", "tracker_config.json"), encoding="utf-8"))

UA = "Mozilla/5.0 (compatible; ai-monetization-tracker; +https://github.com)"
DAY_MS = 86400000
HTTP_TIMEOUT = int(os.environ.get("TRACKER_HTTP_TIMEOUT", "60"))
HTTP_RETRIES = int(os.environ.get("TRACKER_HTTP_RETRIES", "2"))


def log(*a):
    print("[update_data]", *a, flush=True)


def http_get(url, headers=None, timeout=None, retries=None):
    timeout = HTTP_TIMEOUT if timeout is None else timeout
    retries = HTTP_RETRIES if retries is None else retries
    h = {"User-Agent": UA}
    if headers:
        h.update(headers)
    last = None
    for i in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers=h)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:  # noqa: BLE001
            last = e
            if i < retries:
                time.sleep(2 * (i + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


def http_json(url, headers=None, timeout=None):
    return json.loads(http_get(url, headers, timeout).decode("utf-8"))


def today_utc():
    return datetime.now(timezone.utc)


def iso(d):
    return d.strftime("%Y-%m-%d")


def ms(dt_or_str):
    if isinstance(dt_or_str, str):
        dt_or_str = datetime.strptime(dt_or_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt_or_str.timestamp() * 1000)


def load_previous():
    try:
        txt = open(DATA_JS, encoding="utf-8").read()
        txt = txt[txt.index("=") + 1:].strip().rstrip(";")
        return json.loads(txt)
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------- ARR model
def build_arr():
    """Log-linear interpolation through public run-rate checkpoints, then a
    damped-growth extrapolation with a widening uncertainty fan."""
    acfg = CFG["arr"]
    now = today_utc()
    t_now = ms(now)
    out = {"render": True, "updated": iso(now), "companies": {}}

    for key, c in acfg["companies"].items():
        pts = [(ms(c["start"]["date"]), c["start"]["value_b"])] + [
            (ms(p["date"]), p["value_b"]) for p in c["checkpoints"]
        ]
        pts.sort()

        (t1, v1), (t2, v2) = pts[-2], pts[-1]
        months_span = max((t2 - t1) / (DAY_MS * 30.4375), 0.25)
        g_month = (math.log(v2) - math.log(v1)) / months_span
        g0 = g_month * acfg["growth_damping"]          # damped starting growth (per month)
        tau = acfg.get("growth_decay_months", 2.0)     # growth pace decays with this time constant

        def ext_v(dt_months):
            """Value dt months after the last checkpoint, with decaying growth."""
            return pts[-1][1] * math.exp(g0 * tau * (1 - math.exp(-dt_months / tau)))

        def value_at(t):
            if t <= pts[0][0]:
                return pts[0][1]
            for (ta, va), (tb, vb) in zip(pts, pts[1:]):
                if t <= tb:
                    f = (t - ta) / (tb - ta)
                    return math.exp(math.log(va) + f * (math.log(vb) - math.log(va)))
            return ext_v((t - pts[-1][0]) / (DAY_MS * 30.4375))

        hist, t = [], pts[0][0]
        while t < t_now:
            hist.append([t, round(value_at(t), 2)])
            t += 7 * DAY_MS
        v_now = value_at(t_now)
        hist.append([t_now, round(v_now, 2)])

        dt_now = (t_now - pts[-1][0]) / (DAY_MS * 30.4375)
        ext, fan_lo, fan_hi = [[t_now, round(v_now, 2)]], [[t_now, round(v_now, 2)]], [[t_now, round(v_now, 2)]]
        n_steps = int(acfg["extrapolation_months"] * 3)
        for i in range(1, n_steps + 1):
            mth = acfg["extrapolation_months"] * i / n_steps
            te = t_now + int(mth * 30.4375 * DAY_MS)
            v = ext_v(dt_now + mth)
            band = acfg["fan_pct_per_month"] * mth
            ext.append([te, round(v, 2)])
            fan_lo.append([te, round(v * (1 - band), 2)])
            fan_hi.append([te, round(v * (1 + band), 2)])

        out["companies"][key] = {
            "label": c["label"],
            "color": c["color"],
            "hist": hist,
            "ext": ext,
            "fanLo": fan_lo,
            "fanHi": fan_hi,
            "cps": [{"t": ms(p["date"]), "v": p["value_b"], "src": p["src"]} for p in c["checkpoints"]],
            "counter": {"tLast": t_now, "vLast": round(v_now, 3),
                        "rMs": g0 * math.exp(-dt_now / tau) / (DAY_MS * 30.4375)},
            "yoyDen": round(value_at(t_now - 365 * DAY_MS), 2),
        }
    return out


# ------------------------------------------------------------- OpenRouter
LAB_SET = set(CFG["openrouter"]["labs"])


def fetch_openrouter(prev):
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        log("openrouter: no OPENROUTER_API_KEY set")
        return prev if prev and not prev.get("sample") else sample_openrouter()

    start = CFG["openrouter"]["start_date"]
    rows = []
    cur = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = today_utc()
    while cur < end:
        chunk_end = min(cur + timedelta(days=182), end)
        url = ("https://openrouter.ai/api/v1/datasets/rankings-daily"
               f"?start_date={iso(cur)}&end_date={iso(chunk_end)}")
        j = http_json(url, headers={"Authorization": f"Bearer {key}"})
        data = j.get("data", j) if isinstance(j, dict) else j
        rows.extend(data)
        cur = chunk_end + timedelta(days=1)

    by_day = defaultdict(list)
    for r in rows:
        by_day[r["date"][:10]].append((r["model_permaslug"], float(r["total_tokens"])))
    days = sorted(by_day)
    if not days:
        raise RuntimeError("openrouter: empty dataset")

    daily_totals = [[d, round(sum(t for _, t in by_day[d]) / 1e9, 2)] for d in days]

    share_days = days[-180:]
    lab_series = defaultdict(lambda: [0.0] * len(share_days))
    for i, d in enumerate(share_days):
        total = sum(t for _, t in by_day[d]) or 1.0
        for slug, t in by_day[d]:
            if slug == "other":
                lab = "long-tail"
            else:
                vendor = slug.split("/")[0]
                lab = vendor if vendor in LAB_SET else "others"
            lab_series[lab][i] += t / total * 100
    labs = {k: [round(x, 2) for x in v] for k, v in lab_series.items()}

    watchlist = {}
    for w in CFG["openrouter"]["watch_models"]:
        series = []
        for d in days:
            tot = sum(t for slug, t in by_day[d] if slug.startswith(w["match"]))
            if tot > 0:
                series.append([d, round(tot / 1e9, 2)])
        watchlist[w["label"]] = series

    def top_models(day_list, n=15):
        agg = defaultdict(float)
        for d in day_list:
            for slug, t in by_day[d]:
                if slug != "other":
                    agg[slug] += t
        top = sorted(agg.items(), key=lambda x: -x[1])[:n]
        return [{"slug": s, "tokens_b": round(t / 1e9, 2)} for s, t in top]

    return {
        "sample": False,
        "as_of": today_utc().isoformat(),
        "citation": "Source: OpenRouter (openrouter.ai/rankings), official Datasets API",
        "tokenizer_note": "Token counts use each provider's own tokenizer — not fully comparable across providers.",
        "daily_totals": daily_totals,
        "daily_totals_unit": "B tokens/day",
        "lab_share": {"dates": share_days, "labs": labs},
        "watchlist": watchlist,
        "top_models_latest": top_models(days[-1:]),
        "top_models_7d": top_models(days[-7:]),
        "latest_date": days[-1],
    }


def sample_openrouter():
    log("openrouter: generating SAMPLE data")
    rnd = lambda i, a, b: a + (b - a) * (0.5 + 0.5 * math.sin(i * 0.7 + 1.3))  # noqa: E731
    end = today_utc()
    days = [iso(end - timedelta(days=i)) for i in range(179, -1, -1)]
    daily_totals = [[d, round(800 + i * 12 + 150 * math.sin(i / 9), 2)] for i, d in enumerate(days)]
    labs = {
        "google": [round(rnd(i, 22, 30), 2) for i in range(180)],
        "deepseek": [round(rnd(i + 2, 15, 26), 2) for i in range(180)],
        "anthropic": [round(rnd(i + 4, 8, 14), 2) for i in range(180)],
        "openai": [round(rnd(i + 6, 8, 13), 2) for i in range(180)],
        "x-ai": [round(rnd(i + 8, 4, 8), 2) for i in range(180)],
        "others": [round(rnd(i + 10, 15, 25), 2) for i in range(180)],
    }
    watchlist = {}
    for k, w in enumerate(CFG["openrouter"]["watch_models"]):
        watchlist[w["label"]] = [[d, round(rnd(i + k * 3, 20, 120), 2)] for i, d in enumerate(days[-120:])]
    tops = [{"slug": f"sample/model-{i}", "tokens_b": round(600 / (i + 1), 2)} for i in range(15)]
    return {
        "sample": True,
        "as_of": end.isoformat(),
        "citation": "SAMPLE DATA — set OPENROUTER_API_KEY and run the update workflow to replace",
        "tokenizer_note": "",
        "daily_totals": daily_totals,
        "daily_totals_unit": "B tokens/day",
        "lab_share": {"dates": days, "labs": labs},
        "watchlist": watchlist,
        "top_models_latest": tops,
        "top_models_7d": tops,
        "latest_date": days[-1],
    }


# ------------------------------------------------------------------ GPU
def fetch_gpu(prev):
    series = {}
    for name in CFG["gpu"]["models"]:
        url = f"https://api.ornnai.com/api/gpu/{urllib.parse.quote(name)}/index-history"
        j = http_json(url)
        pts = [[p["timestamp"][:10], round(float(p["index_value"]), 4)] for p in j.get("data", [])]
        if pts:
            series[name] = pts
    if not series:
        raise RuntimeError("gpu: no series returned")
    return {
        "as_of": iso(today_utc()),
        "source": "Ornn Compute Price Index (OCPI) — api.ornnai.com public API",
        "unit": "USD per GPU-hour (index value)",
        "series": series,
    }


# ------------------------------------------------------------------ SDK
def fetch_sdk(prev):
    start = CFG["sdk"]["start_date"]
    end = today_utc()
    npm = {}
    for pkg in CFG["sdk"]["npm"]:
        pts, cur = [], datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        while cur < end:
            chunk_end = min(cur + timedelta(days=530), end)
            url = f"https://api.npmjs.org/downloads/range/{iso(cur)}:{iso(chunk_end)}/{urllib.parse.quote(pkg, safe='@/')}"
        # noqa: E501
            j = http_json(url)
            for d in j.get("downloads", []):
                if d["downloads"] > 0:
                    pts.append([d["day"], round(d["downloads"] / 1e6, 4)])
            cur = chunk_end + timedelta(days=1)
        npm[pkg] = pts
    pypi = {}
    for pkg in CFG["sdk"]["pypi"]:
        j = http_json(f"https://pypistats.org/api/packages/{pkg}/overall?mirrors=false")
        pts = [[d["date"], round(d["downloads"] / 1e6, 4)]
               for d in j.get("data", [])
               if d.get("category") == "without_mirrors" and d["date"] >= start and d["downloads"] > 0]
        pypi[pkg] = sorted(pts)
    return {
        "as_of": iso(today_utc()),
        "source": "npm: api.npmjs.org/downloads/range · PyPI: pypistats.org (without mirrors)",
        "unit": "M downloads/day",
        "note": "Developer adoption proxy. Strong weekly seasonality — the front-end plots a 7-day moving average. Zero-value days (upstream gaps) removed.",
        "npm": npm,
        "pypi": pypi,
    }


# ----------------------------------------------------------- Data centers
def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def fetch_datacenters(prev):
    base = "https://epoch.ai/data/data_centers"
    snap = list(csv.DictReader(io.StringIO(http_get(f"{base}/data_centers.csv").decode("utf-8"))))
    tl = list(csv.DictReader(io.StringIO(http_get(f"{base}/data_center_timelines.csv").decode("utf-8"))))

    clean = lambda s: re.sub(r"\s*#\w+", "", (s or "")).strip()  # noqa: E731

    sites = [r for r in snap if _f(r.get("Current power (MW)")) > 0 or _f(r.get("Current H100 equivalents")) > 0]
    totals = {
        "sites": len(sites),
        "it_power_gw": round(sum(_f(r.get("Current power (MW)")) for r in sites) / 1000, 1),
        "h100_eq_m": round(sum(_f(r.get("Current H100 equivalents")) for r in sites) / 1e6, 1),
    }

    by_owner = defaultdict(lambda: {"mw": 0.0, "sites": 0})
    for r in sites:
        o = clean(r.get("Owner")) or "Unknown"
        by_owner[o]["mw"] += _f(r.get("Current power (MW)"))
        by_owner[o]["sites"] += 1
    owners = sorted(
        ({"owner": k, "mw": round(v["mw"]), "sites": v["sites"]} for k, v in by_owner.items()),
        key=lambda x: -x["mw"],
    )[:8]

    top_sites = []
    for r in sorted(sites, key=lambda r: -_f(r.get("Current power (MW)")))[:15]:
        top_sites.append({
            "name": r.get("Name", "").strip(),
            "owner": clean(r.get("Owner")),
            "user": clean((r.get("Users") or "").split(",")[0]),
            "location": (r.get("Country") or "").strip(),
            "status": "operational" if _f(r.get("Current power (MW)")) > 0 else "construction",
            "mw": round(_f(r.get("Current power (MW)"))),
            "h100e_k": round(_f(r.get("Current H100 equivalents")) / 1000),
        })

    per_site = defaultdict(list)
    for r in tl:
        d = (r.get("Date") or "")[:7]
        if not d:
            continue
        per_site[r.get("Data center", "")].append((d, _f(r.get("IT power (MW)")), _f(r.get("H100 equivalents"))))
    months = sorted({m for pts in per_site.values() for m, _, _ in pts if m >= "2024-01"})
    timeline = []
    for m in months:
        tot_mw = tot_h = 0.0
        for pts in per_site.values():
            best = None
            for pm, mw, h in sorted(pts):
                if pm <= m:
                    best = (mw, h)
            if best:
                tot_mw += best[0]
                tot_h += best[1]
        timeline.append([m, str(round(tot_mw)), str(round(tot_h / 1000))])

    return {
        "as_of": iso(today_utc()),
        "source": "Epoch AI, 'AI Data Centers' (https://epoch.ai/data/ai-data-centers), CC-BY 4.0",
        "generated_by": "update_data.py",
        "totals": totals,
        "industry_timeline": timeline,
        "timeline_unit": ["month", "cumulative IT power (MW)", "cumulative compute (k H100-eq)"],
        "by_owner": owners,
        "top_sites": top_sites,
        "notes": [
            "'Current power (MW)' in the snapshot CSV is IT power for the covered sample only — not the whole industry.",
            "Timeline includes announced future milestones; treat the right side of the chart as planned, not built.",
        ],
    }


# ---------------------------------------------------------------- Vercel
def fetch_vercel(prev):
    api = http_json("https://vercel.com/api/ai/v4/gateway-model-leaderboard?view=models&metric=tokens")
    hist = {"tokens": {}, "cost": {}}
    days = sorted({r["day"][:10] for r in api})
    focus = CFG["vercel"]["history_labs"]
    for metric in ("tokens", "cost"):
        series = {lab: [] for lab in focus}
        for d in days:
            row = next((r for r in api if r["day"][:10] == d and r["metric"] == metric), None)
            vals = dict(row["chef_values"]) if row else {}
            for lab in focus:
                series[lab].append(round(float(vals.get(lab, 0.0)), 2))
        hist[metric] = {"days": days, "labs": series}

    html = http_get("https://vercel.com/ai-gateway/leaderboards/models").decode("utf-8", "replace")
    tuples = re.findall(
        r'tabular-nums[^>]*>(\d+)</span>.*?<span class="truncate[^"]*">([^<]+)</span>'
        r'.*?font-mono[^>]*>([\d.]+)%</span>',
        html, re.S)
    groups, cur = [], []
    for rank, name, pct in tuples:
        if rank == "1" and cur:
            groups.append(cur)
            cur = []
        cur.append([name.strip(), float(pct)])
    if cur:
        groups.append(cur)
    snapshot = {"date": iso(today_utc())}
    names = ["token_share", "request_share", "spend_share"]
    for i, g in enumerate(groups[:3]):
        snapshot[names[i]] = g
    if "token_share" not in snapshot:
        raise RuntimeError("vercel: failed to parse leaderboard page")

    snapshots = [s for s in (prev or {}).get("snapshots", []) if s["date"] != snapshot["date"]]
    snapshots.append(snapshot)
    return {
        "as_of": iso(today_utc()),
        "source": "Vercel AI Gateway Leaderboards (vercel.com/ai-gateway/leaderboards) — production traffic of 200K+ teams",
        "window_note": "share % per day; model top-10 parsed from the public leaderboard page, lab history from the public API",
        "snapshots": snapshots[-60:],
        "history": hist,
    }


# ------------------------------------------------------------------ News
def fetch_news(prev):
    q = urllib.parse.quote(CFG["news"]["query"])
    xml_bytes = http_get(f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en")
    root = ET.fromstring(xml_bytes)
    items = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        src = (it.findtext("source") or "").strip()
        pub = it.findtext("pubDate") or ""
        try:
            date = iso(datetime.strptime(pub[5:16], "%d %b %Y"))
        except ValueError:
            date = iso(today_utc())
        items.append({"date": date, "title": title, "news_source": src, "url": link, "pinned": False})

    prev_items = (prev or {}).get("items", [])
    pinned = [p for p in prev_items if p.get("pinned")] + [
        dict(p, pinned=True) for p in CFG["news"].get("pinned", [])
    ]
    seen = {p["title"] for p in pinned}
    merged = pinned[:]
    for it in items + prev_items:
        if it["title"] not in seen:
            seen.add(it["title"])
            merged.append(it)
    merged.sort(key=lambda x: x["date"], reverse=True)
    return {
        "as_of": iso(today_utc()),
        "source": "Google News RSS (daily incremental merge; pinned items are kept manually)",
        "items": merged[: CFG["news"]["max_items"]],
    }


# ------------------------------------------------------- sample fallbacks
def _wave(i, lo, hi, phase=0.0):
    return round(lo + (hi - lo) * (0.5 + 0.5 * math.sin(i * 0.31 + phase)), 3)


def sample_gpu():
    end = today_utc()
    days = [iso(end - timedelta(days=i)) for i in range(159, -1, -1)]
    base = {"H100 SXM": 2.4, "H200": 3.6, "B200": 5.1, "A100 SXM4": 1.0, "RTX 5090": 0.6}
    return {"sample": True, "as_of": iso(end),
            "source": "SAMPLE DATA — run scripts/update_data.py with network access to replace (Ornn OCPI)",
            "unit": "USD per GPU-hour (index value)",
            "series": {k: [[d, _wave(i, v * 0.85, v * 1.15, hash(k) % 6)] for i, d in enumerate(days)]
                       for k, v in base.items()}}


def sample_sdk():
    end = today_utc()
    days = [iso(end - timedelta(days=i)) for i in range(179, -1, -1)]
    mkpts = lambda a, b, ph: [[d, _wave(i, a + i * (b - a) / 180, a + i * (b - a) / 180 + 0.4, ph)]  # noqa: E731
                              for i, d in enumerate(days)]
    return {"sample": True, "as_of": iso(end),
            "source": "SAMPLE DATA — run scripts/update_data.py with network access to replace",
            "unit": "M downloads/day", "note": "",
            "npm": {p: mkpts(1.5, 6.0, i) for i, p in enumerate(CFG["sdk"]["npm"])},
            "pypi": {p: mkpts(2.0, 9.0, i + 2) for i, p in enumerate(CFG["sdk"]["pypi"])}}


def sample_datacenters():
    months = []
    d = datetime(2024, 1, 1, tzinfo=timezone.utc)
    while d <= today_utc():
        months.append(d.strftime("%Y-%m"))
        d += timedelta(days=31)
    n = len(months)
    return {"sample": True, "as_of": iso(today_utc()),
            "source": "SAMPLE DATA — run scripts/update_data.py with network access to replace (Epoch AI)",
            "totals": {"sites": 3, "it_power_gw": 2.5, "h100_eq_m": 2.4},
            "industry_timeline": [[m, str(200 + int(2300 * (i / n) ** 2)), str(150 + int(2200 * (i / n) ** 2))]
                                  for i, m in enumerate(months)],
            "by_owner": [{"owner": "Sample Cloud", "mw": 1500, "sites": 2},
                         {"owner": "Sample Labs", "mw": 1000, "sites": 1}],
            "top_sites": [
                {"name": "Sample Site A", "owner": "Sample Cloud", "user": "Sample Labs",
                 "location": "United States", "status": "operational", "mw": 950, "h100e_k": 900},
                {"name": "Sample Site B", "owner": "Sample Cloud", "user": "Sample Labs",
                 "location": "United States", "status": "construction", "mw": 800, "h100e_k": 760},
                {"name": "Sample Site C", "owner": "Sample Labs", "user": "Sample Labs",
                 "location": "United States", "status": "operational", "mw": 750, "h100e_k": 740}],
            "notes": ["SAMPLE placeholder data."]}


def sample_vercel():
    end = today_utc()
    days = [iso(end - timedelta(days=i)) for i in range(59, -1, -1)]
    tok = [["Sample Model " + s, v] for s, v in
           [("A", 21.0), ("B", 13.0), ("C", 10.0), ("D", 6.0), ("E", 5.5), ("F", 4.0),
            ("G", 4.0), ("H", 3.8), ("I", 3.4)]] + [["Other", 29.3]]
    return {"sample": True, "as_of": iso(end),
            "source": "SAMPLE DATA — run scripts/update_data.py with network access to replace (Vercel AI Gateway)",
            "window_note": "",
            "snapshots": [{"date": iso(end), "token_share": tok, "spend_share": tok, "request_share": tok}],
            "history": {"tokens": {"days": days,
                                   "labs": {l: [_wave(i, 8, 30, j) for i in range(60)]
                                            for j, l in enumerate(CFG["vercel"]["history_labs"])}},
                        "cost": {"days": days,
                                 "labs": {l: [_wave(i, 10, 45, j + 1) for i in range(60)]
                                          for j, l in enumerate(CFG["vercel"]["history_labs"])}}}}


def sample_news():
    return {"sample": True, "as_of": iso(today_utc()),
            "source": "SAMPLE DATA — run scripts/update_data.py with network access to replace (Google News RSS)",
            "items": [{"date": iso(today_utc()), "title": "Sample headline — news feed populates on first successful update run",
                       "news_source": "sample", "url": "https://news.google.com", "pinned": False}]}


# ------------------------------------------------------------------ main
def main():
    prev = load_previous()
    out = {}
    sections = {
        "arr": (lambda: build_arr(), None),
        "openrouter": (lambda: fetch_openrouter(prev.get("openrouter")), sample_openrouter),
        "gpu": (lambda: fetch_gpu(prev.get("gpu")), sample_gpu),
        "datacenters": (lambda: fetch_datacenters(prev.get("datacenters")), sample_datacenters),
        "vercel": (lambda: fetch_vercel(prev.get("vercel")), sample_vercel),
        "sdk": (lambda: fetch_sdk(prev.get("sdk")), sample_sdk),
        "news": (lambda: fetch_news(prev.get("news")), sample_news),
    }
    for name, (fn, sample_fn) in sections.items():
        try:
            out[name] = fn()
            log(f"{name}: OK")
        except Exception as e:  # noqa: BLE001
            prev_sec = prev.get(name)
            if prev_sec and not prev_sec.get("sample") and not prev_sec.get("error"):
                log(f"{name}: FAILED ({e}) — keeping previous data")
                out[name] = prev_sec
            elif sample_fn:
                log(f"{name}: FAILED ({e}) — using SAMPLE data")
                out[name] = sample_fn()
            else:
                log(f"{name}: FAILED ({e})")
                out[name] = {"error": str(e), "as_of": iso(today_utc())}

    out["signals"] = dict(CFG["signals"], as_of=iso(today_utc()),
                          note="Manually curated KOL / podcast takes — edit config/tracker_config.json")

    os.makedirs(os.path.dirname(DATA_JS), exist_ok=True)
    with open(DATA_JS, "w", encoding="utf-8") as f:
        f.write("window.__DATA__=")
        json.dump(out, f, separators=(",", ":"), ensure_ascii=False)
        f.write(";")
    log(f"wrote {DATA_JS} ({os.path.getsize(DATA_JS)} bytes)")


if __name__ == "__main__":
    sys.exit(main())
