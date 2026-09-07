# AI Monetization Tracker

A self-hosted dashboard tracking the marginal signals of AI monetization, inspired by the format of [ai.castoramoney.com](https://ai.castoramoney.com/) and rebuilt from scratch in English:

- **Frontier Lab ARR** — live-ticking estimated ARR for Anthropic and OpenAI, modeled from sourced evidence anchors (log-linear history + damped-growth extrapolation with an uncertainty fan)
- **Token usage & adoption** — OpenRouter daily tokens (per model family, lab share, top models), Vercel AI Gateway token/$-spend leaderboards, npm/PyPI SDK downloads
- **GPU rental prices** — Ornn Compute Price Index ($/GPU-hr, daily)
- **AI data-center buildout** — Epoch AI satellite/permit dataset + Google News feed

It is a **static site** (single `index.html` + ECharts from CDN + one generated `data/data.js`) refreshed by a **GitHub Actions** cron job. No server, no build step, no dependencies beyond Python 3 stdlib.

The repository also owns the producer for Funda's AI Lab ARR play. That path
uses the `anthropic_arr` package to maintain the richer Estimate, Live, and
Models datasets, captures them as one immutable snapshot, and publishes the
snapshot through funda-api-service. It is independent of the play's Price feed,
which the Funda app reads from its existing API endpoint.

## Quick start (local)

```bash
python scripts/update_data.py   # generate data/data.js (works without any keys — see below)
python -m http.server 8000      # then open http://localhost:8000
```

Don't double-click `index.html` — browsers block `file://` data loading. Serve it as above.

On the first run without an OpenRouter key, the token-usage charts contain clearly flagged **SAMPLE data**; everything else (ARR model, GPU, data centers, Vercel, SDK, news) is real, fetched live.

## Put it on GitHub (with auto-refresh + free hosting)

```bash
cd ai-monetization-tracker
git init -b main
git add -A
git commit -m "AI monetization tracker"
gh repo create ai-monetization-tracker --public --source=. --push
```

(No GitHub CLI? Create an empty repo on github.com, then `git remote add origin <url> && git push -u origin main`.)

Then, in the repo settings on github.com:

1. **Actions secret** (optional but recommended): *Settings → Secrets and variables → Actions → New repository secret* → name `OPENROUTER_API_KEY`, value = a key from [openrouter.ai/keys](https://openrouter.ai/settings/keys) (free account works; the datasets API doesn't cost tokens).
2. **Run the workflow once**: *Actions → Update tracker data → Run workflow*. This replaces sample data with real OpenRouter data and commits it.
3. **GitHub Pages**: *Settings → Pages → Deploy from a branch → `main` / root*. Your dashboard is now live at `https://<you>.github.io/ai-monetization-tracker/` and refreshes daily (~13:17 UTC).

## Customizing

Most static-tracker settings live in `config/tracker_config.json`:

- `config/openai_arr_anchors.json` — the only OpenAI ARR anchor input for both the static tracker and Funda producer. Do not add OpenAI checkpoints to `tracker_config.json`.
- `arr.companies.anthropic.checkpoints` — Anthropic run-rate figures. `growth_damping` (how much of the recent growth pace the extrapolation assumes) and `fan_pct_per_month` (uncertainty band width) are tunable.
- `openrouter.watch_models` — the model families charted; `hero_model` gets its own panel.
- `sdk.npm` / `sdk.pypi` — packages tracked as adoption proxies.
- `news.query` / `news.pinned` — news feed query and manually pinned items.
- `signals.kol` — the manually curated quotes panel.

The front-end is one dependency-free `index.html`; colors and the retro window styling are CSS variables at the top of the file.

## Publish the Funda play snapshot

The OpenAI ARR anchor manifest is
`config/openai_arr_anchors.json`. Each anchor records its date, USD-billion
unit, source title/URL, and `evidence_type`. Update this manifest instead of
editing model constants, `tracker_config.json`, SQLite, or generated output.

From a persistent checkout:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

export OPENROUTER_API_KEY="<openrouter-key>"
export ANTHROPIC_ARR_PRIVATE_ANCHORS_JSON="<json-array-from-secret-manager>"
.venv/bin/python -m scripts.refresh_snapshot_data

export FUNDA_API_BASE_URL="https://<api-host>"
export FUNDA_ADMIN_API_KEY="<admin-key>"
scripts/publish_snapshot.sh
```

`publish_snapshot.sh` starts this checkout's tracker API on an isolated local
port, captures all
required Estimate/Live/Models endpoints, validates that none is partial, calls
the Funda API dry-run endpoint, publishes, and verifies the exact API readback.
`scripts/daily_sync_snapshot.sh`
combines sync and publish for an existing scheduler. The SQLite database and
dated captures are runtime state and are intentionally ignored by Git.

The complete anchor-update, validation, idempotency, release, and rollback
procedure is in [`docs/ailab-arr-snapshot-runbook.md`](docs/ailab-arr-snapshot-runbook.md).

`ANTHROPIC_ARR_PRIVATE_ANCHORS_JSON` is confidential deployment input. Its
values must remain in the approved secret manager: never place them in source,
task docs, logs, shell history, generated examples, or committed captures. The
publisher refuses to run without this injection.

Release order is producer first: publish and read back the beta snapshot before
deploying the Funda app consumer. A failed publish leaves the previous published
row active; rollback the app version if the consumer has already shipped.

## Data sources & credits

| Section | Source | Access |
|---|---|---|
| Token usage | [OpenRouter Datasets API](https://openrouter.ai/docs/api/api-reference/datasets/get-rankings-daily) | API key (free) |
| Gateway share | [Vercel AI Gateway Leaderboards](https://vercel.com/ai-gateway/leaderboards/models) | public |
| SDK downloads | [npm API](https://api.npmjs.org) · [pypistats](https://pypistats.org) | public |
| GPU prices | [Ornn Compute Price Index](https://dashboard.ornnai.com/docs) | public |
| Data centers | [Epoch AI — AI Data Centers](https://epoch.ai/data/ai-data-centers) (CC-BY 4.0) | public |
| News | Google News RSS | public |
| ARR checkpoints | public press reports, editable in config | — |

ARR figures are unaudited run-rate estimates for research reference only — not investment advice. Token counts use each provider's own tokenizer and are not fully comparable across providers.
