# AI Lab ARR snapshot runbook

## Ownership and data flow

`ai-monetization-tracker` owns the Estimate, Live, and Models snapshot used by
the `ailab-arr` Funda play. The Price tab remains on its separate
`/v1/ailab/aa-models/latest` Funda API feed.

The snapshot path is:

1. `scripts.refresh_snapshot_data` refreshes model/provider metadata first,
   then daily sources and ARR materializations in SQLite.
2. `scripts/snapshot.py` captures every required local API response in one file.
3. `scripts/publish_snapshot.py` validates the file, reads the existing API
   coordinate, dry-runs the write, publishes it to
   `/v1/plays/ailab-arr/snapshot`, and verifies an exact GET readback from
   `/v1/plays/ailab-arr/snapshot`.
4. `funda-app` reads only the latest published row. It has no Blob fallback.

## Update an OpenAI ARR anchor

Edit only `config/openai_arr_anchors.json`. Each active row must contain:

- `date`: exact ISO date, and `arr_bn`: positive, finite value;
- `currency: USD` and `unit: billions`;
- a reviewable `source_title`, `source`, and HTTPS `url`;
- `evidence_type: reported` only when the linked evidence supports a reported
  figure, or `evidence_type: third_party_estimate` for an external estimate.

Never edit `openai_arr_known`, `tracker_config.json`, or generated captures to
change OpenAI anchors. Manifest sync removes retired/legacy managed rows and
rebuilds the published materialized monthly series in the same transaction;
an exact-coordinate local/admin row is reclaimed by the manifest, while other
local/admin rows are preserved but are not active model inputs.

Run the local gates:

```bash
.venv/bin/python -m unittest discover -s tests -v
uvx ruff check --isolated --select E4,E7,E9,F,I \
  anthropic_arr scripts/publish_snapshot.py scripts/snapshot.py \
  scripts/refresh_snapshot_data.py tests
uvx ruff format --isolated --check \
  anthropic_arr scripts/publish_snapshot.py scripts/snapshot.py \
  scripts/refresh_snapshot_data.py tests
.venv/bin/python -m compileall -q anthropic_arr scripts tests
bash -n scripts/publish_snapshot.sh scripts/daily_sync_snapshot.sh
```

For a normal scheduled refresh, configure `OPENROUTER_API_KEY`,
`ANTHROPIC_ARR_PRIVATE_ANCHORS_JSON`, `FUNDA_API_BASE_URL`, and
`FUNDA_ADMIN_API_KEY`, then run:

```bash
scripts/daily_sync_snapshot.sh
```

Do not print or commit those values. The script stops before capture if a
required credential is absent, and stops before publish if any endpoint or
per-model request is incomplete. The publisher always starts this checkout's
API on `SNAPSHOT_SERVER_PORT` (default `18301`) and refuses an occupied port, so
it cannot silently capture another checkout. Starting the local API only
initializes its schema; refresh is always an explicit pipeline step and never a
startup side effect.

The private-anchor variable must be a non-empty JSON array of objects with only
`month` (`YYYY-MM`) and positive, finite `arr_b_usd` fields, ordered strictly by
month. Every private month must have a later public anchor, and a private/public
month collision fails closed. Store the entire JSON value in the approved
secret manager. Do not add real values to this public repository or to task
evidence. Server startup transactionally reconciles code-owned ARR rows with
the injected set, while preserving `is_local=1` admin rows; scheduled refresh
performs the same reconciliation before capture.

PyPI model features correct the index-wide count break observed on 2026-08-25
at read time; stored rows and the public SDK downloads endpoint remain the raw
upstream feed. Five mature control packages are collected under the separate
`pypictl:` namespace to audit future feed-wide shifts. They are not model
signals and add about 7.5 seconds of intentional throttling to a refresh.

## Publish safety

- Captures with an unsupported schema version, future timestamp,
  missing/null/failed endpoint, incomplete Estimate/Live/Models UI shape, empty
  score/metric series, partial model roster, any missing or stale manifest
  anchor, or serialized size above 8 MiB are rejected.
- The publisher reads both latest and exact-version state before writing, and
  rejects a capture older than the current latest row. Re-running identical
  `(version, rev, snapshot)` content is a successful no-op.
- Reusing a `(version, rev)` coordinate for different content is rejected; bump
  `SNAPSHOT_REV` instead.
- Publishing behind a newer revision is rejected.
- A successful new coordinate always follows GET preflight → API dry-run → API
  publish → exact GET readback.

## Release and rollback

The release order is producer first:

1. Run the pipeline against beta.
2. GET the published beta snapshot and verify its version, revision, captured
   timestamp, required endpoints, payload size, and OpenAI anchor classification.
3. Re-run the same capture to prove the idempotent no-op.
4. Deploy the app consumer and verify Estimate, Live, Models, and Price in beta.
5. Proceed to production only after the task release gate is approved.

If a published snapshot is bad, first list rows and identify its numeric `id`:

```bash
curl -fsS -H "Authorization: Bearer $FUNDA_ADMIN_API_KEY" \
  "$FUNDA_API_BASE_URL/v1/plays/ailab-arr/snapshot/versions?include_drafts=true"
```

Then flip only that row to draft and read back the default snapshot:

```bash
BAD_SNAPSHOT_ID="<id-from-versions>"
curl -fsS -X PATCH \
  -H "Authorization: Bearer $FUNDA_ADMIN_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"status":"draft"}' \
  "$FUNDA_API_BASE_URL/v1/plays/ailab-arr/snapshot/$BAD_SNAPSHOT_ID/status"
curl -fsS -H "Authorization: Bearer $FUNDA_ADMIN_API_KEY" \
  "$FUNDA_API_BASE_URL/v1/plays/ailab-arr/snapshot"
```

GET must now expose the previous published row. If the consumer is bad, roll
back the app deployment. Do not delete the legacy Blob as part of this
migration.

## Verification recorded on 2026-09-07

The current v2 producer captured 17 successful endpoints, including 112/112
provider models, in a 2,789,348-byte file. The app transform and summary
projection both resolved OpenAI July 2026 to `$42.6B` with a
`third_party_estimate:TickerTrends` source. Beta row `id=312` completed publish,
exact GET readback, and an identical rerun as an idempotent no-op. The earlier
rollback drill demoted beta row `id=306` to draft and verified the prior default.

Production remains gated on explicit release approval. Only beta may validate
the secret-manager-injected private-anchor path without copying its values into
task evidence.
