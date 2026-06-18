# Running This in Production

Your question: *"after this, how does it work in production? My code runs in
production — or is there another way?"*

Short answer: this prediction pipeline is a **batch job**, not a web service.
It does NOT run inside your backend's request path. It runs on a schedule
(e.g. nightly), writes forecasts to `card_predictions`, and your existing
backend just reads that table. Below are the ways to host it, simplest first.

---

## The core idea (don't skip this)

```
   [your scraper]            [this pipeline]              [your backend]
  apify/pricecharting  ->  runs nightly, writes  ->  reads card_predictions
   writes card_history      card_predictions           serves to the app

         every night                                    every request
```

The ML never runs when a user opens a card. By the time they look, the
forecast is already sitting in the table. This keeps your app fast and the ML
completely decoupled from your API.

---

## Option 1 — Cron on your existing server (start here)

If your backend already runs on a Linux server/VM, just add a cron entry.
This is the simplest thing that works and what I recommend you start with.

```bash
# install once on the server
cd /opt/tcg_predictor
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# crontab -e  — run at 11:30 (after your scrape finishes)
30 11 * * *  cd /opt/tcg_predictor && \
  TCG_DATABASE_URL="postgresql+psycopg2://user:pass@host:5432/db" \
  ./.venv/bin/python -m predictor.run_all >> /var/log/tcg_predict.log 2>&1
```

Pros: zero new infrastructure, easy to reason about.
Cons: tied to that one machine; you watch the log file yourself.

---

## Option 2 — Managed scheduled job

Run the same command as a scheduled container on a platform, so you don't
babysit a server:

- **Render / Railway / Fly.io** — "cron job" service type
- **AWS** — ECS Scheduled Task or Lambda (Lambda only if the run is short)
- **GCP** — Cloud Run Job + Cloud Scheduler
- **Heroku** — Scheduler add-on
- **Kubernetes** — a `CronJob` resource

A minimal Dockerfile is included (`Dockerfile`). The platform runs
`python -m predictor.run_all` on the schedule you set, with `TCG_DATABASE_URL`
as a secret env var.

Pros: managed, observable, isolated from your web app.
Cons: one more deploy target.

---

## Option 3 — Worker alongside your backend

If your backend is containerised (docker-compose / k8s), add this as a second
service in the same project sharing the same database:

```yaml
# docker-compose snippet
  predictor:
    build: ./tcg_predictor
    environment:
      TCG_DATABASE_URL: ${DATABASE_URL}
    # run via an external scheduler, or an in-container cron, or:
    command: sh -c "while true; do python -m predictor.run_all; sleep 86400; done"
```

Pros: lives with your stack, shares config/secrets.
Cons: the simple `sleep` loop is crude; prefer a real scheduler for retries.

---

## How often to run what

Not everything needs to run nightly. A good cadence:

| Task                         | Cadence            | Why                              |
|------------------------------|--------------------|----------------------------------|
| Monte Carlo                  | nightly (or more)  | instant, no training             |
| Prophet                      | nightly            | cheap per-series fit             |
| SARIMA                       | nightly or weekly  | slowest; weekly is often fine    |
| XGBoost **train**            | weekly             | patterns drift slowly            |
| XGBoost **predict**          | nightly            | reuse the weekly-trained model   |
| Ensemble                     | nightly            | after the others write           |

`run_all` does the full thing in one shot. If you want the split cadence,
call the individual runners on their own schedules and run the ensemble last.
(For very large catalogues, SARIMA is the bottleneck — shard card ids across
parallel processes, or drop it from the nightly and run it weekly.)

---

## Performance note

The per-card loop is sequential. For a few thousand cards that's fine overnight.
If your catalogue is large and runs get slow, the easiest win is to shard:

```bash
# run N workers, each taking a slice of card ids (add a --shard flag if needed)
python -m predictor.run_all   # or split by collection_id / category
```

I kept the code single-process for clarity; parallelising is a small change
when you actually need it — don't do it preemptively.

---

## Is there "another way" (without managing any of this)?

Yes — a few, depending on how much you want to own:

1. **Managed scheduler only (Option 2).** You still own the code, but no server
   to maintain. This is the sweet spot for most small teams.

2. **A managed ML/orchestration platform** (Airflow, Prefect, Dagster, Modal).
   Overkill for four models and one table, but worth it once you have many
   pipelines, want retries/alerting/backfills, and a dashboard. Don't reach
   for this yet.

3. **Push prediction into the DB/warehouse layer.** If you later move analytics
   to BigQuery/Snowflake, some forecasting can run as scheduled SQL/UDFs. Big
   architectural change; only relevant at much larger scale.

For where you are now — one app, one Postgres, four models — **Option 1 (cron)
to start, graduating to Option 2 (managed job) when you want it hands-off** is
the right path. You do not need anything fancier.

---

## Checklist to go live

1. `card_predictions` table created (the pipeline does this on first run, or
   run the CREATE TABLE manually if the DB user lacks CREATE rights).
2. `TCG_DATABASE_URL` points at your production Postgres, stored as a secret.
3. Schedule `run_all` to fire after your scrape completes.
4. Backend reads `model_used='ensemble'` rows for display.
5. Watch the first few runs' logs; confirm row counts grow as history builds.
EOF
echo "PRODUCTION.md written"