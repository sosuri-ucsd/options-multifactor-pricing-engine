# Scheduling the pipeline

`main.py` should run once per trading day during market hours (equity
options: 9:30-16:00 America/New_York, weekdays). Pick whichever of these
fits how you're actually running this -- none of them are wired up
automatically; each is a copy-paste starting point.

## Option 1: cron

Runs at 9:45am ET on weekdays (giving the market 15 minutes to settle
after the open). Adjust the `TZ` line if your cron daemon's host isn't
already in America/New_York.

```cron
TZ=America/New_York
45 9 * * 1-5 cd /path/to/options-multifactor-pricing-engine && /path/to/venv/bin/python main.py >> logs/cron.out 2>&1
```

## Option 2: systemd timer

`/etc/systemd/system/options-pricing-engine.service`:

```ini
[Unit]
Description=Options multi-factor pricing engine, one run

[Service]
Type=oneshot
WorkingDirectory=/path/to/options-multifactor-pricing-engine
ExecStart=/path/to/venv/bin/python main.py
```

`/etc/systemd/system/options-pricing-engine.timer`:

```ini
[Unit]
Description=Run the options pricing engine at 9:45am ET on weekdays

[Timer]
OnCalendar=Mon-Fri 09:45 America/New_York
Persistent=true

[Install]
WantedBy=timers.target
```

Enable with `systemctl enable --now options-pricing-engine.timer`.

## Option 3: GitHub Actions

```yaml
name: Run pricing engine
on:
  schedule:
    - cron: "45 13 * * 1-5"  # 9:45am ET = 13:45 UTC (EST); adjust for DST
  workflow_dispatch: {}
jobs:
  run:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r options-multifactor-pricing-engine/requirements.txt
      - run: python options-multifactor-pricing-engine/main.py
        env:
          POLYGON_API_KEY: ${{ secrets.POLYGON_API_KEY }}
          TRADIER_API_KEY: ${{ secrets.TRADIER_API_KEY }}
          TRADIER_ACCOUNT_ID: ${{ secrets.TRADIER_ACCOUNT_ID }}
          FRED_API_KEY: ${{ secrets.FRED_API_KEY }}
          ALERT_SLACK_WEBHOOK_URL: ${{ secrets.ALERT_SLACK_WEBHOOK_URL }}
```

GitHub Actions cron is UTC and doesn't observe daylight saving time, so the
offset from ET drifts by an hour twice a year -- either accept the drift or
maintain two cron entries (one for EST, one for EDT) if that matters.

Not committed as an actual `.github/workflows/` file in this repo: this
project lives in a subdirectory of an existing repo without its own GitHub
remote, so wiring up a real workflow file is a decision for wherever this
ends up actually deployed, not something to do silently as part of scaffolding.

## Logging and alerting

Whichever option is used, `main.py` should call
`deployment.logging_setup.configure_logging()` once at startup, and wrap
its top-level run in a try/except that calls
`deployment.alerting.alert_unhandled_exception(exc)` before re-raising --
see `main.py`.
