@echo off
REM Tier-3 scheduled research regression for the Homerun trading harness.
REM Runs rolling-window backtests over the candidate strategies via the
REM backtest job queue, aggregates results across windows, and reports a
REM promotion recommendation. Deterministic, no LLM. Promotion updates stay
REM OFF until explicitly enabled in homerun_caretaker_research_policy.json —
REM this tier is observation-only by default. Scheduled every 4h by Windows
REM Task Scheduler (one cycle = max_runs_per_cycle backtests, each may take
REM up to poll_timeout_seconds — do not schedule tighter than 2h).
cd /d "%~dp0\..\.."
if not exist output\caretaker mkdir output\caretaker
backend\venv\Scripts\python.exe scripts\monitoring\homerun_caretaker.py research ^
  --policy scripts\monitoring\homerun_caretaker_research_policy.json ^
  --report-dir output\caretaker >> output\caretaker\research_cron.log 2>&1
