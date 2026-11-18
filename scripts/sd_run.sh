#!/usr/bin/env bash
# Resumable: download Dukascopy M1 data from START, then backtest the supply/demand strategy.
# Usage: scripts/sd_run.sh [start_date] [max_download_minutes]
set -uo pipefail
cd "$(dirname "$0")/.."
START="${1:-2025-11-01}"
MAX_MIN="${2:-110}"

timeout "${MAX_MIN}m" python3 -u -c "
from xauusd_ai.dukascopy import download_minutes
download_minutes('$START')
" > data/sd_download.log 2>&1
status=$?
tail -3 data/sd_download.log

git add -f data/cache >/dev/null 2>&1
git commit -q -m "Cache XAUUSD minute data from Dukascopy" >/dev/null 2>&1 || true

if [ "$status" -eq 0 ] && ! grep -q "skipped" data/sd_download.log; then
  echo "DOWNLOAD COMPLETE - running strategy backtest"
  python3 -u -m xauusd_ai.sd_strategy --start "$START" 2>&1 | tee data/sd_run.log
  git add reports && git commit -q -m "Update supply/demand strategy backtest report" || true
  echo "SD BACKTEST DONE"
else
  echo "DOWNLOAD INCOMPLETE (exit $status) - run again to resume"
fi
