#!/usr/bin/env bash
# Automatic pipeline: resume the XAUUSD download, save progress to git, and once the
# history is complete run the full research and commit the report.
# Usage: scripts/auto_run.sh [max_download_minutes]
set -uo pipefail
cd "$(dirname "$0")/.."
MAX_MIN="${1:-100}"

timeout "${MAX_MIN}m" python3 -u -c "
from xauusd_ai.dukascopy import download_hours
download_hours('2015-01-01')
" 2>&1 | tee -a data/download.log
status=${PIPESTATUS[0]}

git add -f data/cache >/dev/null 2>&1
git commit -q -m "Cache XAUUSD hourly data from Dukascopy" >/dev/null 2>&1 || true

if [ "$status" -eq 0 ] && ! grep -q "skipped" <(tail -n 200 data/download.log | sed -n '/hourly:/,$p'); then
  echo "DOWNLOAD COMPLETE - running research"
  python3 -u -m xauusd_ai.auto --out reports/latest 2>&1 | tee data/auto.log
  git add reports && git commit -q -m "Update automatic XAUUSD backtest report" || true
  echo "RESEARCH DONE"
else
  echo "DOWNLOAD INCOMPLETE (exit $status) - run again to resume"
fi
