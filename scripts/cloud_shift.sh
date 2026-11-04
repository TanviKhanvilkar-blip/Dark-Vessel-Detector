#!/usr/bin/env bash
# One cloud shift: restore state, record AIS continuously, and every 15 minutes process new passes, rebuild the site
# and save everything. State (databases and the website) lives on the `state` branch as a single commit that is
# replaced each time, so the repository does not grow; GitHub Pages serves the site from its docs/ folder.
set -u
MINUTES="${1:-345}"
REPO_URL="https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"
DBS="live.sqlite mumbai_results.sqlite denmark_hist.sqlite"

# never continue without the saved state: a shift that started empty would overwrite it with empty databases
git fetch -q --depth=1 origin state && git checkout -q FETCH_HEAD -- $DBS docs || { echo "could not restore state; stopping"; exit 1; }
git reset -q                                    # restored files stay in place, unstaged
[ -n "${AISSTREAM_KEY:-}" ] || echo "WARNING: AISSTREAM_KEY secret is not set, so no live AIS will be recorded this shift"

save_state() {
  rm -rf _state && mkdir _state
  # copy through SQLite's backup API: the recorder may be writing to live.sqlite right now
  python - <<'PY'
import sqlite3
for name in ("live.sqlite", "mumbai_results.sqlite", "denmark_hist.sqlite"):
    src, dst = sqlite3.connect(name, timeout=30), sqlite3.connect(f"_state/{name}")
    src.backup(dst)
    dst.close(); src.close()
PY
  cp -r docs _state/docs
  (cd _state && git init -q -b state \
    && git -c user.name="dark-vessel-bot" -c user.email="41898282+github-actions[bot]@users.noreply.github.com" add -A \
    && git -c user.name="dark-vessel-bot" -c user.email="41898282+github-actions[bot]@users.noreply.github.com" commit -q -m "State $(date -u '+%Y-%m-%d %H:%M UTC')" \
    && git push -q --force "$REPO_URL" state)
  echo "$(date -u '+%H:%M') state saved: $?"
}

cycle() {
  timeout 900 python -m dvd auto || echo "pass processing failed; next cycle retries"
  python -m dvd prune-ais --days 5 || true
  timeout 300 python -m dvd build-site || echo "site build failed; next cycle retries"
  save_state
}

(while true; do python -u -m dvd record-ais --db live.sqlite; echo "recorder exited; restarting"; sleep 10; done) &
RECORDER=$!
trap 'kill $RECORDER 2>/dev/null' EXIT

END=$((SECONDS + MINUTES * 60))
sleep 120                                       # let some live AIS arrive before the first rebuild
while [ "$SECONDS" -lt "$END" ]; do
  cycle
  LEFT=$((END - SECONDS))
  [ "$LEFT" -gt 0 ] && sleep $(( LEFT < 900 ? LEFT : 900 ))
done
kill $RECORDER 2>/dev/null
pkill -f "dvd record-ais" 2>/dev/null
sleep 6                                         # the recorder flushes at least every 5 s
cycle
