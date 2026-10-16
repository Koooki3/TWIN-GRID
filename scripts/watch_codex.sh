#!/usr/bin/env bash
# Print a line whenever a Codex review file (anything except my claude-response.txt) is new or updated.
D=/home/kooki3/docs/projects/attention-review/2026-10-16
snap() { find "$D" -maxdepth 1 -type f ! -name claude-response.txt ! -name 'cadence-*' ! -name '*.lock' ! -name 'review-monitor.py' ! -name '*.tmp' -printf '%f %T@\n' | sort; }
prev=$(snap)
while true; do
  cur=$(snap)
  if [ "$cur" != "$prev" ]; then
    diff <(echo "$prev") <(echo "$cur") | grep '^>' | awk '{print "codex file new/updated:", $2}'
    prev=$cur
  fi
  sleep 30
done
