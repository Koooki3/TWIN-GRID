#!/usr/bin/env bash
# Print a line whenever a third-party review file (anything except our researcher-response.md) is new or updated.
D=/home/kooki3/docs/projects/twin-grid-review/2026-10-16
snap() { find "$D" -maxdepth 2 -type f ! -name researcher-response.md ! -name '*.lock' ! -name '*.tmp' -printf '%P %T@\n' | sort; }
prev=$(snap)
while true; do
  cur=$(snap)
  if [ "$cur" != "$prev" ]; then
    diff <(echo "$prev") <(echo "$cur") | grep '^>' | awk '{print "review file new/updated:", $2}'
    prev=$cur
  fi
  sleep 30
done
