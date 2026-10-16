#!/usr/bin/env bash
# Print GPU memory every 15 s for N iterations (default 40), plus rows of a JSONL; stop on an error in the latest job log.
OUT=${1:-/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis/runs/force_probe_16k.jsonl}
N=${2:-40}
for i in $(seq 1 "$N"); do
  m=$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader)
  p=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader | head -1)
  echo "$(date +%T) total=$m mine=${p:-none} rows=$(wc -l < "$OUT" 2>/dev/null || echo 0)"
  L=$(ls -t /home/kooki3/.local/state/work-handoff/runs/ | head -1)
  if grep -qE "OutOfMemory|Traceback" "/home/kooki3/.local/state/work-handoff/runs/$L/stdout.log" 2>/dev/null; then
    echo "ERROR in $L"; grep -E "Error" "/home/kooki3/.local/state/work-handoff/runs/$L/stdout.log" | tail -2; break
  fi
  sleep 15
done
