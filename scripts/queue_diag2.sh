#!/usr/bin/env bash
# After the round-4 queue: outlier-criterion diagnostics (budget 512 only).
set -u
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
while pgrep -f queue_round4.sh >/dev/null; do sleep 30; done
/home/kooki3/.local/bin/work-handoff run --project dual-channel-attention --owner claude:a0424013 --label diag2-16k \
  --input $WT/src/sparse_attn.py --input $WT/src/run_diag.py -- \
  taskset -c 0-5 nice -n 10 /home/kooki3/projects/dual-channel-attention/.venv/bin/python $WT/src/run_diag.py \
  --ctx 16384 --n 20 --budgets 512 --rank 8 --out $WT/results/diag2_16k.json
echo "[queue_diag2 $(date '+%T')] rc=$?"
