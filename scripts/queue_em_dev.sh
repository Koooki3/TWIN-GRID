#!/usr/bin/env bash
# Exploratory: Edmundson-Madansky scorer (em, em+g2f) on the phase-1 development seeds (same args as phase 1), low
# footprint. Serial order (fixes the f3/mech waiter deadlock): mech check -> this -> remaining F3.
set -u
set -o pipefail
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
while pgrep -f "queue_mech_check.sh" >/dev/null; do sleep 30; done
/home/kooki3/.local/bin/work-handoff run --project dual-channel-attention --owner claude:a0424013 --label em-dev \
  --input $WT/src/sparse_attn.py --input $WT/src/run_e2e.py --input $WT/src/tasks.py -- \
  taskset -c 0-5 nice -n 15 /home/kooki3/projects/dual-channel-attention/.venv/bin/python $WT/src/run_e2e.py \
  --ctx 16384 --budgets 512 --tasks multikey,single_essay --n 40 --seed0 100000 --seed-stream \
  --rank 8 --prefill-chunk 2048 --no-stats --vram-frac 0.6 --throttle 0.3 --strict-budget \
  --methods em,em+g2f --out $WT/runs/em_dev.jsonl
echo "[em-dev $(date '+%T')] rc=$?"
