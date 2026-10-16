#!/usr/bin/env bash
# Phase 1 (development) of prereg draft v2 — runs after the round-5d diagnostics queue finishes.
# Development seeds: seed0 = 100000, per-task streams; 40 prompts per task; strict budget invariant.
set -u
set -o pipefail
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
while pgrep -f queue_round5d.sh >/dev/null; do sleep 30; done
/home/kooki3/.local/bin/work-handoff run --project dual-channel-attention --owner claude:a0424013 --label phase1-g2f-dev \
  --input $WT/src/sparse_attn.py --input $WT/src/run_e2e.py --input $WT/src/tasks.py -- \
  taskset -c 0-5 nice -n 15 /home/kooki3/projects/dual-channel-attention/.venv/bin/python $WT/src/run_e2e.py \
  --ctx 16384 --budgets 512 --tasks multikey,single_essay --n 40 --seed0 100000 --seed-stream \
  --rank 8 --prefill-chunk 2048 --no-stats --vram-frac 0.6 --throttle 0.3 --strict-budget \
  --methods cobs,cobs+g2f,nf_pt1,nf_pt1+g2f,quest,quest+g2f --out $WT/runs/phase1_g2f_dev.jsonl
echo "[phase1 $(date '+%T')] rc=$?"
