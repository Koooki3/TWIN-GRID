#!/usr/bin/env bash
# Per-step mechanism diagnostic (budget 512) through work-handoff run.
set -u
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
/home/kooki3/.local/bin/work-handoff run --project dual-channel-attention --owner claude:a0424013 --label mech-steps-16k \
  --input $WT/src/sparse_attn.py --input $WT/src/run_mech_steps.py -- \
  taskset -c 0-5 nice -n 10 /home/kooki3/projects/dual-channel-attention/.venv/bin/python $WT/src/run_mech_steps.py \
  --ctx 16384 --n 20 --budget 512 --rank 8 --out $WT/results/mech_steps_16k.json
echo "[run_mech $(date '+%T')] rc=$?"
