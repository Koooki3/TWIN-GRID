#!/usr/bin/env bash
# Per-step mechanism diagnostic with the target mask restricted to the answer-value digits (dense trajectory).
set -u
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
/home/kooki3/.local/bin/work-handoff run --project dual-channel-attention --owner claude:a0424013 --label mech-value-16k \
  --input $WT/src/sparse_attn.py --input $WT/src/run_mech_steps.py -- \
  taskset -c 0-5 nice -n 10 /home/kooki3/projects/dual-channel-attention/.venv/bin/python $WT/src/run_mech_steps.py \
  --ctx 16384 --n 20 --budget 512 --rank 8 --mask value --out $WT/results/mech_value_16k.json
echo "[mech_value $(date '+%T')] rc=$?"
