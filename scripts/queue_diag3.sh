#!/usr/bin/env bash
# After the ell/qsub e2e run: mechanism diagnostics (retrieval-weighted oracle capture, needle-block hit).
set -u
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
while pgrep -f "runs/ell_16k" >/dev/null; do sleep 20; done
/home/kooki3/.local/bin/work-handoff run --project dual-channel-attention --owner claude:a0424013 --label diag3-16k \
  --input $WT/src/sparse_attn.py --input $WT/src/run_diag.py -- \
  taskset -c 0-5 nice -n 10 /home/kooki3/projects/dual-channel-attention/.venv/bin/python $WT/src/run_diag.py \
  --ctx 16384 --n 20 --budgets 256,512 --rank 8 --out $WT/results/diag3_16k.json
echo "[queue_diag3 $(date '+%T')] rc=$?"
