#!/usr/bin/env bash
# After diag2: e2e test of the moments+radii summary (ell_hi, ell_dual) at 16k, same samples.
set -u
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
while pgrep -f "queue_round4.sh|queue_diag2.sh" >/dev/null; do sleep 30; done
/home/kooki3/.local/bin/work-handoff run --project dual-channel-attention --owner claude:a0424013 --label ell-16k \
  --input $WT/src/sparse_attn.py --input $WT/src/run_e2e.py --input $WT/src/tasks.py -- \
  taskset -c 0-5 nice -n 10 /home/kooki3/projects/dual-channel-attention/.venv/bin/python $WT/src/run_e2e.py \
  --ctx 16384 --n 20 --tasks multikey,single_essay,multivalue --budgets 512 --rank 8 --prefill-chunk 4096 \
  --methods ell_hi,ell_dual,qsub2c,qsub2k --out $WT/runs/ell_16k.jsonl
echo "[queue_ell $(date '+%T')] rc=$?"
