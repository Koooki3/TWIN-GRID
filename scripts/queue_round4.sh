#!/usr/bin/env bash
# Round-4 GPU queue: wait for the diagnostics job, then the interval-summary e2e comparison,
# then the equal-memory Quest (16-token blocks) control. Each step goes through work-handoff run.
set -u
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
PY=/home/kooki3/projects/dual-channel-attention/.venv/bin/python
WH=/home/kooki3/.local/bin/work-handoff
OWNER=claude:a0424013

while pgrep -f "run_diag.py --ctx 16384" >/dev/null; do sleep 20; done
echo "[queue $(date '+%T')] diag finished"

$WH run --project dual-channel-attention --owner $OWNER --label nf-16k \
  --input $WT/src/sparse_attn.py --input $WT/src/run_e2e.py --input $WT/src/tasks.py -- \
  taskset -c 0-5 nice -n 10 $PY $WT/src/run_e2e.py --ctx 16384 --n 20 --tasks multikey,single_essay,multivalue \
  --budgets 256,512 --rank 8 --prefill-chunk 4096 \
  --methods quest,cobs,sum_dual,nf_lo1,nf_hi1,nf_pt1,nf_dual1,nf_hi2,nf_dual2 \
  --out $WT/runs/nf_16k.jsonl
echo "[queue $(date '+%T')] nf-16k finished rc=$?"

$WH run --project dual-channel-attention --owner $OWNER --label quest-b16-16k \
  --input $WT/src/sparse_attn.py --input $WT/src/run_e2e.py --input $WT/src/tasks.py -- \
  taskset -c 0-5 nice -n 10 $PY $WT/src/run_e2e.py --ctx 16384 --n 20 --tasks multikey,single_essay,multivalue \
  --budgets 256,512 --rank 8 --block 16 --prefill-chunk 4096 --methods quest \
  --out $WT/runs/quest_b16_16k.jsonl
echo "[queue $(date '+%T')] quest-b16 finished rc=$?"
