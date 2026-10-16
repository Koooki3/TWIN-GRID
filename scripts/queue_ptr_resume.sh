#!/usr/bin/env bash
# Exploratory: online read-pointer successor (+ptr, KVFetch-style baseline) on the phase-1 development seeds (same args as phase 1), low
# footprint. RESUME after the 07:58Z CUDA "unknown error" (job 3e27e7ee failed at 32/80); runs last.
set -u
set -o pipefail
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
while pgrep -f "queue_mech_full.sh|queue_accept.sh|queue_f3_rest.sh" >/dev/null; do sleep 30; done
/home/kooki3/.local/bin/work-handoff run --project dual-channel-attention --owner claude:a0424013 --label ptr-dev-resume \
  --input $WT/src/sparse_attn.py --input $WT/src/run_e2e.py --input $WT/src/tasks.py -- \
  taskset -c 0-5 nice -n 15 /home/kooki3/projects/dual-channel-attention/.venv/bin/python $WT/src/run_e2e.py \
  --ctx 16384 --budgets 512 --tasks multikey,single_essay --n 40 --seed0 100000 --seed-stream \
  --rank 8 --prefill-chunk 2048 --no-stats --vram-frac 0.6 --throttle 0.3 --strict-budget \
  --methods cobs+ptr,nf_pt1+ptr,quest+ptr --out $WT/runs/ptr_dev.jsonl
rc=$?
echo "[ptr-dev-resume $(date '+%T')] rc=$rc"
exit "$rc"
