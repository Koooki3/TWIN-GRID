#!/usr/bin/env bash
# Remaining F3 configurations with chunked prefill (2048), low-footprint mode, after the mechanism check.
# f3-ch0-r16-perhead (one-shot prefill + per-head subspace, rank 16) OOM'd under --vram-frac 0.6; it is deferred to a
# time when the user is away (needs a higher VRAM cap) and is NOT run here.
set -u
set -o pipefail
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
PY=/home/kooki3/projects/dual-channel-attention/.venv/bin/python
while pgrep -f "queue_mech_full.sh|queue_accept.sh" >/dev/null; do sleep 30; done
for R in 8 16; do
  for WP in shared perhead; do
    FLAG=""; [ "$WP" = perhead ] && FLAG="--w-per-head"
    /home/kooki3/.local/bin/work-handoff run --project dual-channel-attention --owner claude:a0424013 --label f3-ch2048-r$R-$WP \
      --input $WT/src/sparse_attn.py --input $WT/src/run_e2e.py --input $WT/src/tasks.py -- \
      taskset -c 0-5 nice -n 15 $PY $WT/src/run_e2e.py --ctx 16384 --budgets 512 --no-stats --vram-frac 0.6 \
      --throttle 0.3 --tasks multikey,single_essay --n 20 --rank $R --prefill-chunk 2048 $FLAG --methods cobs,quest \
      --out $WT/runs/f3_ch2048_r${R}_${WP}.jsonl
    echo "[f3-rest ch2048 r$R $WP $(date '+%T')] rc=$?"
  done
done
