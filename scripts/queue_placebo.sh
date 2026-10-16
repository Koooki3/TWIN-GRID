#!/usr/bin/env bash
# After orchestrate_round5c.sh: placebo grid-shift arm (offset keeps the straddling status; Codex G3).
set -u
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
while pgrep -f orchestrate_round5c.sh >/dev/null; do sleep 30; done
/home/kooki3/.local/bin/work-handoff run --project dual-channel-attention --owner claude:a0424013 --label shift-placebo-16k \
  --input $WT/src/sparse_attn.py --input $WT/src/run_e2e.py --input $WT/src/tasks.py -- \
  taskset -c 0-5 nice -n 15 /home/kooki3/projects/dual-channel-attention/.venv/bin/python $WT/src/run_e2e.py \
  --ctx 16384 --budgets 512 --tasks multikey,single_essay --n 20 --rank 8 --prefill-chunk 2048 \
  --shift-intervention --shift-mode placebo --methods cobs,nf_pt1,quest --out $WT/runs/shift_placebo_16k.jsonl
echo "[placebo $(date '+%T')] rc=$?"
