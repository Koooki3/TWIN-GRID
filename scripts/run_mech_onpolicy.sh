#!/usr/bin/env bash
# On-policy per-step mechanism diagnostics: decode with nf_pt1, then with nf_hi1 (budget 512).
set -u
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
for M in nf_pt1 nf_hi1; do
  /home/kooki3/.local/bin/work-handoff run --project dual-channel-attention --owner claude:a0424013 --label mech-onpolicy-$M \
    --input $WT/src/sparse_attn.py --input $WT/src/run_mech_steps.py -- \
    taskset -c 0-5 nice -n 10 /home/kooki3/projects/dual-channel-attention/.venv/bin/python $WT/src/run_mech_steps.py \
    --ctx 16384 --n 20 --budget 512 --rank 8 --method $M --out $WT/results/mech_onpolicy_${M}_16k.json
  echo "[onpolicy $M $(date '+%T')] rc=$?"
done
