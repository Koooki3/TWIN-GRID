#!/usr/bin/env bash
# Round 5 (run under scripts/idle_gate.sh: only while the Windows desktop is idle; stopped when the user
# returns; every step resumes from its JSONL).
#  1. Neighbour-expansion remedy for block-boundary straddling (baselines in the same run for pairing).
#  2. Causal layer probe: sparse selection only in one layer group, moment (nf_pt1) vs box (nf_hi1).
set -u
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
PY=/home/kooki3/projects/dual-channel-attention/.venv/bin/python
WH=/home/kooki3/.local/bin/work-handoff
COMMON="--ctx 16384 --n 20 --tasks multikey,single_essay,multivalue --budgets 512 --rank 8 --prefill-chunk 2048"

$WH run --project dual-channel-attention --owner claude:a0424013 --label nb-16k \
  --input $WT/src/sparse_attn.py --input $WT/src/run_e2e.py --input $WT/src/tasks.py -- \
  taskset -c 0-5 nice -n 15 $PY $WT/src/run_e2e.py $COMMON \
  --methods cobs,cobs+nb,cobs+g2,nf_pt1,nf_pt1+nb,nf_pt1+g2,quest,quest+nb,quest+g2 --out $WT/runs/nb_16k.jsonl
echo "[round5 nb $(date '+%T')] rc=$?"

for G in 2-7 8-13 14-19 20-27; do
  $WH run --project dual-channel-attention --owner claude:a0424013 --label causal-L$G \
    --input $WT/src/sparse_attn.py --input $WT/src/run_e2e.py --input $WT/src/tasks.py -- \
    taskset -c 0-5 nice -n 15 $PY $WT/src/run_e2e.py $COMMON --sparse-layers $G --methods nf_pt1,nf_hi1 \
    --out $WT/runs/causal_L${G}_16k.jsonl
  echo "[round5 L$G $(date '+%T')] rc=$?"
done
