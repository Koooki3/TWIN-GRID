#!/usr/bin/env bash
# Variants development run (exploratory; reviewer batch 7, user follow-up): after the delim run and P-grid rerun 3
# (pass their background output file to wait for "[pgrid-mv-nf"), on the P-grid data (7 unit tasks x frozen lines
# 0..19, B32, U32, budget 512): process A = cobs+nb, cobs+nbr, cobs+nbh; process B = mean, mean+g2f, quest.
# Same protocol as P-grid (ruler_pred main unchanged via src/variants_dev.py); low-footprint resources,
# expandable_segments; every run under the work-handoff ledger.
set -u
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
PY=/home/kooki3/projects/dual-channel-attention/.venv/bin/python
WAIT_FOR=${1:-}
if [ -n "$WAIT_FOR" ]; then until grep -q '\[pgrid-mv-nf' "$WAIT_FOR" 2>/dev/null; do sleep 60; done; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { /home/kooki3/.local/bin/work-handoff run --project dual-channel-attention --owner claude:a0424013 --label "$1" \
          --input $WT/src/ruler_pred.py --input $WT/src/sparse_attn.py --input $WT/src/nbh_attn.py \
          --input $WT/src/variants_dev.py -- taskset -c 0-5 nice -n 15 "${@:2}"; }
V=($PY $WT/src/variants_dev.py --ruler /home/kooki3/tools/sources/ruler --model /home/kooki3/models/qwen2.5-1.5b-instruct \
   --data_dir $WT/runs/ruler/pgrid_qwen1.5b_16k --budget 512 --strict-budget --limit 20 --layout-unit 32 \
   --vram-frac 0.75 --throttle 0 --block 32)   # full speed: user away (2026-10-17); resource-only flags
rc_all=0
for T in niah_single_1 niah_single_2 niah_multikey_1 niah_multikey_2 niah_multivalue niah_multiquery vt; do
  run var-nb-$T "${V[@]}" --save_dir $WT/runs/ruler/variants_pred/nb --task $T --methods cobs+nb,cobs+nbr,cobs+nbh || rc_all=1
  run var-sum-$T "${V[@]}" --save_dir $WT/runs/ruler/variants_pred/sum --task $T --methods mean,mean+g2f,quest || rc_all=1
done
echo "[variants $(date '+%T')] rc_all=$rc_all"
exit $rc_all
