#!/usr/bin/env bash
# Box-summary P-grid development run (exploratory; reviewer batch 10, E4): after the variants run (pass its background
# output file to wait for "[variants"), on the P-grid data (7 unit tasks x frozen lines 0..19, U32, budget 512):
# quest at B16 (G3-matched single grid) and quest+g2f at B32. ruler_pred unchanged; full speed (user away).
set -u
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
PY=/home/kooki3/projects/dual-channel-attention/.venv/bin/python
WAIT_FOR=${1:-}
if [ -n "$WAIT_FOR" ]; then until grep -q '\[variants' "$WAIT_FOR" 2>/dev/null; do sleep 60; done; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { /home/kooki3/.local/bin/work-handoff run --project dual-channel-attention --owner claude:a0424013 --label "$1" \
          --input $WT/src/ruler_pred.py --input $WT/src/sparse_attn.py -- taskset -c 0-5 nice -n 15 "${@:2}"; }
P=($PY $WT/src/ruler_pred.py --ruler /home/kooki3/tools/sources/ruler --model /home/kooki3/models/qwen2.5-1.5b-instruct \
   --data_dir $WT/runs/ruler/pgrid_qwen1.5b_16k --budget 512 --strict-budget --limit 20 --layout-unit 32 \
   --vram-frac 0.75 --throttle 0)
rc_all=0
for T in niah_single_1 niah_single_2 niah_multikey_1 niah_multikey_2 niah_multivalue niah_multiquery vt; do
  run qpg-b16-$T "${P[@]}" --block 16 --save_dir $WT/runs/ruler/quest_pgrid/b16 --task $T --methods quest || rc_all=1
  run qpg-b32-$T "${P[@]}" --block 32 --save_dir $WT/runs/ruler/quest_pgrid/b32 --task $T --methods quest+g2f || rc_all=1
done
echo "[quest-pgrid $(date '+%T')] rc_all=$rc_all"
exit $rc_all
