#!/usr/bin/env bash
# P-grid rerun 2 for niah_multivalue at B32 only: the 4-method process OOMs at the 0.6 VRAM cap even with
# expandable_segments (4.66 GiB allocated, 21:5x). Same protocol, split into two processes with their own save dirs;
# scripts/pgrid_mv_merge.sh then archives the 1-row partial files and places the outputs where src/pgrid_dev.py reads.
set -u
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
PY=/home/kooki3/projects/dual-channel-attention/.venv/bin/python
RULER=/home/kooki3/tools/sources/ruler
M=/home/kooki3/models/qwen2.5-1.5b-instruct
D=$WT/runs/ruler/pgrid_qwen1.5b_16k
O=$WT/runs/ruler/pgrid_pred
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { /home/kooki3/.local/bin/work-handoff run --project dual-channel-attention --owner claude:a0424013 --label "$1" \
          --input $WT/src/ruler_pred.py --input $WT/src/sparse_attn.py -- taskset -c 0-5 nice -n 15 "${@:2}"; }
PRED=($PY $WT/src/ruler_pred.py --ruler $RULER --model $M --data_dir $D --budget 512 --strict-budget --limit 20 \
      --layout-unit 32 --vram-frac 0.6 --throttle 0.3 --block 32 --task niah_multivalue)
rc_all=0
run pgrid3-b32-mv-cobs "${PRED[@]}" --save_dir $O/b32_mv_cobs --methods cobs,cobs+g2f || rc_all=1
run pgrid3-b32-mv-nf "${PRED[@]}" --save_dir $O/b32_mv_nf --methods nf_pt1,nf_pt1+g2f || rc_all=1
echo "[pgrid-mv $(date '+%T')] rc_all=$rc_all"
exit $rc_all
