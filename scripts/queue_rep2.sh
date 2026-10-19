#!/usr/bin/env bash
# rep2 replication run (development; reviewer batch 12; analysis src/rep2_analysis.py committed before this runs).
# Budget 512: 9 arms x 7 unit tasks, --limit 100 (analysis uses lines 20..99; lines 0..19 are a determinism check
# against round 1). Then R7: budget 1024, 5 arms x 7 tasks, --limit 60 (analysis uses lines 20..59).
# One method per process; low-footprint resources (the user is using the machine): --vram-frac 0.6 --throttle 0.3.
set -u
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
PY=/home/kooki3/projects/dual-channel-attention/.venv/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { /home/kooki3/.local/bin/work-handoff run --project dual-channel-attention --owner claude:a0424013 --label "$1" \
          --input $WT/src/ruler_pred.py --input $WT/src/sparse_attn.py -- taskset -c 0-5 nice -n 15 "${@:2}"; }
P=($PY $WT/src/ruler_pred.py --ruler /home/kooki3/tools/sources/ruler --model /home/kooki3/models/qwen2.5-1.5b-instruct \
   --data_dir $WT/runs/ruler/rep2_qwen1.5b_16k --strict-budget --layout-unit 32 --vram-frac 0.6 --throttle 0.3)
# arm label : method : block
ARMS512="dense:dense:32 cobs:cobs:32 cobs+g2f:cobs+g2f:32 mean:mean:32 mean+g2f:mean+g2f:32 quest32:quest:32 quest32+g2f:quest+g2f:32 cobs16:cobs:16 quest16:quest:16"
ARMS1024="cobs:cobs:32 cobs+g2f:cobs+g2f:32 cobs16:cobs:16 quest32:quest:32 mean+g2f:mean+g2f:32"
rc_all=0
for T in niah_single_1 niah_single_2 niah_multikey_1 niah_multikey_2 niah_multivalue niah_multiquery vt; do
  for A in $ARMS512; do
    IFS=: read -r L M BL <<< "$A"
    run rep2-512-$L-$T "${P[@]}" --budget 512 --limit 100 --block $BL --save_dir $WT/runs/ruler/rep2_pred/b512/$L \
        --task $T --methods $M || rc_all=1
  done
done
echo "[rep2-512 $(date '+%T')] rc_all=$rc_all"
for T in niah_single_1 niah_single_2 niah_multikey_1 niah_multikey_2 niah_multivalue niah_multiquery vt; do
  for A in $ARMS1024; do
    IFS=: read -r L M BL <<< "$A"
    run rep2-1024-$L-$T "${P[@]}" --budget 1024 --limit 60 --block $BL --save_dir $WT/runs/ruler/rep2_pred/b1024/$L \
        --task $T --methods $M || rc_all=1
  done
done
echo "[rep2 $(date '+%T')] rc_all=$rc_all"
exit $rc_all
