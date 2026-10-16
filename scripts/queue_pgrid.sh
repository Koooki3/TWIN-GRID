#!/usr/bin/env bash
# P-grid development run (exploratory; plan v1.4 §3.4 P-grid, theory L4): local Qwen2.5-1.5B, official RULER data at
# 16k (runs/ruler/pgrid_qwen1.5b_16k, prepared for unit-length development), the 7 unit tasks, frozen lines 0..19,
# read budget 512. One process per (task, block): B32 arms cobs, cobs+g2f, nf_pt1, nf_pt1+g2f; B16 arms cobs, nf_pt1
# with --layout-unit 32 (G3 matched: same fixed tokens and candidate token count as B32). Low-footprint resources;
# every run under the work-handoff ledger; resumable (ruler_pred resumes by sample_uid).
set -u
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
PY=/home/kooki3/projects/dual-channel-attention/.venv/bin/python
RULER=/home/kooki3/tools/sources/ruler
M=/home/kooki3/models/qwen2.5-1.5b-instruct
D=$WT/runs/ruler/pgrid_qwen1.5b_16k
O=$WT/runs/ruler/pgrid_pred
run() { /home/kooki3/.local/bin/work-handoff run --project dual-channel-attention --owner claude:a0424013 --label "$1" \
          --input $WT/src/ruler_pred.py --input $WT/src/sparse_attn.py -- taskset -c 0-5 nice -n 15 "${@:2}"; }
PRED=($PY $WT/src/ruler_pred.py --ruler $RULER --model $M --data_dir $D --budget 512 --strict-budget --limit 20 \
      --layout-unit 32 --vram-frac 0.6 --throttle 0.3)
rc_all=0
for T in niah_single_1 niah_single_2 niah_multikey_1 niah_multikey_2 niah_multivalue niah_multiquery vt; do
  run pgrid-b32-$T "${PRED[@]}" --block 32 --save_dir $O/b32 --task $T --methods cobs,cobs+g2f,nf_pt1,nf_pt1+g2f || rc_all=1
  run pgrid-b16-$T "${PRED[@]}" --block 16 --save_dir $O/b16 --task $T --methods cobs,nf_pt1 || rc_all=1
done
echo "[pgrid $(date '+%T')] rc_all=$rc_all"
exit $rc_all
