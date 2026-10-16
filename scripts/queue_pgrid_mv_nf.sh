#!/usr/bin/env bash
# P-grid rerun 3 (B32 niah_multivalue, nf_pt1 arms): nf_pt1+g2f OOMed even in a 2-method process (job 6069fa18,
# >= 4.65 GiB at the 0.6 cap). Semantics-preserving workaround only (reviewer): one method per process, same protocol.
# 2026-10-17 00:2x: the user is away and asked for full speed, so the resource-only flags are --vram-frac 0.75
# --throttle 0 (they cap allocation / add sleeps; they do not change any computation); nf_pt1 runs in its own process. Each method keeps its own save dir (the manifest records argv),
# so scripts/pgrid_mv_merge.sh's manifest check compares each with the original apart from --methods/--save_dir.
set -u
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
PY=/home/kooki3/projects/dual-channel-attention/.venv/bin/python
O=$WT/runs/ruler/pgrid_pred
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { /home/kooki3/.local/bin/work-handoff run --project dual-channel-attention --owner claude:a0424013 --label "$1" \
          --input $WT/src/ruler_pred.py --input $WT/src/sparse_attn.py -- taskset -c 0-5 nice -n 15 "${@:2}"; }
PRED=($PY $WT/src/ruler_pred.py --ruler /home/kooki3/tools/sources/ruler --model /home/kooki3/models/qwen2.5-1.5b-instruct \
      --data_dir $WT/runs/ruler/pgrid_qwen1.5b_16k --budget 512 --strict-budget --limit 20 --layout-unit 32 \
      --vram-frac 0.75 --throttle 0 --block 32 --task niah_multivalue)
rc_all=0
run pgrid4-b32-mv-nfg "${PRED[@]}" --save_dir $O/b32_mv_nfg --methods nf_pt1+g2f || rc_all=1
run pgrid4-b32-mv-nf1 "${PRED[@]}" --save_dir $O/b32_mv_nf1 --methods nf_pt1 || rc_all=1
echo "[pgrid-mv-nf $(date '+%T')] rc_all=$rc_all"
exit $rc_all
