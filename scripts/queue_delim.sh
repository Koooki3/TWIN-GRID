#!/usr/bin/env bash
# Delimiter-chunking development run (exploratory; docs/THEORY_ADDENDUM_ZH.md §5). Optionally waits until the P-grid
# multivalue rerun has finished (pass its background output file), then runs src/delim_dev.py (ruler_pred protocol
# unchanged) with methods cobs (fidelity re-run) and delim_cobs on the P-grid data: 7 unit tasks x frozen lines 0..19,
# B32, U32, budget 512. Low-footprint resources, expandable_segments; every run under the work-handoff ledger.
set -u
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
PY=/home/kooki3/projects/dual-channel-attention/.venv/bin/python
WAIT_FOR=${1:-}
if [ -n "$WAIT_FOR" ]; then until grep -q '\[pgrid-mv' "$WAIT_FOR" 2>/dev/null; do sleep 60; done; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { /home/kooki3/.local/bin/work-handoff run --project dual-channel-attention --owner claude:a0424013 --label "$1" \
          --input $WT/src/ruler_pred.py --input $WT/src/sparse_attn.py --input $WT/src/delim_attn.py \
          --input $WT/src/delim_dev.py -- taskset -c 0-5 nice -n 15 "${@:2}"; }
rc_all=0
for T in niah_single_1 niah_single_2 niah_multikey_1 niah_multikey_2 niah_multivalue niah_multiquery vt; do
  run delim-$T $PY $WT/src/delim_dev.py --ruler /home/kooki3/tools/sources/ruler \
      --model /home/kooki3/models/qwen2.5-1.5b-instruct --data_dir $WT/runs/ruler/pgrid_qwen1.5b_16k --budget 512 \
      --strict-budget --limit 20 --layout-unit 32 --vram-frac 0.6 --throttle 0.3 --block 32 \
      --save_dir $WT/runs/ruler/delim_pred --task $T --methods cobs,delim_cobs || rc_all=1
done
echo "[delim $(date '+%T')] rc_all=$rc_all"
exit $rc_all
