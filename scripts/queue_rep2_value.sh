#!/usr/bin/env bash
# Value-criteria appendix arms (docs/VALUE_CRITERIA_ZH.md A4/A5; analysis src/rep2_value_analysis.py committed before
# this runs). After the rep2 queue (pass its background output file to wait for "[rep2 "): em (B32), quest16+g2f (B16),
# quest8 (B8), budget 512, --limit 100 on the rep2 data, U = 32; one method per process; low footprint.
set -u
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
PY=/home/kooki3/projects/dual-channel-attention/.venv/bin/python
WAIT_FOR=${1:-}
if [ -n "$WAIT_FOR" ]; then until grep -q '^\[rep2 ' "$WAIT_FOR" 2>/dev/null; do sleep 60; done; fi
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { /home/kooki3/.local/bin/work-handoff run --project dual-channel-attention --owner claude:a0424013 --label "$1" \
          --input $WT/src/ruler_pred.py --input $WT/src/sparse_attn.py -- taskset -c 0-5 nice -n 15 "${@:2}"; }
P=($PY $WT/src/ruler_pred.py --ruler /home/kooki3/tools/sources/ruler --model /home/kooki3/models/qwen2.5-1.5b-instruct \
   --data_dir $WT/runs/ruler/rep2_qwen1.5b_16k --strict-budget --layout-unit 32 --vram-frac 0.6 --throttle 0.3 \
   --budget 512 --limit 100)
rc_all=0
for T in niah_single_1 niah_single_2 niah_multikey_1 niah_multikey_2 niah_multivalue niah_multiquery vt; do
  for A in em:em:32 quest16+g2f:quest+g2f:16 quest8:quest:8; do
    IFS=: read -r L M BL <<< "$A"
    run rep2v-$L-$T "${P[@]}" --block $BL --save_dir $WT/runs/ruler/rep2_pred/b512/$L --task $T --methods $M || rc_all=1
  done
done
echo "[rep2-value $(date '+%T')] rc_all=$rc_all"
exit $rc_all
