#!/usr/bin/env bash
# Re-run only step 3 of scripts/queue_accept.sh (the comparator) on the existing predictions. The 20:01 attempt failed
# before any comparison: RULER's model_wrappers.py imports `requests`, which the project venv lacked (installed 20:10).
set -u
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
PY=/home/kooki3/projects/dual-channel-attention/.venv/bin/python
RULER=/home/kooki3/tools/sources/ruler
M=/home/kooki3/models/qwen2.5-1.5b-instruct
D=$WT/runs/ruler/accept2_qwen1.5b_16k
O=$WT/runs/ruler/accept3_pred
/home/kooki3/.local/bin/work-handoff run --project dual-channel-attention --owner claude:a0424013 --label accept-compare-2 \
    --input $WT/src/ruler_pred.py --input $WT/src/sparse_attn.py --input $WT/cluster/accept_adapter.py -- \
    taskset -c 0-5 nice -n 15 $PY $WT/cluster/accept_adapter.py --ruler $RULER --model $M --data_dir $D \
    --tasks niah_single_2,cwe,qa_1 --pred_root $O --order_task cwe --pairs cobs:cobs+g2f,quest:quest+g2f \
    --budget 512 --report $WT/results/ruler_adapter_acceptance.json > $WT/results/ruler_adapter_acceptance.log 2>&1
rc=$?; echo "[accept $(date '+%T')] compare rc=$rc"
exit $rc
