#!/usr/bin/env bash
# RULER adapter GPU acceptance on the local Qwen2.5-1.5B (plan §3.2; Codex R1/R3, batch-1 A2), after the mechanism run.
# 1) ruler_pred dense on the acceptance data (3 tasks x 5); 2) same-family order runs on cwe (long generation):
#    for (cobs, cobs+g2f) and (quest, quest+g2f): A→B, B→A and each single arm, with --local 16 so that generated
#    tokens reach selectable candidate blocks (cache-isolation test only, not an accuracy setting); 3) accept_adapter.py.
# Low-footprint resources; every run under the work-handoff ledger; exit code propagated.
set -u
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
PY=/home/kooki3/projects/dual-channel-attention/.venv/bin/python
RULER=/home/kooki3/tools/sources/ruler
M=/home/kooki3/models/qwen2.5-1.5b-instruct
D=$WT/runs/ruler/accept2_qwen1.5b_16k
O=$WT/runs/ruler/accept3_pred
run() { /home/kooki3/.local/bin/work-handoff run --project dual-channel-attention --owner claude:a0424013 --label "$1" \
          --input $WT/src/ruler_pred.py --input $WT/src/sparse_attn.py -- taskset -c 0-5 nice -n 15 "${@:2}"; }
# the prediction command as an array (a shell function cannot be exec'd by work-handoff: that bug made every run
# exit 127 in the first attempt, 19:35)
PRED=($PY $WT/src/ruler_pred.py --ruler $RULER --model $M --data_dir $D --budget 512 --block 32 --strict-budget \
      --vram-frac 0.6 --throttle 0.3)
rc_all=0
for T in niah_single_2 cwe qa_1; do
  run accept-dense-$T "${PRED[@]}" --save_dir $O/dense --task $T --methods dense || rc_all=1
done
for P in "cobs cobs+g2f" "quest quest+g2f"; do
  set -- $P; A=$1; B=$2; TAG=${A}__${B}
  run accept-ab-$TAG "${PRED[@]}" --save_dir $O/order_ab_$TAG --task cwe --methods $A,$B --local 16 || rc_all=1
  run accept-ba-$TAG "${PRED[@]}" --save_dir $O/order_ba_$TAG --task cwe --methods $B,$A --local 16 || rc_all=1
  for X in $A $B; do
    # always run: ruler_pred resumes by sample_uid (a directory left by an interrupted run is not "done")
    run accept-single-$X "${PRED[@]}" --save_dir $O/single_$X --task cwe --methods $X --local 16 || rc_all=1
  done
done
run accept-compare $PY $WT/cluster/accept_adapter.py --ruler $RULER --model $M --data_dir $D \
    --tasks niah_single_2,cwe,qa_1 --pred_root $O --order_task cwe --pairs cobs:cobs+g2f,quest:quest+g2f \
    --budget 512 --report $WT/results/ruler_adapter_acceptance.json > $WT/results/ruler_adapter_acceptance.log 2>&1
rc=$?; echo "[accept $(date '+%T')] compare rc=$rc"; [ $rc -ne 0 ] && rc_all=$rc
echo "[accept $(date '+%T')] rc_all=$rc_all"
exit $rc_all
