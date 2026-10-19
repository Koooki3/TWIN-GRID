#!/usr/bin/env bash
# After the rep2 queue ends (wait for "[rep2 " in its background output):
#   1. re-run every failed rep2 job with the identical command, at most 3 attempts per job (prereg v3 convention;
#      ruler_pred resumes by sample_uid and refuses a different manifest; no new seed, no deleted sample);
#   2. run the value-appendix queue, then re-run its failed jobs the same way (at most 3 attempts);
#   3. run the completeness gate src/rep2_gate.py --value and record its rc (results/rep2_gate.log).
# Every attempt runs under the work-handoff ledger, so every receipt is kept. Strictly serial: one GPU job at a time.
# Known failure: rep2-512-mean+g2f-niah_multikey_2 (job 8de48c96, "CUDA error: unknown error", 14/100 rows written).
set -u
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
PY=/home/kooki3/projects/dual-channel-attention/.venv/bin/python
REP2_OUT=$1
VAL_OUT=$WT/results/rep2_value_queue.log
until grep -q '^\[rep2 ' "$REP2_OUT" 2>/dev/null; do sleep 60; done
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { /home/kooki3/.local/bin/work-handoff run --project dual-channel-attention --owner claude:a0424013 --label "$1" \
          --input $WT/src/ruler_pred.py --input $WT/src/sparse_attn.py -- taskset -c 0-5 nice -n 15 "${@:2}"; }
P=($PY $WT/src/ruler_pred.py --ruler /home/kooki3/tools/sources/ruler --model /home/kooki3/models/qwen2.5-1.5b-instruct \
   --data_dir $WT/runs/ruler/rep2_qwen1.5b_16k --strict-budget --layout-unit 32 --vram-frac 0.6 --throttle 0.3)
declare -A METHOD=([dense]=dense [cobs]=cobs [cobs+g2f]=cobs+g2f [mean]=mean [mean+g2f]=mean+g2f [quest32]=quest
                   [quest32+g2f]=quest+g2f [cobs16]=cobs [quest16]=quest [em]=em [quest16+g2f]=quest+g2f [quest8]=quest)
declare -A BLOCK=([cobs16]=16 [quest16]=16 [quest16+g2f]=16 [quest8]=8)
failed() {   # labels whose last recorded status is not succeeded
  $PY - "$1" <<'EOF'
import json, sys
st = {}
for l in open(sys.argv[1]):
    l = l.strip()
    if l.startswith('{"job":'):
        j = json.loads(l)["job"]
        st[j["label"]] = j["status"]
print(" ".join(k for k, v in st.items() if v != "succeeded"))
EOF
}
retry_all() {   # $1 = queue output file; labels rep2-<budget>-<arm>-<task> or rep2v-<arm>-<task>
  local rc=0
  for LBL in $(failed "$1"); do
    case "$LBL" in
      rep2v-*) BUD=512; REST=${LBL#rep2v-};;
      *) BUD=$(echo "$LBL" | cut -d- -f2); REST=${LBL#rep2-$BUD-};;
    esac
    case "$REST" in *-vt) T=vt; ARM=${REST%-vt};; *) T=niah_${REST##*-niah_}; ARM=${REST%-niah_*};; esac
    LIM=100; [ "$BUD" = 1024 ] && LIM=60
    ok=0
    for k in 1 2 3; do
      echo "[retry] $LBL attempt $k -> arm $ARM task $T budget $BUD"
      if run retry$k-$LBL "${P[@]}" --budget $BUD --limit $LIM --block ${BLOCK[$ARM]:-32} \
           --save_dir $WT/runs/ruler/rep2_pred/b$BUD/$ARM --task $T --methods ${METHOD[$ARM]}; then ok=1; break; fi
    done
    [ $ok = 1 ] || { echo "[retry] $LBL still failing after 3 attempts -> incomplete"; rc=1; }
  done
  return $rc
}
retry_all "$REP2_OUT"; echo "[rep2-retry $(date '+%T')] rc=$?"
bash $WT/scripts/queue_rep2_value.sh > "$VAL_OUT" 2>&1; echo "[rep2-value-queue $(date '+%T')] rc=$?"
retry_all "$VAL_OUT"; echo "[value-retry $(date '+%T')] rc=$?"
$PY $WT/src/rep2_gate.py $WT --value > $WT/results/rep2_gate.log 2>&1; G=$?
echo "rc=$G" >> $WT/results/rep2_gate.log
echo "[rep2-gate $(date '+%T')] rc=$G"
