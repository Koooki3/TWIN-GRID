#!/usr/bin/env bash
# Resume rep2 + R7 + value appendix after the RECAP GPU window (2026-10-18), at full speed where possible (user asleep:
# "你可以全速去跑实验了"). ruler_pred refuses to resume a (save_dir, task) whose manifest records different argv, so:
#   - a (arm, task, budget) that already has a manifest keeps the resource flags recorded there (low footprint);
#   - one without a manifest starts at full speed (--vram-frac 0.75 --throttle 0).
# Resource flags only cap allocation / add sleeps; they do not change computation (reviewer, rerun 3). Same order and
# commands as queue_rep2.sh / queue_rep2_value.sh otherwise; failed jobs retried up to 3 attempts; gate last.
set -u
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
PY=/home/kooki3/projects/dual-channel-attention/.venv/bin/python
OUT=$WT/results/rep2_resume_queue.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { /home/kooki3/.local/bin/work-handoff run --project dual-channel-attention --owner claude:a0424013 --label "$1" \
          --input $WT/src/ruler_pred.py --input $WT/src/sparse_attn.py -- taskset -c 0-5 nice -n 15 "${@:2}"; }
BASE=($PY $WT/src/ruler_pred.py --ruler /home/kooki3/tools/sources/ruler --model /home/kooki3/models/qwen2.5-1.5b-instruct
      --data_dir $WT/runs/ruler/rep2_qwen1.5b_16k --strict-budget --layout-unit 32)
res_flags() {   # $1 = save dir, $2 = task -> resource flags (recorded ones if a manifest exists, else full speed)
  local man=$1/manifest.$2.json
  if [ -f "$man" ]; then
    $PY -c "import json,sys; a=json.load(open(sys.argv[1]))['argv']; print('--vram-frac', a[a.index('--vram-frac')+1], '--throttle', a[a.index('--throttle')+1])" "$man"
  else
    echo "--vram-frac 0.75 --throttle 0"
  fi
}
job() {   # label budget limit block arm method task
  local D=$WT/runs/ruler/rep2_pred/b$2/$5
  local R; R=$(res_flags "$D" "$7")
  local ok=0
  for k in 1 2 3; do
    if run "$1" "${BASE[@]}" $R --budget $2 --limit $3 --block $4 --save_dir $D --task $7 --methods $6; then ok=1; break; fi
    echo "[resume] $1 attempt $k failed"
  done
  [ $ok = 1 ] || echo "[resume] $1 still failing after 3 attempts -> incomplete"
}
TASKS="niah_single_1 niah_single_2 niah_multikey_1 niah_multikey_2 niah_multivalue niah_multiquery vt"
{
for T in $TASKS; do
  for A in dense:dense:32 cobs:cobs:32 cobs+g2f:cobs+g2f:32 mean:mean:32 mean+g2f:mean+g2f:32 quest32:quest:32 \
           quest32+g2f:quest+g2f:32 cobs16:cobs:16 quest16:quest:16; do
    IFS=: read -r L M BL <<< "$A"; job rep2-512-$L-$T 512 100 $BL $L $M $T
  done
done
echo "[rep2-512 $(date '+%T')] done"
for T in $TASKS; do
  for A in cobs:cobs:32 cobs+g2f:cobs+g2f:32 cobs16:cobs:16 quest32:quest:32 mean+g2f:mean+g2f:32; do
    IFS=: read -r L M BL <<< "$A"; job rep2-1024-$L-$T 1024 60 $BL $L $M $T
  done
done
echo "[rep2-1024 $(date '+%T')] done"
for T in $TASKS; do
  for A in em:em:32 quest16+g2f:quest+g2f:16 quest8:quest:8; do
    IFS=: read -r L M BL <<< "$A"; job rep2v-$L-$T 512 100 $BL $L $M $T
  done
done
echo "[rep2-value $(date '+%T')] done"
$PY $WT/src/rep2_gate.py $WT --value > $WT/results/rep2_gate.log 2>&1; G=$?
echo "rc=$G" >> $WT/results/rep2_gate.log
echo "[rep2-gate $(date '+%T')] rc=$G"
} 2>&1 | tee -a "$OUT"
