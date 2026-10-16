#!/usr/bin/env bash
# Mechanism design v2 — full diagnostic on the 63 pre-specified prompts (low-footprint mode), then acceptance.
# Order: ptr dev -> phase 2 confirmation -> this -> F3 rest (no concurrent GPU jobs). Acceptance gets the full prompt list.
set -u
set -o pipefail
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
IDS=/home/kooki3/docs/projects/attention-review/2026-10-16/hourly-mech-v2-1426/original-prompt-identities.json
while pgrep -f "queue_ptr_dev.sh|queue_phase2.sh" >/dev/null; do sleep 30; done
/home/kooki3/.local/bin/work-handoff run --project dual-channel-attention --owner claude:a0424013 --label mech-full-16k \
  --input $WT/src/sparse_attn.py --input $WT/src/run_digit_diag.py --input $WT/src/tasks.py \
  --input $WT/results/mech_prompts_full.json --input $IDS -- \
  taskset -c 0-5 nice -n 15 /home/kooki3/projects/dual-channel-attention/.venv/bin/python $WT/src/run_digit_diag.py \
  --prompts $WT/results/mech_prompts_full.json --identities $IDS --vram-frac 0.6 --throttle 0.3 --logger-check \
  --out $WT/runs/mech_full_16k.jsonl
echo "[mech-check $(date '+%T')] run rc=$?"
/home/kooki3/projects/dual-channel-attention/.venv/bin/python $WT/src/mech_check_accept.py $WT/runs/mech_full_16k.jsonl $WT/results/mech_prompts_full.json \
  > $WT/results/mech_full_accept.log 2>&1
rc=$?
echo "[mech-check $(date '+%T')] acceptance rc=$rc"
exit $rc
