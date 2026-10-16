#!/usr/bin/env bash
# Direct-exit-code checks of src/kv_prediction_confirm.py (Codex K1):
#   development file in confirm mode (gate must refuse) -> 2
#   partial confirmation file in confirm mode (gate must refuse: keys missing) -> 2 — reads only the gate's verdict,
#     prints nothing about effects (stdout discarded)
#   failed job receipt with the dev file -> 2
#   --dev on the development file -> 0 (labelled DEV — NOT CONFIRMATORY)
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
PY=/home/kooki3/projects/dual-channel-attention/.venv/bin/python
T=/home/kooki3/.claude/jobs/a0424013/tmp
cd $WT
echo '{"status": "succeeded", "exit_code": 0}' > $T/ok_receipt.json
echo '{"status": "failed", "exit_code": 1}' > $T/bad_receipt.json
taskset -c 0-5 nice -n 15 $PY src/kv_prediction_confirm.py theory/phase2_spec.json runs/phase1_g2f_dev.jsonl $T/ok_receipt.json > /dev/null 2>&1
echo "confirm_mode_on_dev_file_rc=$?"
cp runs/phase2_g2f_confirm.jsonl $T/partial_copy.jsonl
cp runs/phase2_g2f_confirm.jsonl.manifest.json $T/partial_copy.jsonl.manifest.json
taskset -c 0-5 nice -n 15 $PY src/kv_prediction_confirm.py theory/phase2_spec.json $T/partial_copy.jsonl $T/ok_receipt.json > /dev/null 2>&1
echo "confirm_mode_on_partial_file_rc=$?"
rm -f $T/partial_copy.jsonl $T/partial_copy.jsonl.manifest.json
taskset -c 0-5 nice -n 15 $PY src/kv_prediction_confirm.py theory/phase2_spec.json runs/phase1_g2f_dev.jsonl $T/bad_receipt.json > /dev/null 2>&1
echo "failed_receipt_rc=$?"
taskset -c 0-5 nice -n 15 $PY src/kv_prediction_confirm.py --dev runs/phase1_g2f_dev.jsonl > results/kv_confirm_dev.log 2>&1
echo "dev_mode_rc=$?"
