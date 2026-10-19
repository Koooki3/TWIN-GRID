#!/usr/bin/env bash
# Stop the job currently running under the given queue's output (for a GPU window handed to another project, e.g.
# RECAP 2026-10-17). The queue scripts themselves must be stopped first so they start nothing new. ruler_pred resumes
# by sample_uid, so at most the sample in progress is lost; the ledger records the job as interrupted.
# Usage: stop_current_job.sh QUEUE_OUTPUT_FILE REASON
set -u
OUT=$1; REASON=${2:-gpu window for another project}
JOB=$(grep -o '"job_id": "[0-9a-f]*", "log": "[^"]*", "status": "starting"' "$OUT" | tail -1 | cut -d'"' -f4)
[ -n "$JOB" ] || { echo "no starting job found in $OUT"; exit 1; }
/home/kooki3/.local/bin/work-handoff stop --project dual-channel-attention --owner claude:a0424013 --job "$JOB" \
  --signal INT --reason "$REASON"
echo "stopped $JOB"; nvidia-smi --query-gpu=memory.used --format=csv,noheader
