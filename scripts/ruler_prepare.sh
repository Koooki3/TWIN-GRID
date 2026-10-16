#!/usr/bin/env bash
# Official RULER v1 data preparation (unchanged prepare.py; only the qwen2.5-chat template was added, see
# docs/ruler_template_qwen.diff). CPU only (cores 0-5, nice). Usage:
#   ruler_prepare.sh SEQ_LEN NUM_SAMPLES OUT_DIR [TASK ...]   (default tasks: all 13 of the official `synthetic` list)
# A preparation manifest (src/ruler_prep_manifest.py) pins the configuration of OUT_DIR: a different length / N /
# tokenizer / template / RULER commit / source data in an existing OUT_DIR is refused (Codex R4); after preparation
# every file is checked (line count, schema) and its sha256 and sample uids are recorded.
# Model-specific settings come from the environment (defaults = the local development model):
#   RULER_ROOT (official checkout), RULER_TOKENIZER (model dir), RULER_TEMPLATE (template name in template.py),
#   RULER_VENV (venv whose `python` prepare.py should call; unset = `python` on PATH), PROJECT_PY (python with this
#   project's deps; unset = python3 on PATH),
#   CPU_SET (taskset core list; empty = no pinning).
set -u
export RULER_ROOT=${RULER_ROOT:-/home/kooki3/tools/sources/ruler}
export RULER_TOKENIZER=${RULER_TOKENIZER:-/home/kooki3/models/qwen2.5-1.5b-instruct}
export RULER_TEMPLATE=${RULER_TEMPLATE:-qwen2.5-chat}
RULER=$RULER_ROOT/scripts
WT=$(cd "$(dirname "$0")/.." && pwd)
# defaults below are the development machine's; elsewhere they fall back to the interpreters on PATH
_DEV_PY=/home/kooki3/projects/dual-channel-attention/.venv/bin/python
_DEV_VENV=/home/kooki3/tools/envs/ruler
if [ -n "${PROJECT_PY:-}" ]; then PYP=$PROJECT_PY; elif [ -x "$_DEV_PY" ]; then PYP=$_DEV_PY; else PYP=$(command -v python3 || command -v python); fi
if [ -n "${RULER_VENV:-}" ]; then export PATH=$RULER_VENV/bin:$PATH        # prepare.py shells out to `python`
elif [ -d "$_DEV_VENV" ]; then export PATH=$_DEV_VENV/bin:$PATH; fi
CPU_SET=${CPU_SET-0-5}
PIN=""; [ -n "$CPU_SET" ] && PIN="taskset -c $CPU_SET"
LEN=$1; N=$2; OUT=$3; shift 3
TASKS=${@:-niah_single_1 niah_single_2 niah_single_3 niah_multikey_1 niah_multikey_2 niah_multikey_3 niah_multivalue niah_multiquery vt cwe fwe qa_1 qa_2}
$PYP $WT/src/ruler_prep_manifest.py begin $OUT $LEN $N $TASKS || exit 1
cd $RULER/data
rc_all=0
for T in $TASKS; do
  $PIN nice -n 15 python prepare.py --save_dir $OUT --benchmark synthetic --task $T \
    --tokenizer_path $RULER_TOKENIZER --tokenizer_type hf --max_seq_length $LEN \
    --model_template_type $RULER_TEMPLATE --num_samples $N > $OUT.$T.log 2>&1
  rc=$?
  # prepare.py prints child errors but still exits 0: verify the output file and its line count ourselves
  F=$OUT/$T/validation.jsonl
  if [ $rc -eq 0 ] && { [ ! -f $F ] || [ "$(wc -l < $F)" -ne $N ]; }; then rc=3; fi
  echo "$T rc=$rc"; [ $rc -ne 0 ] && rc_all=$rc
done
$PYP $WT/src/ruler_prep_manifest.py finish $OUT || rc_all=1
exit $rc_all
