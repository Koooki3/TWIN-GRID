#!/usr/bin/env bash
# Prereg v3 confirmation (phase 2), frozen SPEC theory/phase2_spec.json (commit a217f16), low-footprint resources as
# the user asked. Order: ptr dev -> THIS -> mech full -> F3 rest. Refuses to start if the SPEC check fails.
# While this runs and until the analysis is done, the frozen files (src/sparse_attn.py, src/run_e2e.py, src/tasks.py,
# src/phase2_*.py, src/newcombe.py, src/make_phase2_spec.py, theory/prereg_g2f_v3.md) must not change.
set -u
set -o pipefail
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
PY=/home/kooki3/projects/dual-channel-attention/.venv/bin/python
while pgrep -f "queue_ptr_dev.sh" >/dev/null; do sleep 30; done
$PY $WT/src/make_phase2_spec.py /tmp/phase2_spec_recheck_$$.json -- --ctx 16384 --budgets 512 --tasks multikey,single_essay \
  --n 150 --seed0 200000 --seed-stream --haystack v2 --slot0 100 --rank 8 --prefill-chunk 2048 --no-stats --vram-frac 0.6 \
  --throttle 0.3 --strict-budget --methods cobs,cobs+g2f,nf_pt1,nf_pt1+g2f,quest,quest+g2f \
  --out $WT/runs/phase2_g2f_confirm.jsonl > /dev/null || exit 1
$PY - "$WT/theory/phase2_spec.json" "/tmp/phase2_spec_recheck_$$.json" <<'EOF' || { echo "SPEC drift: refusing to launch"; exit 1; }
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
sys.exit(0 if a == b else 1)
EOF
/home/kooki3/.local/bin/work-handoff run --project dual-channel-attention --owner claude:a0424013 --label phase2-confirm \
  --input $WT/theory/phase2_spec.json --input $WT/src/sparse_attn.py --input $WT/src/run_e2e.py --input $WT/src/tasks.py -- \
  taskset -c 0-5 nice -n 15 $PY $WT/src/run_e2e.py --ctx 16384 --budgets 512 --tasks multikey,single_essay --n 150 \
  --seed0 200000 --seed-stream --haystack v2 --slot0 100 --rank 8 --prefill-chunk 2048 --no-stats --vram-frac 0.6 \
  --throttle 0.3 --strict-budget --methods cobs,cobs+g2f,nf_pt1,nf_pt1+g2f,quest,quest+g2f \
  --out $WT/runs/phase2_g2f_confirm.jsonl
rc=$?
echo "[phase2 $(date '+%T')] rc=$rc"
exit "$rc"
