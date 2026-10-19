#!/usr/bin/env bash
# Final rep2 verdict (R1–R7, value appendix A4/A5), written and committed BEFORE the full gate has run and before any
# value-appendix output is read. Waits for the queue's own gate (results/rep2_gate.log with an "rc=" line, written
# after this script started), then re-runs rep2_gate.py --value itself; only if that rc is 0 runs the registered
# src/rep2_analysis.py and src/rep2_value_analysis.py (both unchanged). Writes results/rep2_final.log and
# results/rep2_value.log; per-(budget, arm, task) resource flags are listed as the deviation record.
# Exit code: 0 only if the gate passed, both analyses returned 0 and every line R1..R7, A4, A5 is present;
# 3 gate failed (nothing judged: every item stays "无法判定（不完整）"), 5 a verdict line is missing, else an analysis rc.
# Amendment 2026-10-19 (Codex T5, review of 7d669b4): a missing verdict line set the process/log rc to 5 but left
# results/rep2_final.rc at the earlier analysis rc; the final rc is now written back. Not triggered in the 7d669b4 run
# (all lines present); its original log, .rc and receipts are kept as written.
set -u
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
PY=/home/kooki3/projects/dual-channel-attention/.venv/bin/python
LOG=$WT/results/rep2_final.log
VLOG=$WT/results/rep2_value.log
GLOG=$WT/results/rep2_gate.log
START=$(date +%s)
while true; do
  if [ -f "$GLOG" ] && [ "$(stat -c %Y "$GLOG")" -ge "$START" ] && grep -q '^rc=' "$GLOG"; then break; fi
  sleep 60
done
RC=0
{
  echo "== $(date '+%F %T') final rep2 verdict"
  echo "-- queue gate log ($(grep '^rc=' "$GLOG")):"; cat "$GLOG"
  echo "-- gate --value re-run here:"; $PY $WT/src/rep2_gate.py $WT --value; gate_rc=$?; echo "rc=$gate_rc"
  echo "-- resource flags per (budget, arm, task) from the manifests:"
  for man in $WT/runs/ruler/rep2_pred/b*/*/manifest.*.json; do
    $PY -c "import json,sys; a=json.load(open(sys.argv[1]))['argv']; p=sys.argv[1].split('/'); print(p[-3], p[-2], p[-1][9:-5], '--vram-frac', a[a.index('--vram-frac')+1], '--throttle', a[a.index('--throttle')+1])" "$man"
  done
  if [ $gate_rc -ne 0 ]; then
    echo "-- STATUS: failed (gate rc $gate_rc): no verdict; affected items 无法判定（不完整）"
    echo 3 > $WT/results/rep2_final.rc
  else
    echo "-- sha256 of the analysis inputs:"; sha256sum $WT/src/rep2_analysis.py $WT/src/rep2_value_analysis.py $WT/src/rep2_gate.py
    echo "-- rep2_analysis.py (R1–R7):"; $PY $WT/src/rep2_analysis.py $WT; a_rc=$?; echo "analysis rc=$a_rc"
    $PY $WT/src/rep2_value_analysis.py $WT > "$VLOG" 2>&1; v_rc=$?; echo "rc=$v_rc" >> "$VLOG"
    echo "-- rep2_value_analysis.py (A4, A5; also in $VLOG):"; cat "$VLOG"
    r=$a_rc; [ $r -eq 0 ] && r=$v_rc
    echo $r > $WT/results/rep2_final.rc
  fi
} > "$LOG" 2>&1
RC=$(cat $WT/results/rep2_final.rc)
if [ "$RC" = 0 ]; then
  for k in R1 R2 R3 R4 R5 R6 R7 A4 A5; do grep -q "^$k " "$LOG" || { echo "-- missing $k line" >> "$LOG"; RC=5; }; done
fi
echo "final rc=$RC" >> "$LOG"
echo "$RC" > $WT/results/rep2_final.rc   # amendment 2026-10-19 (Codex T5): the .rc file carries the FINAL rc
echo "[rep2-final $(date '+%T')] rc=$RC"
exit $RC
