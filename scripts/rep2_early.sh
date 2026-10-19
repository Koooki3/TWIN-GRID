#!/usr/bin/env bash
# Early R1–R6 verdict (reviewer, after the restart): as soon as the budget-512 rep2 arms are complete
# (rep2_gate.py --b512-only rc 0), assert that at least one of the five R7 arms has NO budget-1024 output file yet
# (otherwise rep2_analysis's R7 branch could print a direction from incomplete 1024 data) and, only if the assertion
# holds, run the registered rep2_analysis.py. Writes results/rep2_early.log with: the gate output and rc, the b1024
# directory listing at that moment, the assertion result, the per-(arm, task, budget) resource flags actually used
# (deviation record), and the analysis output (its R7 line is NOT a verdict: R7 waits for the full gate).
# Exit code (Codex T2): 0 only if the gate passed, the analysis returned 0 and every R1..R6 line is present; 3 gate
# failed, 4 assertion failed (no early analysis), 5 an R1..R6 line is missing, otherwise the analysis rc.
# Amendment 2026-10-19 (Codex T5): same tail pattern as rep2_final.sh; the final rc is now written back to the .rc
# file. Not triggered in the 70d0779 run (all lines present); its original log/.rc/receipts are kept as written.
set -u
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
PY=/home/kooki3/projects/dual-channel-attention/.venv/bin/python
LOG=$WT/results/rep2_early.log
rm -f $WT/results/rep2_early.rc          # no stale rc from an earlier attempt
while true; do
  $PY $WT/src/rep2_gate.py $WT --b512-only > /dev/null 2>&1 && break
  sleep 60
done
{
  echo "== $(date '+%F %T') early R1–R6 verdict"
  echo "-- gate --b512-only:"; $PY $WT/src/rep2_gate.py $WT --b512-only; gate_rc=$?; echo "rc=$gate_rc"
  echo "-- b1024 listing at this moment:"; find $WT/runs/ruler/rep2_pred/b1024 -name '*.jsonl' 2>/dev/null | sort || true
  n_full=0
  for A in cobs:cobs cobs+g2f:cobs+g2f cobs16:cobs quest32:quest mean+g2f:mean+g2f; do
    L=${A%%:*}; M=${A##*:}
    ls $WT/runs/ruler/rep2_pred/b1024/$L/$M@1024/*.jsonl > /dev/null 2>&1 && n_full=$((n_full + 1))
  done
  echo "-- assertion: R7 arms with any b1024 output = $n_full of 5 (need < 5)"
  echo "-- resource flags per (budget, arm, task) from the manifests:"
  for man in $WT/runs/ruler/rep2_pred/b*/*/manifest.*.json; do
    $PY -c "import json,sys; a=json.load(open(sys.argv[1]))['argv']; p=sys.argv[1].split('/'); print(p[-3], p[-2], p[-1][9:-5], '--vram-frac', a[a.index('--vram-frac')+1], '--throttle', a[a.index('--throttle')+1])" "$man"
  done
  if [ $gate_rc -ne 0 ]; then
    echo "-- STATUS: failed (gate rc $gate_rc)"; echo 3 > $WT/results/rep2_early.rc
  elif [ $n_full -lt 5 ]; then
    echo "-- assertion holds: running rep2_analysis.py (R7 line below is not a verdict)"
    $PY $WT/src/rep2_analysis.py $WT; a_rc=$?; echo "analysis rc=$a_rc"
    echo $a_rc > $WT/results/rep2_early.rc
    if [ $a_rc -ne 0 ]; then echo "-- STATUS: failed (analysis rc $a_rc): R1–R6 incomplete"; else echo "-- STATUS: analysis rc 0"; fi
  else
    echo "-- assertion FAILED: no early analysis; wait for the full gate"; echo 4 > $WT/results/rep2_early.rc
  fi
} > "$LOG" 2>&1
RC=$(cat $WT/results/rep2_early.rc)
# completion also requires every R1..R6 line in the log (reviewer/Codex T2); R7 is listed as not_run here
for r in R1 R2 R3 R4 R5 R6; do grep -q "^$r " "$LOG" || { echo "-- missing $r line" >> "$LOG"; [ "$RC" = 0 ] && RC=5; }; done
echo "-- R7: not_run (waits for the full gate)" >> "$LOG"
echo "final rc=$RC" >> "$LOG"
echo "$RC" > $WT/results/rep2_early.rc   # amendment 2026-10-19 (Codex T5): the .rc file carries the FINAL rc
echo "[rep2-early $(date '+%T')] rc=$RC"
exit $RC
