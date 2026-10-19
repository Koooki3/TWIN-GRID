#!/usr/bin/env bash
# T5 regression (Codex review of 7d669b4): run each wrapper's tail (from 'RC=$(cat' to the end) in a temp dir, with all
# verdict lines present vs the last one missing, for several initial .rc values; assert that the process rc, the .rc
# file and the log's "final rc=" agree and equal the expected value (0 if complete and initial 0, 5 if a line is
# missing and initial 0, else the initial rc). Exit 1 on any mismatch (amendment after Codex T5 closure: assertions).
S=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis/scripts
TMPROOT=${TMPDIR:-/tmp}
bad=0; n=0
for name in final early; do
  if [ $name = final ]; then keys="R1 R2 R3 R4 R5 R6 R7 A4 A5"; drop=A5; else keys="R1 R2 R3 R4 R5 R6"; drop=R6; fi
  tailsrc=$(sed -n '/^RC=\$(cat/,$p' $S/rep2_$name.sh)
  for init in 0 3 4 7 137; do
    for mode in full missing; do
      T=$(mktemp -d "$TMPROOT/t5.XXXXXX"); mkdir -p $T/results
      LOG=$T/log; : > $LOG
      for k in $keys; do [ $mode = missing ] && [ $k = $drop ] && continue; echo "$k x" >> $LOG; done
      echo $init > $T/results/rep2_$name.rc
      bash -c "WT=$T; LOG=$LOG; $tailsrc" > /dev/null; prc=$?
      frc=$(cat $T/results/rep2_$name.rc); lrc=$(tail -1 $LOG | sed 's/^final rc=//')
      if [ $init = 0 ] && [ $mode = missing ]; then want=5; else want=$init; fi
      ok=OK; { [ $prc = $want ] && [ $frc = $want ] && [ $lrc = $want ]; } || { ok=FAIL; bad=$((bad + 1)); }
      n=$((n + 1))
      echo "$ok $name init=$init $mode: process $prc, file $frc, log $lrc (want $want)"
      rm -rf $T
    done
  done
done
echo "$n cases, $bad failed"
[ $bad -eq 0 ]
