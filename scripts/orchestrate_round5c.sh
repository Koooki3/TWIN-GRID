#!/usr/bin/env bash
# Round 5c (after Codex review F1-F6): wait for nb_16k, then
#  1. forced-block probe on straddled samples (diagnostic first step, F2)
#  2. grid-shift intervention: same inputs/positions, selection grid offset flips straddling (F2e)
#  3. minimal factorial: prefill chunk {one-shot, 2048} x rank {8,16} x subspace {per q-head, shared per KV head} (F3)
set -u
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
PY=/home/kooki3/projects/dual-channel-attention/.venv/bin/python
WH=/home/kooki3/.local/bin/work-handoff
IN="--input $WT/src/sparse_attn.py --input $WT/src/run_e2e.py --input $WT/src/tasks.py"
RUN="taskset -c 0-5 nice -n 15 $PY $WT/src/run_e2e.py --ctx 16384 --budgets 512"

until [ "$(wc -l < $WT/runs/nb_16k.jsonl 2>/dev/null || echo 0)" -ge 60 ]; do sleep 20; done
while pgrep -f "runs/nb_16k.jsonl" >/dev/null; do sleep 10; done
echo "[orch5c $(date '+%T')] nb_16k done"

$WH run --project dual-channel-attention --owner claude:a0424013 --label force-probe-16k $IN -- \
  $RUN --tasks multikey,single_essay --n 60 --rank 8 --prefill-chunk 2048 --straddle-only \
  --methods cobs,cobs+ftail,cobs+fhead,cobs+fadj,cobs+nbr,nf_pt1,nf_pt1+ftail,nf_pt1+fhead,quest \
  --out $WT/runs/force_probe_16k.jsonl
echo "[orch5c force $(date '+%T')] rc=$?"

$WH run --project dual-channel-attention --owner claude:a0424013 --label shift-intervention-16k $IN -- \
  $RUN --tasks multikey,single_essay --n 20 --rank 8 --prefill-chunk 2048 --shift-intervention \
  --methods cobs,nf_pt1,quest --out $WT/runs/shift_16k.jsonl
echo "[orch5c shift $(date '+%T')] rc=$?"

for CH in 0 2048; do
  for R in 8 16; do
    for WP in shared perhead; do
      FLAG=""; [ "$WP" = perhead ] && FLAG="--w-per-head"
      $WH run --project dual-channel-attention --owner claude:a0424013 --label f3-ch$CH-r$R-$WP $IN -- \
        $RUN --tasks multikey,single_essay --n 20 --rank $R --prefill-chunk $CH $FLAG --methods cobs,quest \
        --out $WT/runs/f3_ch${CH}_r${R}_${WP}.jsonl
      echo "[orch5c f3 ch$CH r$R $WP $(date '+%T')] rc=$?"
    done
  done
done
