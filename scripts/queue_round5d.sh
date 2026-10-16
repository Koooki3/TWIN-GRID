#!/usr/bin/env bash
# Round 5d — low-footprint mode while the user uses the machine (VRAM cap 0.6, no offline stats, 30% idle).
# Each step resumes from its JSONL (same command and config each time).
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
PY=/home/kooki3/projects/dual-channel-attention/.venv/bin/python
WH=/home/kooki3/.local/bin/work-handoff
IN="--input $WT/src/sparse_attn.py --input $WT/src/run_e2e.py --input $WT/src/tasks.py"
RUN="taskset -c 0-5 nice -n 15 $PY $WT/src/run_e2e.py --ctx 16384 --budgets 512 --no-stats --vram-frac 0.6 --throttle 0.3"

$WH run --project dual-channel-attention --owner claude:a0424013 --label force-probe-16k $IN -- \
  $RUN --tasks multikey,single_essay --n 60 --rank 8 --prefill-chunk 2048 --straddle-only \
  --methods cobs,cobs+ftail,cobs+fhead,cobs+fadj,cobs+nbr,nf_pt1,nf_pt1+ftail,nf_pt1+fhead,quest \
  --out $WT/runs/force_probe_16k.jsonl || exit 1

$WH run --project dual-channel-attention --owner claude:a0424013 --label shift-intervention-16k $IN -- \
  $RUN --tasks multikey,single_essay --n 20 --rank 8 --prefill-chunk 2048 --shift-intervention \
  --methods cobs,nf_pt1,quest --out $WT/runs/shift_16k.jsonl || exit 1

$WH run --project dual-channel-attention --owner claude:a0424013 --label shift-placebo-16k $IN -- \
  $RUN --tasks multikey,single_essay --n 20 --rank 8 --prefill-chunk 2048 --shift-intervention \
  --shift-mode placebo --methods cobs,nf_pt1,quest --out $WT/runs/shift_placebo_16k.jsonl || exit 1

for CH in 0 2048; do
  for R in 8 16; do
    for WP in shared perhead; do
      FLAG=""; [ "$WP" = perhead ] && FLAG="--w-per-head"
      $WH run --project dual-channel-attention --owner claude:a0424013 --label f3-ch$CH-r$R-$WP $IN -- \
        $RUN --tasks multikey,single_essay --n 20 --rank $R --prefill-chunk $CH $FLAG --methods cobs,quest \
        --out $WT/runs/f3_ch${CH}_r${R}_${WP}.jsonl || exit 1
    done
  done
done
echo "[round5d $(date '+%T')] all steps finished"
