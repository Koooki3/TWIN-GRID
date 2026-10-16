#!/usr/bin/env bash
# After nb_16k finishes: stop queue_round5.sh before its layer probes, run the decisive forced-block probe
# on straddled samples, then run the layer probes.
set -u
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
PY=/home/kooki3/projects/dual-channel-attention/.venv/bin/python
WH=/home/kooki3/.local/bin/work-handoff
COMMON="--ctx 16384 --tasks multikey,single_essay --budgets 512 --rank 8 --prefill-chunk 2048"

until [ "$(wc -l < $WT/runs/nb_16k.jsonl 2>/dev/null || echo 0)" -ge 60 ]; do sleep 20; done
sleep 10
pkill -TERM -f "$WT/scripts/queue_round5.sh"
pkill -TERM -f "causal_L"
sleep 5
echo "[orch $(date '+%T')] nb_16k done; queue_round5 stopped before layer probes"

$WH run --project dual-channel-attention --owner claude:a0424013 --label force-probe-16k \
  --input $WT/src/sparse_attn.py --input $WT/src/run_e2e.py --input $WT/src/tasks.py -- \
  taskset -c 0-5 nice -n 15 $PY $WT/src/run_e2e.py $COMMON --n 60 --straddle-only \
  --methods cobs,cobs+ftail,cobs+fhead,cobs+fadj,cobs+nbr,nf_pt1,nf_pt1+ftail,nf_pt1+fhead,quest \
  --out $WT/runs/force_probe_16k.jsonl
echo "[orch force $(date '+%T')] rc=$?"

for G in 2-7 8-13 14-19 20-27; do
  $WH run --project dual-channel-attention --owner claude:a0424013 --label causal-L$G \
    --input $WT/src/sparse_attn.py --input $WT/src/run_e2e.py --input $WT/src/tasks.py -- \
    taskset -c 0-5 nice -n 15 $PY $WT/src/run_e2e.py --ctx 16384 --n 20 --tasks multikey,single_essay,multivalue \
    --budgets 512 --rank 8 --prefill-chunk 2048 --sparse-layers $G --methods nf_pt1,nf_hi1 \
    --out $WT/runs/causal_L${G}_16k.jsonl
  echo "[orch L$G $(date '+%T')] rc=$?"
done
