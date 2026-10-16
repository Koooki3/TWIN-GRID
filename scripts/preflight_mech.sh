#!/usr/bin/env bash
# CPU-only identity preflight of a mechanism prompt list (no model load, no GPU). Exit code from run_digit_diag.
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
export CUDA_VISIBLE_DEVICES=""
taskset -c 0-5 nice -n 15 /home/kooki3/projects/dual-channel-attention/.venv/bin/python $WT/src/run_digit_diag.py \
  --prompts ${1:-$WT/results/mech_prompts_full.json} \
  --identities /home/kooki3/docs/projects/attention-review/2026-10-16/hourly-mech-v2-1426/original-prompt-identities.json \
  --out /dev/null --preflight-only
