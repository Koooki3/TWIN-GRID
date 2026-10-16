#!/usr/bin/env bash
# Round 5 under the idle gate: start after 10 min of desktop idle, stop within ~10 s of user input.
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
export STOP_PATTERN="$WT/src/run_e2e.py"
exec "$WT/scripts/idle_gate.sh" 600 30 -- "$WT/scripts/queue_round5.sh"
