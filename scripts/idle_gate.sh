#!/usr/bin/env bash
# Run a resumable GPU job only while the Windows desktop is idle.
#   idle_gate.sh <start_after_idle_s> <stop_below_idle_s> -- <command...>
# Starts the command once keyboard/mouse have been idle for start_after_idle_s; if the
# user comes back (idle drops below stop_below_idle_s) the job gets SIGTERM and the gate
# waits for the next idle period, then reruns the same (resumable) command.
# Exits when the command finishes on its own. `idle_gate.sh --probe` prints idle seconds.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PS1_WIN="$(wslpath -w "$HERE/win_idle_seconds.ps1")"

idle() {
  timeout 20 powershell.exe -NoProfile -ExecutionPolicy Bypass -File "$PS1_WIN" 2>/dev/null | tr -dc '0-9'
}

if [ "${1:-}" = "--probe" ]; then idle; echo; exit 0; fi

START=$1; STOP=$2; shift 3
log() { echo "[idle_gate $(date '+%F %T')] $*"; }

while true; do
  log "waiting for ${START}s of desktop idle"
  while :; do
    i=$(idle); [ -n "$i" ] && [ "$i" -ge "$START" ] && break
    sleep 30
  done
  log "idle ${i}s -> starting job"
  setsid "$@" &
  pid=$!
  stopped=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 10
    i=$(idle)
    if [ -n "$i" ] && [ "$i" -lt "$STOP" ]; then
      log "user active (idle ${i}s) -> stopping job"
      kill -TERM -- -"$pid" 2>/dev/null
      # children started in their own session (e.g. by work-handoff run) are matched by path
      if [ -n "${STOP_PATTERN:-}" ]; then
        pkill -TERM -f "$STOP_PATTERN" 2>/dev/null
        for _ in $(seq 20); do pgrep -f "$STOP_PATTERN" >/dev/null || break; sleep 1; done
        pkill -KILL -f "$STOP_PATTERN" 2>/dev/null
      fi
      wait "$pid" 2>/dev/null
      stopped=1
      break
    fi
  done
  if [ "$stopped" = 0 ]; then
    wait "$pid"; rc=$?
    log "job exited rc=$rc"
    exit "$rc"
  fi
done
