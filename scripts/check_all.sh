#!/usr/bin/env bash
# Run the CPU check suite of the cluster delivery package (no GPU, no model weights except tokenizer files).
# Usage: scripts/check_all.sh [PYTHON]
WT=$(cd "$(dirname "$0")/.." && pwd)
PY=${1:-python}
rc_all=0
for t in check_cluster_aggregate check_gain_law check_g3_layout check_invariant check_model_identity; do
  out=$($PY "$WT/theory/$t.py" 2>&1); rc=$?
  echo "$t rc=$rc: $(echo "$out" | tail -1)"
  [ $rc -ne 0 ] && rc_all=1
done
# default path guard: selection / budget / fixed region / candidate counts vs the confirmation-time sparse_attn
# (e194213, sha256 c9b56b7f..., theory/phase2_spec.json) — not a bitwise model-output comparison (reviewer batch 4)
OLD=$(mktemp -d)/sparse_attn_e194213.py
# local history has e194213; the public snapshot carries the same file as results/sparse_attn_e194213.py
git -C "$WT" show e194213:src/sparse_attn.py > "$OLD" 2>/dev/null || cp "$WT/results/sparse_attn_e194213.py" "$OLD"
case "$(sha256sum "$OLD")" in
  c9b56b7f89bc99281a10a4bcc439f8d736ea149980f35768bb93063654005fed*) ;;     # = theory/phase2_spec.json sparse_attn.py
  *) echo "confirmation-time sparse_attn.py missing or altered"; rc_all=1 ;;
esac
for t in check_g2f check_fixed_layout; do
  out=$($PY "$WT/theory/$t.py" "$OLD" "$WT/src/sparse_attn.py" 2>&1); rc=$?
  echo "$t (e194213 vs HEAD) rc=$rc: $(echo "$out" | tail -1)"
  [ $rc -ne 0 ] && rc_all=1
done
$PY "$WT/cluster/make_jobs.py" --tier 3 --models-dir /m --out-root /o --dry_run || rc_all=1
echo "rc_all=$rc_all"
exit $rc_all
