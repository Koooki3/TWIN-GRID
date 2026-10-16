#!/usr/bin/env bash
# After scripts/queue_pgrid_mv.sh: archive the OOM-interrupted 1-row B32 niah_multivalue files (and their manifest)
# under pgrid_pred/b32/_oom_partial/, check the split-process outputs, then copy them to the paths src/pgrid_dev.py
# reads. Checks (reviewer batch 5, Y1; none of them looks at scores):
#  (1) each split output has exactly 20 rows, no duplicate uid, uids = the frozen lines 0..19 of the data file;
#  (2) the split manifests differ from the original 4-method manifest only in --methods and --save_dir and the
#      resource-only flags --vram-frac / --throttle (full-speed rerun 3; they do not change computation) (diff archived);
#  (nf_pt1 arms come from one-method processes b32_mv_nf1 / b32_mv_nfg — rerun 3; the 2-method b32_mv_nf OOMed)
#  (3) for the uid(s) in the archived 1-row OOM files, pred equality with the split rerun (a free determinism check of
#      expandable_segments + process split); inequality is reported, not hidden.
set -eu
WT=/home/kooki3/projects/dual-channel-attention/.claude/worktrees/stage1-analysis
PY=/home/kooki3/projects/dual-channel-attention/.venv/bin/python
O=$WT/runs/ruler/pgrid_pred
D=$WT/runs/ruler/pgrid_qwen1.5b_16k/niah_multivalue/validation.jsonl
A=$O/b32/_oom_partial
[ -e "$A" ] && { echo "refusing: $A exists (merge already done?)"; exit 2; }
$PY - "$O" "$D" <<'EOF'
import hashlib, json, sys
O, D = sys.argv[1], sys.argv[2]
sha = hashlib.sha256(open(D, "rb").read()).hexdigest()
want = [f"{sha[:16]}:niah_multivalue:{i}" for i in range(20)]
bad = []
for sub, arms in (("b32_mv_cobs", ("cobs", "cobs+g2f")), ("b32_mv_nf1", ("nf_pt1",)), ("b32_mv_nfg", ("nf_pt1+g2f",))):
    for a in arms:
        uids = [json.loads(l)["sample_uid"] for l in open(f"{O}/{sub}/{a}@512/niah_multivalue.jsonl")]
        if len(uids) != 20 or len(set(uids)) != 20 or sorted(uids, key=lambda u: int(u.rsplit(":", 1)[1])) != want:
            bad.append(f"{sub}/{a}: {len(uids)} rows, {len(set(uids))} unique, uid set == frozen 0..19: {set(uids) == set(want)}")
        else:
            print(f"(1) {sub}/{a}: 20 rows, unique, = frozen lines 0..19")
orig = json.load(open(f"{O}/b32/manifest.niah_multivalue.json"))
def strip(m):
    m = dict(m); av = list(m.pop("argv"))
    for k in ("--methods", "--save_dir", "--vram-frac", "--throttle"):   # resource-only flags ignored (rerun 3, full speed)
        if k in av:
            i = av.index(k); del av[i:i + 2]
    return m, av
for sub in ("b32_mv_cobs", "b32_mv_nf1", "b32_mv_nfg"):
    new = json.load(open(f"{O}/{sub}/manifest.niah_multivalue.json"))
    same = strip(orig) == strip(new)
    print(f"(2) manifest {sub} vs original: identical except --methods/--save_dir: {same}")
    if not same:
        bad.append(f"{sub}: manifest differs beyond --methods/--save_dir")
if bad:
    print("CHECK FAILED:", *bad, sep="\n  ")
    sys.exit(1)
EOF
mkdir -p $A
for sub in b32_mv_cobs b32_mv_nf1 b32_mv_nfg; do
  diff <($PY -m json.tool $O/b32/manifest.niah_multivalue.json) <($PY -m json.tool $O/$sub/manifest.niah_multivalue.json) \
    > $A/manifest_diff_$sub.txt || true
done
mv $O/b32/manifest.niah_multivalue.json $A/
for a in cobs cobs+g2f nf_pt1 nf_pt1+g2f; do
  mv "$O/b32/$a@512/niah_multivalue.jsonl" "$A/$a@512.niah_multivalue.jsonl"
done
$PY - "$O" "$A" <<'EOF'
import json, sys
O, A = sys.argv[1], sys.argv[2]
src = {"cobs": "b32_mv_cobs", "cobs+g2f": "b32_mv_cobs", "nf_pt1": "b32_mv_nf1", "nf_pt1+g2f": "b32_mv_nfg"}
for a, sub in src.items():
    old = [json.loads(l) for l in open(f"{A}/{a}@512.niah_multivalue.jsonl")]
    new = {json.loads(l)["sample_uid"]: json.loads(l)["pred"] for l in open(f"{O}/{sub}/{a}@512/niah_multivalue.jsonl")}
    for r in old:
        print(f"(3) {a} {r['sample_uid']}: pred equal to split rerun: {r['pred'] == new.get(r['sample_uid'])}")
    if not old:
        print(f"(3) {a}: archived OOM file is empty (nothing to compare)")
EOF
for a in cobs cobs+g2f; do cp "$O/b32_mv_cobs/$a@512/niah_multivalue.jsonl" "$O/b32/$a@512/"; done
cp "$O/b32_mv_nf1/nf_pt1@512/niah_multivalue.jsonl" "$O/b32/nf_pt1@512/"
cp "$O/b32_mv_nfg/nf_pt1+g2f@512/niah_multivalue.jsonl" "$O/b32/nf_pt1+g2f@512/"
# the 2-method nf process (job 6069fa18) OOMed after 1 nf_pt1 row: archived, not used
mv "$O/b32_mv_nf" "$A/b32_mv_nf_oom"
wc -l $O/b32/*@512/niah_multivalue.jsonl
sha256sum $O/b32/*@512/*.jsonl $O/b16/*@512/*.jsonl > $WT/results/pgrid_outputs_sha256.txt
echo "merge done; output sha256 -> results/pgrid_outputs_sha256.txt"
