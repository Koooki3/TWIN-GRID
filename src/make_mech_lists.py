"""Build the mech-design-v2 prompt lists: 34 straddled (force_probe_16k) + 29 non-straddled (shift_16k rows with
straddle_orig = False). Check list: the first straddled and first non-straddled prompt per task (sorted by seed).
Usage: make_mech_lists.py FORCE.jsonl SHIFT.jsonl OUT_FULL.json OUT_CHECK.json"""
import json
import sys

force = [json.loads(l) for l in open(sys.argv[1])]
shift = [json.loads(l) for l in open(sys.argv[2])]
strad = sorted({(r["task"], r["seed"]) for r in force})
inside = sorted({(r["task"], r["seed"]) for r in shift if r["straddle_orig"] is False})
assert not set(strad) & set(inside)
full = [list(x) for x in strad + inside]
check = []
for task in ("multikey", "single_essay"):
    check.append(list(next(x for x in strad if x[0] == task)))
    check.append(list(next(x for x in inside if x[0] == task)))
json.dump(full, open(sys.argv[3], "w"))
json.dump(check, open(sys.argv[4], "w"))
print(f"full: {len(full)} ({len(strad)} straddled, {len(inside)} inside); check: {check}")
