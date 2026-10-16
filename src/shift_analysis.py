"""Grid-shift intervention: pair shifted-grid rows with a reference arm on the same (task, ctx, seed) and split by
the value's straddling status under the ORIGINAL grid. Descriptive counts only (exploration seeds; label-guided).
Refuses missing or duplicate keys; prints the actual number of pairs; fixed-region logs absent -> "unknown".
Usage: shift_analysis.py SHIFTED.jsonl REFERENCE.jsonl [reference label]"""
import json
import sys
from collections import defaultdict


def load(path):
    rows = {}
    for r in map(json.loads, open(path)):
        k = (r["task"], r["ctx"], r["seed"])
        if k in rows:
            raise SystemExit(f"duplicate key {k} in {path}")
        rows[k] = r
    return rows


def main(shifted, reference, ref_label="original grid", methods=("cobs", "nf_pt1", "quest")):
    sh, ref = load(shifted), load(reference)
    missing = [k for k in sh if k not in ref]
    if missing:
        raise SystemExit(f"{len(missing)} shifted rows have no reference row, e.g. {missing[:3]}")
    table = defaultdict(lambda: [0, 0, 0, 0, 0])   # n, reference correct, shifted correct, improved, worsened
    fixed_viol, fixed_unknown = 0, 0
    for k, r in sh.items():
        b = ref[k]
        if r.get("shift_mode") == "placebo":
            grp = "crossing (kept)" if r["straddle_orig"] else "inside (kept)"
        else:
            grp = "crossing→inside" if r["straddle_orig"] else "inside→crossing"
        for m in methods:
            key = f"{m}@512"
            o, s_ = b["res"][key]["score"], r["res"][key]["score"]
            t = table[(m, grp)]
            t[0] += 1; t[1] += o; t[2] += s_; t[3] += (s_ > o); t[4] += (s_ < o)
            tk = r["res"][key].get("tokens")
            if not tk or "fixed_region_violations" not in tk:
                fixed_unknown += 1
            else:
                fixed_viol += tk["fixed_region_violations"]
    print(f"pairs: {len(sh)} (reference = {ref_label}); fixed-region violations: {fixed_viol}; "
          f"method records without the log: {fixed_unknown} (unknown)")
    print(f"| scorer | group | n | {ref_label} | shifted | improved / worsened (reference → shifted) |")
    print("|---|---|---|---|---|---|")
    for (m, grp), t in sorted(table.items()):
        print(f"| {m} | {grp} | {t[0]} | {t[1]:.0f}/{t[0]} | {t[2]:.0f}/{t[0]} | {t[3]} / {t[4]} |")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(sys.argv[3:4]))
