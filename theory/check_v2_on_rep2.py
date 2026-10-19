"""Consistency check of src/v2_stats.py against the already-reviewed rep2 numbers (CPU; existing predictions; no model).
Only point estimates that rep2 already reported are recomputed: H1 = cobs16 - cobs+g2f (+0.065, R1), H2 = quest8 -
quest16+g2f (+0.024, A5 sign flipped), H3 = G_cobs - G_quest32 (+0.201, R3). No new analysis (no H4) is run on rep2.
Exit 1 if any differs from the logged value by more than 0.0005. Usage: check_v2_on_rep2.py [ROOT]"""
import os
import sys

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.argv = [sys.argv[0], ROOT]
import rep2_analysis as R  # noqa: E402
import v2_stats as V  # noqa: E402

R.ARMS.update({"quest16+g2f": "quest+g2f", "quest8": "quest"})
recs = R.units(("cobs", "cobs+g2f", "cobs16", "quest32", "quest16+g2f", "quest8"), 512, range(20, 100))
got = {"H1": V.diff(recs, "cobs16", "cobs+g2f"), "H2": V.diff(recs, "quest8", "quest16+g2f"),
       "H3": V.gdiff(recs, "cobs", "quest32")}
want = {"H1": 0.065, "H2": 0.024, "H3": 0.201}   # rep2_final.log R1, rep2_value.log A5 (sign flipped), R3
bad = []
for k in got:
    ok = abs(got[k] - want[k]) <= 0.0005
    print(f"{'OK  ' if ok else 'FAIL'} {k}: v2_stats {got[k]:+.4f}  rep2 log {want[k]:+.3f}")
    if not ok:
        bad.append(k)
d = {k: V.contributions(recs, *ab, kind) for k, ab, kind in
     (("H1", ("cobs16", "cobs+g2f"), "diff"), ("H2", ("quest8", "quest16+g2f"), "diff"), ("H3", ("cobs", "quest32"), "gdiff"))}
for k in d:
    ok = abs(d[k].sum() - got[k]) < 1e-12
    print(f"{'OK  ' if ok else 'FAIL'} {k}: per-sample contributions sum to the statistic")
    if not ok:
        bad.append(k + " contributions")
print("PROBLEMS", bad if bad else "none")
sys.exit(1 if bad else 0)
