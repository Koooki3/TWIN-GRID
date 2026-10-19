"""Value-criteria appendix to rep2 (docs/VALUE_CRITERIA_ZH.md, A4 and A5; reviewer batch 13, V3). Written and committed
BEFORE any output of its three arms exists. Same data and evaluation lines as rep2 (runs/ruler/rep2_qwen1.5b_16k,
lines 20..99), budget 512, U = 32, one method per process. New arms: em (B32), quest16+g2f (B16), quest8 (B8);
quest16 comes from the rep2 run. Scoring, units and bootstrap exactly as src/rep2_analysis.py (2000 resamples,
seed 0, sample-level, stratified by task).
Rules (fixed now):
  A4 em - quest16 (em: 12 floats per token per KV head = 75% of quest16's 16): 95% lower > -0.02 -> "内存效率收益成立";
     else "不成立".
  A5 quest16+g2f - quest8 (equal memory, 32 floats per token): 95% lower > 0 -> "成立"; else "不成立".
     quest16+g2f - quest16 reported (descriptive only).
Usage: rep2_value_analysis.py [ROOT]"""
import os
import sys

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
sys.path.insert(0, os.path.join(ROOT, "src"))
import rep2_analysis as R  # noqa: E402

R.ARMS.update({"em": "em", "quest16+g2f": "quest+g2f", "quest8": "quest"})


def main():
    arms = ("em", "quest16", "quest16+g2f", "quest8")
    recs = R.units(arms, 512, range(20, 100))
    ci, ns = R.booter(recs)
    print(f"value appendix @512: {len(recs)} units, {ns} samples (lines 20..99)")
    for a in arms:
        print(f"  {a:12s} {R.acc(recs, a):.3f}")

    def d(a, b):
        return lambda u: R.acc(u, a) - R.acc(u, b)
    a4 = R.acc(recs, "em") - R.acc(recs, "quest16")
    lo, hi = ci(d("em", "quest16"))
    print(f"A4 em - quest16 = {a4:+.3f} [{lo:+.3f}, {hi:+.3f}] (rule: lower > -0.02) -> "
          f"{'内存效率收益成立' if lo > -0.02 else '不成立'}")
    a5 = R.acc(recs, "quest16+g2f") - R.acc(recs, "quest8")
    lo, hi = ci(d("quest16+g2f", "quest8"))
    print(f"A5 quest16+g2f - quest8 = {a5:+.3f} [{lo:+.3f}, {hi:+.3f}] (rule: lower > 0) -> {'成立' if lo > 0 else '不成立'}")
    lo, hi = ci(d("quest16+g2f", "quest16"))
    print(f"   quest16+g2f - quest16 = {R.acc(recs, 'quest16+g2f') - R.acc(recs, 'quest16'):+.3f} [{lo:+.3f}, {hi:+.3f}] "
          f"(descriptive)")


if __name__ == "__main__":
    main()
