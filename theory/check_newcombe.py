"""Checks of src/newcombe.py (method 10). Exit 1 on failure.
0. Published values: Newcombe (1998) Table III method 10 for (27,0,0,27) and (20,12,2,16).
1. Boundary cases: no discordant pairs, all pairs discordant one way, n = 1, all both-right: finite, ordered, in
   [-1, 1], contains D.
2. Antisymmetry: swapping b and c negates the interval.
3. Coverage sanity by simulation (not a guarantee): at n = 300 for a few cell-probability settings, the 95% interval
   covers the true difference in roughly 93-97% of 4000 simulated tables.
4. Phase-1 dev numbers (cobs+g2f vs cobs: a=62, b=10, c=2, d=6) are printed for the record."""
import random
import sys

sys.path.insert(0, __file__.rsplit("/theory/", 1)[0] + "/src")
from newcombe import newcombe10  # noqa: E402

bad = []
# Newcombe (1998) Table III, method 10 (p. 2642), 4-decimal values; also method 8 differs on the first case.
for cells, want in [((27, 0, 0, 27), (-0.0351, 0.0351)), ((20, 12, 2, 16), (0.0562, 0.3292))]:
    _, lo, hi = newcombe10(*cells)
    print("Table III", cells, "->", round(lo, 4), round(hi, 4), "published", want)
    if abs(lo - want[0]) > 5e-5 or abs(hi - want[1]) > 5e-5:
        bad.append(f"Table III mismatch {cells}: {lo:.4f} {hi:.4f} vs {want}")
for cells in [(0, 0, 0, 1), (5, 0, 0, 5), (0, 10, 0, 0), (0, 0, 10, 0), (1, 0, 0, 0), (300, 0, 0, 0), (0, 150, 150, 0),
              (62, 10, 2, 6)]:
    D, lo, hi = newcombe10(*cells)
    if not (-1 <= lo <= D <= hi <= 1):
        bad.append(f"{cells}: {lo} {D} {hi}")
    a, b, c, d = cells
    D2, lo2, hi2 = newcombe10(a, c, b, d)
    if abs(D2 + D) > 1e-12 or abs(lo2 + hi) > 1e-9 or abs(hi2 + lo) > 1e-9:
        bad.append(f"{cells}: not antisymmetric")
rng = random.Random(1)
for probs in [(0.80, 0.10, 0.05, 0.05), (0.70, 0.125, 0.025, 0.15), (0.6, 0.1, 0.1, 0.2)]:
    true = probs[1] - probs[2]
    cov = 0
    for _ in range(4000):
        cnt = [0, 0, 0, 0]
        for _ in range(300):
            u, acc = rng.random(), 0
            for k, p in enumerate(probs):
                acc += p
                if u < acc:
                    cnt[k] += 1
                    break
            else:
                cnt[3] += 1
        _, lo, hi = newcombe10(*cnt)
        cov += lo <= true <= hi
    print(f"coverage at {probs}: {cov / 4000:.3f}")
    if not 0.93 <= cov / 4000 <= 0.975:
        bad.append(f"coverage {cov / 4000} at {probs}")
print("phase-1 dev cobs+g2f vs cobs (a,b,c,d)=(62,10,2,6):", tuple(round(x, 4) for x in newcombe10(62, 10, 2, 6)))
print("PROBLEMS", bad if bad else "none")
sys.exit(1 if bad else 0)
