"""Self-check of cluster/gain_law.analyse on simulated cells where the gain law holds exactly in expectation
(R_repair = 1, R_out = 0, no inside change) and where it fails (R_repair = 0.5): kappa must cover 1 in the first case
and exclude 1 in the second; the cross-fitted kappa must not be noticeably worse (bias check: at most 0.02 farther from 1 than the same-sample one on average) — this says nothing about independence (erratum E1, docs/THEORY_ADDENDUM_ZH.md).
Exit 1 on failure."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cluster"))
import gain_law  # noqa: E402


def simulate(rng, R, n=100, tasks=7, scorers=(("cobs", 0.9, 0.5), ("nf_pt1", 0.9, 0.6), ("quest32", 0.95, 0.9))):
    data = {}
    for t in range(tasks):
        k = 1 if t < 4 else 4
        cls = [[rng.choice(["in0", "repair", "out"], p=[0.68, 0.3, 0.02]) for _ in range(k)] for _ in range(n)]
        for s, a_in, a_cut in scorers:
            b, c = [], []
            for units in cls:
                bb, cc = [], []
                for u in units:
                    p = a_in if u == "in0" else a_cut
                    x = int(rng.random() < p)
                    if u == "repair" and not x and rng.random() < R * (a_in - a_cut) / (1 - a_cut):
                        y = 1
                    else:
                        y = x
                    bb.append(x)
                    cc.append(y)
                b.append(bb)
                c.append(cc)
            data[s, f"t{t}"] = (cls, b, c)
    return data


bad = []
rng = np.random.default_rng(1)
_, p1, _, ci1 = gain_law.analyse(simulate(rng, 1.0), boot=400)
print(f"law holds:  kappa {p1['kappa']:.3f} [{ci1['kappa'][0]:.3f}, {ci1['kappa'][1]:.3f}]  R_repair {p1['R_repair']:.3f}"
      f"  R_out {p1['R_out']:.3f}  inside {p1['inside']:+.4f}  slope {p1['slope']:.3f}")
if not ci1["kappa"][0] <= 1 <= ci1["kappa"][1]:
    bad.append("kappa interval misses 1 when the law holds")
_, p2, _, ci2 = gain_law.analyse(simulate(rng, 0.5), boot=400)
print(f"R = 0.5:    kappa {p2['kappa']:.3f} [{ci2['kappa'][0]:.3f}, {ci2['kappa'][1]:.3f}]  R_repair {p2['R_repair']:.3f}")
if ci2["kappa"][1] >= 1:
    bad.append("kappa interval covers 1 when R = 0.5")
# cross-fitting vs same-sample G: mean kappa over replications
cf, ss = [], []
for r in range(40):
    d = simulate(np.random.default_rng(100 + r), 1.0)
    full = {k: gain_law.cell_stats(*v, list(range(len(v[0]))), [i % 2 for i in range(len(v[0]))]) for k, v in d.items()}
    cf.append(sum(x["delta"] for x in full.values()) / sum(x["pred"] for x in full.values()))
    ss.append(sum(x["delta"] for x in full.values()) / sum(x["pi"] * x["G"] for x in full.values()))
print(f"replications: cross-fitted kappa mean {np.mean(cf):.3f}; same-sample kappa mean {np.mean(ss):.3f}")
if abs(np.mean(cf) - 1) > abs(np.mean(ss) - 1) + 0.02:
    bad.append("cross-fitted kappa noticeably farther from 1 than same-sample (bias check)")
print("PROBLEMS", bad if bad else "none")
sys.exit(1 if bad else 0)
