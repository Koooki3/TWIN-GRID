"""Planning power for the v2 draft (docs/PLAN_V2_COGNITIVE_ZH.md §3.5; Codex v2 review V6). CPU only; reads the
existing rep2 predictions (Qwen2.5-1.5B, 7 RULER tasks with units, lines 20..99, budget 512); runs no model.
For H1 (cobs16 - cobs+g2f), H2 (quest8 - quest16+g2f) and H3 (G_cobs - G_quest32), with the v2 estimators
(answer-unit weighted over the analysis set, sample-level resampling stratified by task):
  SE at n = 100 samples per task is estimated directly by resampling 100 samples per task with replacement from the
  80 rep2 samples (B = 4000, seed 20261019) -- not by sqrt(80/100) scaling. These are 1.5B / budget-512 SEs; the
  8B / budget-1024 SE is unknown, hence the variance scenarios.
  Decision: one-sided test at the strictest Holm step alpha = 0.05/3 (normal approximation, z = 2.128).
  Scenarios: true effect = rep2 point x {1, 0.5, 0.25}; variance x {1, 1.5}; and, for H1/H2, dilution by the two UUID
  tasks (niah_single_3, niah_multikey_3; they have units, so they are in the v2 analysis set): 2 x 100 extra
  single-unit samples whose paired difference has zero mean and the empirical per-sample variance of the rep2
  single-unit tasks (an assumption, stated). H3 is not diluted here (UUID class counts under the 8B tokenizer are
  unknown). MDE is a proportion (0.044 = 4.4 percentage points).
Output: results/power_v2.log. Usage: power_v2.py [ROOT]"""
import os
import sys
from math import erf, sqrt

import numpy as np

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else ".")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.argv = [sys.argv[0], ROOT]
import rep2_analysis as R  # noqa: E402

R.ARMS.update({"quest16+g2f": "quest+g2f", "quest8": "quest"})
ARMS = ("cobs", "cobs+g2f", "cobs16", "quest32", "quest16+g2f", "quest8")
N, B, SEED, Z = 100, 4000, 20261019, 2.128
SINGLE = ("niah_single_1", "niah_single_2", "niah_multikey_1", "niah_multikey_2")


def phi(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def stats(u, extra=None):
    """H1, H2 as unit-weighted differences (extra = list of zero-mean paired diffs of added single-unit samples)."""
    def d(a, b):
        s = sum(r["ok"][a] - r["ok"][b] for r in u)
        n = len(u)
        if extra is not None:
            s += sum(extra)
            n += len(extra)
        return s / n
    return d("cobs16", "cobs+g2f"), d("quest8", "quest16+g2f"), R.G(u, "cobs") - R.G(u, "quest32")


def main():
    recs = R.units(ARMS, 512, range(20, 100))
    by = {}
    for r in recs:
        by.setdefault((r["task"], r["i"]), []).append(r)
    tasks = sorted({t for t, _ in by})
    per_task = {t: [s for s in by if s[0] == t] for t in tasks}
    point = stats(recs)
    # per-sample paired diffs of the single-unit tasks, centred, as the UUID dilution model
    dil = {}
    for k, (a, b) in enumerate((("cobs16", "cobs+g2f"), ("quest8", "quest16+g2f"))):
        v = np.array([sum(r["ok"][a] - r["ok"][b] for r in by[s]) for t in SINGLE for s in per_task[t]], float)
        dil[k] = v - v.mean()
    rng = np.random.default_rng(SEED)
    draws, draws_dil = [], []
    for _ in range(B):
        u = [r for t in tasks for j in rng.integers(0, len(per_task[t]), N) for r in by[per_task[t][j]]]
        draws.append(stats(u))
        h = []
        for k in (0, 1):
            ex = list(rng.choice(dil[k], 2 * N))
            h.append(stats(u, ex)[k])
        draws_dil.append(h)
    se = np.nanstd(np.array(draws, float), axis=0, ddof=1)
    se_dil = np.std(np.array(draws_dil, float), axis=0, ddof=1)
    n_units = len([r for t in tasks for s in per_task[t][:1] for r in by[s]])
    out = [f"rep2 inputs: {len(recs)} units, {len(by)} samples, tasks {tasks}; resampled to {N}/task, B={B}, seed {SEED}",
           f"decision: one-sided, strictest Holm step alpha 0.05/3, z={Z}; MDE(power 0.8) = (z + 0.8416) * SE"]
    names = ("H1 cobs16 - cobs+g2f", "H2 quest8 - quest16+g2f", "H3 G_cobs - G_quest32")
    for k, nm in enumerate(names):
        rows = [("as rep2 tasks", se[k], point[k])]
        if k < 2:
            f = (sum(len(by[s]) for s in by) * N / 80) / (sum(len(by[s]) for s in by) * N / 80 + 2 * N)
            rows.append(("+2 UUID tasks (zero-mean diffs)", se_dil[k], point[k] * f))
        for label, s, eff in rows:
            out.append(f"{nm} [{label}]: rep2 point {point[k]:+.3f}, effect used {eff:+.3f}, SE(n={N}) {s:.4f}, "
                       f"MDE {(Z + 0.8416) * s:.3f}")
            for vm in (1.0, 1.5):
                ss = s * sqrt(vm)
                pw = "  ".join(f"x{m}: {1 - phi(Z - eff * m / ss):.2f}" for m in (1, 0.5, 0.25))
                out.append(f"    variance x{vm}: power at effect {pw}")
    print("\n".join(out))
    open(os.path.join(ROOT, "results/power_v2.log"), "w").write("\n".join(out) + "\n")
    _ = n_units


if __name__ == "__main__":
    main()
