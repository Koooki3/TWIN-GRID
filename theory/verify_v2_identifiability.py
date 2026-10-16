"""Identifiability gap of moment summaries.

Claim (to check): over all score vectors x in R^n with mean m and (population) variance s^2,
    min A = m + log n + log cosh s            (n even; half at m+s, half at m-s)    [lower]
    max A = m + log( e^{s sqrt(n-1)} + (n-1) e^{-s/sqrt(n-1)} )    (one spike)     [upper]
where A = logsumexp(x). So any estimator that sees only (m, s, n) has worst-case error at
least half the gap, which grows like s sqrt(n).

Check by projected gradient ascent/descent on the constraint set from many random starts.
"""
import numpy as np

rng = np.random.default_rng(2)


def lse(x):
    mx = x.max()
    return mx + np.log(np.exp(x - mx).sum())


def project(x, s):
    x = x - x.mean()
    nrm = np.sqrt((x ** 2).mean())
    return x * (s / nrm) if nrm > 0 else x


def optimise(n, s, sign, starts=200, steps=3000, lr=0.05):
    best = -np.inf if sign > 0 else np.inf
    for _ in range(starts):
        x = project(rng.standard_normal(n), s)
        for _ in range(steps):
            p = np.exp(x - x.max()); p /= p.sum()
            x = project(x + sign * lr * s * p * n, s)
        v = lse(x)
        best = max(best, v) if sign > 0 else min(best, v)
    return best


print(" n    s   min_num  min_formula   max_num  max_formula   gap")
for n in (8, 32):
    for s in (0.5, 1.0, 2.0, 4.0):
        lo_f = np.log(n) + np.log(np.cosh(s))
        hi_f = np.log(np.exp(s * np.sqrt(n - 1)) + (n - 1) * np.exp(-s / np.sqrt(n - 1)))
        lo_n = optimise(n, s, -1, starts=60, steps=1500)
        hi_n = optimise(n, s, +1, starts=60, steps=1500)
        print(f"{n:2d} {s:4.1f} {lo_n:8.3f} {lo_f:11.3f} {hi_n:9.3f} {hi_f:11.3f} {hi_f - lo_f:6.2f}")
