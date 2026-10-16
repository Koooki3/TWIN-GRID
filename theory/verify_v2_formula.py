"""Accuracy of closed forms for E log sum_{i<=n} exp(s z_i), z_i iid N(0,1).

  cumulant:  log n + s^2/2                                   (COBS)
  tangent:   two-phase branch formula of theory v1
  F_n:       log( e^{c_n s} + (n-1) e^{s^2/2} Phi(c_n - s) ) (max + truncated bulk)
c_n = E[max of n standard normals] via Blom's approximation.
"""
import numpy as np
from scipy.stats import norm

rng = np.random.default_rng(1)


def lse(x):
    m = x.max(-1, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(-1, keepdims=True)))[..., 0]


def c_blom(n):
    return norm.ppf((n - 0.375) / (n + 0.25))


def formulas(s, n):
    c = c_blom(n)
    cum = np.log(n) + s * s / 2
    tan = np.where(s <= c, cum, np.log(n) - c * c / 2 + c * s)
    F = np.logaddexp(c * s, np.log(n - 1) + s * s / 2 + norm.logcdf(c - s))
    return cum, tan, F


print(" n     s    E[A]    err_cum  err_tan  err_F")
worst = {"cum": 0, "tan": 0, "F": 0}
for n in (8, 16, 31, 32, 64, 128):
    for s in (0.25, 0.5, 1, 1.5, 2, 2.5, 3, 4, 6, 8, 12):
        A = lse(s * rng.standard_normal((40000, n))).mean()
        cum, tan, F = formulas(s, n)
        worst["cum"] = max(worst["cum"], abs(cum - A))
        worst["tan"] = max(worst["tan"], abs(tan - A))
        worst["F"] = max(worst["F"], abs(F - A))
        if n in (32, 128):
            print(f"{n:3d} {s:5.2f} {A:8.3f} {cum - A:8.3f} {tan - A:8.3f} {F - A:8.3f}")
print("max |error| over n in {8..128}, s in [0.25, 12]:", {k: round(v, 3) for k, v in worst.items()})
