"""Numerical checks for theory v1 (CPU only).

Claims checked:
  P1 sandwich: max_i x_i <= logsumexp(x) <= max_i x_i + log n, and the beta-family
     A_beta = logsumexp(beta x)/beta lies in [max, max + log(n)/beta].
  P2 two phases (random-energy-model picture): for x_i = s z_i, z_i iid N(0,1), n points,
     E[logsumexp(x)] is close to log n + s^2/2 for small s (the cumulant/COBS formula) and
     grows linearly ~ s*sqrt(2 log n) - corrections for large s, so the quadratic formula
     overestimates by a gap that grows ~ s^2/2.
  P3 outlier decomposition: for a block = Gaussian bulk + one outlier, the estimator
     log(exp(x_out) + (n-1) exp(mu_bulk + s_bulk^2/2)) is far more accurate than the
     whole-block cumulant formula and than the max alone.
"""
import numpy as np

rng = np.random.default_rng(0)


def lse(x, axis=-1):
    m = x.max(axis, keepdims=True)
    return (m + np.log(np.exp(x - m).sum(axis, keepdims=True))).squeeze(axis)


def p1():
    worst = 0.0
    for n in (2, 8, 32, 128):
        x = rng.standard_t(2, size=(20000, n)) * rng.uniform(0.1, 30, size=(20000, 1))
        a, h = lse(x), x.max(-1)
        assert np.all(a >= h - 1e-9) and np.all(a <= h + np.log(n) + 1e-9)
        for beta in (0.5, 2.0, 8.0):
            ab = lse(beta * x) / beta
            assert np.all(ab >= h - 1e-9) and np.all(ab <= h + np.log(n) / beta + 1e-9)
        worst = max(worst, ((a - h) / np.log(n)).max())
    print(f"P1 sandwich holds on 320k random blocks; max (A-h)/log n = {worst:.4f}")


def p2(n=32):
    print(f"P2 n={n}: E[A] vs cumulant formula log n + s^2/2 vs REM branch; sqrt(2 log n)={np.sqrt(2*np.log(n)):.2f}")
    print("   s    E[A]   cumulant  err_cum   E[max]  log n+s*sqrt(2logn)")
    for s in (0.25, 0.5, 1, 1.5, 2, 2.63, 3, 4, 6, 8):
        z = rng.standard_normal((40000, n))
        A = lse(s * z).mean()
        cum = np.log(n) + s * s / 2
        print(f"  {s:4.2f} {A:7.3f} {cum:8.3f} {cum - A:8.3f} {(s * z).max(-1).mean():7.3f} {s * np.sqrt(2 * np.log(n)):7.3f}")


def p3(n=32, trials=20000):
    print(f"P3 bulk N(0, sb^2) + one outlier at x_out (n={n}): mean |error| of log Z estimates")
    print("  sb  x_out | cumulant(whole) | max only | outlier+bulk cumulant")
    for sb in (0.5, 1.0, 2.0):
        for xo in (2.0, 5.0, 10.0):
            z = rng.standard_normal((trials, n - 1)) * sb
            x = np.concatenate([z, np.full((trials, 1), xo)], 1)
            A = lse(x)
            mu, var = x.mean(-1), x.var(-1)
            cum = np.log(n) + mu + var / 2
            mx = x.max(-1)
            bmu, bvar = z.mean(-1), z.var(-1)
            dec = np.logaddexp(xo, np.log(n - 1) + bmu + bvar / 2)
            print(f"  {sb:3.1f} {xo:5.1f} | {np.abs(cum - A).mean():15.3f} | {np.abs(mx - A).mean():8.3f} | "
                  f"{np.abs(dec - A).mean():8.3f}")


if __name__ == "__main__":
    p1(); p2(); p3()
