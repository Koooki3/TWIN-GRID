"""Finite-n exact upper end of sum exp(s_i) given equal weights, s_i in [a, b], mean mu (Codex T2):
t = n(mu-a)/(b-a), r = floor(t), theta = t - r;
Zmax = (n-r-1) e^a + r e^b + e^(a + theta(b-a))   (theta = 0 reduces to (n-r) e^a + r e^b).
Checks: (1) the construction is feasible and attains Zmax; (2) no random feasible point exceeds it; (3) Zmax <= the
Edmundson-Madansky value n((b-mu)e^a + (mu-a)e^b)/(b-a), with Codex's strict example n=2, a=0, b=1, mu=1/4.
Exit 1 on failure."""
import math
import random
import sys


def zmax(n, a, b, mu):
    if b == a:
        return n * math.exp(a)
    t = n * (mu - a) / (b - a)
    r = min(int(math.floor(t)), n)
    th = t - r
    if r == n:
        return n * math.exp(b)
    return (n - r - 1) * math.exp(a) + r * math.exp(b) + math.exp(a + th * (b - a))


def em(n, a, b, mu):
    return n * ((b - mu) * math.exp(a) + (mu - a) * math.exp(b)) / (b - a)


def feasible_sample(n, a, b, mu, rng):
    """Random point of {s in [a,b]^n, mean mu}: random start, then project by alternating clipping/shifting."""
    s = [rng.uniform(a, b) for _ in range(n)]
    for _ in range(200):
        shift = mu - sum(s) / n
        s = [min(b, max(a, x + shift)) for x in s]
    return s if abs(sum(s) / n - mu) < 1e-9 else None


rng = random.Random(0)
bad = []
ex = zmax(2, 0.0, 1.0, 0.25), 1 + math.exp(0.5), em(2, 0.0, 1.0, 0.25)
if not (abs(ex[0] - ex[1]) < 1e-12 and ex[2] > ex[0] + 1e-6):
    bad.append(f"Codex example {ex}")
for trial in range(3000):
    n = rng.randint(1, 8)
    a = rng.uniform(-3, 1)
    b = a + rng.uniform(0.01, 5)
    mu = rng.uniform(a, b)
    Z = zmax(n, a, b, mu)
    t = n * (mu - a) / (b - a)
    r = min(int(t), n)
    cons = [a] * (n - r - 1) + [b] * r + [a + (t - r) * (b - a)] if r < n else [b] * n
    if abs(sum(cons) / n - mu) > 1e-9 or abs(sum(map(math.exp, cons)) - Z) > 1e-9 * Z:
        bad.append(f"construction n={n}")
    if Z > em(n, a, b, mu) * (1 + 1e-12):
        bad.append(f"zmax above EM n={n}")
    for _ in range(20):
        s = feasible_sample(n, a, b, mu, rng)
        if s is not None and sum(map(math.exp, s)) > Z * (1 + 1e-9):
            bad.append(f"feasible point above zmax n={n}")
            break
print("Codex example: zmax", round(ex[0], 6), "= 1+e^0.5", round(ex[1], 6), "< EM", round(ex[2], 6))
print("PROBLEMS", bad[:5] if bad else "none")
sys.exit(1 if bad else 0)
