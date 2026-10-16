"""Numerical check of theory v8 §2 (two-level block): exact log-mass, second-order cumulant, and the split drops.
Exit 1 on failure."""
import math
import sys

bad = []
B = 32
for Delta in (2.0, 4.0, 6.0, 8.0):
    for m in range(1, 17):
        f = m / B
        s = [Delta] * m + [0.0] * (B - m)
        exact = math.log(sum(math.exp(x) for x in s)) - math.log(B)
        mu = sum(s) / B
        var = sum((x - mu) ** 2 for x in s) / B
        if abs(exact - math.log(1 - f + f * math.exp(Delta))) > 1e-12:
            bad.append(f"F1 Delta={Delta} m={m}")
        if abs((mu + 0.5 * var) - (f * Delta + 0.5 * f * (1 - f) * Delta ** 2)) > 1e-12:
            bad.append(f"F2 Delta={Delta} m={m}")
        if max(s) != Delta:
            bad.append("F3")


def dA(m, m1, D):
    f, f1 = m / B, m1 / B
    return math.log((1 - f + f * math.exp(D)) / (1 - f1 + f1 * math.exp(D)))


def dK(m, m1, D):
    f, f1 = m / B, m1 / B
    return (f - f1) * D + 0.5 * (f * (1 - f) - f1 * (1 - f1)) * D ** 2


for D in (4.0, 6.0, 8.0):
    for m in range(2, 17):
        for m1 in range((m + 1) // 2, m):
            if dA(m, m1, D) > math.log(m / m1) + 1e-12:
                bad.append(f"dA bound m={m} m1={m1} D={D}")
# exact closed form of dK and Codex's counter-example to "dK > dA in general"
for D in (2.0, 6.0, math.log(33)):
    for m, m1 in ((7, 4), (2, 1), (12, 6)):
        f, f1 = m / B, m1 / B
        if abs(dK(m, m1, D) - (f - f1) * (D + 0.5 * (1 - f - f1) * D * D)) > 1e-12:
            bad.append(f"dK closed form m={m} m1={m1}")
ce = dA(2, 1, math.log(33)), dK(2, 1, math.log(33))
print(f"counter-example m=2->1, Delta=log33: dA={ce[0]:.3f} > dK={ce[1]:.3f}")
if not (abs(ce[0] - math.log(1.5)) < 1e-12 and ce[0] > ce[1]):
    bad.append("counter-example")
# mass-oracle rank flip (one candidate slot): whole block 30+2e^D, each fragment 31+e^D, competitor in between
D = 3.0
whole, frag = 30 + 2 * math.exp(D), 31 + math.exp(D)
comp = (whole + frag) / 2
if not (frag < comp < whole):
    bad.append("oracle flip example")
print(f"oracle flip example (D=3): fragment {frag:.2f} < competitor {comp:.2f} < whole {whole:.2f}")
print(f"example m=7->4, Delta=6: dA={dA(7, 4, 6):.3f} nats (log 7/4={math.log(7 / 4):.3f}), dK={dK(7, 4, 6):.3f} nats")
print("PROBLEMS", bad[:5] if bad else "none")
sys.exit(1 if bad else 0)
