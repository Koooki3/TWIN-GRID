"""Newcombe (1998, Stat Med 17:2635-2650) method 10: hybrid score interval for the difference of paired proportions,
no continuity correction. Frozen for prereg v3 at 95% (z = 1.959964).
Cells over n pairs: a = both right, b = A1 right / A0 wrong, c = A1 wrong / A0 right, d = both wrong.
  p1 = (a+b)/n, p2 = (a+c)/n, D = p1 - p2 = (b-c)/n;  (l_i, u_i) = Wilson score interval of p_i (no CC);
  phi = N / sqrt((a+b)(c+d)(a+c)(b+d)), N = max(ad - bc - n/2, 0) if ad > bc else ad - bc; phi = 0 when the
  product is 0 (the uncorrected phi would be method 8; Wilson margins are uncorrected in both);
  delta = sqrt((p1-l1)^2 - 2 phi (p1-l1)(u2-p2) + (u2-p2)^2),
  eps   = sqrt((u1-p1)^2 - 2 phi (u1-p1)(p2-l2) + (p2-l2)^2);
  interval = [D - delta, D + eps], clipped to [-1, 1]."""
import math

Z95 = 1.959963984540054


def wilson(x, n, z=Z95):
    p = x / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def newcombe10(a, b, c, d, z=Z95):
    n = a + b + c + d
    if n == 0:
        raise ValueError("no pairs")
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    prod = (a + b) * (c + d) * (a + c) * (b + d)
    num = a * d - b * c
    if num > 0:                                     # method 10's correction of phi (method 8 would keep num)
        num = max(num - n / 2, 0.0)
    phi = 0.0 if prod == 0 else num / math.sqrt(prod)
    delta = math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    eps = math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    D = p1 - p2
    return D, max(-1.0, D - delta), min(1.0, D + eps)
