"""Theorem 1, last step: among two-value configurations (k scores at a, n-k at b) with mean 0
and population variance s^2, A = logsumexp is maximised at k = 1 (a > b) and minimised at
k = n-1 (i.e. one low value). Enumerate all k for a grid of (n, s)."""
import numpy as np

viol = 0
for n in range(2, 129):
    for s in np.linspace(0.05, 12, 120):
        vals = []
        for k in range(1, n):
            a = s * np.sqrt((n - k) / k)          # k values at +a, n-k at -b, mean 0, var s^2
            b = s * np.sqrt(k / (n - k))
            vals.append(np.logaddexp(np.log(k) + a, np.log(n - k) - b))
        vals = np.array(vals)
        if vals.argmax() != 0 or vals.argmin() != n - 2:
            viol += 1
print("grid points violating (max at k=1, min at k=n-1):", viol)
