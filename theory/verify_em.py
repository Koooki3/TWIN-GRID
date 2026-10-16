"""Check the Edmundson-Madansky block score: for random/heavy-tailed key blocks and queries,
log mean exp(s) <= em_upper(mu, a, b) <= b (Quest), with a/b from the per-channel box. Exit 1 on failure."""
import math
import sys

import torch

sys.path.insert(0, __file__.rsplit("/theory/", 1)[0] + "/src")
from sparse_attn import em_upper  # noqa: E402

torch.manual_seed(0)
fails, gap_q, gap_em = 0, [], []
for trial in range(2000):
    B, d = 32, 64
    K = torch.randn(B, d, dtype=torch.float64)
    if trial % 2:
        K[torch.randint(B, (1,))] += 4 * torch.randn(d, dtype=torch.float64)     # one outlier token
    q = torch.randn(d, dtype=torch.float64) * (0.5 + 2 * torch.rand(1, dtype=torch.float64))
    sc = 1 / math.sqrt(d)
    s = K @ q * sc
    true = (s.logsumexp(0) - math.log(B)).item()
    mn, mx = K.amin(0), K.amax(0)
    b = (torch.maximum(q * mn, q * mx).sum() * sc)
    a = (torch.minimum(q * mn, q * mx).sum() * sc)
    mu = (K.mean(0) @ q * sc)
    em = em_upper(mu, a, b).item()
    if not (true <= em + 1e-9 and em <= b.item() + 1e-9):
        fails += 1
    gap_q.append(b.item() - true)
    gap_em.append(em - true)
print(f"violations {fails}/2000; mean slack quest {sum(gap_q)/len(gap_q):.3f}, em {sum(gap_em)/len(gap_em):.3f}")
sys.exit(1 if fails else 0)
