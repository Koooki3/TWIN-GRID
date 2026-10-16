"""CPU smoke test for sparse_attn._diag on a toy cache with one planted needle."""
import math
import sys

import torch

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "src")
import sparse_attn  # noqa: E402
from sparse_attn import S  # noqa: E402

torch.manual_seed(0)
H, d, B = 2, 64, 32
n = B * 40 + B + 64
K = torch.randn(H, n, d) * 0.3
q = torch.randn(H, d)
needle = B * 7 + 5
K[:, needle] += 9 * q / q.norm(dim=-1, keepdim=True)
s = torch.einsum("hd,hnd->hn", q, K) / math.sqrt(d)
S.W = {3: torch.linalg.qr(torch.randn(H, d, 8))[0]}
S.summ, S.block, S.local, S.budget = {}, B, 64, B * 6 + B + 64
S.needle_mask = torch.zeros(n + 64, dtype=torch.bool)
S.needle_mask[needle] = True
S.diag = {}
S.C = {3: torch.eye(d).expand(H, d, d).clone()}
sparse_attn._diag(3, s, q, K)
for k, v in S.diag.items():
    if k == 3:
        v = {kk: (vv if kk != "margin" else len(vv)) for kk, vv in v.items()}
    print(k, v)
