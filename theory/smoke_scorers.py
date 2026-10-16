"""CPU smoke test for the new block scorers on a toy cache with one planted needle."""
import math
import sys

import torch

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "src")
from sparse_attn import S, block_scores  # noqa: E402

torch.manual_seed(0)
H, d, B = 2, 64, 32
n = B * 20 + B + 64
K = torch.randn(H, n, d) * 0.3
q = torch.randn(H, d)
needle = B * 7 + 5                                   # candidate block index 6
K[:, needle] += 3 * q / q.norm(dim=-1, keepdim=True) * 3
s = torch.einsum("hd,hnd->hn", q, K) / math.sqrt(d)
S.W = {0: torch.linalg.qr(torch.randn(H, d, 16))[0]}
S.summ, S.block, S.local = {}, B, 64
S.budget = B * 4 + B + 64                            # 4 candidate blocks
for m in ["o_mass", "cobs", "cobs+nb", "quest", "quest+nb", "nf_pt1", "nf_pt1+nb"]:
    sel = block_scores(m, q, K, s, 0)
    print(f"{m:10s} picks per head: {[torch.nonzero(sel[h]).flatten().tolist() for h in range(H)]}")
