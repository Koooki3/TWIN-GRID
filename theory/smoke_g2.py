"""CPU smoke test for +g2 (two offset grids) and +nb: a 3-token spike straddling a block boundary."""
import math
import sys

import torch

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "src")
from sparse_attn import S, block_scores, token_mask  # noqa: E402

torch.manual_seed(0)
H, d, B = 2, 64, 32
n = B * 40 + B + 64
K = torch.randn(H, n, d) * 0.3
q = torch.randn(H, d)
for pos in (B * 8 - 1, B * 8, B * 8 + 1):            # spike straddles the boundary at 8B
    K[:, pos] += 6 * q / q.norm(dim=-1, keepdim=True)
s = torch.einsum("hd,hnd->hn", q, K) / math.sqrt(d)
S.W = {0: torch.linalg.qr(torch.randn(H, d, 8))[0]}
S.summ, S.block, S.local, S.budget = {}, B, 64, B * 6 + B + 64
S.base_shift = int(sys.argv[2]) if len(sys.argv) > 2 else 0
for m in ["cobs", "cobs+nb", "cobs+g2", "quest+g2", "nf_pt1+g2"]:
    tm = token_mask(block_scores(m, q, K, s, 0), n)
    covered = [bool(tm[h, B * 8 - 1] and tm[h, B * 8] and tm[h, B * 8 + 1]) for h in range(H)]
    print(f"{m:10s} tokens read per head {tm.sum(1).tolist()} (budget {S.budget}); whole spike covered {covered}")
