"""CPU check of src/nbh_attn.py (cobs+nbh): strict per-call budget / fixed-region assertion at several n for cobs+nbh,
cobs+nb and cobs+nbr (U = 32); routing of other methods unchanged (cobs equal to the original function); L1' by brute
force: for B in {16, 32, 64}, every span of length <= B/2 + 1 intersecting block [b0, b0+B) lies in [b0-B/2, b0+B+B/2),
and some span of length B/2 + 2 intersecting the block does not. Exit 1 on failure."""
import math
import sys
import types

import torch

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "src")
import sparse_attn  # noqa: E402
_orig = sparse_attn.block_scores
import nbh_attn  # noqa: E402
from sparse_attn import S, blocksparse_attention  # noqa: E402

torch.manual_seed(0)
Hkv, g, d = 2, 6, 64
H = Hkv * g
mod = types.SimpleNamespace(layer_idx=3)
ok = True
for n in (2050, 4100, 4111):
    key = torch.randn(1, Hkv, n, d) * 0.3
    value = torch.randn(1, Hkv, n, d)
    query = torch.randn(1, H, 1, d)
    res = {}
    for m in ("cobs+nbh", "cobs+nb", "cobs+nbr"):
        S.summ, S.W, S.block, S.unit, S.local, S.budget, S.base_shift, S.shift = {}, {}, 32, 32, 64, 512, 0, 0
        S.C = {3: torch.eye(d).expand(H, d, d).clone()}
        S.method, S.strict_budget, S.tokens, S.swaps, S.force = m, True, {}, {}, None
        try:
            blocksparse_attention(mod, query, key, value, None, scaling=1 / math.sqrt(d))
            t = S.tokens[m]
            res[m] = t[4] == 0 and t[6] == 0 and t[7] > 0
        except RuntimeError as e:
            print(f"n={n} {m}: {e}")
            res[m] = False
    ok &= all(res.values())
    print(f"n={n}: strict budget {res}")
n = 4100
key = torch.randn(1, Hkv, n, d) * 0.3
S.summ, S.method = {}, "cobs"
S.W = {3: torch.linalg.qr(torch.randn(H, d, 8))[0]}
Kr = sparse_attn._repeat(key[0], g)
q = torch.randn(H, d)
s = torch.einsum("hd,hnd->hn", q, Kr) / math.sqrt(d)
a = sparse_attn.block_scores("cobs", q, Kr, s, 3)
S.summ = {}
same = bool(torch.equal(a, _orig("cobs", q, Kr, s, 3)))
ok &= same
print(f"cobs routed to the original scorer: {same}")
for B in (16, 32, 64):
    h, b0 = B // 2, 4 * B
    inside = all(x >= b0 - h and x + l - 1 < b0 + B + h
                 for l in range(1, h + 2) for x in range(b0 - l + 1, b0 + B))
    sharp = any(not (x >= b0 - h and x + h + 1 < b0 + B + h) for x in range(b0 - h - 1, b0 + B))
    ok &= inside and sharp
    print(f"L1' B={B}: all spans <= h+1 intersecting the block inside the window: {inside}; length h+2 not always: {sharp}")
print("ALL OK" if ok else "PROBLEM")
sys.exit(0 if ok else 1)
