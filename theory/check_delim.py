"""CPU check of src/delim_attn.py (delimiter chunking, development baseline): chunks tile the candidate region [U, fe)
exactly, every chunk has 1 <= length <= B; delim_cobs passes sparse_attn's strict per-call budget / fixed-region
assertion at several n; other methods are routed to the original scorer (cobs mask unchanged with the module loaded);
fill statistics are collected. Exit 1 on failure."""
import math
import sys
import types

import torch

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "src")
import sparse_attn  # noqa: E402
_orig = sparse_attn.block_scores
import delim_attn  # noqa: E402
from sparse_attn import S, blocksparse_attention  # noqa: E402

torch.manual_seed(0)
Hkv, g, d = 2, 6, 64
H = Hkv * g
mod = types.SimpleNamespace(layer_idx=3)
ok = True
for n in (2050, 4100, 4111):
    gaps = torch.randint(3, 60, (n // 3,))
    ends = torch.cumsum(gaps, 0)
    delim_attn.DELIM.ends = ends[ends < n]
    delim_attn.DELIM.cache = {}
    delim_attn.DELIM.stats = dict(calls=0, fill_sum=0.0, fill_max=0.0, budget_sum=0, chunks_sum=0, cand_tok_sum=0,
                                  layer_calls=0)
    S.summ, S.W, S.block, S.unit, S.local, S.budget, S.base_shift, S.shift = {}, {}, 32, 32, 64, 512, 0, 0
    fe = sparse_attn._fixed_end(n)
    cid, off, lens = delim_attn.chunks(32, fe, 32)
    tiles = int(lens.sum()) == fe - 32 and bool((lens >= 1).all()) and bool((lens <= 32).all()) \
        and cid.numel() == fe - 32 and bool((off < lens[cid]).all())
    ok &= tiles
    key = torch.randn(1, Hkv, n, d) * 0.3
    value = torch.randn(1, Hkv, n, d)
    query = torch.randn(1, H, 1, d)
    res = {}
    for m in ("delim_cobs", "cobs"):
        S.C = {3: torch.eye(d).expand(H, d, d).clone()}
        S.W = {}
        S.method, S.strict_budget, S.tokens, S.swaps, S.force = m, True, {}, {}, None
        try:
            blocksparse_attention(mod, query, key, value, None, scaling=1 / math.sqrt(d))
            t = S.tokens[m]
            res[m] = t[4] == 0 and t[6] == 0 and t[7] > 0
        except RuntimeError as e:
            print(f"n={n} {m}: {e}")
            res[m] = False
    ok &= all(res.values())
    st = delim_attn.DELIM.stats
    print(f"n={n}: chunks {lens.numel()} (len {int(lens.min())}-{int(lens.max())}) tile [U, fe): {tiles}; strict budget "
          f"{res}; fill share mean {st['fill_sum'] / max(st['budget_sum'], 1):.3f} max {st['fill_max']:.3f}; "
          f"chunks per candidate token {st['chunks_sum'] / max(st['cand_tok_sum'], 1):.4f}")
# routing: with the module loaded, cobs gives the same selection as the original function
n = 4100
key = torch.randn(1, Hkv, n, d) * 0.3
S.summ, S.method = {}, "cobs"
S.W = {3: torch.linalg.qr(torch.randn(H, d, 8))[0]}
Kr = sparse_attn._repeat(key[0], g)
q = torch.randn(H, d)
s = torch.einsum("hd,hnd->hn", q, Kr) / math.sqrt(d)
a = sparse_attn.block_scores("cobs", q, Kr, s, 3)
S.summ = {}
b = _orig("cobs", q, Kr, s, 3)
same = bool(torch.equal(a, b))
ok &= same
print(f"cobs routed to the original scorer: {same}")
print("ALL OK" if ok else "PROBLEM")
sys.exit(0 if ok else 1)
