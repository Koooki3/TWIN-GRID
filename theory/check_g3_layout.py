"""G3 layout matching (plan v1.4; Codex G3): with layout unit U = 32, block B = 16 must have exactly the fixed token
set, the selectable candidate region and the per-call read count of B = 32 (U = 32), for every n tried; the strict
per-call budget invariant must hold for cobs / nf_pt1 / quest at B = 16, U = 32 (and for cobs+g2f at B = 32); U not
a multiple of B is refused. CPU only. Exit 1 on any failure."""
import math
import sys
import types

import torch

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "src")
import sparse_attn  # noqa: E402
from sparse_attn import S, blocksparse_attention, token_mask  # noqa: E402

torch.manual_seed(0)
Hkv, g, d = 2, 6, 64
H = Hkv * g
mod = types.SimpleNamespace(layer_idx=3)
ok = True


def setup(B, U, m, budget=512):
    S.summ, S.W, S.block, S.unit, S.local, S.budget, S.base_shift, S.shift = {}, {}, B, U, 64, budget, 0, 0
    S.C = {3: torch.eye(d).expand(H, d, d).clone()}
    S.method, S.strict_budget, S.tokens, S.swaps, S.force = m, True, {}, {}, None


for n in (2050, 4100, 4111, 16376, 16413):
    for budget in (512, 1024):
        lay = {}
        for B, U in ((32, 0), (16, 32)):
            setup(B, U, "cobs", budget)
            fe = sparse_attn._fixed_end(n)
            nc, c0, c1 = sparse_attn._layout(n)
            k = min(nc, sparse_attn._kfix(n))
            lay[B] = dict(fixed=(sparse_attn._U(), fe), region=(c0, c1), read=sparse_attn._U() + (n - fe) + B * k)
        same = lay[32] == lay[16]
        ok &= same
        print(f"n={n:5d} budget={budget}: B32 {lay[32]}  B16/U32 {lay[16]}  {'match' if same else 'MISMATCH'}")
n = 4100
key = torch.randn(1, Hkv, n, d) * 0.3
key[:, :, 40 * 32 + 30: 40 * 32 + 34] += 1.5
value = torch.randn(1, Hkv, n, d)
query = torch.randn(1, H, 1, d)
for B, U, m in ((16, 32, "cobs"), (16, 32, "nf_pt1"), (16, 32, "quest"), (32, 0, "cobs+g2f"), (32, 32, "cobs+g2f")):
    setup(B, U, m)
    try:
        blocksparse_attention(mod, query, key, value, None, scaling=1 / math.sqrt(d))
        t = S.tokens.get(m)
        good = t is not None and t[4] == 0 and t[6] == 0 and t[7] > 0
    except RuntimeError as e:
        print(f"B={B} U={U} {m}: {e}")
        good = False
    ok &= good
    print(f"B={B} U={U or B} {m:9s} strict budget {'ok' if good else 'VIOLATED'} tokens={S.tokens.get(m)}")
S.block, S.unit = 16, 24
try:
    sparse_attn._fixed_end(4100)
    print("U not a multiple of B accepted (unexpected)")
    ok = False
except ValueError:
    print("U not a multiple of B refused (expected)")
S.unit = 0
print("ALL OK" if ok else "PROBLEM")
sys.exit(0 if ok else 1)
