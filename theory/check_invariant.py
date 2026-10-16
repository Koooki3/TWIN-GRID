"""V1: per-call budget/fixed-region invariant inside blocksparse_attention (CPU, strict mode).
Expect: no violation for cobs/quest/nf_pt1/cobs+g2f/cobs+ftail/cobs+nbr; old +g2 violates (under-fill);
g2f refuses odd B. Exit 1 on any unexpected outcome."""
import math
import sys
import types

import torch

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "src")
import sparse_attn  # noqa: E402
from sparse_attn import S, blocksparse_attention  # noqa: E402

torch.manual_seed(0)
Hkv, g, d, B, n = 2, 6, 64, 32, 4100
H = Hkv * g
key = torch.randn(1, Hkv, n, d) * 0.3
key[:, :, B * 40 + 30: B * 40 + 34] += 1.5
value = torch.randn(1, Hkv, n, d)
query = torch.randn(1, H, 1, d)
mod = types.SimpleNamespace(layer_idx=3)
ok = True
for m, expect_violation in (("cobs", False), ("quest", False), ("nf_pt1", False), ("cobs+g2f", False),
                            ("cobs+ftail", False), ("cobs+nbr", False), ("quest+g2", True)):
    S.summ, S.W, S.block, S.local, S.budget, S.base_shift, S.shift = {}, {}, B, 64, 512, 0, 0
    S.C = {3: torch.eye(d).expand(H, d, d).clone()}
    S.force = {"head": 40, "tail": 41, "adj": 39}
    S.method, S.strict_budget, S.tokens, S.swaps = m, True, {}, {}
    try:
        blocksparse_attention(mod, query, key, value, None, scaling=1 / math.sqrt(d))
        violated = False
    except RuntimeError as e:
        if "budget invariant violated" not in str(e):
            raise                                   # unexpected errors must not count as "no violation"
        violated = True
    t = S.tokens.get(m)
    print(f"{m:11s} violation={violated} (expected {expect_violation}) tokens={t} swaps={S.swaps.get(m)}")
    ok &= violated == expect_violation
    if m == "cobs+ftail":                           # forced tail block absent from every head's selection here
        ok &= S.swaps.get(m) == [H, H]
S.block = 31
S.method, S.strict_budget = "cobs+g2f", False
try:
    blocksparse_attention(mod, query, key, value, None, scaling=1 / math.sqrt(d))
    print("odd B accepted (unexpected)"); ok = False
except ValueError:
    print("odd B refused for g2f (expected)")
print("ALL OK" if ok else "PROBLEM")
sys.exit(0 if ok else 1)
