"""Equivalence/consistency check for the fixed sink/tail layout (Codex G1).
Usage: check_fixed_layout.py OLD.py NEW.py"""
import importlib.util
import math
import sys

import torch


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


old, new = load(sys.argv[1], "sa_old"), load(sys.argv[2], "sa_new")
torch.manual_seed(0)
H, d, B = 4, 64, 32
ok = True
for n in (1301, 1333, 1365, 2050):
    K = torch.randn(H, n, d) * 0.3
    q = torch.randn(H, d)
    s = torch.einsum("hd,hnd->hn", q, K) / math.sqrt(d)
    for mod in (old, new):
        S = mod.S
        S.W = {0: torch.linalg.qr(torch.randn(H, d, 8, generator=torch.Generator().manual_seed(1)))[0]}
        S.summ, S.block, S.local, S.budget, S.base_shift, S.shift = {}, B, 64, 512, 0, 0
    for m in ("cobs", "quest", "nf_pt1", "nf_hi1", "cobs+nb", "cobs+nbr"):
        a = old.token_mask(old.block_scores(m, q, K, s, 0), n)
        b = new.token_mask(new.block_scores(m, q, K, s, 0), n)
        same = bool((a == b).all())
        ok &= same
        if not same:
            print("MISMATCH at shift 0:", n, m)
    S = new.S
    fixed0 = None
    for sh in (0, 5, 16, 31):
        S.base_shift, S.summ = sh, {}
        tm = new.token_mask(new.block_scores("cobs", q, K, s, 0), n)
        f = torch.zeros(n, dtype=torch.bool)
        f[:B] = True
        f[new._fixed_end(n):] = True
        fixed_ok = bool(tm[:, f].all())
        cand = int(tm[0].sum() - f.sum())
        print(f"n={n} shift={sh:2d} read={tm.sum(1).tolist()} fixed={int(f.sum())} candidate_tokens={cand} "
              f"fixed_kept={fixed_ok} k_fix*B={new._kfix(n) * B}")
        ok &= fixed_ok and int(tm.sum(1).max()) <= 512
    S.base_shift = 0
print("ALL OK" if ok else "PROBLEM")
sys.exit(0 if ok else 1)
