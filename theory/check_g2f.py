"""Check (1) old vs new identical for existing methods, (2) +g2f reads exactly the baseline's tokens,
(3) swap logging counts. Usage: check_g2f.py OLD.py NEW.py"""
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
Hkv, g, d, B = 2, 3, 64, 32
H = Hkv * g
ok = True
for n in (1301, 1349, 2050, 16376):
    Kkv = torch.randn(Hkv, n, d) * 0.3
    for pos in (B * 9 - 2, B * 9 - 1, B * 9, B * 9 + 1):
        Kkv[:, pos] += 2.0
    K = Kkv.repeat_interleave(g, 0)
    q = torch.randn(H, d)
    s = torch.einsum("hd,hnd->hn", q, K) / math.sqrt(d)
    Wkv = torch.linalg.qr(torch.randn(Hkv, d, 8, generator=torch.Generator().manual_seed(1)))[0]
    for mod in (old, new):
        S = mod.S
        S.W = {0: Wkv.repeat_interleave(g, 0)}
        S.summ, S.block, S.local, S.budget, S.base_shift, S.shift = {}, B, 64, 512, 0, 0
        S.force = {"head": 8, "tail": 9, "adj": 7}
        S.group = g
    for m in ("cobs", "quest", "nf_pt1", "cobs+ftail", "cobs+nbr", "quest+g2"):
        a = old.token_mask(old.block_scores(m, q, K, s, 0), n)
        b = new.token_mask(new.block_scores(m, q, K, s, 0), n)
        if not bool((a == b).all()):
            ok = False
            print("MISMATCH", n, m)
    for base in ("cobs", "nf_pt1", "quest"):
        tb = new.token_mask(new.block_scores(base, q, K, s, 0), n).sum(1)
        tf = new.token_mask(new.block_scores(base + "+g2f", q, K, s, 0), n).sum(1)
        tg = new.token_mask(new.block_scores(base + "+g2", q, K, s, 0), n).sum(1)
        eq = bool((tb == tf).all())
        ok &= eq
        print(f"n={n} {base:7s} baseline {tb.tolist()[:3]} g2f {tf.tolist()[:3]} old-style g2 {tg.tolist()[:3]} equal={eq}")
print("swap log:", new.S.swaps)
print("ALL OK" if ok else "PROBLEM")
sys.exit(0 if ok else 1)
