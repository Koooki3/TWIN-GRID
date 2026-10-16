"""Old vs new sparse_attn: identical selections with per-KV-group summary dedupe (grouped toy, shared W).
Usage: check_dedupe.py OLD.py NEW.py"""
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
for n in (1301, 2050):
    Kkv = torch.randn(Hkv, n, d) * 0.3
    Kkv[:, B * 9 + 3] += 2.0
    K = Kkv.repeat_interleave(g, 0)
    q = torch.randn(H, d)
    s = torch.einsum("hd,hnd->hn", q, K) / math.sqrt(d)
    Wkv = torch.linalg.qr(torch.randn(Hkv, d, 8, generator=torch.Generator().manual_seed(1)))[0]
    for mod in (old, new):
        S = mod.S
        S.W = {0: Wkv.repeat_interleave(g, 0)}
        S.summ, S.block, S.local, S.budget, S.base_shift, S.shift = {}, B, 64, 512, 0, 0
        S.force = {"head": 8, "tail": 9, "adj": 7}
        if hasattr(S, "group"):
            S.group = g
    for m in ("cobs", "quest", "nf_pt1", "nf_hi1", "zdec1", "cobs+ftail", "cobs+nbr", "quest+g2", "mean", "rem"):
        for rep in range(2):                       # second call exercises the cached (deduped) path
            a = old.token_mask(old.block_scores(m, q, K, s, 0), n)
            b = new.token_mask(new.block_scores(m, q, K, s, 0), n)
            same = bool((a == b).all())
            ok &= same
            if not same:
                print("MISMATCH", n, m, rep)
    big_old = sum(v.numel() for v in old.S.summ.values() if torch.is_tensor(v))
    big_new = sum(v.numel() for v in new.S.summ.values() if torch.is_tensor(v))
    print(f"n={n}: cached tensor elements old {big_old} new {big_new}")
print("ALL OK" if ok else "PROBLEM")
sys.exit(0 if ok else 1)
