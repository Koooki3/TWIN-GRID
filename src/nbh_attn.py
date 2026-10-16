"""cobs+nbh — half-block expansion on both sides, the zero-summary-memory dual of TWIN-GRID (docs/THEORY_ADDENDUM_ZH.md
L1'; user follow-up and reviewer batch 7). Development only; sparse_attn.py is frozen, so this module wraps
sparse_attn.block_scores at import time and handles one new method, "cobs+nbh"; every other method goes to the original
function unchanged (delim_attn's wrapper, if also imported, stays in the chain).
Seeds: grid-0 candidate blocks ranked by the cobs score (the same scores cobs ranks by). Each seed b (tokens
[b0, b0+B)) reads the window [b0-h, b0+B+h) clipped to the candidate region [U, fe) (h = B/2). Seeds are taken in
descending score; a window's tokens not yet selected are added while the de-duplicated candidate token count stays
<= C = B * min(nc, kfix) (the budget every method reads); the first window that does not fit contributes its
not-yet-selected tokens in position order up to C, so exactly C candidate tokens are read and sparse_attn's per-call
strict budget / fixed-region assertion applies unchanged. Summary memory = cobs (no extra summaries); per-seed read
= up to 2B, so about half as many seeds as cobs.
Geometric property (L1'): any unit of length <= h+1 that intersects the seed block lies inside its window.
"""
import torch

import sparse_attn
from sparse_attn import S, _U, _fixed_end, _kfix, _layout

_prev_block_scores = sparse_attn.block_scores
NBH = {"fill": None}          # optional per-sample stats: [calls, fill tokens, budget tokens]


def cobs_nbh(q, K, s, layer):
    H, n, _ = K.shape
    B, U, h = S.block, _U(), S.block // 2
    fe = _fixed_end(n)
    nc, c0, _ = _layout(n)
    C = B * min(nc, _kfix(n))
    mask = torch.zeros(H, n, dtype=torch.bool, device=K.device)
    mask[:, :U] = True
    mask[:, fe:] = True
    if C == 0 or nc == 0:
        return mask
    prev = S.capture
    S.capture = {}
    _prev_block_scores("cobs", q, K, s, layer)
    sc = S.capture["primary"].float()
    S.capture = prev
    order = torch.argsort(-sc, dim=1, stable=True).cpu()
    cand = torch.zeros(H, n, dtype=torch.bool)
    for hd in range(H):
        cnt = 0
        for j in order[hd].tolist():
            a, b = max(U, c0 + j * B - h), min(fe, c0 + j * B + B + h)
            new = (~cand[hd, a:b]).nonzero().flatten() + a
            if cnt + new.numel() <= C:
                cand[hd, new] = True
                cnt += new.numel()
            else:
                cand[hd, new[:C - cnt]] = True
                if NBH["fill"] is not None:
                    NBH["fill"][1] += C - cnt
                cnt = C
            if cnt == C:
                break
        if NBH["fill"] is not None:
            NBH["fill"][0] += 1
            NBH["fill"][2] += C
    return mask | cand.to(K.device)


def block_scores(method, q, K, s, layer):
    if method == "cobs+nbh":
        return cobs_nbh(q, K, s, layer)
    return _prev_block_scores(method, q, K, s, layer)


sparse_attn.block_scores = block_scores                       # blocksparse_attention looks the name up at call time
