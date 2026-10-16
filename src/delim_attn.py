"""Delimiter chunking — the main competing approach to TWIN-GRID (docs/THEORY_ADDENDUM_ZH.md; reviewer batch 5).
Development only. sparse_attn.py is frozen (plan v1.4), so this module wraps sparse_attn.block_scores at import time
and handles one new method, "delim_cobs"; every other method goes to the original function unchanged.

Chunks (recomputed when the tail start fe changes): the candidate region [U, fe) is split after every sentence-end
token (DELIM.ends: prompt positions whose decoded text contains . ? ! or a newline); a piece longer than B is cut
into B-token pieces from its start; consecutive pieces are then merged greedily while the merged length stays <= B.
Score per chunk c (same summary family as cobs, with the length term that variable lengths require, THEORY §5 P2):
  log|c| + mean_{i in c} s_i + 1/2 var_{i in c} u_i,   s = exact logits (mean only, = q . mean key), u = projected
  logits pq . W^T k (the cobs subspace S.W).
Ranking target: the estimated block LOG-MASS (hence log|c|), not a per-token density — fixed before any result
(reviewer batch 5). Sentence end = the decoded token text contains a character of {. ? ! \n}; over-splitting at decimal
points or abbreviations is absorbed by the forward merge of short pieces.
Selection per head: chunks in descending score while the cumulative length fits the candidate token budget
C = B * min(nc, kfix) (the same as every other method); the first chunk that does not fit contributes a prefix of the
remaining length, so exactly C candidate tokens are read; the per-call strict budget / fixed-region assertion of
sparse_attn.blocksparse_attention applies to this mask unchanged (it checks every method's mask). Per sample the runner
collects DELIM.stats: fill tokens (share of the candidate budget, mean and max) and chunks per candidate token (summary
count; single grid = 1/B, TWIN-GRID = 2/B).
"""
import math

import torch

import sparse_attn
from sparse_attn import S, _U, _fixed_end, _kfix, _layout


class Delim:
    ends = None          # sorted LongTensor of prompt positions ending a sentence (set per sample by the runner)
    stats = None         # per sample: dict(calls, fill_sum, fill_max, budget_sum, chunks_sum, cand_tok_sum)
    cache = {}           # fe -> (chunk_id [m], tok_off [m], lens [nch]) for the candidate region [U, fe)


DELIM = Delim()
_orig_block_scores = sparse_attn.block_scores


def sentence_ends(tok, ids):
    """Positions whose token text contains a sentence end (. ? ! or newline)."""
    pieces = tok.convert_ids_to_tokens(ids)
    txt = [tok.convert_tokens_to_string([p]) for p in pieces]
    return torch.tensor([i for i, t in enumerate(txt) if any(c in t for c in ".?!\n")], dtype=torch.long)


def chunks(c0, fe, B):
    key = (c0, fe, B)
    if key in DELIM.cache:
        return DELIM.cache[key]
    ends = [int(e) for e in (DELIM.ends.tolist() if DELIM.ends is not None else []) if c0 <= e < fe]
    seg, start = [], c0
    for e in ends + [fe - 1]:
        if e >= start:
            seg.append((start, e + 1))
            start = e + 1
    pieces = []
    for a, b in seg:                                   # cut long sentences into B-token pieces
        while b - a > B:
            pieces.append((a, a + B))
            a += B
        if b > a:
            pieces.append((a, b))
    merged = []
    for a, b in pieces:                                # merge short pieces forward while <= B
        if merged and b - merged[-1][0] <= B:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    lens = torch.tensor([b - a for a, b in merged], dtype=torch.long)
    cid = torch.repeat_interleave(torch.arange(len(merged)), lens)
    off = torch.cat([torch.arange(int(l)) for l in lens]) if len(merged) else torch.zeros(0, dtype=torch.long)
    DELIM.cache = {key: (cid, off, lens)}              # one live layout (fe changes rarely)
    return cid, off, lens


def delim_cobs(q, K, s, layer):
    H, n, d = K.shape
    B, U = S.block, _U()
    fe = _fixed_end(n)
    nc, _, _ = _layout(n)
    C = B * min(nc, _kfix(n))
    mask = torch.zeros(H, n, dtype=torch.bool, device=K.device)
    mask[:, :U] = True
    mask[:, fe:] = True
    if C == 0 or fe <= U:
        return mask
    cid, off, lens = chunks(U, fe, B)
    dev = K.device
    cid, off, lensd = cid.to(dev), off.to(dev), lens.to(dev).float()
    nch = lens.numel()
    scale = 1 / math.sqrt(d)
    pq = torch.einsum("hd,hdr->hr", q.float(), S.W[layer])
    u = torch.einsum("hr,hnr->hn", pq, torch.einsum("hnd,hdr->hnr", K[:, U:fe].float(), S.W[layer])) * scale
    sc_ = s[:, U:fe].float()
    idx = cid.expand(H, -1)
    zero = torch.zeros(H, nch, device=dev)
    mean = zero.scatter_add(1, idx, sc_) / lensd
    mu_u = zero.clone().scatter_add(1, idx, u) / lensd
    var = (zero.clone().scatter_add(1, idx, u * u) / lensd - mu_u ** 2).clamp_min(0)
    score = lensd.log() + mean + 0.5 * var                       # [H, nch]
    order = torch.argsort(-score, dim=1, stable=True)
    cum = torch.cumsum(lensd[order], dim=1)
    full = cum <= C                                              # prefix of the order that fits whole
    nfull = full.sum(1)                                          # [H]
    sel = torch.zeros(H, nch, dtype=torch.bool, device=dev)
    sel.scatter_(1, order, full)
    used = torch.where(nfull > 0, cum.gather(1, (nfull - 1).clamp_min(0)[:, None])[:, 0], torch.zeros_like(nfull,
                                                                                                        dtype=cum.dtype))
    rem = (C - used).long()                                      # tokens still to read
    has_next = nfull < nch
    nxt = order.gather(1, nfull.clamp_max(nch - 1)[:, None])[:, 0]
    tok_sel = sel.gather(1, idx)                                 # [H, m]
    part = (idx == nxt[:, None]) & (off[None] < rem[:, None]) & has_next[:, None]
    mask[:, U:fe] = tok_sel | part
    if DELIM.stats is not None:                                  # reviewer batch 5: fill share and summary count
        st = DELIM.stats
        fill = (rem * has_next.long()).float()
        st["calls"] += H
        st["fill_sum"] += float(fill.sum())
        st["fill_max"] = max(st["fill_max"], float(fill.max()) / C)
        st["budget_sum"] += C * H
        st["chunks_sum"] += nch
        st["cand_tok_sum"] += fe - U
        st["layer_calls"] += 1
    return mask


def block_scores(method, q, K, s, layer):
    if method == "delim_cobs":
        return delim_cobs(q, K, s, layer)
    return _orig_block_scores(method, q, K, s, layer)


sparse_attn.block_scores = block_scores                       # blocksparse_attention looks the name up at call time
