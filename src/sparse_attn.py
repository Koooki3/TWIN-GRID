"""Block-sparse decode attention for HF transformers, with pluggable block scorers.

Prefill (q_len > 1) is dense SDPA. Decode steps (q_len == 1) select, per query
head, `k` candidate blocks of size B plus an always-kept sink block and local
window; attention is the exact softmax restricted to the selected tokens
(simulated by masking, so this measures accuracy, not speed).

Scorers (higher = keep):
  oracle (true scores s_i = q.k_i * scale):
    o_mass  logsumexp_b(s)            o_peak  max_b(s)        o_beta2  logsumexp(2s)/2
    o_dual  ceil(rho*k) by peak, rest by mass
  summary-based (query-independent per-block summaries):
    quest   sum_c max(q_c*min_c, q_c*max_c)        (per-channel min/max)
    mean    q.mu_b
    cobs    q.mu_b + 1/2 * var_b((P q).(P k))      (2nd order, rank-r query subspace P)
    sum_dual  ceil(rho*k) by quest, rest by cobs
    em      Edmundson-Madansky upper bound on block mass from mean + box (tightest given mean and range)
  sketch-based (per-token r-dim sketch W^T k, W = top-r query PCA of the head):
    sk_mass, sk_peak, sk_beta2, sk_dual
"""
import math

import torch
import torch.nn.functional as F
from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward


class State:
    method = "dense"
    budget = 1024          # tokens read per head per step, incl. sink + local
    block = 32
    unit = 0               # layout unit U (0 = block): sink [0, U), tail start and candidate token budget on the U grid
    local = 128
    rho = 0.5              # share of slots chosen by the peak channel
    rank = 16              # sketch / COBS subspace rank
    skip_layers = 2        # first layers stay dense (as in Quest)
    sparse_layers = None   # optional set of layers where sparse selection is applied (causal probes)
    shift = 0              # grid offset used by _layout (set temporarily by the +g2 wrapper)
    base_shift = 0         # per-sample selection-grid offset (grid-shift intervention); positions unchanged
    w_per_head = False     # query subspace per query head (True) or shared per KV head (False, default)
    group = 1              # query heads per KV head (set at each decode step)
    tokens = {}            # method -> [min, max, sum, count, budget_viol, max_abs_err, fixed_viol] per (step, layer, head)
    strict_budget = False  # raise on any per-call budget / fixed-region violation (phase 1/2 runs)
    dd = None              # per-digit mechanism logging state (see _digit_diag)
    in_diag = False        # True while _digit_diag re-evaluates scorers (no swap logging then)
    swaps = {}             # forced-block method -> [(layer, head) calls, actual swaps]
    force = None           # {"head": blk, "tail": blk, "adj": blk} candidate-block indices for +f* probes
    em_viol = [0, 0]       # em: [block values, tolerance violations] (see em_upper)
    ptr_stats = {}         # +ptr: method -> [head-calls, ready, copy, successor already selected, swapped in]
    ptr = {}               # +ptr: layer -> (n of last step, [H, PTR_WIN] recent attention-peak positions)
    W = {}                 # layer -> [H, d, r] query-PCA basis, built at the first decode step
    C = {}                 # layer -> [H, d, d] prefill query second moment (summed over chunks)
    summ = {}              # (layer, nc, name) -> cached block summary
    collect = None         # dict to fill with offline stats on the next decode step
    needle_mask = None     # bool [n] of target-needle token positions (prompt coordinates)
    diag = None            # dict to fill with H1 / outlier diagnostics on decode steps
    capture = None         # set to {} to record the scores a scorer ranks by
    mech = None            # set to {} for per-decode-step mechanism metrics (see _mech)
    mech_step = 0
    stat_methods = ()


S = State()


def _repeat(x, n_rep):
    return x if n_rep == 1 else x.repeat_interleave(n_rep, dim=0)


def _U():
    """Layout unit U (plan v1.4, Codex G3 matching): the sink [0, U), the tail start and the candidate token budget are
    defined on the U grid, candidate blocks keep length B (U a multiple of B). U = B (default, unit = 0) is the original
    layout, bit for bit; B = 16 with U = 32 has exactly the fixed token set, the selectable candidate region and the
    candidate token count of B = 32."""
    U = S.unit or S.block
    if U % S.block:
        raise ValueError(f"layout unit {U} is not a multiple of the block length {S.block}")
    return U


def _fixed_end(n):
    """Start of the always-kept tail, from the UNSHIFTED U grid. The always-kept set is
    [0, U) + [_fixed_end(n), n) for every grid offset (Codex G1: shifts must not change sink/local)."""
    U = _U()
    return U + max(0, (n - S.local) // U - 1) * U


def _kfix(n):
    """Number of candidate blocks of length B the budget allows: the candidate token budget left by the fixed sink and
    tail, floored to whole U units (same for all offsets; = the original // B when U = B)."""
    U = _U()
    return max(0, ((S.budget - U - (n - _fixed_end(n))) // U) * U // S.block)


def _layout(n):
    B, U = S.block, _U()
    sh = (S.base_shift + S.shift) % B          # grid offset: per-sample base (intervention) + B/2 for the +g2 grid
    c1f = _fixed_end(n)
    nc = max(0, (c1f - U - sh) // B)           # full candidate blocks between the fixed sink and fixed tail
    c0 = U + sh
    c1 = c0 + nc * B                           # candidate token range; [U, c0) and [c1, c1f) unselectable
    return nc, c0, c1


def _topk_union(primary, secondary, k, k1):
    """Pick k1 by primary, then fill to k by secondary. [H, nc] -> bool [H, nc]."""
    if S.capture is not None:
        S.capture["primary"], S.capture["secondary"] = primary, secondary
    sel = torch.zeros_like(primary, dtype=torch.bool)
    if k1 > 0:
        sel.scatter_(1, primary.topk(k1, dim=1).indices, True)
    if k - k1 > 0:
        sec = secondary.masked_fill(sel, float("-inf"))
        sel.scatter_(1, sec.topk(k - k1, dim=1).indices, True)
    return sel


def _adaptive(mass, peak, k):
    """Mass top-k, then swap in "surprise" blocks: unselected blocks whose peak exceeds
    every peak already selected (per head), up to k/2, replacing the lowest-mass picks."""
    sel = _topk_union(mass, mass, k, 0)
    if k == 0:
        return sel
    cap = k // 2
    pmax = peak.masked_fill(~sel, float("-inf")).amax(1, keepdim=True)
    sur = peak.masked_fill(sel | (peak <= pmax), float("-inf"))
    n_sur = (sur > float("-inf")).sum(1).clamp(max=cap)           # [H]
    for h in torch.nonzero(n_sur).flatten().tolist():
        m = int(n_sur[h])
        add = sur[h].topk(m).indices
        keep = ~sel[h] | (peak[h] == pmax[h])                      # never drop the top-peak pick
        drop = mass[h].masked_fill(keep, float("inf")).topk(m, largest=False).indices
        sel[h, drop] = False
        sel[h, add] = True
    return sel


def _dedupe(x, g, store):
    """Keep one query head per KV group (store=True) or expand back to all heads (store=False)."""
    if isinstance(x, tuple):
        return tuple(_dedupe(v, g, store) for v in x)
    if not torch.is_tensor(x) or g == 1:
        return x
    return x[::g].contiguous() if store else x.repeat_interleave(g, dim=0)


def _summ(layer, nc, name, fn):
    """Per-sample cache of query-independent block summaries (cleared via S.summ = {}).
    When the query subspace is shared per KV head, every summary is identical within a KV group, so only one
    head per group is stored (memory /g); per-query-head subspaces disable this."""
    key = (layer, nc, name, (S.base_shift + S.shift) % S.block)
    g = 1 if S.w_per_head else S.group
    if key not in S.summ:
        val = fn()
        S.summ[key] = _dedupe(val, g, True)
        return val
    return _dedupe(S.summ[key], g, False)


def _dilate(sc, k, side="best"):
    """Neighbour expansion: take the top k//2 blocks by sc, add for each a neighbour
    (side = "best": the better-scored one; "right": the following block, where a straddled tail
    lies), fill the rest by sc."""
    H, nc = sc.shape
    sel = torch.zeros_like(sc, dtype=torch.bool)
    k1 = k // 2
    if k1 > 0:
        top = sc.topk(k1, dim=1).indices                                  # [H,k1]
        left = (top - 1).clamp(min=0)
        right = (top + 1).clamp(max=nc - 1)
        if side == "right":
            nbr = right
        else:
            sl, sr = torch.gather(sc, 1, left), torch.gather(sc, 1, right)
            nbr = torch.where(sr >= sl, right, left)
        sel.scatter_(1, top, True)
        sel.scatter_(1, nbr, True)
    rem = k - int(sel.sum(1).min())
    if rem > 0:
        filled = sel.sum(1)
        rest = sc.masked_fill(sel, float("-inf"))
        order = rest.argsort(1, descending=True)
        for h in range(H):
            need = k - int(filled[h])
            if need > 0:
                sel[h, order[h, :need]] = True
    return sel


def _g2_fill(base, q, K, s, layer):
    """g2_fill_v1. Candidates = blocks of grid 0 (offset 0) and grid 1 (offset B/2) inside the fixed region,
    scored by `base` (same block length B, so scores are comparable). Greedy in descending score (ties: grid 0
    first, then lower index); a block is added if its NEW tokens fit the remaining capacity; capacity is the
    baseline's candidate-token count C = B * _kfix(n), so every method reads exactly the same number of
    candidate tokens when enough candidates exist. Coverage is tracked in half-blocks (B/2 tokens)."""
    if S.block % 2 or S.base_shift:
        raise ValueError("g2_fill_v1 is defined (and proved exact) only for even B and base_shift = 0")
    n = K.shape[1]
    B, half, U = S.block, S.block // 2, _U()
    C_units = 2 * min(_layout(n)[0], _kfix(n))                 # capacity in half-block units
    scored = []
    for gi, sh in ((0, 0), (1, half)):
        S.shift = sh
        prev = S.capture
        S.capture = {}
        block_scores(base, q, K, s, layer)
        sc = S.capture["primary"]
        S.capture = prev
        ncs, c0s, c1s = _layout(n)
        S.shift = 0
        u0 = (c0s - U) // half                                 # first half-unit index of this grid's block 0
        scored.append((gi, sc, u0, ncs))
    H = K.shape[0]
    n_units = (_fixed_end(n) - U) // half
    mask = torch.zeros(H, n, dtype=torch.bool, device=K.device)
    mask[:, :U] = True
    mask[:, _fixed_end(n):] = True
    allsc = torch.cat([sc for _, sc, _, _ in scored], 1).float().cpu()
    units = torch.cat([torch.arange(ncs) * 2 + u0 for _, _, u0, ncs in scored])
    order = torch.argsort(-allsc, dim=1, stable=True)          # ties keep grid-0 / lower-index first
    for h in range(H):
        cov = torch.zeros(n_units + 2, dtype=torch.bool)
        used = 0
        for j in order[h].tolist():
            if used >= C_units:
                break
            u = int(units[j])
            new_u = int(not cov[u]) + int(not cov[u + 1])
            if new_u == 0 or used + new_u > C_units:
                continue
            cov[u] = cov[u + 1] = True
            used += new_u
        idx = torch.nonzero(cov[:n_units]).flatten()
        for u in idx.tolist():
            mask[h, U + u * half:U + (u + 1) * half] = True
    return mask


def em_upper(mu, a, b, tol=1e-3):
    """log of the Edmundson-Madansky bound on mean(exp(s_i)) given the exact mean mu and s_i in [a, b] (log B omitted:
    the block log-mass bound is log B + this). U = log((1-w) e^a + w e^b), w = (mu-a)/(b-a); U <= b (Quest), U = mu
    when a = b. Exploration-grade float32, not a rounding-safe certificate: mu outside [a-tol, b+tol] or non-finite
    inputs are counted in S.em_viol = [values, violations] and then clipped (Codex C1)."""
    width = b - a
    viol = ~torch.isfinite(mu) | ~torch.isfinite(a) | ~torch.isfinite(b) | (width < -tol) | (mu < a - tol) | (mu > b + tol)
    S.em_viol[0] += viol.numel()
    S.em_viol[1] += int(viol.sum())
    flat = width.abs() <= 1e-6 * (1 + b.abs())
    w = ((mu - a) / torch.where(flat, torch.ones_like(width), width)).clamp(0, 1)
    U = torch.logaddexp(torch.log(w) + b, torch.log1p(-w) + a)
    return torch.where(flat, mu, U)


PTR_WIN, PTR_MOVES = 8, 2   # KVFetch-style copy-mode test: pointer advanced >= PTR_MOVES times in the last PTR_WIN steps


def _ptr_update(layer, s_sel, n):
    """Record this step's attention peak (over the candidate region, from the scores actually attended)."""
    nc, c0, c1 = _layout(n)
    if c1 <= c0:
        return
    pk = s_sel[:, c0:c1].argmax(1) + c0                      # [H]
    last = S.ptr.get(layer)
    hist = pk[:, None] if last is None or last[0] != n - 1 else torch.cat([last[1], pk[:, None]], 1)[:, -PTR_WIN:]
    S.ptr[layer] = (n, hist)


def _ptr_swap(method, sel, sc, layer, n):
    """Online successor rule, a block-level baseline INSPIRED BY KVFetch (2610.08811), not a faithful reproduction:
    for heads in copy mode, force the candidate block holding pointer+1 into the selection, swapping out the
    lowest-scored selected block (budget unchanged).
    Copy mode (design choice): the history keeps the last PTR_WIN = 8 peak positions (so at most 7 moves); copy mode
    needs >= PTR_MOVES of those moves to be 1..4 tokens forward AND the latest move to be 0..4 tokens forward. A large
    or backward latest move suppresses copy mode for that step only; earlier history is kept (this is NOT "restart
    counting after a jump").
    Counters S.ptr_stats[method] = [head-calls, history ready, copy mode, successor already selected, swapped in]
    (not updated under diagnostics, which still see the same mask). Runners reset S.ptr / S.ptr_stats at the start of
    every prompt/method/mode (Codex: length continuity alone is not identity)."""
    H, nc = sel.shape
    st = None if S.in_diag else S.ptr_stats.setdefault(method, [0, 0, 0, 0, 0])
    if st is not None:
        st[0] += H
    last = S.ptr.get(layer)
    if nc == 0 or not bool(sel.any()) or last is None or last[0] != n - 1 or last[1].shape[1] < 3:
        return sel                                            # nothing to swap / new sequence / no history yet
    hist = last[1]
    step = hist[:, 1:] - hist[:, :-1]
    copy = (((step >= 1) & (step <= 4)).sum(1) >= PTR_MOVES) & (step[:, -1] >= 0) & (step[:, -1] <= 4)
    _, c0, c1 = _layout(n)
    b = (hist[:, -1] + 1 - c0) // S.block
    ok = copy & (hist[:, -1] + 1 < c1) & (b >= 0) & (b < nc)
    have = sel.gather(1, b.clamp(0, nc - 1)[:, None])[:, 0]
    rows = torch.nonzero(ok & ~have).flatten()
    if rows.numel():
        low = sc.masked_fill(~sel, float("inf")).argmin(1)
        sel[rows, low[rows]] = False
        sel[rows, b[rows]] = True
    if st is not None:
        st[1] += H
        st[2] += int(copy.sum())
        st[3] += int((ok & have).sum())
        st[4] += int(rows.numel())
        rec = S.swaps.setdefault(method, [0, 0])
        rec[0] += H
        rec[1] += int(rows.numel())
    return sel


def block_scores(method, q, K, s, layer):
    """q [H,d] (unscaled), K [H,n,d] (repeated to q heads), s [H,n] true scaled scores.
    Returns bool selection [H, nc] over candidate blocks."""
    if "+ptr" in method and not method.endswith("+ptr"):         # e.g. quest+ptr+g2f: refused, never half-applied
        raise ValueError(f"{method}: +ptr must be the only wrapper")
    if method.endswith("+ptr"):                                # online read-pointer successor (single grid)
        base = method[:-4]
        if "+" in base:
            raise ValueError(f"{method}: +ptr must be the only wrapper")
        prev = S.capture
        S.capture = {}
        sel = block_scores(base, q, K, s, layer)
        sc = S.capture["primary"]
        S.capture = prev
        return _ptr_swap(method, sel, sc, layer, K.shape[1])
    if method.endswith("+g2f"):                                # g2_fill_v1: exact candidate-token budget
        return _g2_fill(method[:-4], q, K, s, layer)
    if method.endswith("+g2"):                                 # two grids offset by B/2 -> token mask
        n = K.shape[1]
        nc0, c00, c10 = _layout(n)
        k_ = min(nc0, _kfix(n))
        mask = torch.zeros(K.shape[0], n, dtype=torch.bool, device=K.device)
        for sh, kk in ((0, k_ - k_ // 2), (S.block // 2, k_ // 2)):
            S.shift = sh
            prev = S.capture
            S.capture = {}
            block_scores(method[:-3], q, K, s, layer)
            sc = S.capture["primary"]
            S.capture = prev
            ncs, c0s, c1s = _layout(n)
            sel = torch.zeros_like(sc, dtype=torch.bool)
            if kk > 0:
                sel.scatter_(1, sc.topk(min(kk, ncs), dim=1).indices, True)
            mask[:, c0s:c1s] |= sel.repeat_interleave(S.block, dim=1)
            S.shift = 0
        nc0, c00, c10 = _layout(n)
        mask[:, :_U()] = True                                  # fixed sink
        mask[:, _fixed_end(n):] = True                         # fixed tail (local window)
        return mask
    if method.endswith(("+nb", "+nbr")):                       # neighbour expansion of a single-score scorer
        base, side = (method[:-3], "best") if method.endswith("+nb") else (method[:-4], "right")
        prev = S.capture
        S.capture = {}
        block_scores(base, q, K, s, layer)
        sc = S.capture["primary"]
        S.capture = prev
        nc_, c0_, c1_ = _layout(K.shape[1])
        k_ = min(nc_, _kfix(K.shape[1]))
        return _dilate(sc, k_, side)
    if method.endswith(("+fhead", "+ftail", "+fadj")):        # causal probe: force one block into the selection
        base, which = method.rsplit("+f", 1)
        sel = block_scores(base, q, K, s, layer)
        prev = S.capture
        S.capture = {}
        block_scores(base, q, K, s, layer)
        sc = S.capture["primary"]
        S.capture = prev
        b = S.force.get(which) if S.force else None            # candidate-block index (grid 0)
        if b is not None and 0 <= b < sel.shape[1]:
            miss = ~sel[:, b]
            rec = S.swaps.setdefault(method, [0, 0])                # [(layer, head) calls, actual swaps]
            rec[0] += sel.shape[0]
            if miss.any():                                     # swap out the lowest-scored selected block
                low = sc.masked_fill(~sel, float("inf")).argmin(1)
                rows = torch.nonzero(miss).flatten()
                sel[rows, low[rows]] = False
                sel[rows, b] = True
                rec[1] += int(rows.numel())
                if S.dd is not None and not S.in_diag:          # per-swap log for the mechanism diagnostics
                    S.dd.setdefault("swap_log", []).append(dict(
                        step=S.dd.get("step"), layer=layer, heads=rows.tolist(), added=b,
                        removed=low[rows].tolist(), added_score=sc[rows, b].float().tolist(),
                        removed_score=sc[rows, low[rows]].float().tolist()))
        return sel
    H, n, d = K.shape
    B = S.block
    nc, c0, c1 = _layout(n)
    scale = 1 / math.sqrt(d)
    nb_budget = _kfix(n)
    k = min(nc, nb_budget)
    k1 = min(k, math.ceil(S.rho * k))
    Kc = K[:, c0:c1].reshape(H, nc, B, d)

    def from_tok(tok_scores):
        t = tok_scores[:, c0:c1].reshape(H, nc, B)
        return t.logsumexp(-1), t.amax(-1), (2 * t).logsumexp(-1) / 2

    if method.startswith("o_"):
        mass, peak, b2 = from_tok(s.float())
    elif method.startswith("sk_"):
        W = S.W[layer]                                         # [H,d,r]
        p = torch.einsum("hd,hdr->hr", q.float(), W)
        sk = _summ(layer, nc, "sk", lambda: torch.einsum("hnbd,hdr->hnbr", Kc.float(), W))
        t = torch.einsum("hr,hnbr->hnb", p, sk) * scale
        mass, peak, b2 = t.logsumexp(-1), t.amax(-1), (2 * t).logsumexp(-1) / 2
    else:
        mass = peak = b2 = None

    if method in ("o_mass", "sk_mass"):
        return _topk_union(mass, mass, k, 0)
    if method in ("o_peak", "sk_peak"):
        return _topk_union(peak, peak, k, k)
    if method in ("o_beta2", "sk_beta2"):
        return _topk_union(b2, b2, k, 0)
    if method in ("o_dual", "sk_dual"):
        return _topk_union(peak, mass, k, k1)
    if method.startswith(("o_dual:", "sk_dual:")):           # fixed peak share rho
        return _topk_union(peak, mass, k, min(k, math.ceil(float(method.split(":")[1]) * k)))
    if method in ("o_dual_ad", "sk_dual_ad"):
        return _adaptive(mass, peak, k)

    qf = q.float()
    if method in ("quest", "em") or method.startswith("sum_dual"):
        mn = _summ(layer, nc, "mn", lambda: Kc.float().amin(2))  # [H,nc,d]
        mx = _summ(layer, nc, "mx", lambda: Kc.float().amax(2))
        quest = (torch.maximum(qf[:, None] * mn, qf[:, None] * mx)).sum(-1) * scale
        if method == "quest":
            return _topk_union(quest, quest, k, k)
    mu = _summ(layer, nc, "mu", lambda: Kc.float().mean(2))   # [H,nc,d]
    mean = torch.einsum("hd,hnd->hn", qf, mu) * scale
    if method == "mean":
        return _topk_union(mean, mean, k, 0)
    if method == "em":                                         # Edmundson-Madansky: mean + box upper bound
        sc = em_upper(mean, (torch.minimum(qf[:, None] * mn, qf[:, None] * mx)).sum(-1) * scale, quest)
        return _topk_union(sc, sc, k, 0)
    W = S.W[layer]
    pq = torch.einsum("hd,hdr->hr", qf, W)
    cen = _summ(layer, nc, "cen", lambda: torch.einsum("hnbd,hdr->hnbr", Kc.float() - mu[:, :, None], W))
    var = (torch.einsum("hr,hnbr->hnb", pq, cen) * scale).pow(2).mean(-1)
    cobs = mean + 0.5 * var
    if method == "cobs":
        return _topk_union(cobs, cobs, k, 0)
    if method == "sum_dual":
        return _topk_union(quest, cobs, k, k1)
    if method.startswith("sum_dual:"):
        return _topk_union(quest, cobs, k, min(k, math.ceil(float(method.split(":")[1]) * k)))
    if method == "sum_dual_ad":
        return _adaptive(cobs, quest, k)
    if method == "mix_cobs_mean":                              # control: two mass estimators
        return _topk_union(cobs, mean, k, k1)
    if method == "rem":                                        # two-phase log-partition, whole block
        sc = two_phase(var.clamp_min(0).sqrt(), B) + mean
        return _topk_union(sc, sc, k, 0)
    if method.startswith("zdec"):                              # zdec{m}[_cum]: m exact outliers + bulk
        m = int(method[4])
        out_idx, k_out, mu_b, cen_b = _summ(layer, nc, f"zdec{m}", lambda: _outlier_split(Kc, mu, cen, W, m))
        s_out = torch.einsum("hd,hnmd->hnm", qf, k_out) * scale                  # exact outlier scores
        mean_b = torch.einsum("hd,hnd->hn", qf, mu_b) * scale
        var_b = (torch.einsum("hr,hnbr->hnb", pq, cen_b) * scale).pow(2).sum(-1) / (B - m)
        if method.endswith("_cum"):
            bulk = math.log(B - m) + var_b / 2
        else:
            bulk = two_phase(var_b.clamp_min(0).sqrt(), B - m)
        sc = torch.cat([s_out, (mean_b + bulk)[..., None]], -1).logsumexp(-1)
        return _topk_union(sc, sc, k, 0)
    if method in ("qsub2c", "qsub2k"):                         # two sub-boxes per block, same memory as B/2
        U = _summ(layer, nc, method, lambda: _subboxes(Kc, method[-1]))
        mn2, mx2, cnt = U
        h = (torch.maximum(qf[:, None, None] * mn2, qf[:, None, None] * mx2)).sum(-1) * scale   # [H,nc,2]
        ub = (h + cnt.log()).logsumexp(-1)
        return _topk_union(ub, ub, k, 0)
    if method.startswith("ell_"):                              # moments + two extreme radii
        U = ell_upper(layer, nc, Kc, mu, cen, W, qf, pq, mean, scale, B)
        if method == "ell_hi":
            return _topk_union(U, U, k, 0)
        if method == "ell_dual":                               # recall-safe bound + calibrated estimate
            return _topk_union(U, cobs + math.log(B), k, k1)
    if method.startswith("nf_"):                               # nf_{lo,hi,pt,dual}{m}: interval summary
        kind, m = method[3:-1], int(method[-1])
        out_idx, k_out, mu_b, cen_b = _summ(layer, nc, f"zdec{m}", lambda: _outlier_split(Kc, mu, cen, W, m))
        bmn, bmx = _summ(layer, nc, f"box{m}", lambda: _bulk_box(Kc, out_idx))
        s_out = torch.einsum("hd,hnmd->hnm", qf, k_out) * scale
        mean_b = torch.einsum("hd,hnd->hn", qf, mu_b) * scale
        logn = math.log(B - m)

        def with_out(bulk):                                     # log(sum_C e^x + e^bulk)
            return torch.cat([s_out, bulk[..., None]], -1).logsumexp(-1)
        hi = with_out(logn + (torch.maximum(qf[:, None] * bmn, qf[:, None] * bmx)).sum(-1) * scale)
        if kind == "hi":
            return _topk_union(hi, hi, k, 0)
        if kind == "lo":
            lo = with_out(logn + mean_b)
            return _topk_union(lo, lo, k, 0)
        var_b = (torch.einsum("hr,hnbr->hnb", pq, cen_b) * scale).pow(2).sum(-1) / (B - m)
        pt = with_out(mean_b + two_phase(var_b.clamp_min(0).sqrt(), B - m))
        if kind == "pt":
            return _topk_union(pt, pt, k, 0)
        if kind == "dual":
            return _topk_union(hi, pt, k, k1)
    raise ValueError(method)


def _subboxes(Kc, how):
    """Split each block's tokens into 2 groups (c: contiguous halves; k: sign of the projection on
    the block's top principal direction) and return per-group box [H,nc,2,d] x2 and counts [H,nc,2]."""
    H, nc, B, d = Kc.shape
    Kf = Kc.float()
    if how == "c":
        g = (torch.arange(B, device=Kf.device) >= B // 2).expand(H, nc, B)
    else:
        cen = Kf - Kf.mean(2, keepdim=True)
        v = cen.abs().sum(2) + 1e-3                                             # power iteration for the
        for _ in range(8):                                                      # top principal direction
            v = torch.einsum("hnbd,hnb->hnd", cen, torch.einsum("hnbd,hnd->hnb", cen, v))
            v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        g = torch.einsum("hnbd,hnd->hnb", cen, v) > 0
    big = torch.finfo(Kf.dtype).max
    mns, mxs, cnts = [], [], []
    for side in (False, True):
        m = (g == side)[..., None]
        mns.append(Kf.masked_fill(~m, big).amin(2))
        mxs.append(Kf.masked_fill(~m, -big).amax(2))
        cnts.append(m[..., 0].sum(-1).float())
    cnt = torch.stack(cnts, -1)
    mn2, mx2 = torch.stack(mns, 2), torch.stack(mxs, 2)
    empty = cnt == 0                                                          # degenerate split
    mn2 = mn2.masked_fill(empty[..., None], 0.0)
    mx2 = mx2.masked_fill(empty[..., None], 0.0)
    return mn2, mx2, cnt.clamp_min(1e-30)


def _ell_summary(Kc, mu, cen, W):
    """Per block: whitened covariance factor in the query subspace and two extreme radii.
    R_w  = max_i || Sigma_W^{-1/2} W^T (k_i - mu) ||   (<= sqrt(B-1); = sqrt(B-1) iff one spike)
    R_res = max_i || (I - W W^T)(k_i - mu) ||"""
    H, nc, B, r = cen.shape
    Sig = torch.einsum("hnbr,hnbs->hnrs", cen, cen) / B + 1e-6 * torch.eye(r, device=cen.device)
    Lc = torch.linalg.cholesky(Sig)                                        # Sigma = L L^T
    white = torch.linalg.solve_triangular(Lc, cen.transpose(-1, -2), upper=False)   # [H,nc,r,B]
    R_w = white.pow(2).sum(-2).sqrt().amax(-1)                             # [H,nc]
    resid = Kc.float() - mu[:, :, None] - torch.einsum("hnbr,hdr->hnbd", cen, W)
    R_res = resid.pow(2).sum(-1).sqrt().amax(-1)                           # [H,nc]
    return Lc, R_w, R_res


def ell_upper(layer, nc, Kc, mu, cen, W, qf, pq, mean, scale, B):
    """Certified upper bound on the block log-partition from (mean, Sigma_W, R_w, R_res):
    x_i - m <= R_w ||L^T W^T q|| + R_res ||q_res||  (Cauchy-Schwarz in whitened / residual parts),
    so A <= m + R_w s_w + R_res ||q_res|| + log B."""
    Lc, R_w, R_res = _summ(layer, nc, "ell", lambda: _ell_summary(Kc, mu, cen, W))
    s_w = torch.einsum("hnrs,hr->hns", Lc, pq).pow(2).sum(-1).sqrt() * scale   # ||L^T p|| = sqrt(p^T Sigma p)
    q_res = (qf - torch.einsum("hdr,hr->hd", W, pq)).norm(dim=-1)              # [H]
    return mean + R_w * s_w + R_res * q_res[:, None] * scale + math.log(B)


def _bulk_box(Kc, out_idx):
    """Per-channel min/max of the bulk keys (outliers excluded): Quest's box on the bulk."""
    H, nc, B, d = Kc.shape
    keep = torch.ones(H, nc, B, dtype=torch.bool, device=Kc.device)
    keep.scatter_(2, out_idx, False)
    Kf = Kc.float()
    big = torch.finfo(Kf.dtype).max
    bmn = Kf.masked_fill(~keep[..., None], big).amin(2)
    bmx = Kf.masked_fill(~keep[..., None], -big).amax(2)
    return bmn, bmx


def two_phase(s, n):
    """E log sum_{i<=n} exp(s z_i), z_i ~ N(0,1), in closed form: the largest term (expected
    value c_n s, c_n = E max of n standard normals) plus the other n-1 terms truncated below it,
        F_n(s) = log( e^{c_n s} + (n-1) e^{s^2/2} Phi(c_n - s) ).
    Covers both phases (cumulant regime and frozen regime); max error 0.14 for n in [8,128],
    s in [0.25,12] (theory/verify_v2_formula.py)."""
    c = float(torch.special.ndtri(torch.tensor((n - 0.375) / (n + 0.25))))      # Blom
    return torch.logaddexp(c * s, math.log(n - 1) + s * s / 2 + torch.special.log_ndtr(c - s))


def _outlier_split(Kc, mu, cen, W, m):
    """Query-independent: per block keep the m keys farthest from the block mean in the
    query subspace; return their indices, keys, and the bulk mean / centred sketches."""
    H, nc, B, d = Kc.shape
    dev = cen.pow(2).sum(-1)                                   # [H,nc,B]
    idx = dev.topk(m, dim=-1).indices                          # [H,nc,m]
    Kf = Kc.float()
    k_out = torch.gather(Kf, 2, idx[..., None].expand(H, nc, m, d))
    mask = torch.ones(H, nc, B, device=Kc.device)
    mask.scatter_(2, idx, 0.0)
    mu_b = (Kf * mask[..., None]).sum(2) / (B - m)
    cen_b = torch.einsum("hnbd,hdr->hnbr", Kf - mu_b[:, :, None], W) * mask[..., None]
    return idx, k_out, mu_b, cen_b


def _diag(layer, s, q, K):
    """H1 / outlier diagnostics at one decode step (accumulated into S.diag[layer]):
    - margin: A of the k-th vs (k+1)-th candidate block by exact log-partition; count < log B
    - swaps: blocks in the mass top-k but not in the peak top-k
    - hit_m: among mass top-k blocks, how often the block's argmax token is one of the m
      query-independent outliers (m = 1, 2)."""
    H, n, d = K.shape
    B = S.block
    nc, c0, c1 = _layout(n)
    k = min(nc, _kfix(n))
    if k == 0 or k >= nc:
        return
    t = s[:, c0:c1].float().reshape(H, nc, B)
    A, h = t.logsumexp(-1), t.amax(-1)
    As = A.sort(-1, descending=True).values
    margin = As[:, k - 1] - As[:, k]
    top = A.topk(k, -1).indices
    ptop = h.topk(k, -1).indices
    swaps = sum(len(set(top[i].tolist()) - set(ptop[i].tolist())) for i in range(H))
    Kc = K[:, c0:c1].reshape(H, nc, B, d)
    mu = Kc.float().mean(2)
    cen = torch.einsum("hnbd,hdr->hnbr", Kc.float() - mu[:, :, None], S.W[layer])
    dev = cen.pow(2).sum(-1)
    arg = t.argmax(-1)                                          # [H,nc]
    est = {}
    for m in ("cobs", "rem", "nf_lo1", "nf_hi1", "nf_pt1", "nf_pt2", "quest", "mean", "ell_hi"):
        S.capture = {}
        block_scores(m, q, K, s, layer)
        est[m] = S.capture["primary"]
        S.capture = None
    sd = t.std(-1, unbiased=False).clamp_min(1e-6)
    spike = (h - t.mean(-1)) / sd                               # (max - mean) / std per block
    bins = torch.bucketize(spike, torch.tensor([2.0, 3.0, 4.0], device=spike.device))
    cal = S.diag.setdefault("cal", {})
    logB = math.log(B)
    for m, e in est.items():
        # cobs/mean/quest omit the shared log n constant; add it back so errors are comparable
        err = e + (logB if m in ("cobs", "mean", "quest") else 0.0) - A
        for b in range(4):
            sel_b = bins == b
            if sel_b.any():
                r = cal.setdefault(f"{m}|{b}", [0, 0.0, 0.0])
                r[0] += int(sel_b.sum()); r[1] += float(err[sel_b].sum()); r[2] += float(err[sel_b].abs().sum())
    # certified selection: read blocks in decreasing U until the captured mass is certified
    # >= 1 - eps: (Z_fixed + sum_read e^A) / (Z_fixed + sum_read e^A + sum_unread e^U)
    p_all = s.float()
    fixed = torch.cat([p_all[:, :c0], p_all[:, c1:]], 1).logsumexp(-1)       # sink + local, exact
    S.capture = {}
    block_scores("nf_hi2", q, K, s, layer)
    est["nf_hi2"] = S.capture["primary"]
    S.capture = None
    uppers = {"quest": est["quest"] + logB, "nf1": est["nf_hi1"], "nf2": est["nf_hi2"], "ell": est["ell_hi"],
              "oracle": A}
    cert = S.diag.setdefault("certify", {})
    for name, U in uppers.items():
        order = U.argsort(-1, descending=True)
        A_o = torch.gather(A, 1, order)
        U_o = torch.gather(U, 1, order)
        read = torch.cat([fixed[:, None], A_o], 1).logcumsumexp(-1)[:, 1:]          # after j+1 reads
        rest = torch.cat([U_o.flip(-1).logcumsumexp(-1).flip(-1)[:, 1:],
                          torch.full_like(U_o[:, :1], float("-inf"))], 1)         # unread uppers
        frac = (read - torch.logaddexp(read, rest)).exp()                          # certified fraction
        for eps in (0.1, 0.05, 0.01):
            need = ((frac >= 1 - eps).float().argmax(-1) + 1).float()
            need = torch.where((frac >= 1 - eps).any(-1), need, torch.full_like(need, nc))
            r = cert.setdefault(f"{name}|{eps}", [0, 0.0])
            r[0] += H; r[1] += float((need / nc).sum())
    cov = S.diag.setdefault("cover", [0, 0])
    cov[0] += int(((est["nf_lo1"] <= A + 1e-3) & (A <= est["nf_hi1"] + 1e-3) & (A <= est["ell_hi"] + 1e-3)).sum())
    cov[1] += A.numel()
    if S.needle_mask is not None:
        nm = S.needle_mask[c0:c1].to(t.device).reshape(nc, B)
        nb = torch.nonzero(nm.any(-1)).flatten()
        rr = S.diag.setdefault("needle_recall", [0, 0, 0])
        for m_ in (1, 2):
            out = dev.topk(m_, -1).indices                      # [H,nc,m]
            for b_ in nb.tolist():
                hit = nm[b_][out[:, b_]].any(-1)                # [H]
                rr[m_] += int(hit.sum())
        rr[0] += H * len(nb)
    # alternative query-independent outlier criteria: needle-block recall @1/@2
    if S.needle_mask is not None and len(nb) > 0:
        Kf = Kc.float()
        dev_full = (Kf - mu[:, :, None]).pow(2).sum(-1)
        # weight each subspace direction by its query variance E[(w_j.q)^2] from the prefill
        lam = torch.einsum("hdr,hde,her->hr", S.W[layer], S.C[layer], S.W[layer])   # [H,r]
        dev_mahal = (cen.pow(2) * lam[:, None, None, :]).sum(-1)
        # "box-defining" tokens: how many channels' min or max each token attains
        ones = torch.ones(H, nc, d, device=Kf.device)
        ext = torch.zeros(H, nc, B, device=Kf.device)
        ext.scatter_add_(2, Kf.argmax(2), ones)
        ext.scatter_add_(2, Kf.argmin(2), ones)
        crit = {"full": dev_full, "mahal": dev_mahal, "boxcount": ext}
        rc = S.diag.setdefault("needle_recall_by_criterion", {})
        for cname, score_c in crit.items():
            for m_ in (1, 2):
                out = score_c.topk(m_, -1).indices
                r = rc.setdefault(f"{cname}@{m_}", [0, 0])
                for b_ in nb.tolist():
                    r[0] += int(nm[b_][out[:, b_]].any(-1).sum())
                r[1] += H * len(nb)
    # mechanism: per scorer, captured share of the oracle top-k mass and needle-block hit,
    # weighted by each head's attention mass on the target needle (retrieval-head weighting)
    if S.needle_mask is not None and len(nb) > 0:
        pfull = s.float().softmax(-1)
        w = pfull[:, S.needle_mask[:n].to(pfull.device)].sum(-1)              # [H] needle mass per head
        eA = (A - A.amax(-1, keepdim=True)).exp()
        orc = torch.zeros_like(A, dtype=torch.bool).scatter_(1, A.topk(k, -1).indices, True)
        orc_mass = (eA * orc).sum(-1)
        nblk = torch.zeros(nc, dtype=torch.bool, device=A.device)
        nblk[nb] = True
        mech = S.diag.setdefault("mechanism", {})
        for m in ("quest", "cobs", "sum_dual", "mix_cobs_mean", "nf_hi1", "nf_pt1", "ell_hi", "ell_dual",
                  "qsub2c", "qsub2k", "o_mass"):
            sel = block_scores(m, q, K, s, layer)
            cap = ((eA * sel).sum(-1) / orc_mass).clamp(max=1.0)
            hit = (sel & nblk[None]).any(-1).float()
            r = mech.setdefault(m, [0.0, 0.0, 0.0, 0.0])
            r[0] += float((w * cap).sum()); r[1] += float((w * hit).sum()); r[2] += float(w.sum())
            r[3] += float(cap.sum()) / H
    p = s.float().softmax(-1)[:, c0:c1].reshape(H, nc, B)
    thr = p.flatten(1).quantile(0.99, dim=1)[:, None, None]
    toppct = p >= thr
    tr = S.diag.setdefault("top1pct_recall", [0, 0, 0])
    tr[0] += int(toppct.sum())
    for m_ in (1, 2):
        out = dev.topk(m_, -1).indices
        in_c = torch.zeros_like(toppct)
        in_c.scatter_(2, out, True)
        tr[m_] += int((toppct & in_c).sum())
    rec = S.diag.setdefault(layer, dict(n=0, small=0, swaps=0, k=0, hit1=0, hit2=0, margin=[]))
    rec["n"] += H
    rec["small"] += int((margin < math.log(B)).sum())
    rec["margin"] += margin.tolist()
    rec["swaps"] += swaps
    rec["k"] += H * k
    for m in (1, 2):
        out = dev.topk(m, -1).indices                           # [H,nc,m]
        hit = (out == arg[..., None]).any(-1)                   # [H,nc]
        rec[f"hit{m}"] += int(torch.gather(hit, 1, top).sum())


MECH_SCORERS = ("quest", "cobs", "sum_dual", "nf_hi1", "nf_pt1", "qsub2c")


def _mech(layer, s, q, K):
    """Light per-step mechanism metrics (S.mech[step][scorer] += [w*cap, w*hit, w, cap_unweighted, 1]),
    where w = this head's attention mass on the target needle at this step."""
    H, n, d = K.shape
    B = S.block
    nc, c0, c1 = _layout(n)
    k = min(nc, _kfix(n))
    if k == 0 or k >= nc:
        return
    t = s[:, c0:c1].float().reshape(H, nc, B)
    A = t.logsumexp(-1)
    nm = S.needle_mask[c0:c1].to(t.device).reshape(nc, B)
    nblk = nm.any(-1)
    if not nblk.any():
        return
    pfull = s.float().softmax(-1)
    w = pfull[:, S.needle_mask[:n].to(pfull.device)].sum(-1)
    eA = (A - A.amax(-1, keepdim=True)).exp()
    orc = torch.zeros_like(A, dtype=torch.bool).scatter_(1, A.topk(k, -1).indices, True)
    orc_mass = (eA * orc).sum(-1)
    rec_step = S.mech.setdefault(S.mech_step, {})
    for m in MECH_SCORERS:
        sel = block_scores(m, q, K, s, layer)
        cap = ((eA * sel).sum(-1) / orc_mass).clamp(max=1.0)
        hit = (sel & nblk[None]).any(-1).float()
        r = rec_step.setdefault(m, [0.0, 0.0, 0.0, 0.0, 0])
        r[0] += float((w * cap).sum()); r[1] += float((w * hit).sum()); r[2] += float(w.sum())
        r[3] += float(cap.sum()); r[4] += H


def _digit_diag(layer, s, q, K):
    """Per-digit mechanism logging (mech design v2). S.dd = dict(mode, step, target_pos, scorers, out, weights).
    Target block = candidate block holding the prompt token of the digit being predicted at this step.
    mode "shared": state is the dense TF state; every scorer in S.dd["scorers"] is evaluated on it, and the dense
      attention mass of each head on the target token is stored as the reference weight for (step, layer).
    mode "own": state is the active arm's own trajectory; only S.method is evaluated (its actual selection).
    Per (step, layer) a compact record is appended; no attention tensors are stored."""
    dd = S.dd
    tp = dd.get("target_pos")
    if tp is None:
        return
    H, n, d = K.shape
    B = S.block
    nc, c0, c1 = _layout(n)
    k = min(nc, _kfix(n))
    # memoised summaries are reused across steps: legitimate only while the candidate region is inside the frozen
    # prefill prefix (local window and generation lengths of this design guarantee it; asserted here)
    assert c1 <= dd["frozen_len"], f"candidate region {c1} beyond frozen prefix {dd['frozen_len']}"
    if not (c0 <= tp < c1) or k == 0:
        dd["out"].append(dict(step=dd["step"], layer=layer, region="fixed" if tp < c0 or tp >= _fixed_end(n)
                              else "unselectable_edge"))
        return
    tb = (tp - c0) // B
    t = s[:, c0:c1].float().reshape(H, nc, B)
    A = t.logsumexp(-1)                                          # exact block log-partition [H, nc]

    def gt_eq(score, idx):
        """#candidates strictly greater than, and #others equal to, the score at block idx (per head)."""
        v = score[:, idx:idx + 1]
        return (score > v).sum(1), (score == v).sum(1) - 1

    def membership(g, e):
        """1 = certainly in the top-k, 0 = certainly out, 0.5 = a tie straddles the k-th boundary."""
        inside = (g + e) < k
        out = g >= k
        return torch.where(inside, 1.0, torch.where(out, 0.0, 0.5))

    g_ex, e_ex = gt_eq(A, tb)
    kth = A.topk(k, -1).values[:, -1]
    blk = t[:, tb]                                               # [H, B] target block scores
    mean_all, var_all = t.mean(-1), t.var(-1, unbiased=False)
    full_mom = mean_all + 0.5 * var_all + math.log(B)            # COBS form with exact (full) moments
    g_fm, e_fm = gt_eq(full_mom, tb)
    Kc = K[:, c0 + tb * B:c0 + (tb + 1) * B].float()
    W = S.W[layer]
    pq = torch.einsum("hd,hdr->hr", q.float(), W)
    cen = torch.einsum("hbd,hdr->hbr", Kc - Kc.mean(1, keepdim=True), W)
    proj_var = (torch.einsum("hr,hbr->hb", pq, cen) / math.sqrt(d)).pow(2).mean(-1)
    key = (dd["step"], layer)
    if dd["mode"] == "shared":
        w = s.float().softmax(-1)[:, tp]                         # dense mass on the target token per head
        dd["weights"][key] = w.tolist()
        scorers = dd["scorers"]
    else:
        w = torch.tensor(dd["weights"].get(key, [float("nan")] * H))
        scorers = [S.method]
    others = {}                                                  # head / tail / key-phrase blocks (candidate idx)
    for name, pos in dd.get("probe_pos", {}).items():
        others[name] = (pos - c0) // B if c0 <= pos < c1 else None
    rec = dict(step=dd["step"], layer=layer, region="candidate", k=k, nc=nc, target_block=tb,
               exact=dict(lse=A[:, tb].tolist(), mean=blk.mean(-1).tolist(), var=var_all[:, tb].tolist(),
                          peak=blk.amax(-1).tolist(), kth_lse=kth.tolist(), greater=g_ex.tolist(),
                          ties=e_ex.tolist(), in_topk=membership(g_ex, e_ex).tolist()),
               full_moment=dict(score=full_mom[:, tb].tolist(), greater=g_fm.tolist(), ties=e_fm.tolist(),
                                in_topk=membership(g_fm, e_fm).tolist()),
               proj_var=proj_var.tolist(), dense_weight=w.tolist(), probe_blocks=others, scorers={})
    swaps_saved = {kk: list(v) for kk, v in S.swaps.items()}   # diagnostics must not change swap accounting
    S.in_diag = True
    for m in scorers:
        prev = S.capture
        S.capture = {}
        sel = block_scores(m, q, K, s, layer)
        sc = S.capture.get("primary")
        S.capture = prev
        blockwise = sel.shape[1] == nc

        def chosen(idx):
            if idx is None:
                return None
            return (sel[:, idx] if blockwise else sel[:, c0 + idx * B:c0 + (idx + 1) * B].any(1)).tolist()
        r = dict(selected=chosen(tb), probe_selected={nm: chosen(ix) for nm, ix in others.items()})
        if sc is not None and sc.shape[1] == nc:
            g_s, e_s = gt_eq(sc, tb)
            r.update(score=sc[:, tb].float().tolist(), kth_score=sc.float().topk(k, -1).values[:, -1].tolist(),
                     greater=g_s.tolist(), ties=e_s.tolist(), in_topk=membership(g_s, e_s).tolist())
        rec["scorers"][m] = r
    S.in_diag = False
    S.swaps = swaps_saved
    dd["out"].append(rec)


def token_mask(sel, n):
    if sel.shape[1] == n:                                      # already a token mask (+g2)
        return sel
    H, nc = sel.shape
    nc_, c0, c1 = _layout(n)
    m = torch.zeros(H, n, dtype=torch.bool, device=sel.device)
    m[:, :_U()] = True                                         # fixed sink
    m[:, _fixed_end(n):] = True                                # fixed tail
    m[:, c0:c1] = sel.repeat_interleave(S.block, dim=1)
    return m


def _offline_stats(layer, s, Vr, q, Kr):
    """Per-method stats at one decode step: mass captured, relative output error,
    target-needle mass retained (mass-weighted over heads)."""
    p = s.float().softmax(-1)                                  # [H,n]
    o = torch.einsum("hn,hnd->hd", p, Vr.float())
    n = s.shape[1]
    nm = S.needle_mask[:n].to(p.device) if S.needle_mask is not None else None
    for m in S.stat_methods:
        sel = token_mask(block_scores(m, q, Kr, s, layer), n)
        pk = p * sel
        mass = pk.sum(-1)
        oh = torch.einsum("hn,hnd->hd", pk / mass[:, None], Vr.float())
        err = (o - oh).norm(dim=-1) / o.norm(dim=-1)
        rec = S.collect.setdefault(m, {}).setdefault(layer, dict(mass=0., err=0., nkeep=0., ntot=0., h=0))
        rec["mass"] += mass.sum().item()
        rec["err"] += err.sum().item()
        rec["h"] += p.shape[0]
        if nm is not None:
            rec["nkeep"] += (pk[:, nm]).sum().item()
            rec["ntot"] += (p[:, nm]).sum().item()


def blocksparse_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
    layer = module.layer_idx
    b, H, ql, d = query.shape
    if ql > 1:
        # prefill: dense; record query-PCA basis per head for sketch/COBS scorers
        # accumulate the query second moment (prefill may arrive in chunks); basis is built lazily
        qf = query[0].float()                                   # [H,L,d]
        C = torch.einsum("hld,hle->hde", qf, qf)
        S.C[layer] = S.C[layer] + C if layer in S.C else C
        return sdpa_attention_forward(module, query, key, value, attention_mask,
                                      scaling=scaling, dropout=dropout, **kw)
    if layer not in S.W and layer in S.C:
        # one subspace per KV head (shared by its query-head group, as in COBS), repeated to query heads
        if S.w_per_head:                                         # one subspace per query head (older runs)
            _, vecs = torch.linalg.eigh(S.C[layer])
            S.W[layer] = vecs[:, :, -S.rank:].contiguous()
        else:
            g = H // key.shape[1]
            Cg = S.C[layer].reshape(key.shape[1], g, d, d).sum(1)
            _, vecs = torch.linalg.eigh(Cg)
            S.W[layer] = vecs[:, :, -S.rank:].repeat_interleave(g, dim=0).contiguous()
    n_rep = H // key.shape[1]
    S.group = n_rep
    Kr = _repeat(key[0], n_rep)                                 # [H,n,d]
    Vr = _repeat(value[0], n_rep)
    q = query[0, :, 0]                                          # [H,d]
    scale = scaling if scaling is not None else 1 / math.sqrt(d)
    s = torch.einsum("hd,hnd->hn", q.float(), Kr.float()) * scale
    n = s.shape[1]
    if S.collect is not None and layer >= S.skip_layers:
        _offline_stats(layer, s, Vr, q, Kr)
    if S.diag is not None and layer >= S.skip_layers:
        _diag(layer, s, q, Kr)
    if S.mech is not None and layer >= S.skip_layers:
        _mech(layer, s, q, Kr)
    if S.dd is not None and layer >= S.skip_layers:
        _digit_diag(layer, s, q, Kr)
    if S.method != "dense" and layer >= S.skip_layers and (S.sparse_layers is None or layer in S.sparse_layers):
        sel = token_mask(block_scores(S.method, q, Kr, s, layer), n)
        cnt = sel.sum(1)
        # [min, max, sum, count, budget_violations, max_abs_budget_error, fixed_region_violations]
        t = S.tokens.setdefault(S.method, [10 ** 9, 0, 0, 0, 0, 0, 0, 0])   # ... + budget_checked_calls
        t[0] = min(t[0], int(cnt.min())); t[1] = max(t[1], int(cnt.max()))
        t[2] += int(cnt.sum()); t[3] += int(cnt.numel())
        # per-call invariant: every sparse head reads R(n) = |F(n)| + B*min(nc0, k_fix) tokens and keeps all of F(n)
        fe = _fixed_end(n)
        fixed_ok = sel[:, :_U()].all(1) & sel[:, fe:].all(1)
        t[6] += int((~fixed_ok).sum())
        if True:                                              # also under a base_shift (shifted candidate grid)
            sh_saved, S.shift = S.shift, 0
            target = _U() + (n - fe) + S.block * min(_layout(n)[0], _kfix(n))
            S.shift = sh_saved
            t[7] += int(cnt.numel())
            err = (cnt - target).abs()
            t[4] += int((err != 0).sum())
            t[5] = max(t[5], int(err.max()))
            if S.strict_budget and (bool((err != 0).any()) or not bool(fixed_ok.all())):
                raise RuntimeError(f"budget invariant violated: method={S.method} n={n} layer={layer} "
                                   f"target={target} counts={cnt.tolist()} fixed_ok={fixed_ok.tolist()}")
        s = s.masked_fill(~sel, float("-inf"))
        if S.method.endswith("+ptr"):
            _ptr_update(layer, s, n)
    p = s.softmax(-1).to(value.dtype)
    out = torch.einsum("hn,hnd->hd", p, Vr)                     # [H,d]
    return out[None, None], None                                # [b, q, H, d]


AttentionInterface.register("blocksparse", blocksparse_attention)
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask  # noqa: E402
AttentionMaskInterface.register("blocksparse", sdpa_mask)
