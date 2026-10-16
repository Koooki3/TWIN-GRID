"""CPU check of the +ptr wrapper: with a synthetic pointer that advances one token per step, after the history fills
the block holding pointer+1 is selected for every head, the per-call token budget equals the base method's, and a
non-consecutive step resets the history. Exit 1 on failure."""
import math
import sys

import torch

sys.path.insert(0, __file__.rsplit("/theory/", 1)[0] + "/src")
import sparse_attn as sa  # noqa: E402

S = sa.S
S.block, S.budget, S.local = 32, 512, 64
torch.manual_seed(0)
H, d, n0 = 4, 64, 4096
K = torch.randn(H, n0 + 40, d)
bad = []
ptr0 = 1000                                         # pointer walks 1000, 1001, ... (block of pointer+1 changes at 1023→1024)
for t in range(30):
    n = n0 + t
    q = torch.randn(H, d)
    s = torch.einsum("hd,hnd->hn", q, K[:, :n]) / math.sqrt(d)
    base = sa.token_mask(sa.block_scores("quest", q, K[:, :n], s, 5), n).sum(1)
    sel = sa.block_scores("quest+ptr", q, K[:, :n], s, 5)
    tm = sa.token_mask(sel, n)
    if not torch.equal(tm.sum(1), base):
        bad.append(f"t={t}: budget {tm.sum(1).tolist()} vs base {base.tolist()}")
    nc, c0, c1 = sa._layout(n)
    p = ptr0 + t
    if t >= 3:                                      # history of >= 3 consecutive peaks with 2+ forward moves
        b = (p - 1 + 1 - c0) // S.block             # pointer recorded last step was p-1
        if not sel[:, b].all():
            bad.append(f"t={t}: successor block {b} not selected")
    peak = torch.full((H, n), -1e9)
    peak[:, p] = 10.0
    sa._ptr_update(5, peak, n)
S.ptr[5] = (10 ** 9, S.ptr[5][1])                   # non-consecutive step -> history must reset
sa._ptr_update(5, peak, n0)
if S.ptr[5][1].shape[1] != 1:
    bad.append("history not reset")
# Codex: a new prompt whose length happens to continue the old one must not inherit history once runners reset S.ptr.
n = n0 + 31
S.ptr = {5: (n - 1, torch.tensor([[ptr0, ptr0 + 1, ptr0 + 2]] * H))}   # stale "consecutive" history
S.ptr = {}                                                             # what run_e2e / run_digit_diag do per method
q = torch.randn(H, d)
s = torch.einsum("hd,hnd->hn", q, K[:, :n]) / math.sqrt(d)
if not torch.equal(sa.block_scores("quest+ptr", q, K[:, :n], s, 5), sa.block_scores("quest", q, K[:, :n], s, 5)):
    bad.append("reset state still swapped")
src_e2e = open(__file__.rsplit("/theory/", 1)[0] + "/src/run_e2e.py").read()
src_dd = open(__file__.rsplit("/theory/", 1)[0] + "/src/run_digit_diag.py").read()
if "S.tokens, S.swaps, S.ptr = {}, {}, {}" not in src_e2e or src_dd.count("S.ptr, S.ptr_stats = method, {}, {}, {}, {}") != 2:
    bad.append("runner does not reset S.ptr per method/mode")
# Codex P2: a large jump in the latest pointer move ends copy mode at once.
S.ptr = {5: (n - 1, torch.tensor([[ptr0, ptr0 + 1, ptr0 + 2, 3000]] * H))}
if not torch.equal(sa.block_scores("quest+ptr", q, K[:, :n], s, 5), sa.block_scores("quest", q, K[:, :n], s, 5)):
    bad.append("large jump still treated as copy mode")
# counters [head-calls, ready, copy, already selected, swapped]: ordered and swaps match S.swaps
st = S.ptr_stats.get("quest+ptr")
if st is None or not (st[0] >= st[1] >= st[2] >= st[3] + st[4]) or st[4] != S.swaps["quest+ptr"][1]:
    bad.append(f"inconsistent counters {st} vs swaps {S.swaps.get('quest+ptr')}")
# nesting under the two-grid wrapper is refused
for nested in ("quest+g2f+ptr", "quest+ptr+g2f", "quest+ptr+g2", "quest+ptr+nb", "quest+ftail+ptr"):
    try:
        sa.block_scores(nested, q, K[:, :n], s, 5)
        bad.append(f"nested {nested} accepted")
    except ValueError:
        pass
print("swaps", S.swaps.get("quest+ptr"), "ptr_stats", st, "PROBLEMS", bad if bad else "none")
sys.exit(1 if bad else 0)
