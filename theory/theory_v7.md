# Theory v7 — after the literature survey and phase 1 (2026-10-16)

Status words: classical (known result, cited) / verified (proof or exact recount) / measured (development seeds,
not confirmatory) / hypothesis / withdrawn. "COBS-style" = this project's projected-variance scorer, not arXiv
2607.09052. Survey: reports/稀疏注意力块选择相关工作.md.

## 0. One frame

A block summary S defines a feasible set F(S) of score multisets {s_i} (s_i = q·k_i/√d). For the block log-mass
A = log Σ e^{s_i}, the summary can certify at most the interval [inf_F A, sup_F A]: that interval describes what the summary can
certify; a heuristic score need not lie in it. Two questions then cover the whole project:
  (Q1, what to store) how wide is that interval for a given summary family, and where must the score sit in it?
  (Q2, where to cut) which blocks exist at all — a unit the decoder needs (a 7-token number) can be split by the
      grid, in which case no summary of any block holds it whole.
Working hypothesis (to be confirmed, not a theorem): selection should rank by an upper end of the interval (so
rare extreme tokens stay visible) and over a partition in which the needed unit lies inside one candidate.

## 1. Q1 — intervals for three summary families

P1 (mean + variance; classical). With mean m and population variance s² of n scores,
    m + φ_n(−s) ≤ A − log 1 ≤ m + φ_n(s),  φ_n(t) = log(e^{t√(n−1)} + (n−1)e^{−t/√(n−1)}),
sharp (one high/one low value). Upper end = Samuelson's bound x_i − m ≤ s√(n−1) fed into the two-point
Bennett–Hoeffding MGF bound; lower end = the classical Chebyshev-system / equal-variable two-point extremum.
Our proof (Codex, theorem1-proof.txt) is a self-contained route to a known fact. Gap Δ_n(s) =
s(n−2)/√(n−1) − log(n−1) + o(1) as s → ∞: an estimator that sees only (m, s, n) is off by ≥ Δ/2 on some input.
What we keep: this worst-case non-identifiability reading for second-order (COBS-style) block summaries.

P2 (mean + range; classical bound, new use). If every s_i ∈ [a, b] and the mean μ is exact,
    log n + μ ≤ A ≤ log n + log(((b−μ)e^a + (μ−a)e^b)/(b−a))      (Jensen | Edmundson–Madansky)
The EM upper end is always valid and ≤ log n + b (Quest's bound), equal only when μ = b. It is attained for an
arbitrary-weight distribution (two points {a, b}), but NOT in general for n equal-weight scores: equality needs every
s_i at an endpoint and n(μ−a)/(b−a) an integer. Exact finite-n upper end (Codex T2; theory/verify_em_finite.py:
construction attains it, 3000 random cases, no feasible point above it, exit 0): with t = n(μ−a)/(b−a),
r = ⌊t⌋, θ = t − r,
    Z_max = (n−r−1)e^a + r e^b + e^{a+θ(b−a)}   (θ = 0: (n−r)e^a + r e^b),   A_max = log Z_max;  a = b: A = log n + a.
Reason: the polytope {s ∈ [a,b]^n, Σs = nμ} has vertices with at most one coordinate off the endpoints, and the
convex Σe^{s_i} is maximised at a vertex. Example n = 2, a = 0, b = 1, μ = 1/4: Z_max = 1 + e^{1/2} < EM = 3/2 + e/2.
The lower end log n + μ is attained by s_i ≡ μ when (a, b) are only bounds; if a and b must be attained, it changes.
In a block summary, (a, b) come from the per-channel box: a = Σ_c min(q_c lo_c, q_c hi_c)/√d, b likewise with max
(same 1/√d as s_i). Then the interval is valid but not sharp for the box itself (box corners are generally not keys).
Comparing log-masses across blocks or scorers needs every scorer's omitted log n restored (it cancels only between
equal-length blocks). We use EM (not Z_max) as the scorer; no new scorer or arm is added for Z_max.
Implemented as scorer `em` = log((1−w)e^a + w e^b) via logaddexp, w = (μ−a)/(b−a) clipped to [0,1] (log n omitted;
equal-length blocks); near-flat boxes return μ; non-finite inputs or μ outside [a−10⁻³, b+10⁻³] are counted per
(prompt, method) in em_viol. Exploration-grade float32 ordering, not a rounding-safe certificate.
theory/verify_em.py: 0/2000 violations; on random 64-d blocks the box is loose (mean slack 22.3 nats for b vs 21.7
for em), so em mainly re-orders blocks by where the mean sits in the box. Summary memory: 3 vectors / 32 tokens
= 75% of Quest-16 (2 vectors / 16 tokens); with the two-grid wrapper ≈ 150% (Codex) — not equal memory.

P3 (exact anomalies + bulk mean + bulk box; verified, v2). Store the m largest-norm tokens exactly; apply P2 to the
remaining n−m tokens using their own (bulk) mean and bulk box; combine by log-sum-exp; if n−m = 0 the block is exact.
Mathematically valid. Within the searched literature (to 2026-10-16) we did not find this combination for block LSE
(components known: Jensen ≈ CEDAR/NSA mean-logit + log n, box ≈ Quest); that is a candidate contribution, not a new
theorem. Upgrade over v2: P2's EM upper end replaces the bulk-box maximum.

Reference prediction (independent Gaussian scores; REM-style truncation, cf. Bovier–Kurkova–Löwe 2002 — the exact
mapping of our F_n to their finite-n statements is NOT yet checked): the second-order estimate m + σ²/2 (+ log n) is
accurate for σ well below √(2 ln n) (≈ 2.63 at n = 32) and overestimates well above it, where A ≈ m + σ√(2 ln n).
√(2 ln n) is a reference crossover scale; a 32-term log-sum-exp is smooth, there is no sharp phase transition, and
Qwen logits are correlated and non-Gaussian. Testable on real keys
(planned, P3 of the survey's priority list); not yet measured.

## 2. Q2 — the two-grid covering lemma (verified, elementary)

Grids G0 (boundaries at kB) and G1 (boundaries at kB + B/2), B even. Any token span of length ℓ ≤ B/2 + 1 lies
inside one block of G0 or one block of G1, and this is sharp (some span of length B/2 + 2 lies in neither).
Proof: a span [x, x+ℓ−1] crosses a boundary p iff p ∈ {x+1, …, x+ℓ−1}, a run of ℓ−1 ≤ B/2 consecutive integers.
The union of both boundary sets is spaced exactly B/2 apart, so the run holds at most one union boundary; it
belongs to only one grid, so the other grid has none inside the span. A run of B/2 + 1 integers can hold one
boundary of each grid. ∎ (theory/verify_cover.py: brute force for B ∈ {4,…,64}, exit 0; a first draft stated
ℓ ≤ B/2 and the tightness check caught it.)
Candidate region (base_shift = 0, C = _fixed_end(n), h = B/2): G0's full candidate blocks cover [B, C), G1's cover
[B+h, C−h); a unit lying in the common interior [B+h, C−h) is sufficient for the lemma; units nearer the edges are
handled by G0 alone and may be split.
With B = 32 this covers any unit of ≤ 17 tokens (a 7-digit number is 7 tokens in Qwen2.5).
Three separate statements: (i) some whole candidate block contains the unit (this lemma); (ii) g2_fill_v1 spends
exactly the same candidate-token budget as one grid (Codex g2-fill-budget-proof.txt); (iii) the selector actually
keeps the block holding the unit — NOT proved: within budget it may pick unrelated high-scoring blocks. So the
lemma increases the set of representable whole units at a proven read budget (summary memory doubles); any
selection benefit, and any causal role of cutting, are empirical questions. Overlapping windows themselves are not
new (NSA compression l=32, d=16; InfLLM-V2).

## 3. Measured (development seeds; 16k/512 unless noted; strict scoring)

M1 (phase 1, 80 prompts, seeds 110000–110039 / 120000–120039, gate passed). Two-grid exact-budget g2f helps all
    three scorers: COBS-style 64→72 (10 better / 2 worse), nf_pt1 69→73 (7/3), Quest 67→73 (6/0); dense 79.
    Natural crossing 15/80. Confirmation planned at N = 300, single primary contrast cobs+g2f vs cobs (Codex agreed).
M2 (grid-shift flip and placebo, 40 + 40). Moving a natural-inside answer onto a cut lowers COBS-style 24→16, nf
    27→22, Quest 25→14; placebo shifts that keep the status leave them ≈ unchanged (24, 26, 26). The two grid
    policies differ; the effect of the cut itself is not isolated from that (S1, Codex).
M3 (forced-block probe, 34 straddled prompts; oracle intervention). Forcing the tail block in: COBS-style 11→27
    (16/0), nf_pt1 15→30. Oracle diagnostic (uses the answer label). Related hypothesis: KVFetch's "sequential
    forgetting" (2610.08811, KV eviction setting); not the same evidence — we keep the full KV and re-select each
    step. A block-level online rule inspired by it (+ptr) is a baseline, not a reproduction.
M4 (earlier). Offline selection metrics did not predict end-to-end differences here. (COBS reports an exact-mass
    oracle ≈ dense in its own trained setting; that is a setting-specific observation, not evidence about ours.)
M5 (earlier, historical). Quest with 16-token pages scored 0.838, the best single scorer in that run; sink/tail,
    candidate tokens and memory accounting were NOT matched, so it is not an equal-memory result or a threshold.
M6 (mechanism implementation check, 4 prompts) ACCEPTED (a85b7d6); 63-prompt diagnostic queued.

## 4. Hypotheses and planned tests

H-unit. Failures concentrate where the needed unit is split (Q2), across summary types; the size of the effect
        depends on the summary (interaction). Test: confirmation (N = 300, v2 independent haystacks) + controlled
        shift sweep crossing {moment, box, em}.
H-em.   Ranking by the P2 upper end (em) is at least as good as Quest at ≤ Quest-16 memory. Dev run em, em+g2f queued.
H-ptr.  A read-pointer successor rule (KVFetch-style) recovers most of M3's oracle gain online.
H-σ.    The REM prediction in §1 holds on Qwen keys and the overestimated blocks are where COBS-style fails.

## Withdrawn or demoted

Demoted to classical: T1 (P1), F_n (REM truncation). Withdrawn as novel: union of box and moment scorers (RBS
2609.20971 mean dilution + union), "tail rescue" as a new mechanism (KVFetch). Withdrawn: optimism/UCB; certified
selection on real keys; "the cut is the main cause" (not isolated, M2); "gain grows with length"; inferring the mean
from the box midpoint (breaks P2's exact-mean premise); "em+g2f is equal memory".
