# Theory v2 — moment summaries cannot see anomalies; an interval summary can

Status: revision after adversarial review of v1 (review notes: what was wrong, what changed,
at the end). 2026-10-16. [V] = verified numerically (theory/verify_*.py); [E] = measured on
Qwen2.5-1.5B; [H] = hypothesis under test.

## 0. Setting (unchanged, credited)

Block b holds n keys; for a query q the scores are x_i = q·k_i/√d and the block's share of
attention is governed by its log-partition A_b = log Σ_i e^{x_i}. Treating block selection as
estimating A_b from a query-independent summary is the explicit basis of COBS
(ln m_b = ln L + K_X(q)); Tactic and Twilight target cumulative attention mass. Nothing in this
section is new. What follows asks a sharper question: WHAT INFORMATION must a block summary
keep for A_b to be estimable at all?

## 1. Theorem 1 — the identifiability gap of moment summaries

Let x ∈ R^n have mean m and population variance s². Over all such x,

    m + φ_n(−s) ≤ A(x) ≤ m + φ_n(s),
    φ_n(t) = log( e^{t√(n−1)} + (n−1) e^{−t/√(n−1)} ),

and both ends are attained: the upper by a single spike (one score at m + s√(n−1), the rest
at m − s/√(n−1)), the lower by a single "anti-spike" (one score at m − s√(n−1)).

Proof sketch. Maximise/minimise the convex function Σ e^{x_i} on the sphere
{Σx_i = nm, Σ(x_i−m)² = ns²}. Stationarity gives e^{x_i} = λ + μ x_i; a line meets the
exponential in at most two points, so extremisers take at most two values, say a (k times)
and b (n−k times). Among the two-value configurations with fixed mean and variance, the
value is maximised at k = 1 with a > b and minimised at k = 1 with a < b. [V: projected
gradient optimisation from 60 random starts matches the formula to 3 decimals for
n ∈ {8, 32}, s ∈ {0.5, 1, 2, 4}; theory/verify_v2_identifiability.py. The two-value step —
maximum at one high value, minimum at one low value, over all k — holds at every point of a
grid n = 2..128 × 120 values of s ∈ [0.05, 12]; theory/verify_v2_twovalue.py.] A closed-form
proof of the k-step is still to be written.

Gap for n = 32: 0.21 (s = 0.5), 2.05 (s = 1), 7.34 (s = 2), 18.1 (s = 4) nats. The gap grows
like s(√(n−1) + 1/√(n−1)).

Corollary 1 (all mean+g(variance) scorers fail on anomalies). Any scorer that depends on a
block only through (m, s, n) — mean key, COBS, the two-phase/REM correction, F_n — cannot
distinguish a one-spike block from a dispersed block with the same moments, and its
worst-case error is at least half the gap. Adding the variance term can only help by betting
on the typical shape; a needle block is the atypical shape.
[V, reviewer] one needle block among 200 Gaussian distractor blocks with equal-or-larger s:
the oracle ranks the needle block first 77–82% of the time, COBS / two-phase / F_n 0%.
[V, reviewer] On a spike block the empirical cumulant UNDER-estimates A (−4.6 nats for a
+10 spike), while dispersed blocks are over-estimated: COBS errs in both directions, for the
same reason.

Interpretation. Moments are what an organisation's monthly average reports: they can never
reveal the single incident that matters, however many moments are reported, unless the
summary names extremes explicitly. This is the mathematical content of the "report the
largest anomaly" intuition from round 1.

## 2. Theorem 2 — a certified interval from (anomalies, bulk mean, bulk box)

Choose a set C of |C| = c keys per block without the query (e.g. farthest from the block mean
in the head's query subspace). Store: the keys in C exactly, the bulk mean μ' and the
per-channel bulk box [ℓ, u] of the remaining n − c keys. Then for every query

    L_b = log( Σ_{j∈C} e^{x_j} + (n−c) e^{q·μ'/√d} )
    U_b = log( Σ_{j∈C} e^{x_j} + (n−c) e^{Σ_ch max(q_ch ℓ_ch, q_ch u_ch)/√d} )

satisfy L_b ≤ A_b ≤ U_b. Proof: split the sum (exact); Jensen on the bulk for L; each bulk
score is at most the box support function for U. ∎

Special cases: c = 0 gives L = mean key + log n (NSA-style compressed key) and U = Quest +
log n. So the two most common summaries are the two ends of one interval, and the
Quest+COBS "dual channel" of round 3 is a heuristic way of using both ends. Removing the
anomalies from the box both tightens U (the box no longer stretches to the outlier) and puts
the anomaly into the exact term, where it can never be under-estimated.

Point estimate inside the interval: replace the bulk term by e^{q·μ'/√d + F_{n−c}(s')} with s'
the bulk's projected score std and F_n the max-plus-truncated-bulk closed form (§3). By
Theorem 1 the bulk's moments are now trustworthy only to the extent the bulk has no further
anomalies — which is exactly what removing C aims at.

## 3. A closed form for the Gaussian bulk (heuristic, credited)

For Gaussian bulk scores m + s z_i, E A ≈ m + F_n(s) with
F_n(s) = log( e^{c_n s} + (n−1) e^{s²/2} Φ(c_n − s) ), c_n = E max of n standard normals:
the largest term plus the others truncated below it (the standard REM truncation device;
Bovier–Kurkova–Löwe 2002, Ben Arous–Bogachev–Molchanov 2005). In-model error ≤ 0.14 nats
for n ∈ [8, 128], s ≤ 12, versus up to 57 for the cumulant [V]. With an empirical s the
error std is 0.4–4.4 nats and it is biased low under heavy tails (reviewer) — it is a
heuristic for the bulk only, not a contribution.

## 4. Mass vs peak (v1 P1, corrected)

h ≤ A^β ≤ h + (log n)/β for every β > 0 [V]. Top-k sets by different β coincide when the
k-th/(k+1)-th margin exceeds (log n)·max(1, 1/β) (v1 omitted the 1/β for β < 1). The
factor-n captured-mass bound of v1 is correct but vacuous; whether real heads have such
margins is measured, not assumed [H1, being measured].

## 5. What is claimed now, and what would refute it

Claims:
1. Theorem 1: moment summaries have an identifiability gap Θ(s√n); [V].
2. Theorem 2: (anomalies + bulk mean + bulk box) gives a certified interval containing Quest
   and mean-key as the c = 0 endpoints; [proof].
3. [H3'] On real heads, a single query-independent anomaly per block recovers the needle
   token in most needle blocks (needle-block recall), and nf intervals are much tighter
   than Quest on high-spikiness blocks.
4. [H4'] At equal per-KV-head memory, selecting by the interval (U, or U/point hedge) beats
   Quest with half-size blocks and Quest+COBS.

Refutation: if needle-block recall of c ∈ {1, 2} is low, or nf at equal memory loses to
Quest with 16-token blocks, the interval summary has no practical edge and Theorem 2
remains a reformulation.

Memory, per KV head per 32-token block (d = 128, query subspace shared per KV head, r = 8):
Quest 2d = 256 floats; COBS d + r(r+1)/2 = 164; Quest+COBS 420; interval c = 1: d (anomaly)
+ d (bulk mean) + 2d (bulk box) = 512 (+1 index); Quest with 16-token blocks: 512 per 32
tokens. Selection is per query head; KV reads are the union over the 6 heads of a group for
every method alike.

## 6. Direction (narrowed)

Strongest defensible version (reviewer's suggestion, adopted): certified selection. With
[L_b, U_b] per block, any block with U_b below the k-th largest L is provably not in the
exact top-k, and the captured-mass fraction of a selection S is certified by
Σ_{b∈S} e^{L_b} / (Σ_{b∈S} e^{L_b} + Σ_{b∉S} e^{U_b}). This gives an adaptive budget with a
guarantee (top-p in mass), and refinement only where intervals overlap. A training-time
replacement for attention is NOT claimed; trainable hierarchical/FMM attention already exists
(FMMformer, H-Transformer-1D, NSA, Multipole Attention).

## Review log (v1 → v2)

| v1 claim | review finding | v2 |
|---|---|---|
| COBS is an upper bound in expectation | only for population Gaussian moments; empirically it under-estimates spikes | dropped; Theorem 1 explains both errors |
| two-phase/F_n fixes COBS | all mean+g(var) scorers share the spike blind spot | F_n demoted to a bulk heuristic |
| selection = log-partition estimation (new) | explicit in COBS; Tactic/Twilight related | credited as setting |
| near/far estimator | the only real lever; hinges on outlier recall | now Theorem 2 interval + recall measured |
| equal-memory table | wrong under GQA (per query head) | subspace shared per KV head, r = 8 as in COBS |
| β-equivalence for all β | false for β < 1 as stated | corrected |
| §6 module claim | overreach | narrowed to certified selection |
