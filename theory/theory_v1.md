# Theory v1 — block selection as free-energy (log-partition) estimation

Status: draft for adversarial review, 2026-10-16. Numbers marked [V] are verified in
`theory/verify_v1.py`; [E] are measured on Qwen2.5-1.5B (results/*.md); [H] are hypotheses.

## 0. Setup

A block b holds n keys k_1..k_n and values v_1..v_n. For a query q let x_i = q·k_i/√d.
Define the block's log-partition (cumulant generating function of its empirical key
distribution, evaluated at q):

    A_b(q) = log Σ_i exp(x_i),     h_b(q) = max_i x_i   (support function of the key hull, scaled)

Exact attention over all blocks factorizes as a mixture of blocks:

    o(q) = Σ_b w_b(q) o_b(q),   w_b = softmax_b(A_b(q)),   o_b = within-block attention output.

With the joint log-partition A_b(q, λ) = log Σ_i exp(x_i + λ·v_i) we have o_b(q) = ∇_λ A_b(q, 0).
So, for attention, a block is fully described by ONE convex function A_b(q, λ): its value
decides the block's share, its λ-gradient is the block's output. Every block summary
(mean key, Quest box, COBS cumulants, centroids, ...) is an approximation of this function,
and block-sparse attention is "evaluate approximations, refine exactly where it matters".

## 1. Proposition 1 — mass and peak are the same target up to log n

For every block, query and β > 0, with A^β = (1/β) log Σ exp(β x_i):

    h_b ≤ A^β_b ≤ h_b + (log n)/β.                                   (P1)

Proof: max ≤ log-sum-exp ≤ max + log n, applied to βx. [V: 320k random blocks, bound tight.]

Corollary 1.1 (selection by any temperature is near-equivalent). Let S_1 and S_∞ be the
top-k blocks by A (mass) and by h (peak), Z(S) = Σ_{b∈S} exp(A_b). Then

    Z(S_∞) ≥ Z(S_1) / n.

Proof: pair each b' ∈ S_1\S_∞ with a distinct b ∈ S_∞\S_1 (equal sizes). Since b' was not
in the peak top-k, h_b' ≤ h_b, so A_b' ≤ h_b' + log n ≤ h_b + log n ≤ A_b + log n.
Summing exp over the pairs: Z(S_1\S_∞) ≤ n Z(S_∞\S_1); add the common part. ∎

The worst case is weak, but the swapped blocks are only those within log n
(= 3.47 nats for n = 32) of the selection threshold. If real attention puts a margin
≫ log n between selected and unselected blocks, every β selects the same set.
[E] On Qwen2.5-1.5B at 16k the mass, peak, β=2 and dual oracles differ by ≤ 0.003 in
accuracy and all retain 0.995 of needle attention mass (budget 512).
[H1] Fraction of (head, step) where the k-th/k+1-th block margin < log n is small.
Consequence: the "which target" question (mass vs peak, round 2) is settled — the target
is A_b; all that matters is how well a compact summary ESTIMATES it.

## 2. Proposition 2 — the cumulant (COBS) estimate has two phases

Model the scores within a block as x_i = m + s z_i, z_i iid N(0,1). Then

    max(log n, c_n s) + m  ≤  E A  ≤  m + log n + s²/2,                 (P2)

where c_n = E[max of n standard normals] (c_32 ≈ 2.07). Upper: Jensen on log. Lower:
log-sum-exp ≥ max and ≥ mean + log n.

The right side is exactly the second-order cumulant (COBS) estimate. It is accurate for
s ≲ 1.5 and overestimates without bound beyond: [V] n = 32,

| s | 1 | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| E A − m | 3.94 | 5.22 | 6.91 | 8.79 | 12.73 | 16.79 |
| cumulant error | 0.02 | 0.25 | 1.05 | 2.68 | 8.74 | 18.68 |

This is the random-energy-model picture: a high-temperature phase (many tokens share the
mass, A ≈ log n + s²/2) and a frozen phase (the maximum dominates, A ≈ c_n s + small).
It explains COBS's self-reported failure: dispersed distractor blocks (large s) receive a
quadratically inflated score. Two-phase estimate (C¹, no new statistics):

    f_n(s) = log n + s²/2           if s ≤ c_n
             log n − c_n²/2 + c_n s  otherwise.

## 3. Proposition 3 — exact near/far decomposition (the "report" structure)

For any subset C of the block (chosen without the query),

    exp A_b = Σ_{j∈C} exp(x_j)  +  Σ_{i∉C} exp(x_i).                     (P3)

Store the keys in C exactly (near field) and the bulk B\C by low-order statistics (far
field): Â_b = log( Σ_{j∈C} e^{x_j} + e^{m' + f_{n−|C|}(s')} ), where m', s' are the bulk's
mean score and projected std. Choose C = the |C| keys farthest from the block mean in the
query subspace. Removing outliers shrinks s', moving the bulk back into the
high-temperature phase where the cumulant is accurate.
[V] Gaussian bulk + one outlier, n = 32: mean |error| of log Z

| bulk std | outlier | whole-block cumulant | max only | outlier + bulk |
|---|---|---|---|---|
| 0.5 | 5 | 1.09 | 0.21 | 0.001 |
| 1.0 | 10 | 4.24 | 0.002 | 0.000 |
| 2.0 | 5 | 0.29 | 0.72 | 0.20 |

This is the organizational analogy made exact: a good report names the largest anomalies
individually and summarizes the rest; "mass" and "peak" are not two channels but two terms
of one sum. It is also the fast-multipole structure (exact near field + expansion far field)
applied to the softmax partition function rather than to a kernel sum.

## 4. Unification of existing summaries

| summary | approximation of A_b(q) | regime where accurate | failure |
|---|---|---|---|
| mean key (NSA-style) | m + log n (tangent at q = 0; lower bound) | s → 0 | any dispersion |
| COBS | m + log n + s²/2 (2nd-order cumulant; upper bound in expectation) | high temperature | frozen phase, outliers |
| Quest | Σ_c max(q_c min_c, q_c max_c) ≥ h_b (box support function) | frozen phase, low dim | loose in high dim |
| peak/mass/β oracles | exact A^β | — | equal up to log n/β (P1) |
| dual union (round 3) | hedge between a high-temperature estimate and a frozen-phase bound | both, heuristically | — |
| near/far (this note) | exact outliers + two-phase bulk | both phases, outliers | outlier not among C |

Memory per block (B = 32, d = 128, subspace r = 16), floats: Quest 2d = 256; COBS
d + r(r+1)/2 = 264; Quest+COBS dual 520; near/far with |C| = 1: 2d + 136 = 392; |C| = 2: 520.

## 5. Testable predictions

- [H2] Two-phase bulk ("rem") ≥ COBS in accuracy at equal memory.
- [H3] Near/far with |C| = 1 matches or beats the Quest+COBS union at lower memory; |C| = 2
  beats it at equal memory. This would replace a two-channel heuristic with one estimator.
- [H4] Near/far with two-phase bulk ≥ near/far with plain cumulant bulk.
- [H1] as above.

## 6. Direction: a module, not a heuristic

Because o_b = ∇_λ A_b, the same near/far summary also approximates the OUTPUT of an
unselected block (exact outlier values + bulk mean value + value–key covariance term),
so unselected blocks need not be dropped. Recursing (blocks of blocks) gives a
hierarchical attention whose every node reports (exact anomalies, bulk free energy), with
refinement where the estimate's uncertainty is largest. Whether such a module can replace
dense attention in training is the open question for stage 3.
