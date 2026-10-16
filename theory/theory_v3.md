# Theory v3 — block selection is detection; rank by the tightest certified upper bound

Status: 2026-10-16 morning, after the round-4 measurements. [V] numerically verified,
[E] measured on Qwen2.5-1.5B (16k, RULER-style needle tasks, 60 paired samples unless noted),
[H] hypothesis. Builds on theory_v2.md (Theorems 1–2) and its review log.

## 1. What the measurements changed

1. Near-ties dominate the threshold [E, diag_16k]. The gap between the k-th and (k+1)-th
   block log-partition is below log B = 3.47 nats in 99.9% of (layer, head) cases; median
   0.14 nats. Mass-top-k and peak-top-k differ in 21–25% of blocks, yet all oracles reach the
   same accuracy. So selection near the threshold is almost free; what decides accuracy is
   not missing the few blocks that are far ABOVE it. v1/v2's "large margins" hypothesis (H1)
   is refuted; the β-equivalence holds for a different reason: swapped blocks are near-ties.
2. Calibration does not predict ranking quality [E]. Block-wise calibration error (MAE vs the
   true A_b): point estimate with 2 exact outliers 0.53–0.76, COBS 0.72–2.95, mean key
   1.6–3.8, Quest/box bounds ≈ +20 (biased upward). Yet as rankers at budget 512:
   bound-based nf_hi1 0.808 > Quest 0.725 > point estimate 0.613 > COBS 0.546 > lower bound
   0.379. The best-calibrated estimator ranks worse than a bound that is 20 nats off.
3. Certification with box bounds is impossible on real keys [E]: reading blocks in order of
   the upper bound needs ≈100% of blocks to certify any ε ≤ 0.1, against 4–24% in oracle
   order. Box bounds are too loose in d = 128 (the √d cost the reviewer predicted).

## 2. The principle: detection under one-sided costs → optimism

Missing a decisive block (false negative) costs the answer; including a mediocre block
(false positive) costs only a slot that near-ties make almost worthless (§1.1). Under this
asymmetry the right score is an upper confidence bound, not an unbiased estimate — the same
"optimism in the face of uncertainty" that governs bandits. A calibrated estimator is
unbiased on average and therefore under-scores exactly the rare spike blocks (Theorem 1);
an upper bound never under-scores them, and its cost is only false positives among
dispersed blocks.

This explains every round-3/4 result with one rule:
- Quest works although its bound is 20 nats loose: it is an upper bound (no false negatives).
- COBS and the point estimates fail: calibrated, so spike blocks are under-scored.
- The Quest+COBS union helps over either alone: half the slots go to the upper bound
  (recall), half to a calibrated score (filling near-ties); with a better bound the union
  stops helping (nf_dual2 0.787 ≈ nf_hi1 0.808).
- Peeling one exact outlier before boxing (nf_hi1) beats Quest at the same block size:
  +0.083 [+0.013, +0.163] at budget 512 — the bound gets tighter exactly on spike blocks.

Design rule: per byte of summary, minimise the slack of a CERTIFIED upper bound on the
blocks that matter. The calibration of the bound is irrelevant; its ordering on high blocks
is what counts.

## 3. Theorem 3 — the missing statistic is the maximum leverage

Theorem 1 says moments cannot see spikes. The minimal fix is one scalar. Let c_i = W^T(k_i − μ)
be centred keys in the query subspace, Σ = (1/n) Σ_i c_i c_i^T, and the maximum leverage
radius R = max_i ||Σ^{-1/2} c_i||. Then

  (a) R ≤ √(n−1), with equality iff the block is a single spike in the subspace
      (leverage scores of centred data are at most 1 − 1/n);
  (b) for every query, x_i − m ≤ R·s + R_res·||q_res||/√d, where s² = q^T W Σ W^T q / d is the
      within-block score variance in the subspace and R_res = max_i ||(I − WW^T)(k_i − μ)||;
  (c) hence A_b ≤ m + R·s + R_res||q_res||/√d + log n (certified), and for a single spike
      aligned with q the subspace part is tight.

So the "spikiness" that moments miss is exactly the block's maximum leverage, R/√(n−1) ∈ (0, 1].
Moments + {R, R_res} (COBS memory + 2 floats) give a certified upper bound. [E pending: ell_16k]

## 4. Granularity vs summary (open)

At equal memory (512 floats per 32 tokens), Quest with 16-token blocks beats every 32-token
summary: 0.838 vs nf_hi1 0.808 (−0.029 [−0.104, +0.046]) at budget 512 and 0.658 vs ≤ 0.450 at
256. Two effects are mixed: finer SELECTION (a smaller sink and more, smaller slots) and a
TIGHTER BOUND (two small boxes). Test running: two sub-boxes per 32-token block, selected at
32-token granularity, split contiguously (qsub2c) or by the block's principal direction
(qsub2k). [E pending]

## 5. Status of claims

| claim | status |
|---|---|
| Theorem 1 (moment identifiability gap) | proved except one step verified on a grid [V] |
| Theorem 2 (anomalies + bulk mean + bulk box interval) | proved; bounds hold on 100% of real blocks [E] |
| Theorem 3 (max leverage) | (a) standard, (b)(c) proved by Cauchy–Schwarz; usefulness pending |
| optimism principle | consistent with all round-3/4 rankings [E]; a formal decision-theoretic statement is still to be written |
| certified selection | refuted for box bounds on real keys [E] |
| margins ≫ log n (H1) | refuted [E] |
