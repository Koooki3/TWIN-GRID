# Theory v4 — what survives two adversarial reviews (2026-10-16 morning)

v3 proposed "selection is detection → rank by the tightest certified upper bound (optimism)".
The second review (theory/review log below) showed that story is contradicted by our own
data. v4 keeps only what survives; everything else is listed as refuted or open.

## Survives

T1. Identifiability gap of moment summaries (v2 Theorem 1). For scores with mean m and
variance s² over n tokens, the log-partition ranges over [m + φ_n(−s), m + φ_n(s)],
φ_n(t) = log(e^{t√(n−1)} + (n−1)e^{−t/√(n−1)}), both ends attained (one spike / one
anti-spike). Width ≈ s√n: 2.05 nats at s = 1, 7.3 at s = 2 (n = 32). Any scorer that sees a
block only through (m, s, n) — mean key, COBS, REM/F_n corrections — cannot tell a spike
block from a dispersed block. [proof + numerical check; one two-value step verified on a grid]

T2. Certified interval from (exact anomalies, bulk mean, bulk box) (v2 Theorem 2); mean key and
Quest are its c = 0 endpoints. Holds on 100% of ~9.5M real blocks.

T3'. Whitened radius bound (v3 Theorem 3 (b)(c), corrected): A ≤ m + R·s + R_res‖q_res‖/√d + log n
is valid. R ≤ √(n−1) (centred leverage ≤ 1 − 1/n); equality iff the point is outside the affine
hull of the others — NOT "iff a single spike"; R carries no direction information. Prior art:
centroid + radius Cauchy–Schwarz bound in RBS-Attention (arXiv 2609.20971); ours is a whitened,
subspace-split refinement. On real keys it is 17–21 nats loose (residual term dominates).

E1. The union of a box-type and a moment-type scorer helps, increasingly with length:
16k/512 +0.084 [+0.022, +0.153] (rank-16 per-head COBS); 32k/1024 +0.225 [+0.125, +0.333].
The 16k control "box + moment" > "two moment estimators" (+0.083 [+0.017, +0.163]) indicates the
gain is not plain ensembling — but it is a single uncorrected comparison.

E2. Box-type scorers rank better than moment estimators at 16k/512 (nf_hi1 > COBS, nf_hi1 > nf_pt1;
Holm-significant), although moment point estimates are far better calibrated (MAE 0.5–0.9 vs
+17–24 nats bias). Calibration error does not predict ranking quality.

E3. Finer pages dominate at low budget: Quest with 16-token pages (Quest's own default) beats
every 32-token summary at budget 256 (Holm-significant). Part of this is budget accounting
(25% more candidate tokens at 256, 8% at 512); token-matched controls are still to run.

## Refuted or withdrawn

- "Mass vs peak target is settled because margins ≫ log n" (H1): the adjacent-gap statistic shrinks
  like 1/k for any smooth tail; it says nothing about decisive blocks. Withdrawn.
- "Optimism / rank by upper bound because bounds never under-score spikes": the box bounds'
  slack PENALISES spike blocks more than COBS does (bias bin 0 → bin 3 drops 4.9–6.5 nats vs
  2.6). A constant bias is irrelevant to ranking. Withdrawn.
- "nf_hi1 wins by putting the anomaly in an exact term": the exact term is ~e^−17 of the box
  term; nf_hi1 is effectively "Quest on the 31 bulk keys" (a de-noised box). Mechanism open.
- Certified selection with box/radius bounds: needs ≈100% of blocks on real keys. Refuted.
- F_n / REM corrections as scorers: they are moment scorers (T1 blind spot). Demoted.
- UCB analogy: one-shot top-k has no exploration; the correct decision rules are Bayes
  (posterior of being the decisive block) or worst-case using both L and U.

## Measured after v4 (2026-10-16, 09:30–10:00)

E4. No offline selection-quality metric we tried predicts end-to-end accuracy. Retrieval-weighted
oracle-mass capture and needle-block hit, at the first decode step, per step along the dense
answer trajectory, on each method's own sparse (on-policy) trajectory, and with the target
restricted to the answer-value digits: the moment point estimate (nf_pt1, e2e 0.613) scores as
well as or better than the box bound (nf_hi1, e2e 0.808) every time; unweighted over all heads
moment scorers capture 0.87–0.90 of oracle mass vs 0.64–0.68 for boxes. Cross-layer drift is
not the explanation (on-policy shifts ≤ 0.03). Files: results/diag3_16k.json,
mech_steps_16k.json, mech_onpolicy_*_16k.json, mech_value_16k.json.
E5. The whitened radius bound is useless in practice (ell_hi 0.308 e2e; residual term dominates);
two sub-boxes at 32-token selection give only +0.037 (n.s.) — page granularity, not bound
tightness, explains most of the 16-token-page advantage.

## Open questions (next experiments)

1. Mechanism of E1/E2 by causal intervention (sparse attention enabled only in chosen layers or
   heads; patching), since offline selection metrics do not predict accuracy (E4).
2. Token-matched page-size controls (B16 with a 32-token sink and budget −32; B32 with a
   16-token sink); sub-box bounds at fixed 32-token selection (qsub2c / qsub2k, running).
3. A Bayes rule for "probability this block holds the decisive token" from (moments, box) —
   the principled version of the union.
4. Power: ≥ 100 samples per task, a second model family, non-needle tasks (CWE/FWE, QA).
