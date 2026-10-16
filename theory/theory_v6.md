# Theory v6 — consolidated after review 3 and the Codex independent review (2026-10-16)

Status words: verified (proof or exact recount) / measured (exploration seeds 1000–1019, not confirmatory) /
hypothesis / withdrawn. "COBS" below = this project's COBS-style projected-variance scorer (query-subspace
dimension s = --rank), NOT the method of arXiv 2607.09052.

## T1 — extremes of the log-partition under fixed first two moments (verified; proof completed by Codex)

For n ≥ 2 scores with mean m and population variance s², A = log Σ e^{x_i} satisfies
    m + φ_n(−s) ≤ A ≤ m + φ_n(s),  φ_n(t) = log(e^{t√(n−1)} + (n−1)e^{−t/√(n−1)}),
with equality at one high value (max) and one low value (min).
Proof (Codex, theorem1-proof.txt; checked): stationary points of Σe^{x_i} on the constraint set satisfy
e^{x_i} = λ + η(x_i − m), so take two values a > b; η is the chord slope, e^b < η < e^a. If a maximiser had two
coordinates equal to a, the tangent direction v = e_i − e_j gives vᵀ∇²L v = 2(e^a − η) > 0, contradicting the
second-order necessary condition for a constrained maximum; symmetric argument for the minimum. ∎
Gap Δ_n(s) = φ_n(s) − φ_n(−s):
  s → ∞ (n > 2):  Δ = s(√(n−1) − 1/√(n−1)) − log(n−1) + o(1) = s(n−2)/√(n−1) − log(n−1) + o(1)
  s → 0:          Δ = (n−2)s³/(3√(n−1)) + O(s⁵);   n = 2: Δ ≡ 0.
Meaning: worst-case non-identifiability for any deterministic estimator that sees only (m, s, n) — at least
Δ/2 error on some feasible input. It does NOT say moment scorers fail on every spike, and it is about the first
two moments only (n power sums determine the multiset).

## T2 — certified interval from (exact anomalies, bulk mean, bulk box) (verified)

Unchanged from v2. The heuristic point estimate nf_pt is NOT guaranteed inside [L, U]
(F_n(0) = log(1 + (n−1)Φ(c_n)) < log n); it is renamed "heuristic point estimate"; past runs keep the original
definition. A clipped variant would be a new method with a new run identity.

## Measured (exploration, 16k unless noted; Qwen2.5-1.5B-Instruct; local RULER-style NIAH; answer containment)

M1. Union of a box-type and a moment-type scorer beats either alone in two settings (16k/512 rank-16
    per-q-head; nominal 32k/1024). Sensitive to implementation configuration (same 60 prompts: union 0.821 under
    the old configuration vs 0.729 under rank-8 shared subspace + chunked prefill). Not established as robust; no
    claim that the gain grows with length.
M2. Box-type scorers rank better than COBS-style at 16k/512 under the current configuration.
M3. At 16k/512, single-value prompts whose 7-digit answer crosses a 32-token block boundary: COBS-style 2/11 vs
    24/29 when not crossing; box-type unaffected; offline re-definition of "crossing" on shifted grids over
    existing outputs shows no such association (review 3; NOT a moved-grid experiment). Not seen at budget 256
    or at nominal 32k (4 crossing prompts). A conditional association, not a general law.
M4. Offline selection-quality metrics (first-step or per-step, any weighting, any-needle-token hits) did not
    predict end-to-end differences.
M5. 16-token pages (Quest's default) beat every 32-token summary at equal summary memory, but sink/tail and
    candidate tokens were not matched; granularity vs accounting is unresolved.

## Hypotheses under test (queued; results will be appended)

H-cut. Crossing a block boundary causes the COBS-style failures in M3. Test: grid-shift intervention with fixed
       sink/tail and matched candidate tokens (flip both directions) + placebo offsets.
H-ride. "Digits ride with the key-phrase block": a straddled tail is reachable only if the block holding it is
       selected; forcing it in rescues COBS-style. Diagnostic only (forced tail/head/left-neighbour).
H-cfg. Configuration non-equivalence (M1) comes from prefill chunking vs rank vs subspace sharing: 2×2×2 factorial.

## Withdrawn

Optimism/UCB explanation; certified selection with box/radius bounds on real keys (needs ≈ all blocks in this
setup); "minimal missing statistic" (radius); margins ≫ log n; "the tail is a spike invisible to moments" as an
established mechanism; "moments cannot see anomalies however many moments"; length-monotone gains.
