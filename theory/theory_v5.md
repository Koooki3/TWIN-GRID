# Theory v5 — the selection unit must contain the information unit

2026-10-16 midday. Builds on theory_v4.md (T1, T2, E1–E5). [E] = measured, [V] = verified,
[H] = hypothesis under test.

## 1. The finding that resolves E4

E4 said no offline selection-quality metric predicts end-to-end accuracy. The answer texts do:
[E, results/nf_16k.jsonl, src/error_taxonomy.py] wrong answers are almost never a distractor's
value (0 cases) — they are near-misses (≥ 4 of 7 digits right) or, in multivalue, the first value
right and the rest missing. Retrieval succeeds; copying fails.

Splitting samples by whether the answer value's tokens straddle a 32-token block boundary
[E, src/boundary_check.py, single/multikey tasks, budget 512]:

| scorer | straddle (n = 11) | no straddle (n = 29) |
|---|---|---|
| COBS (moment) | 0.18 | 0.83 |
| point estimate (moment + 1 anomaly) | 0.45 | 0.90 |
| Quest + COBS union | 0.55 | 0.97 |
| Quest (box) | 0.82 | 0.86 |
| bulk box + 1 anomaly | 1.00 | 0.93 |
| dense | 1.00 | 1.00 |

Without straddling, moment scorers are as good as boxes or better. With straddling, they
collapse; box scorers do not care.

## 2. Mechanism: a straddled fragment is a spike, and moments cannot see spikes (T1)

When a value of ℓ tokens straddles a boundary, the block holding the tail contains t ∈ [1, ℓ−1]
answer tokens among n = 32 others. While the model copies those digits, the true block
log-partition is dominated by a t-token spike. By T1 its mean and variance are compatible with a
dispersed block whose A is up to φ_n(s) − φ_n(−s) ≈ s√n lower; a moment scorer ranks it low. The
head block (key words + first digits) is a broader "mass" event and is found. Box summaries keep
per-channel extremes, so a spike moves the box; they keep finding the tail.

Probability of straddling: a span of ℓ tokens at a uniformly random offset straddles a B-grid with
probability (ℓ − 1)/B. For a 7-digit value plus period (ℓ ≈ 8) and B = 32: 0.22 per value
[observed: 21 of 60 samples straddle, multivalue has 4 values per sample].

## 3. Principle (two conditions)

A block summary used for selection must be
 (C1) spike-complete — able to register a few extreme tokens (T1: moments are not; boxes, exact
      anomalies or per-token sketches are), and
 (C2) span-aligned — information units (a fact, a number, a name) must not be cut by the selection
      grid, or the cut fragments must be recovered.
Moment scorers fail C1, and the fixed grid violates C2; the failure appears exactly where the two
meet. The organisational analogy: reports filed by fixed time slots instead of by matter — a matter
split across two reports leaves a fragment that every average-based summary erases.

## 4. Remedies and predictions

R1. Neighbour expansion: select the top k/2 blocks and, for each, its better-scored neighbour.
    [H5] Moment scorers + neighbour expansion lose (almost) nothing on straddled samples; the
    no-straddle accuracy drops slightly because half the slots become neighbours. (Running: nb_16k.)
R2. Two offset grids (shift B/2): every span of length ≤ B/2 lies wholly inside some block of one
    grid (proof: a span not contained in a grid-0 block contains a multiple of B; the grid-1 block
    centred there contains it if ℓ ≤ B/2). Costs 2× summaries; selection unit = union of blocks.
R3. Content-aligned blocks: cut at sentence / clause boundaries (variable size, still ≤ B).
    Removes straddling by construction for units shorter than a sentence.
[H6] Under R2 or R3, the union with a box scorer stops helping: the dual channel was compensating
    for C2 violations, not providing a distinct "peak" target.
[H7] The 16-token-page advantage (E3) is partly C2: a straddled tail of a 16-token grid sits in a
    smaller block whose box/moment summary is less diluted (fewer bulk tokens).

## 5. Status

| item | status |
|---|---|
| straddle split of accuracy | measured on 60 samples (11 straddled single/multikey); needs n ≥ 100 |
| error taxonomy (no distractor errors) | measured |
| mechanism via T1 | consistent; the per-step diagnostics missed it because "needle block hit" counted any needle token, and value-digit hit counted any digit |
| R1 neighbour expansion | queued (idle-gated GPU) |
| R2/R3 | to implement |
