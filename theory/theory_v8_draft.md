# Theory v8 (DRAFT, 2026-10-16) — why cutting a unit hurts: two candidate mechanisms with separating predictions

Builds on theory_v7.md (interval frame; two-grid covering lemma). Status words as in v7. Nothing here is confirmed;
the purpose is to turn the observed "partition × summary" pattern into falsifiable statements and decide between them
with the 63-prompt mechanism diagnostic (runs/mech_full_16k.jsonl, queued).

## 1. Observations to explain (development data, 16k/512, Qwen2.5-1.5B)

O1 natural crossing (value spans a grid-0 boundary; same 40-prompt table, results/shift_16k.md, original grid):
   COBS-style 2/11 crossing vs 24/29 not crossing; Quest 9/11 vs 25/29. Unpaired small groups — this does not
   establish that Quest is unaffected by natural crossings. (An earlier draft wrote "Quest 9/9"; withdrawn.)
O2 flip (cut placed in the middle of a naturally-inside value) vs placebo shift (same status): improve/worsen
   COBS-style 3/11, nf_pt1 2/6, Quest 0/12. This is a paired difference between two grid policies, not a fixed-Q/K
   causal test of the two-level model; it argues against using that toy model as a complete explanation.
O3 two-grid g2f helps all three scorers (phase 1: 10/2, 7/3, 6/0).
O4 forcing the tail block in rescues COBS-style 11 → 27 of 34 straddled prompts (oracle).

## 2. Mechanism F — fraction sensitivity of the block summary (proved for a model; partial fit)

Two-level block: B tokens, m of them ("unit") at logit μ_bg + Δ, the rest at μ_bg; f = m/B.
F1 (exact) A − log B − μ_bg = log(1 − f + f e^Δ)  (≈ Δ + log f when f e^Δ ≫ 1).
F2 (second-order cumulant, COBS-style without projection) K − μ_bg = fΔ + ½ f(1−f) Δ².
F3 (max / box) the block maximum is μ_bg + Δ for every m ≥ 1; a per-channel box bound is ≥ it and depends on m only
   through the per-channel envelope of the unit tokens present.
Splitting (fixed Q/K, same background and block length; t = e^Δ − 1, 0 < f₁ ≤ f ≤ 1, Δ ≥ 0):
   δA = log((1 + f t)/(1 + f₁ t)),   0 ≤ δA ≤ log(f/f₁)   (exact: f₁(1 + f t) ≤ f(1 + f₁ t) ⟺ f₁ ≤ f);
   with m₁ ≥ m/2, δA ≤ log 2 exactly; δA approaches log(f/f₁) only when the split block is itself spike-dominated
   (f₁ e^Δ ≫ 1).
   δK = (f − f₁)[Δ + ½(1 − f − f₁)Δ²]   (exact; "linear in f" only as a small-f approximation);
   δmax = 0 for the scalar maximum; a per-channel box bound changes with the per-channel envelope of the tokens left.
δK > δA is NOT a general theorem. Counter-example (Codex): B = 32, m = 2 → 1, Δ = log 33: δA = log(3/2) ≈ 0.405 >
δK ≈ 0.282. Conditional statement: if δK > log(f/f₁) then the moment score falls strictly more than the exact
mass; the parameter region where this holds must be stated explicitly. Example m = 7 → 4, B = 32, Δ = 6: δA = 0.55
nats (≤ log 7/4 = 0.56), δK = 1.67 nats (theory/verify_fraction.py, rc 0) — one parameter point only.
A small δA does NOT make mass-oracle selection invariant to the cut: the top-k threshold can fall inside the change.
Strict counter-example (Codex): B = 32, m = 2, Δ > 0, one candidate slot; the whole target block has unnormalised
mass 30 + 2e^Δ, each fragment 31 + e^Δ; a competitor with mass strictly between them is chosen after the cut and
not before. Rank preservation needs an explicit margin condition, and covering the whole unit may need more blocks.
Reading (narrowed): the second-order approximation can turn a log-type dependence on f into a much steeper one in
part of the parameter space; it is a candidate component, not an explanation of O1–O4 by itself.

## 3. Mechanism R — read requirement over decode steps ("digits ride with the key block")

Copying a 7-token value needs, at decode step j, the block holding source token j+1 (the read pointer's successor).
A content scorer ranks blocks by similarity to the current query, which during copying mixes the key phrase and the
recent digits. If the unit lies in the block that also holds the key phrase, one selection serves all steps. If a
cut leaves digits in a block WITHOUT the key phrase, that block must win on its own at the later steps.
Prediction R1: failures concentrate at the first digit after the cut, for every scorer; R2: the probability that the
post-cut block is selected at those steps is what differs between scorers; R3: a mid-value cut (O2) leaves 3–4 digits
in the key-less block, a natural crossing often leaves 1–2 — a box summary, driven by per-channel extremes of a few
tokens, can then score the small key-less fragment low, matching O2 if Quest's post-cut selection rate is lowest.
Mechanism R also predicts O3 (some grid holds unit + key phrase together when the gap fits, cf. covering lemma) and O4.

## 4. Descriptive diagnostic plan (replaces the earlier "separating predictions"; Codex V3–V5)

F and R are not mutually exclusive (scores shaped by f change what is read later; reading and correctness share
prompt difficulty), so no "which mechanism wins" verdict is attempted. src/mech_matrix.py (written before the 63-prompt
diagnostic finished; definitions frozen in its docstring) reports a descriptive evidence matrix:
- state: primarily the shared dense teacher-forced state (dense_tf: every scorer and the exact / full-moment rankings
  on the same hidden states); per-arm trajectories reported separately;
- groups IN / PRE / POST as before, stratified by task and digit position; prompt-level aggregation (mean within
  prompt, then equal-weight mean across prompts; record counts and dense_weight-weighted means alongside);
- a 3×2 table on the shared state, exact ranking {definite-in, definite-out, boundary tie} × actual selection
  {yes, no}, mutually exclusive and exhaustive; marginals derived from it (ties never averaged as half a hit);
- each arm's own TF table separately; outcome link per arm on straddled prompts with explicit denominators (no
  divergence within the gold-continuation window, digit divergence, of which at the first POST digit, non-digit
  divergence; strict correctness separately), and the own-TF selected share at the divergent step vs the same prompt's
  other teacher-forced digit steps on prompts where both exist — association only;
- prompt bootstrap with the same resample across arms; NA for empty groups with valid-replicate counts;
- gate: frozen 63-key list, frozen identities, the existing acceptance on the same file, state_origin semantics,
  one run identity bound to an analysis manifest written after the run and before analysis, valid dense weights;
  candidate-region filtering with counts of other regions; --smoke for the 4-prompt check file never yields findings.
  Script: src/mech_matrix.py v3 (hash in claude-response.txt), manifest by src/make_mech_manifest.py.
Full-moment margins in nats are not recoverable from the stored fields (no k-th threshold), so no nats-level test of
δK vs δA is made here; a fixed-Q/K two-grid margin test would be a separate design.

## 5. Status of claims

Proved: F1–F3 and the δA bounds for the two-level model under fixed Q/K. Not established: that real blocks follow
it; that δK > δA in general (counter-example above); that mass oracles are invariant to cuts (counter-example above);
which mechanism dominates. Withdrawn: "fraction sensitivity unifies the partition × summary pattern"; "Quest 9/9".

## 6. Partition design: k offset grids vs a finer single grid at equal summary memory (verified, elementary)

L1 (k-grid covering). k grids of block B offset by s = B/k (k | B): every span of length ℓ ≤ (k−1)B/k + 1 lies inside
one block of some grid, and this is sharp. Proof: the union of boundaries is spaced s apart; a span crosses the
boundaries in a run of ℓ − 1 consecutive integers, which holds at most ⌈(ℓ−1)/s⌉ ≤ k − 1 union boundaries when
ℓ − 1 ≤ (k−1)s, so at least one of the k grids has none; a run of (k−1)s + 1 integers can hold one boundary of each
grid. k = 2 gives v7's B/2 + 1. (theory/verify_kgrid.py, brute force B ∈ {4, 8, 16, 32, 64}, k | B, k ≤ 8: rc 0.)
L2 (finer single grid). A single grid of block B/2 splits some span of every length ≥ 2, and a span of length
≤ B/2 + 1 meets at most two of its blocks (same script).
Equal-memory comparison. Two grids of block B and one grid of block B/2 store the same number of block summaries per
token (main term over interior full blocks; real bytes also count edges, dtype, metadata, workspace). For a unit of
length ≤ B/2 + 1 inside the common interior candidate region [B + B/2, C − B/2) (where both grids' covering blocks
are selectable; v7 §2): the two-grid layout has one candidate block holding it whole (read cost B tokens); the finer grid holds it in one or two blocks (read cost B/2 or B) and, when split, needs BOTH blocks
selected. Neither layout is better in read cost for a split unit; they differ in whether a single selection suffices.
Prediction P-grid (to test; not yet run under matched conditions): at equal summary memory and equal read budget,
Quest-32+g2f ≥ Quest-16 in accuracy, with the difference concentrated on prompts whose answer is split by the 16-token
grid. The historical "Quest-16 = 0.838" did not match sink/tail, candidate tokens or memory (v7 M5), so it is not
evidence either way. Planned: development run on the phase-1 seeds (quest with --block 16 vs the existing
quest+g2f@32), and arm pair Quest-32+g2f / Quest-16 in the multi-tier plan (tier 2).
L3 (feasible-set inclusion; Codex G2, paper proof). Suppose the fine grid (block h = B/k) is aligned with every
coarse grid's boundaries, the fixed token region is the same, every fine block inside each coarse candidate block is
selectable, and the read budget counts unique tokens. Then every coarse block is a union of k fine blocks, so any
token set reachable by selecting coarse blocks from any number of offset grids is reachable by selecting fine blocks,
at no larger unique-token cost. Hence, for any ideal objective that depends only on the set of tokens read, offset
grids cannot beat the fine grid at its optimum: the two-grid advantage, if any, is NOT representational.
Consequence for the theory: under these conditions and for an objective defined on the set of tokens read, offset
grids have no feasible-set (representational) advantage; observed differences can still come from the summaries,
the ranking, and the in-budget selection rule (overlap resolution, budget filling), and L3 says nothing about
engineering cost. The research question becomes how far each scorer's choice falls from a good feasible token set
under each layout. This narrows P-grid to a statement about selection rules.
Matching conditions for P-grid (Codex G3): besides the sink, _fixed_end(n) rounds to B, _kfix(n) allocates the
candidate budget in blocks of B, and _layout(n) leaves offset edge fragments; a B16-vs-B32 comparison must match the
fixed token set, the selectable candidate region, the candidate read count and the de-duplicated total read, with
the same tail rule as the generation length grows. Same-B comparisons (Quest16 ± g2f) are unaffected; EM32 vs
Quest16 and Quest32+g2f vs Quest16 are affected — existing development data are exploratory only.
What the lemmas do NOT say: that the selector picks the covering block (v7 §2 (iii)); that a single selection is
easier than a joint one for any particular scorer — that is the empirical content of P-grid.

## 7. Candidate retrieval unit: the KEY→VALUE span (post hoc on development data; prediction P-kv registered during data collection)

Read-path view. Copying a value of ℓ_v tokens needs, at each decode step, the block holding the next source token;
the content scorer finds that block through the query's similarity to the key phrase. Candidate retrieval unit: the
key→value span ("{key} is: {value}", ℓ ≈ 11–12 tokens in this template), not the digits alone. Having the span in one
block may improve how selectable it is for a given summary/scorer; the geometry does NOT prove that same-block
co-location is necessary (the value can be read from several selected blocks, and the query already contains the
key) — L1 is geometric coverage, L3 bounds what layouts can represent within budget. If the span's start is uniform
modulo B, a span of ℓ tokens crosses a single grid with probability (ℓ − 1)/B; the generator inserts needles at
sentence boundaries, so uniformity is an assumption checked by the recorded spans, not guaranteed. By L1 two grids
hold the span whole whenever ℓ ≤ B/2 + 1 = 17.
Development data (phase 1, 80 prompts; found AFTER looking at these data, so hypothesis-generating only):
kv span crosses grid 0 in 25/80 = 0.31 (theory 10–11/32 ≈ 0.31–0.34); COBS-style 48/55 inside vs 16/25 crossing;
cobs+g2f 48/55 vs 24/25 — the NET improvement +8 is all in the kv-crossing group (paired: crossing +8/−0, inside
+2/−2 — the inside net 0 hides offsetting changes); nf_pt1 net +5 / −1; Quest +1 / +5 — that Quest's failures have
another source is a hypothesis awaiting its own diagnostic. The two groups also start from different error rates, so a
positive gain interaction supports heterogeneity, not a unique attribution of the gain to co-location. The
digits-only crossing (15/80) explains less of it.
Registered prediction P-kv: src/kv_prediction.py (P1 crossing share in [0.22, 0.42]; P2 COBS-style accuracy lower on
kv-crossing prompts, Newcombe interval excluding 0; P3 the cobs+g2f gain concentrated on kv-crossing prompts,
stratified-bootstrap interaction interval excluding 0; P4 nf_pt1/Quest exploratory), committed while the confirmation
run had written 76/300 rows (the author states no confirmation result had been looked at); to be run unchanged on
runs/phase2_g2f_confirm.jsonl after the frozen prereg-v3 analysis. It is a separate, descriptive secondary analysis
and does not alter prereg v3.
Status wording (Codex K1–K4): "a secondary descriptive prediction registered during data collection; the author states
the confirmation results had not been looked at" — not a pre-registration before the experiment. Confirmation entry:
src/kv_prediction_confirm.py (full frozen gate, succeeded job receipt, prompt+token identity, SPEC tokenizer/B; then
the original script unchanged, then task × crossing tables with paired gains/losses). P1 window [0.22, 0.42] is the
registered acceptance window, not a derived 95% interval; P2/P3 intervals are unadjusted descriptive intervals;
labels mean "met / not met the registered descriptive rule", never a multiplicity-controlled confirmation, and "not
met" is not evidence of no effect. The pooled (two-task) P3 estimand is kept as registered; it is not a task-controlled
interaction nor a causal effect of cutting.

## 8. Mechanism diagnostic results (63 development-selected prompts; descriptive; wording agreed with Codex)

Run: job 1f56da28 succeeded; acceptance ACCEPTED; frozen mech_matrix v4 with pre-run EXPECTED + post-run manifest,
matrix_rc 0 (results/mech_matrix_full.log). IN = 29 prompts, PRE/POST = the same 34 straddled prompts; 11,466
candidate records on the shared dense teacher-forced state.
Agreed summary: "The full diagnostic on 63 development-selected prompts shows that, on the shared dense teacher-forced
state, the POST group has lower exact definite-in, full-moment definite-in and actual selection shares than the IN
group, with the larger group difference for the COBS-style scorer's actual selection; Quest has a higher POST
dense-weighted joint share. Most first free-run divergences did not fall exactly on the first POST digit. These
readings describe associations between candidate-block ranking, selection and generation divergence and are compatible
with parts of the mechanism expectations, but they do not isolate a causal effect of cutting, do not confirm a single
F or R mechanism, and do not establish the mediating mechanism of the TWIN-GRID gain."
Numbers (IN − POST, prompt-bootstrap 95%): exact definite-in +0.109 [+0.065, +0.156]; full-moment definite-in +0.168
[+0.116, +0.223]; selected cobs +0.256 [+0.193, +0.320], nf_pt1 +0.207 [+0.151, +0.266], quest +0.161 [+0.079, +0.238].
POST dense-weighted joint share "exact definite-in AND selected" (per prompt, then prompt-equal): cobs 0.658, nf_pt1
0.760, quest 0.814 — a joint share, not P(selected | exact-in), not a global mass ratio, not free-run retention.
First digit divergences among 34 straddled prompts: cobs 23 (4 at the first POST digit), nf_pt1 19 (3), quest 13 (3),
cobs+ftail 7 (0).
Limits (Codex M1–M3): IN and POST are different prompts and digit positions — not before/after pairs of one prompt;
"exact definite-in" is relative top-k membership, not absolute mass or copy utility; the larger moment/selection
differences are compatible with part of F's expectation but are not decomposed into identified contributions; the
first-divergence counts give no direct support to the simplest R1 form ("fails exactly at the first post-cut digit")
and do not rule out a more general read-requirement mechanism (position-wise at-risk counts would be needed); the
divergence-step link is outcome-conditioned; this diagnostic has no +g2f arm and different prompts / crossing
definition from C3, so it is not a mechanism validation of C3.
