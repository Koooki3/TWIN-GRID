# Pre-registration v3 (DRAFT for Codex review, not yet frozen) — two offset grids with exact budget (g2_fill_v1)

Supersedes draft v2 (which said "NOTHING has been run" and code 121fc2e — both out of date). Changes are only those
listed in "Changes from v2"; nothing here was chosen after seeing any confirmation (phase 2) data — none exists.

## Status at drafting (2026-10-16)
- Phase 1 (development) DONE: job 278258dc39384a558213209c42f8b0e4, succeeded/exit 0; runs/phase1_g2f_dev.jsonl,
  80/80 keys, integrity gate passed (results/phase1_analysis.txt; Codex audit phase1-audit.json).
  Strict accuracy: dense 79; cobs 64 → cobs+g2f 72 (10 better / 2 worse); nf_pt1 69 → 73 (7/3); quest 67 → 73 (6/0).
  Primary pilot discordance d = 0.150 (p10 = 0.125, p01 = 0.025); per task multikey 0.175, single_essay 0.125.
- Phase 2 (confirmation): NOT started. Will start only after Codex review of this draft and the freeze below.

## Question and scope (unchanged)
Does g2_fill_v1 (`+g2f`) improve strict answer accuracy of the COBS-style projected-variance scorer on frozen
Qwen2.5-1.5B-Instruct, local RULER-style NIAH (multikey, single_essay), 16k, budget 512? Not claimed: superiority
over Quest, speed/memory benefit, other models/tasks/lengths, that the effect is independent of summary type, or
that cutting the value is the cause.

## Arms (unchanged)
A0 cobs; A1 cobs+g2f — the single primary contrast A1 vs A0. Secondary family: nf_pt1+g2f vs nf_pt1, quest+g2f vs
quest (Holm, α = 0.05). Reference: dense. Fixed: block 32, local 64, rank 8 shared per KV head, prefill chunk 2048,
bf16, greedy, max 14 new tokens, budget 512, --strict-budget (per-call invariant raises).

## Data (changed: v2 haystack)
- Seeds (phase 2 = 200000 base, per task stream): multikey 210000–210149, single_essay 220000–220149 (150 + 150).
- Haystack v2 (--haystack v2 --slot0 100): 47 public-domain books, cleaned corpus 42.7M chars; every file checked
  against MANIFEST.sha256 at load (loader raises on any mismatch); the manifest hash is part of the run identity.
  Slot = 100 + 2·i + task position (multikey 0, single_essay 1) → slots 100–399, pairwise distinct, each a disjoint
  98,304-char source interval (16k ⇒ 409 slots; 0–99 reserved for development, never used here).
  Each row records hay = {corpus, slot, start, n_chars, books}.
- Unit of inference: the prompt. Population: prompts built by this generator from this fixed corpus/template/model.
  Disjoint source intervals remove shared haystack text between prompts; they do not make the books a random
  sample of documents, and different books may still share passages — no claim beyond this corpus is made.
- Known consequence (Codex F3): phase 1 used the v1 haystack (3 books, random windows). Moving to v2 changes the data
  distribution, so the phase-1 discordance is a borrowed planning scenario, not an estimate on v2. We keep N fixed
  (below) and state this limitation instead of recalibrating; no v2 development run is used to change N.

## Endpoint (unchanged)
Primary: strict match (target 7-digit value is a maximal digit run in the first 14 generated tokens). Secondary:
containment.

## Sample size (fixed, not data-dependent after this freeze)
Total N = 300 (150 per task). Planning model (homogeneous across tasks; an approximation since per-task pilot
discordance was 0.175 / 0.125): M ~ Bin(N, d), G | M = m ~ Bin(m, π), π = (d + Δ)/(2d); exact two-sided McNemar.
Computed with src/phase1_analysis.py power() (cache keyed by (m, α), a6065b4):
| d | Δ = 0.08 | Δ = 0.05 |
|---|---|---|
| 0.150 (pilot) | 0.948 | 0.564 |
| 0.225 (1.5× pilot) | 0.814 | 0.404 |
These are planning values under the model, not guarantees given pilot uncertainty or the v1→v2 shift; N = 300 is
not powered for Δ = 0.05.

## Analysis (single look, fixed N, no interim)
- Primary: exact two-sided McNemar on A1 vs A0 over the 300 prompts, α = 0.05; zero discordant pairs ⇒ no difference.
  Δ estimated with Newcombe's (1998) method 10 hybrid-score paired interval, 95% (z = 1.959964), no continuity
  correction on the Wilson margins, φ = N/√((a+b)(c+d)(a+c)(b+d)) with N = max(ad − bc − n/2, 0) if ad > bc else
  ad − bc (φ = 0 if the product is 0; the uncorrected φ would be method 8), clipped to [−1, 1]; implementation
  src/newcombe.py, checked by theory/check_newcombe.py against Newcombe (1998) Table III method 10 —
  (27,0,0,27) → [−0.0351, 0.0351], (20,12,2,16) → [0.0562, 0.3292] — plus boundary cases, antisymmetry and simulated
  coverage at n = 300. Per-task results reported (descriptive).
- Pre-registered sensitivity (does not change the decision): two-sided mid-p McNemar,
  p_mid = clip(2·P[Bin(m, ½) ≤ min(g, m−g)] − P[Bin(m, ½) = min(g, m−g)], 0, 1), computed from the unclipped exact
  value (Fagerland et al. 2013; no general level guarantee is claimed).
- Secondary family {nf_pt1, quest}: exact McNemar, Holm at α = 0.05. No equivalence / no-interaction claim.
- Straddle strata (grid 0 / grid 1 from the recorded spans): descriptive only.
- Missing/invalid (Codex F3; no drop, no replacement): all 300 pre-registered keys must end with a valid row.
  A non-method failure (process crash, OOM, killed job, host interruption — anything raised before a row is written)
  is rerun for that key with the identical frozen code/config/argv, resuming the same output file under the resume
  guard, at most 3 attempts per key; every attempt's ledger receipt is kept. A per-call budget-invariant violation
  (--strict-budget raise) is a method/implementation failure: it is NOT rerun; the run stops and is reported as an
  implementation defect. If any key lacks a valid row after the allowed attempts, the confirmation is reported as
  INCOMPLETE with no confirmatory claim; there are no substitute seeds and the denominator never changes.
  Wrong answers, empty outputs and dense failures are outcomes, never exclusions.

## Integrity gate (phase 2; src/phase2_gate.py SPEC RUN [--partial]; not the phase-1 80-key gate)
The frozen spec (src/make_phase2_spec.py, written once at freeze, committed) holds the exact launch argv, the code /
corpus / model-file identities computed as run_e2e.py does, the fixed configuration, the 300 keys with slots, and the
corpus book index. Before any analysis the gate exits 2 unless: manifest argv/model/identities equal the SPEC (not
only each other), no dev flags, no legacy rows, no code changes; exactly the 300 keys, no duplicates; every row's
run_id equals the SPEC (a dev flag on a row fails; nothing is stripped); prompt_sha, token_sha, L, answers, spans
present; block/local/rank/w_per_head/prefill_chunk/force/base_shift equal the SPEC; hay corpus v2, slot as frozen,
start = slot·98,304, 0 < n_chars ≤ 98,304, inside the corpus, books equal the index entries overlapping the interval,
slots pairwise distinct; methods exactly {dense + six arms}@512; strict re-computed from out with tasks.score_strict;
every sparse arm: integer calls > 0, all checked, zero violations. --partial only validates present rows and prints
INCOMPLETE; the analysis requires the full gate to pass. Synthetic negative tests: theory/check_phase2_gate.py
(Codex F2 ①–⑤ cases included).

## Freeze checklist (to be completed before launch; recorded in claude-response.txt with the commit)
1. Code commit; SPEC written by make_phase2_spec.py from that commit and the exact launch argv, committed before
   launch; SPEC hash, commit and argv posted together. The argv is parsed with a strict allow-list: experiment flags
   must equal this document (--ctx 16384 --budgets 512 --tasks multikey,single_essay --n 150 --seed0 200000
   --seed-stream --haystack v2 --slot0 100 --rank 8 --prefill-chunk 2048 --no-stats --strict-budget --methods
   cobs,cobs+g2f,nf_pt1,nf_pt1+g2f,quest,quest+g2f); only --vram-frac, --throttle, --out are free (recorded); any
   other, repeated or '='-form flag is refused (theory/check_phase2_spec.py). The SPEC also stores full SHA-256 of
   src/phase2_gate.py, src/phase2_analysis.py, src/newcombe.py, src/make_phase2_spec.py, src/tasks.py and this file;
   the gate and the analysis entry refuse to run if any differs. Weights are identified by size+mtime metadata only.
   Analysis entry: src/phase2_analysis.py SPEC RUN (runs the full gate first; refuses incomplete files).
2. theory/check_haystack_v2.py exit 0 on that commit (all 80 phase-1 v1 prompts reproduce; v2 provenance). The 300
   v2 prompts are checked as they are generated (slot length / identity asserted per prompt in tasks.py; the gate
   re-checks every row); a generation-time failure counts as a non-method failure only if it is a transient host
   error — a slot that cannot fit its haystack is an implementation defect (stop and report).
3. theory/check_phase2_gate.py, theory/check_phase2_spec.py and theory/check_newcombe.py exit 0 on that commit
   (direct exit codes recorded).
4. Codex review of this draft; any change before launch listed here with reason.

## Changes from v2
- Phase 1 complete; N fixed at 300 with the power table above (was "computed from phase 1").
- Haystack v1 → v2 with disjoint slots and provenance; population statement narrowed; limitation stated.
- Mid-p as pre-registered sensitivity; Holm family unchanged; Quest stays secondary.
- New phase-2 integrity gate bound to an independent frozen spec (the phase-1 gate is not reused).
- Missing-data rule: no drops, no substitutes, ≤ 3 same-config attempts for non-method failures, else INCOMPLETE.
- Paired interval frozen: Newcombe method 10, 95%, implementation and checks named.
