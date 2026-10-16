# Pre-registration DRAFT v2 — two offset grids with exact budget (g2_fill_v1) for moment-type block scorers

Status: draft for Codex review; NOTHING here has been run. Supersedes the 12:3x draft in claude-response.txt.
Code: commit 121fc2e (g2_fill_v1 = `+g2f`; fixed sink/tail; per-step token + swap logging; resume guard).

## Question
Does selecting blocks from two grids offset by B/2 (exact candidate-token budget) improve answer accuracy of the
COBS-style projected-variance scorer, in the setting where it was observed (frozen Qwen2.5-1.5B-Instruct, local
RULER-style single-value NIAH, 16k, budget 512)? Scope of inference: these tasks, this corpus, this model config.
Not claimed: superiority over Quest, any speed/memory benefit, generality to other models/tasks/lengths.

## Arms (all on identical prompts; per-step fixed set F(n) = [0,B) ∪ [fixed_end(n), n) identical across arms;
## candidate tokens C(n) = B·_kfix(n) for every arm; actual counts logged and checked for equality)
A0  cobs           (baseline)
A1  cobs+g2f       (primary contrast A1 vs A0)
S1  nf_pt1 vs nf_pt1+g2f, S2 quest vs quest+g2f   (secondary family)
Reference only: dense.
Fixed: block 32, local 64, rank (subspace dim) 8 shared per KV head, prefill chunk 2048, bf16, greedy decode,
max 14 new tokens; budget 512 nominal (actual R(n) = |F(n)| + C(n) logged).

## Data
Tasks: multikey and single_essay, equal weight (same number of prompts each).
Seed streams independent per task and phase: seed = 100000·phase + 10000·task_id + i
(phase 1 = development, phase 2 = confirmation; task_id 1 = multikey, 2 = single_essay). Seeds 1000–1019 and all
previous runs are exploration and are excluded. Every prompt's token-ID hash, target value span and its straddle
status under grid 0 / grid 1 are recorded (computed in the run, no extra pass).

## Endpoint
Primary: strict answer match per prompt (binary): the target 7-digit value appears in the output as a maximal
digit run (digit boundaries), evaluated on the first 14 generated tokens. Secondary: substring containment (the
metric used so far).

## Phase 1 — development (fixed size, not used for confirmation)
40 prompts per task (80). Purpose: verify token-count equality in every step, estimate discordance
p10 = P(A1 right, A0 wrong), p01 = P(A0 right, A1 wrong), and the natural straddle rate. No significance claims.

## Sample size for phase 2 (computed from phase 1, posted for review before phase 2 starts)
Minimal meaningful improvement Δ = p10 − p01 = 0.08 (absolute). Power 0.8, two-sided α = 0.05, exact McNemar.
N from the conditional binomial on discordant pairs with discordance rate from phase 1 (and a sensitivity table
for discordance ±50% and Δ = 0.05). Cap N ≤ 300 per task for compute; if the cap binds, report the achieved power.

## Analysis (phase 2, single look, fixed N, no interim analysis)
Primary: exact McNemar (two-sided, α = 0.05) on A1 vs A0 over prompts; estimate Δ with Newcombe's hybrid-score
interval for paired proportions (method 10, 1998). Task-stratified results reported; since seeds are independent
per task, prompts are the unit (no cross-task seed clusters).
Secondary family {S1, S2}: exact McNemar, Holm at α = 0.05. Straddle-stratified results are descriptive only
(an interaction claim would need its own pre-specified test; none is made).
Missing/invalid: OOM or crash → rerun the same prompt in a new attempt with identical code/config; a prompt is
dropped only if all arms fail it for non-method reasons; never dropped by outcome. All attempts kept.

## Integrity
Run manifest with argv/methods/code hashes; resume refused on any mismatch (no --allow-code-change in phase 2);
outputs per phase in new files; receipts (job ids) posted to claude-response.txt; Codex reviews before phase 2.
