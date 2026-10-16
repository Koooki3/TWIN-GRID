# Mechanism design v1 (DRAFT for Codex review; not run) — why do moment-type scorers lose straddled digits?

## Question
On prompts whose 7-digit answer crosses a 32-token block boundary, at the step where a wrong digit is first
produced (or would be produced), is the block holding the gold digit (a) not selected, and if so, (b) is it
ranked low by the scorer while its true log-partition is high (estimator blind spot), or (c) is the true
log-partition itself low at that step (the model attends elsewhere, e.g. to the key-phrase block)?
Accounts to separate: "tail spike invisible to moments" predicts (a)+(b) for the tail block; "digits ride with
the key-phrase block" predicts the head/key block selected and the tail block's true mass low or replaced by a
distractor; neither predicts errors before the boundary digit.

## Prompts (diagnostic, not confirmatory)
The 34 straddled prompts of force_probe_16k (label-selected, exploration seeds) plus 34 non-straddled prompts
matched by task and seed range (first 17 per task from seeds 1000–1019 whose value lies inside one block).
Same prompt hashes recorded; no new seeds selected on outcomes.

## Arms (each with identical prompt, prefill, fixed sink/tail, budget 512, chunk 2048, rank 8 shared)
cobs, nf_pt1, quest, cobs+ftail (forced tail; swap counts logged), dense.

## Two modes per prompt × arm
1. Teacher forcing (TF): feed the gold continuation token by token (answer prefix as in the prompt template, then
   the 7 digits); at each digit position j record: gold-token log-prob, margin to the best non-gold token, the
   argmax token, per (layer ≥ 2, head) whether the head block / tail block / key-phrase block are selected, the
   tail block's scorer rank and true log-partition rank among candidate blocks, and the actual reads
   (budget invariant on).
2. Free running (FR): greedy generation as in all previous runs; record the first wrong digit index.
Primary descriptive outputs: for FR failures, the TF quantities at the first wrong index vs the same index in
dense; the fraction of heads (weighted by dense attention mass on the gold digit) that select the gold digit's
block. No p-values; counts and per-prompt tables.

## Decision rules written before running
- If, at first-error steps, the gold digit's block is unselected in most mass-weighted heads while its true
  log-partition rank is within the oracle top-k: supports the estimator-blind-spot account.
- If the gold digit's block has low true log-partition rank at those steps (outside oracle top-k) in most cases:
  the failure is not an estimation error; supports "rides with key-phrase block" / dynamics.
- Mixed outcomes are reported as mixed; no account is declared.

## Cost / resources
~68 prompts × 5 arms × (TF 8 steps + FR ≤ 14 steps), 16k, low-footprint mode; estimate ~1–1.5 h of GPU in
the current low-footprint setting. Runs only after F3 and phase 1, unless you advise otherwise.
