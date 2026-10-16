# Provenance of this public snapshot

This repository is a snapshot of the local development branch at commit `e644c8749c91cdf42ba521c80a3eca5e0343aa14`, with `runs/ruler/` omitted
(RULER-prepared inputs and prediction files containing third-party text; regenerate with `scripts/ruler_prepare.sh`).
Commit ids cited in the documents refer to the local history below; the full history is archived locally as a git
bundle. Frozen files are verified by sha256 (`results/freeze_v14_sha256.txt`), independent of commit ids.

| commit | date (+0800) | subject |
|---|---|---|
| e644c87 | 2026-10-17 03:19:53 | PROGRESS: overnight results and resume point |
| 2571fc3 | 2026-10-17 03:19:39 | E4 interpretation in README and appendix 8.5 (unit-fraction effect, not selectable positions); independent stdlib recompute of headline dev numbers (all match) |
| ac44080 | 2026-10-17 03:19:04 | Box-summary P-grid (E4, exploratory): quest32+g2f vs quest16 close (-0.024 [-0.059,+0.009]; 4-target -0.013 [-0.062,+0.037]) -> by the pre-committed rule the moment-summary P-grid gap is the unit-fraction effect, not selectable positions; quest16 best overall (0.888) |
| c497321 | 2026-10-17 02:33:28 | README and appendix 8.4: variants results (Quest far ahead at equal memory; G ordering holds; neighbour-read variants worse) |
| 2d1e27f | 2026-10-17 02:32:39 | Variants development results (exploratory): quest32 0.850 >> mean+g2f 0.591 at equal memory (-0.259 [-0.312,-0.212]); G ordering mean > cobs > quest holds (G_mean - G_cobs +0.142 [0.045,0.244]); nb/nbr/nbh all below cobs and cobs+g2f (nbh single-unit prediction not met) |
| 1ad77d4 | 2026-10-17 01:05:13 | Reviewer batch 10 re-check: wording (three effects: only cutting measured; consistent with L3, not predicted by it; C1 baseline is same-block single grid, 2x vs 1x memory) |
| bbe693b | 2026-10-17 01:04:28 | Reviewer batch 10: README narrative changed by the P-grid result (fine grid beats TWIN-GRID at equal memory; contributions narrowed: block-length three-effect decomposition, L3 supported / L4 premise fails, gain law restricted to fixed B, implementation downgraded); appendix A4 (results, three effects, delim vt-coverage prediction corrected); scenarios must-report |
| 812a117 | 2026-10-17 01:03:42 | Box-summary P-grid (reviewer batch 10 E4): analysis and queue committed before the run (quest16 vs quest32+g2f, equal memory; separates selectable-position vs unit-fraction effects) |
| b89a957 | 2026-10-17 01:00:23 | P-grid and delim development results (exploratory): P-grid prediction twin@32 >= fine@16 NOT met (fine@16 better at equal memory, nf_pt1 -0.131 [-0.181,-0.078] on split units); delim < cobs+g2f (-0.068 [-0.109,-0.026]) with 100% coverage; merge checks passed; output sha256 |
| 6b28dd7 | 2026-10-17 00:31:22 | public_snapshot: parent = current public main when it exists (fast-forward updates, never force); ls-remote check (main currently empty) |
| 104e3ea | 2026-10-17 00:30:04 | Pre-publication: owner's push authorisation recorded verbatim; README/guide release wording; snapshot script with PROVENANCE.md, noreply author, tree/frozen checks, bundle archive, post-push clone checks; check_all falls back to stored confirmation-time sparse_attn (sha c9b56b7f); ruler_prepare.sh portable defaults |
| 351e796 | 2026-10-17 00:26:54 | Public snapshot script (omits runs/ruler/ third-party text; temp index, local history untouched) and README note |
| 971df4f | 2026-10-17 00:26:21 | Full speed overnight (user away): nf rerun 3 and variants use --vram-frac 0.75 --throttle 0 (resource-only flags; merge manifest check ignores them); delim run keeps its original flags |
| 672687b | 2026-10-17 00:05:50 | Reviewer batch 8 re-check: block-length corollary as an upper bound (gain excl. in-block term <= pi <= (l-1)/B for any summary; no constant-ratio claim); kappa < 1 and > 1 narratives in S1' and S2b |
| b6c9ca2 | 2026-10-17 00:04:41 | Reviewer batch 8: README leads with NSA precedent and four contributions; result scenarios exhaustive (S1', S2c, S3a-c, condition table; delta-hat as 13-task macro); addendum corollary gain ~ 1/B |
| 7c163fd | 2026-10-17 00:02:11 | Result scenarios S1-S4 with bound readings (before any cluster result); verified MoBA/InfLLM/NSA citations (NSA compression = offset half-block grids: novelty narrowed); README: absolute accuracies, memory, positioning, generalisation not established |
| 4c0c2f9 | 2026-10-17 00:00:22 | Variants (reviewer batch 7, user follow-up): cobs+nbh (L1' memory-read dual, runtime wrap, CPU check), runner, analysis committed before the run (E1 G ordering, E2 nbh single vs multi-unit, E3 mean+g2f vs quest equal memory), addendum A2/A3, memory table with minimal vs implementation columns |
| e58aa25 | 2026-10-16 23:57:26 | Analytical summary memory per arm (minimal sufficient statistics): cobs+g2f = cobs16 = 1.28x quest32 = 0.64x quest16; mean+g2f = quest32 |
| c20823a | 2026-10-16 23:50:39 | pgrid_mv_merge: nf_pt1 arms from one-method rerun-3 dirs; archive the OOM 2-method dir; checks unchanged |
| f32b0d3 | 2026-10-16 23:50:17 | delim_dev: fix tokenizer wrapper recursion (all 7 launches failed at load, before any prediction; their delim manifests archived in delim_pred/_failed_load); P-grid rerun 3 script (nf_pt1+g2f and nf_pt1 one method per process) |
| 7b8a435 | 2026-10-16 23:49:14 | Guide: measured OOM example (1.5B 16k nf_pt1+g2f multivalue >= 4.65 GiB vs probe 3.9 GiB); only semantics-preserving OOM workarounds |
| f047009 | 2026-10-16 23:45:00 | Reviewer batch 6: token_sha assertions in delim analysis (Z3, before any delim prediction exists); addendum notes on unequal summary memory (Z1) and generated-token chunking (Z2) |
| 3b93ba5 | 2026-10-16 23:42:45 | Delimiter chunking (reviewer batch 5): theory addendum, delim_attn (runtime scorer wrap; log-mass ranking; prefix fill to the strict budget; fill/summary stats), delim_dev runner (ruler_pred main unchanged), analysis script and CPU check committed before the run; README/guide: RBS-decode shelved, LongBench after primary cell |
| 108cacc | 2026-10-16 23:38:49 | pgrid_mv_merge.sh: reviewer batch 5 Y1 checks (row/uid completeness, manifest diff, OOM-partial pred equality) and output sha256 |
| 7cc6fa1 | 2026-10-16 23:23:49 | P-grid OOM handling: rerun scripts (expandable_segments; B32 multivalue split into two processes), merge script, guide note that the memory probe underestimates real peaks |
| b58cb9e | 2026-10-16 20:39:30 | PROGRESS: precise P-grid timing (run started 18 s before the analysis script; fixed before inspecting predictions) |
| 2f61f07 | 2026-10-16 20:38:25 | P-grid dev run: queue and analysis script committed before results; expected direction recorded (twin@32 >= fine@16 on 16-grid-split units, similar otherwise; exploratory) |
| 44122b4 | 2026-10-16 20:37:06 | Reviewer batch 4: code-version note for C1-C3, scoped 'bit-identical' wording (unfrozen docs), check_all old/new guard vs e194213, memory probe, F3/ptr shelved |
| 76ff58c | 2026-10-16 20:33:48 | PROGRESS: v1.4 frozen |
| d65fca7 | 2026-10-16 20:33:24 | Record v1.4 freeze file sha256 (frozen at 8e55275) |
| 8e55275 | 2026-10-16 20:33:16 | Freeze plan v1.4 (reviewer batch 3): bf16 tie rule (accept_tie_diag/accept_ties, conditions a-d); G3 layout matching (--layout-unit 32, check_g3_layout; default path bit-identical); cobs16/nf_pt1@16 ready; P-asym reworded (not a test of R); E4/E5/E7/L4/kappa wording; minimum detectable g; check_all.sh |
| 69cbf51 | 2026-10-16 20:25:01 | Plan v1.4 pre-freeze (reviewer batches 1-2): primary cobs+g2f vs cobs; two-member Holm (nf_pt1, coverable subset); unit_lengths v3 freeze; aggregate v4 (--units/--subset); gain_law (P-gain, P-asym); unified docs/THEORY_ZH.md; power; T4/T5 post hoc diagnostics; acceptance rerun with bf16-tie diagnostic; deps requests/scipy |
| fd705a5 | 2026-10-16 19:53:20 | accept_adapter v3 (Q2-type hardening): re-derived uids, required fields, per-run protocol (local 64/16), token_sha agreement across runs, strict counters, reference model identity bound, coverage per run re-derived from L/n_gen/local/block |
| ffe562f | 2026-10-16 19:49:51 | Watch the TWIN-GRID third-party review directory (Codex paused; reviewer handover) |
| 580aa95 | 2026-10-16 19:45:06 | Codex Q2/Q3: aggregator v3 (pinned RULER source hashes, re-derived uid lists, required code/manifest fields and protocol, strict integer counters, token_sha agreement, evidence identity in JSON); holm v2 (unique members, same plan/cell/data/identity, valid p); tests incl. all named negatives |
| 813d0fd | 2026-10-16 19:42:12 | Theory v8 §8: mechanism diagnostic results with the wording agreed in Codex M1-M3 |
| 010a6c3 | 2026-10-16 19:38:14 | Mechanism evidence matrix on the 63-prompt run (frozen mech_matrix v4, EXPECTED+manifest gate passed, rc 0); acceptance script fix (exec-able command array) |
| dc2669b | 2026-10-16 19:35:50 | Mechanism run sealed: job 1f56da28 succeeded, 63/63, acceptance ACCEPTED; post-run manifest |
| 8948198 | 2026-10-16 19:21:28 | README: TWIN-GRID overview, confirmed results C1-C3 with boundaries, layout, cluster entry points, open items |
| 9b2470b | 2026-10-16 19:17:31 | Codex batch-1 A1/A4/A5/D1: aggregator v2 gate (frozen data, manifests, RULER commit, answers/inputs), unrounded scores vs official rounding, roles PRIMARY/SECONDARY/DESCRIPTIVE, holm.py; plan v1.3 (primary-cell n, same-block secondary family, run tags); doc status/C3 wording |
| 0e32396 | 2026-10-16 19:14:30 | Codex batch-1 A2/A3: strict acceptance (frozen uids/inputs/manifests, same-family order pairs + single arms, coverage evidence via gen_in_candidates, --local 16 for order runs); sharded-checkpoint model identity + revision |
| dece137 | 2026-10-16 19:04:23 | Adapter GPU acceptance queued after mech run; RULER LFS word list verified and pinned in prep manifest; guide LFS step |
| 3991e28 | 2026-10-16 19:01:51 | TWIN-GRID cluster guide v0 (verified vs pending table, env, models, pinned RULER + patch, tier choice, run, aggregate, records) and requirements |
| 6b16f9d | 2026-10-16 19:00:55 | End-to-end check of the cluster aggregator (official RULER metrics, synthetic predictions; gate negatives) |
| c046fcb | 2026-10-16 19:00:28 | Name the project TWIN-GRID (with Codex); cluster package: machine-readable plan, job generator, official-metric aggregator; parameterised data prep |
| e194213 | 2026-10-16 18:50:19 | Confirmation results: gate passed; primary cobs+g2f vs cobs +12.3 pts [8.0,16.9], p=5.7e-8 (43/6); P-kv P1-P3 met on fresh seeds |
| 820f53c | 2026-10-16 17:04:34 | P-kv entry (Codex K1-a/b): receipt bound to this confirmation job/argv/SPEC fingerprint, acceptance record, single encoding for ids+span, empty-group -> insufficient information, banner; §7 wording |
| d50f118 | 2026-10-16 16:58:16 | RULER prep manifest (Codex R4 rest): refuse unmanifested data, refuse sealed-output drift, tokenizer_config/special_tokens identity |
| 448400e | 2026-10-16 16:56:50 | P-kv confirmation entry (Codex K1/K3): full frozen gate + job receipt + token identity before the unchanged registered script; task x crossing tables; v8 §7 wording per K4 |
| 938d896 | 2026-10-16 16:48:36 | PROGRESS: P-kv registration, mech matrix v4 + EXPECTED, RULER adapter status |
| 4af0db6 | 2026-10-16 16:48:09 | RULER adapter (Codex R1-R4): per-arm summary reset, sample_uid by file hash+task+line, stop set from generation_config, actual prompt as input, full manifest and refusal to adopt orphan predictions; prep manifest pins configuration |
| 1710098 | 2026-10-16 16:46:49 | mech_matrix v4 (Codex N1-final-a/b): manifest JSONL/script hashes enforced; independent EXPECTED identity written before the run |
| 4122939 | 2026-10-16 16:42:44 | Register prediction P-kv (key->value span crossing explains the COBS g2f gain on dev data) before confirmation results; v8 §7 |
| 787000b | 2026-10-16 16:38:25 | mech_matrix v3 (Codex N1-N3): manifest-bound run identity, state_origin semantics, weight validity, gold-prefix naming, first-POST count, paired link; v8 §4/§6 wording; RULER adapter + prepare script + Qwen template diff |
| acc23aa | 2026-10-16 16:30:24 | v8 §6: interior-region condition, L3 feasible-set inclusion (two-grid advantage is not representational), P-grid matching conditions; plan v1.1 (G3/P5) |
| 162c5aa | 2026-10-16 16:30:23 | mech_matrix v2 (Codex M1-M4): gate binds frozen list + identities + existing acceptance + shapes; 3x2 exclusive table; E1/E2 estimands; observed diff + bootstrap with NA counts; own-TF table; outcome link |
| e558b73 | 2026-10-16 16:10:50 | PROGRESS: frozen-file list during confirmation, v8 status, P-grid deferral |
| 8b8f831 | 2026-10-16 16:10:37 | Theory v8 §6: k-grid covering lemma (sharp), finer-grid comparison at equal summary memory, prediction P-grid |
| 1122150 | 2026-10-16 16:09:35 | Experiment plan v1 (Codex P1-P4): official task score claim, primary contrast + family, shared primary cell across tiers, arm table, same-input adaptation check |
| ff17c27 | 2026-10-16 16:08:42 | Theory v8 per Codex V1-V5: exact split bounds, counter-examples (dK>dA not general; oracle rank flip), O1 corrected (Quest 9/11 vs 25/29); descriptive evidence matrix on shared dense state replaces mechanism verdict |
| be4640f | 2026-10-16 15:59:18 | ptr dev failed at 32/80 (CUDA unknown error, job 3e27e7ee exit 1); queue same-config resume last with exit code propagated |
| 674303f | 2026-10-16 15:58:46 | Theory v8 draft: fraction-sensitivity (proved for two-level model, contradicted by flip data for Quest) vs read-requirement mechanism; separating analysis with pre-fixed thresholds |
| 100270d | 2026-10-16 15:56:14 | Record user's division of roles (cluster agent picks tier; user creates repo; naming/conclusions by Claude+Codex) |
| d499c24 | 2026-10-16 15:55:25 | Q1: phase2 queue propagates the work-handoff exit code |
| 8346253 | 2026-10-16 15:53:43 | Multi-tier experiment plan draft v0 (official benchmarks x three hardware tiers, claims-evidence matrix) |
| b6de7da | 2026-10-16 15:51:10 | Queue: ptr dev -> phase-2 confirmation (frozen SPEC a217f16, SPEC re-check before launch) -> mech full -> F3 |
| a217f16 | 2026-10-16 15:50:41 | R1: containment secondary endpoint reported descriptively (re-computed with tasks.score); SPEC regenerated |
| 3d22ecc | 2026-10-16 15:43:55 | EM dev 80/80 (job 9eebd362 succeeded): em 68 vs quest 67, em+g2f 74 vs quest+g2f 73 — descriptive |
| 516233d | 2026-10-16 15:43:33 | PROGRESS: official RULER integration plan |
| 7e7ad8c | 2026-10-16 15:42:57 | PROGRESS: benchmark selection principle for the multi-tier plan (official RULER/LongBench/InfiniteBench headline tier) |
| d707f06 | 2026-10-16 15:40:31 | Phase-2 SPEC candidate (low-footprint resource flags) + direct-exit freeze check logs |
| 12c80d6 | 2026-10-16 15:39:26 | Freeze closure C2/C3: strict allow-list argv for SPEC, frozen analysis-file hashes verified by gate/analysis, analysis entry phase2_analysis.py, tests |
| 4a5d7ed | 2026-10-16 15:34:49 | Newcombe: implement method 10 phi correction (was method 8); verified against Table III published values |
| 8665489 | 2026-10-16 15:31:33 | +ptr must be the only wrapper (both nesting orders refused in runner and scorer); exact PTR_WIN wording |
| 34c2ae3 | 2026-10-16 15:31:01 | Prereg v3: gate bound to independent frozen spec (make_phase2_spec), model identity in run id, no-drop missing rule, Newcombe method 10 frozen with checks |
| d7d2ab1 | 2026-10-16 15:28:17 | Theory v7: Codex T1-T3 (finite-n exact upper end, 1/sqrt(d), log n, three separate lemma statements, candidate region, evidence-bounded M3-M5, REM reference scale) |
| ea8ee86 | 2026-10-16 15:27:20 | +ptr: latest large/backward jump ends copy mode; per-method ptr_stats and em_viol; reject nesting; nc/empty guards |
| 3e36aa4 | 2026-10-16 15:21:57 | PROGRESS: round 6 dispositions and next steps |
| ff9c795 | 2026-10-16 15:21:20 | Prereg v3 draft (N=300, v2 haystack slots 100-399, mid-p sensitivity); v2 loader verifies manifest + records provenance; full 80-prompt v1 reproduction check; phase-2 gate with synthetic tests |
| a6065b4 | 2026-10-16 15:19:42 | Codex ptr feedback: reset S.ptr per prompt/method/mode, same mask under diagnostics; power cache keyed by alpha |
| 7441840 | 2026-10-16 15:18:20 | Survey report: apply Codex D/C1/C2 wording corrections (hypothesis status, no causal/novelty overclaims, EM scope, memory, REM) |
| 0ac9fb1 | 2026-10-16 15:17:04 | Codex review: mech identity preflight (23 null token hashes recorded as first_recorded, never filled), explicit EM numerics with violation counter, dev comparison helper |
| 2aa3140 | 2026-10-16 15:14:10 | Add +ptr online read-pointer successor wrapper (KVFetch-style baseline) with CPU check; serial queue em -> ptr -> mech full -> F3 |
| a706516 | 2026-10-16 15:12:19 | Theory v7: one interval frame (mean+var classical, mean+range EM, anomalies+bulk), two-grid covering lemma (sharp l<=B/2+1), demotions |
| a85b7d6 | 2026-10-16 15:10:48 | Mech check ACCEPTED (4/4, 0 failures); propagate acceptance exit code; queue full 63-prompt diagnostic last |
| d0561c6 | 2026-10-16 15:09:50 | Mech-check acceptance: require exact prompt keys, four arms, non-empty steps/counters/logger check |
| 71aef26 | 2026-10-16 15:08:00 | Add v2 haystack: disjoint per-prompt windows from a 47-book public-domain corpus (opt-in, v1 unchanged) |
| fa44566 | 2026-10-16 15:03:01 | Fix mech-check waiter deadlock: serial order mech -> em dev -> F3 |
| 5409d81 | 2026-10-16 15:01:49 | Add Edmundson-Madansky block scorer, phase-1 dev analysis, literature survey; fix queue deadlock |
| 3a6c1b1 | 2026-10-16 14:35:15 | Codex M1-M4: richer per-digit records (exact mean/LSE, scorer values, full-moment ranks, greater/tie counts, probe blocks), per-swap log, frozen-prefix assert; runner identity/digit checks against frozen identities, FR logprobs, throttle, logger on/off check; acceptance script; queue order phase1 -> F3 rest -> mech check |
| 98b97fa | 2026-10-16 14:32:55 | P1: phase-1 analysis hard integrity gate (exit 2), --partial progress-only mode, exact even-N minimum with cached rejection regions |
| e3d0ca6 | 2026-10-16 14:11:51 | Phase-1 analysis script: integrity checks, strict accuracy, paired tables, exact-McNemar planning power (reviewer formula) |
| 319f53c | 2026-10-16 13:42:27 | Remaining F3 configs queued after mechanism check; ch0-r16-perhead deferred (OOM under low-footprint cap) |
| 78add6b | 2026-10-16 13:32:30 | Queue mechanism implementation check (after phase 1) |
| 371cd5a | 2026-10-16 13:32:00 | Mechanism design v2 (Codex D1-D5): per-digit diag hook (shared/own state), run_digit_diag runner, gold-alignment check, prompt lists |
| 3bf0b39 | 2026-10-16 13:22:12 | Mechanism design v1 draft (teacher forcing vs free running on straddled prompts) |
| fcc1442 | 2026-10-16 13:21:43 | Shift analysis hardened (ctx in key, refuse missing/duplicate, unknown logs); flip-vs-placebo table; validate-only fully read-only, GPU config after validation |
| f3f1fa3 | 2026-10-16 13:14:51 | Cut-position table for natural and intervened crossings (descriptive) |
| 0a84451 | 2026-10-16 13:14:04 | Placebo grid-shift arm: no change when the value is not cut; flip-arm losses attributable to cutting the value |
| 651518a | 2026-10-16 13:05:35 | Grid-shift intervention results (flip arm): moment scorers rescued when the value is moved inside a block and hurt when it is cut; Quest also hurt when cut (placebo pending) |
| ea271cf | 2026-10-16 13:02:50 | Budget invariant also checked under base_shift with budget_checked_calls; --validate-only entry with positive sentinel; guard check uses it |
| 3675206 | 2026-10-16 12:57:38 | V2 closure: per-row code/model/required-field identity on resume; V3: swap-count assertion and re-raise of unexpected errors in check_invariant; logs |
| d296ff1 | 2026-10-16 12:54:52 | Phase-1 development queue (after round-5d diagnostics) |
| b76f376 | 2026-10-16 12:54:27 | Codex V1-V3: per-call budget/fixed-region invariant with strict mode, g2f refuses odd B/shift, per-line resume validation before model load, normalised dev flags; checks with logs |
| c0c7006 | 2026-10-16 12:50:14 | Per-call candidate-budget check (budget_mismatch count per method); manifest check before model load, legacy adoption only with --adopt-legacy; check scripts exit nonzero on failure |
| 4fd063a | 2026-10-16 12:45:31 | Phase-1 prerequisites: strict digit-boundary scorer, per-record value spans + straddle labels (grid 0/1), token-ID hash, per-task seed streams |
| 4191aaa | 2026-10-16 12:44:26 | Pre-registration draft v2 for g2_fill_v1 (Codex N2/N3); v6 M3 wording fix |
| 121fc2e | 2026-10-16 12:43:53 | Codex N1/N4: g2_fill_v1 (+g2f, exact candidate-token budget), per-step actual token and swap logging, resume guard with run manifest; checks |
| 6f1cdae | 2026-10-16 12:40:59 | Dedupe block-summary cache per KV group (6x less memory; identical selections, check_dedupe.py); allocator env set in-process |
| 2333079 | 2026-10-16 12:39:30 | Low-footprint GPU mode while the user works (--no-stats, VRAM cap 0.6, throttle 0.3); GPU watch script |
| 7cb17f9 | 2026-10-16 12:36:52 | User back at the machine: remaining round-5 steps moved under the idle gate (round5d); stray causal-L2-7 receipt note |
| ac253b3 | 2026-10-16 12:32:14 | nb_16k results: two offset grids rescue moment scorers on straddled values (exploration) |
| a8e672b | 2026-10-16 11:38:10 | Theory v6: consolidated after review 3 and Codex review (T1 proof credited, measured vs hypotheses vs withdrawn) |
| a336a72 | 2026-10-16 11:30:38 | Placebo shift queue; Codex watcher script |
| 6ec8661 | 2026-10-16 11:30:12 | Per-record run identity: code hashes, argv, model, prompt hash (Codex F6) |
| c3a28e7 | 2026-10-16 11:29:57 | Fixed sink/tail across grid offsets (Codex G1); placebo shift mode (G3); equivalence check |
| 148c495 | 2026-10-16 11:25:03 | Reject wrapper+union scorer combinations at argument parsing (Codex F4) |
| 9470b01 | 2026-10-16 11:24:35 | Grid-shift intervention, per-head subspace switch, budget uses actual sink end; round-5c orchestrator (after Codex F1-F6) |
| ab76259 | 2026-10-16 11:19:20 | Progress: collaboration protocol with Codex reviewer |
| 8b42f41 | 2026-10-16 10:38:11 | Finer error taxonomy (prefix truncation, substitution, splice; first error vs head-block digits) |
| 5bda1c9 | 2026-10-16 10:37:34 | Forced-block causal probes (+ftail/+fhead/+fadj), right-neighbour expansion (+nbr), straddle-only runner, orchestrator |
| 633e545 | 2026-10-16 10:29:06 | Progress: round 5 (idle-gated GPU, straddle finding) |
| 76dcc1e | 2026-10-16 10:27:31 | Two offset grids (+g2); theory v5 (selection unit must contain the information unit); add +g2 to round-5 run |
| eec5786 | 2026-10-16 10:26:05 | Round-5 queue: neighbour-expansion remedy first, then layer probes |
| efae487 | 2026-10-16 10:25:48 | Block-boundary straddle finding; neighbour expansion (+nb); round-5 queue reordered |
| ef5a0ef | 2026-10-16 10:23:22 | Idle-gated execution for GPU work (stop on user activity); round-5 runner |
| 04cf029 | 2026-10-16 10:21:46 | Round 5: layer-restricted sparsity for causal probes |
| bdff882 | 2026-10-16 10:02:18 | Round 4 wrap-up: value-digit diagnostic, theory v4 E4/E5, progress and resume point |
| c44acb1 | 2026-10-16 09:54:16 | On-policy mechanism results; value-digit mask option |
| eeb77fc | 2026-10-16 09:38:39 | Per-step mechanism results along dense trajectory |
| 8cdd947 | 2026-10-16 09:31:09 | diag3 results; per-step mechanism diagnostic along dense trajectory |
| efd65a1 | 2026-10-16 09:22:37 | ell/qsub results: radius bound useless (0.308), sub-boxes +0.037 ns; granularity explains B16 |
| 17d4195 | 2026-10-16 09:06:27 | Progress: round 4 status |
| 1a6523c | 2026-10-16 09:06:12 | Mechanism diagnostic: retrieval-weighted oracle capture and needle-block hit per scorer |
| 5434acc | 2026-10-16 09:03:59 | Theory v4: what survives two adversarial reviews |
| 5700faf | 2026-10-16 09:03:19 | qsub2k: power iteration instead of batched SVD (GPU SVD stalled) |
| 0bfd555 | 2026-10-16 08:52:15 | diag2: outlier criteria recall, ell bound tightness |
| 8752abc | 2026-10-16 08:48:00 | Theory v3 draft (detection/optimism, max-leverage theorem); paired round-4 comparisons |
| 5cbf155 | 2026-10-16 08:46:46 | nf_16k and quest_b16 results; sub-box upper bounds (qsub2c, qsub2k) |
| 99ab374 | 2026-10-16 05:13:36 | Moments + extreme radii summary: certified ellipsoid upper bound (ell_hi, ell_dual) |
| 8083765 | 2026-10-16 05:11:48 | Diag 16k results; outlier-criterion recall diagnostics |
| 947672d | 2026-10-16 05:02:33 | Progress: night session, round-4 queue |
| 0c6bd6a | 2026-10-16 05:02:04 | Lean 32k results: summary dual +0.225 [+0.125,+0.333] over best single |
| ec5e235 | 2026-10-16 04:35:06 | Certified-selection diagnostic (blocks read to certify 1-eps mass) + CPU smoke test |
| c133845 | 2026-10-16 04:34:19 | Theory v2 after adversarial review: identifiability gap theorem, certified interval summary (nf_*), per-KV-head subspace, diagnostics |
| a9661c7 | 2026-10-16 04:23:51 | Closed-form log-partition F_n (max + truncated bulk) replaces branch formula; diagnostics script |
| 0d1fef8 | 2026-10-16 04:21:52 | Theory v1: block selection as log-partition estimation; two-phase and near/far scorers |
| 82a1f1b | 2026-10-16 02:25:50 | Record lean 32k plan (run when user is asleep); optional idle gate script |
| d9c668f | 2026-10-16 02:19:56 | Lean-run options: chunked prefill, VRAM cap, throttle; offline proxy check |
| da5ed50 | 2026-10-16 02:18:12 | Pause: 32k run stopped at user request (14/80 kept) |
| 8499696 | 2026-10-16 01:51:38 | Progress: ablation results, 32k run launched |
| d1308c8 | 2026-10-16 01:48:55 | 16k control and rho/adaptive ablation: paired comparison script and results |
| e872f12 | 2026-10-16 00:44:57 | Update progress: stage 1 16k results, ablation running |
| 5a67693 | 2026-10-16 00:44:28 | Stage 1 16k results (80 samples) and summary-family rho/adaptive/control scorers |
| 710b0d6 | 2026-10-15 20:45:37 | Add analysis script and rho / adaptive dual-channel variants |
| b3058e7 | 2026-10-15 20:40:28 | Stage 1 harness: block-sparse decode scorers, RULER-style NIAH tasks, runner |
