# dual-channel-attention — progress

Source of truth for prior work: Claude Docs doc "从组织类比到注意力机制：层级置信摘要的提出、对抗审查与修正"
https://claude.ai/code/artifact/958bb9a2-f8d2-4547-9a27-31a5f6de1619 (rev 12 read 2026-10-15).
The shared chat (claude.ai/share/5ccf45fc…) and public artifact (9c5ceec8…) could not be read: region-blocked
redirect (claude.com/app-unavailable-in-region). Prior numpy synthetic scripts are therefore not available; reimplemented.

## Status (2026-10-15)
- Taken over by Claude Code bg job a0424013. Stage 1 (training-free validation on a frozen model) is the next step.
- Hardware deviation: RTX 4060 Laptop 8 GB / 11 GB RAM -> Llama-3.1-8B @ 32k-128k infeasible. Using Qwen2.5-1.5B-Instruct
  (GQA, RoPE, 32k ctx) with RULER-style self-generated NIAH tasks; label results accordingly.

## Stage 1 @16k (done 2026-10-16, 80 samples, results/stage1_16k.{jsonl,md})
- Gate passes only for block-summary dual (quest half + cobs fill): 512 budget 0.853 vs best single quest 0.769,
  paired diff +0.084 [+0.022,+0.153]; 256: +0.069 [-0.003,+0.147].
- Sketch (rank-16 query PCA) dual: no gain (-0.006 / -0.013, CIs span 0). Oracles: mass/peak/dual within 0.003.
- Interpretation: gain looks like error complementarity of two imperfect estimators, not distinct targets.
- Written into the Claude Docs doc (section "第三轮", rev 15).

## Ablation @16k (done 2026-10-16, 60 samples, 3 unsaturated tasks, results/ablation_16k.{jsonl,md})
- Reproducible: 420/420 outputs identical to stage1 run.
- Control mix_cobs_mean beats cobs (+0.104 [+0.008,+0.200] @512) -> complementarity is real; but sum_dual beats the
  control by +0.083 [+0.017,+0.163] @512 -> peak channel adds beyond "another estimator".
- rho 0.25/0.75/adaptive all ~0.05 below rho=0.5 @512 (CIs touch 0). Recommendation: keep fixed k/2.
- Doc updated (rev 18); replied in comment thread 5c22629e (reply 674a797c-b901).

## PAUSED 2026-10-16 (user: machine too slow)
- 32k job c2fc3ad581c445e8bbbec66afd7c93e8 stopped with SIGTERM: receipt status interrupted, exit -15; no run_e2e
  process left, GPU memory back to ~1 GB. 14/80 samples (single_noise) kept in worktree runs/stage1_32k.jsonl.
- No automatic resume. Resume only on explicit user request; consider running when the machine is idle.
- Cause of the freeze: the RTX 4060 also drives the Windows desktop; the 32k run held 6.6/8 GB VRAM and kept the GPU
  saturated with long prefill kernels.
- Offline proxy is not a substitute (src/proxy_check.py): needle-mass retention vs accuracy corr 0.35-0.50 per sample;
  it cannot separate cobs (0.944) from sum_dual (0.950) although accuracy is 0.63 vs 0.82 @512.
- Added (untested on GPU): --prefill-chunk, --vram-frac, --throttle. Proposed lean 32k run: tasks
  multikey,single_essay,multivalue; budget 1024; methods quest,cobs,sum_dual (+dense); chunk 4096, vram-frac 0.45,
  throttle 0.5, nice 19. Same seeds, so the 14 done single_noise samples are simply not needed.
- User decision (2026-10-16): run the lean 32k UNTHROTTLED only after the user says they are going to sleep.
  Command (via work-handoff run, owner = the session doing it), from the worktree:
    .venv python src/run_e2e.py --ctx 32768 --n 20 --tasks multikey,single_essay,multivalue --budgets 1024
      --methods quest,cobs,sum_dual --prefill-chunk 4096 --out runs/stage1_32k_lean.jsonl
  (~2h). Optional, unused: scripts/idle_gate.sh stops/restarts a job by Windows desktop idle time.

## Night session 2026-10-16 (user asleep until ~noon; machine shared with FAST: my CPU = cores 0-5,
## GPU mine; heavy load wound down by 11:30)
- Lean 32k done (job 475c7770, succeeded): summary dual 0.863 vs cobs 0.637 / quest 0.629, +0.225
  [+0.125,+0.333]; dense 0.95. In doc rev 19. Code version for that run: commit 82a1f1b (per-query-head W, r=16).
- Theory loop: theory/theory_v1.md -> adversarial review (subagent) -> theory/theory_v2.md.
  v2: Theorem 1 identifiability gap of moment summaries (verified numerically), Theorem 2 certified interval
  summary (anomalies + bulk mean + bulk box), F_n closed form demoted to bulk heuristic, GQA memory fix.
- Code changes after the 32k run: W shared per KV head; nf_{lo,hi,pt,dual}{1,2} scorers; diagnostics.
- GPU queue (scripts/queue_round4.sh): diag_16k (job 4b0a6405) -> nf_16k -> quest_b16_16k.
- Round 4 results (all 16k, 60 paired samples, rank-8 shared W): results/nf_16k.md, quest_b16_16k.md,
  round4_paired.md, diag_16k.json, diag2_16k.json. Second review -> theory/theory_v4.md (what survives).
  Doc rev 20: section "第四轮" (pending block .12505 for qsub/ell results).
- 09:0x: ell_16k (job 37073711; ell_hi, ell_dual, qsub2c, qsub2k @512) running; then diag3 (mechanism) via
  scripts/queue_diag3.sh. First ell attempt (job adae9937) was stopped: batched GPU SVD stalled -> power iteration.

- 09:20–10:01: ell_16k (job 37073711, succeeded): ell_hi 0.308, ell_dual 0.554, qsub2c 0.762, qsub2k 0.692.
  diag3 (mechanism, job via queue_diag3), mech_steps (dense trajectory), on-policy nf_pt1/nf_hi1, value-digit
  mask: none of the offline metrics predicts e2e ranking (theory_v4.md E4). Doc rev 25.
- 10:02: GPU work stopped for the user's noon return; FAST also finished (no jobs).

## Round 5 (2026-10-16 from 10:20; user back at the machine)
- User rule: research continues but must not disturb the user's use of the machine. GPU work now runs only
  under scripts/idle_gate.sh (start after 600 s desktop idle, stop within ~10-30 s of input; resumable);
  CPU work on cores 0-5 at nice 19. FAST agreed (CPU 6-19, no GPU, same idle gate if needed).
- Finding (CPU-only analysis of existing outputs): moment scorers fail when the answer value straddles a 32-token
  block boundary (COBS 0.18 vs 0.83; box scorers unaffected); errors are near-miss digits, never distractor values.
  theory/theory_v5.md; doc rev 26 (section 第五轮). Third adversarial review running.
- Queued (idle-gated, scripts/run_round5_idle.sh -> queue_round5.sh): nb_16k (+nb neighbour expansion, +g2 two
  offset grids, with baselines), then causal layer probes. First attempt at 10:2x OOM'd under --vram-frac 0.55.

## Collaboration with the Codex independent reviewer (user request, 2026-10-16 ~11:20)
- Codex scope: work-handoff project attention-independent-review-20261016 (owner codex:attention-review-20261016),
  dir ~/docs/projects/attention-review/2026-10-16/ (snapshot of git 8b42f41; outputs review.txt,
  collaborator-note.txt). Codex runs no CPU/GPU experiments; validation experiments are delegated to me.
- My channel: claude-response.txt in that directory (Claude-only file, latest entry on top): every finding gets a
  written disposition; requested experiments get scheduled here and reported with job ids/result files.
- User rule: GPU at full speed by default; idle gate only when the user asks.

## Resume point (2026-10-16 ~10:05)
- No owned job running. Theory: theory/theory_v4.md. Doc: rev 25 (sections 第三轮, 第四轮; 结论 第四轮更新).
- Next (only on explicit continue): causal intervention (sparse only in selected layers/heads) to explain why
  box-type scorers win e2e; token-matched page-size controls; Quest 16-token pages as main baseline;
  larger n (>=100/task), second model, non-needle tasks. Naming deferred until a unified rule is validated.
- Earlier: work-handoff job c2fc3ad581c445e8bbbec66afd7c93e8 = 32k run (4 tasks x 20, budgets 512/1024, methods
  o_dual,quest,cobs,sum_dual,mix_cobs_mean) -> worktree runs/stage1_32k.jsonl; rerun same command to resume.
- Then: analyze with src/analyze.py + src/compare.py --ref sum_dual, add 32k rows to doc section "第三轮".

## 后续里程碑：集群交付包（用户 2026-10-16 指示，理论完善后才启动）
- 触发条件：理论成果足够完善、主方法在预注册确认中站得住之后。
- 内容：项目命名与简称（参考 FAST/FASTER）；可移植实验 infra；多层实验计划（按不同显卡集群规模分层）；记录与分析模板/脚本；详细说明指南。用户届时建仓库推送，拷到显卡集群实测效果、落地与性能。
- 参考：/home/kooki3/ocbench-gen/docs/FAST_UR_EXPERIMENT_PLAN.md、FAST_UR_PLANS.md、FAST_UR_HOST_GUIDE_ZH.md、SETUP_NEW_HOST.md。
- 本里程碑不授权推送或对外发布；仓库由用户创建。

## Round 6 (2026-10-16 ~14:50–16:10) — literature grounding, phase 1 done, EM, independent haystacks
- Survey: reports/稀疏注意力块选择相关工作.md (+5 notes in research_notes/稀疏注意力块选择相关工作/). Demoted: union (RBS 2609.20971),
  tail rescue (KVFetch 2610.08811 sequential forgetting), Theorem 1 (Samuelson+Bennett/Hoeffding), F_n (REM). Still new: grid-shift
  controlled experiment, partition × summary interaction, equal-summary-memory vs Quest-16, block-LSE certified interval.
- Phase 1 dev (runs/phase1_g2f_dev.jsonl, 80/80, gate passed; results/phase1_analysis.txt): dense 79; cobs 64→72 (10/2),
  nf_pt1 69→73 (7/3), quest 67→73 (6/0). Plan N: Δ.08,d.15 → 198; proposed N=300, single primary contrast (Codex agreed).
- EM scorer `em` (Edmundson–Madansky, mean+box), theory/verify_em.py ok. Codex: em+g2f ≈150% Quest-16 summary memory (single-grid 75%).
- v2 haystack (--haystack v2 --slot0): 47 books, 42.7M chars, disjoint slots (16k: 409, 32k: 204); theory/check_haystack_v2.py ok; v1 unchanged.
- Fixed two queue deadlocks; serial queue now: mech check (ledger 5fb432bd, running) → em dev (runs/em_dev.jsonl) → F3 ch2048 ×4.
- Doc rev 41: new section 第六轮 + conclusion line. Codex reply: 15:3x entry + 16:0x dispositions.
- Resume: (1) when mech check ends run `python src/mech_check_accept.py runs/mech_check_16k.jsonl` (exit code is authoritative),
  then fix queue_mech_check.sh echo masking; (2) analyse em_dev vs phase1 arms (same seeds); (3) read milestone-phase1-em-review.txt
  and write prereg v3 (v2 haystack, slot0=100, N=300, mid-p sensitivity, disjoint-slot gate); (4) new baselines (Lychee delimiter
  chunks, KVFetch pointer prefetch, RBS decode union, SnapKV dilation = existing +nb); (5) theory v7 with demotions.
- 17:0x update: Codex milestone-phase1-em-review + ptr early feedback all dispositioned (commits a6065b4 0ac9fb1 7441840 ff9c795):
  +ptr resets S.ptr per prompt/method/mode; EM numerics explicit (em_viol per row); mech identity preflight (63 ok: 40 verified,
  23 first_recorded); survey report wording corrected; v2 loader verifies manifest + provenance; 80/80 v1 prompts reproduce;
  prereg v3 draft theory/prereg_g2f_v3.md (N=300: power .948/.814 at Δ.08) + src/phase2_gate.py (synthetic tests pass).
  Theory v7 theory/theory_v7.md (covering lemma sharp at l <= B/2+1). em_dev interim 29/80: em ≈ quest (23 vs 22; +g2f 25 vs 25).
  Next: em_dev finish → dev_compare; ptr_dev; Codex review of v3 → freeze → confirmation run (needs user-away or low-footprint ~10h).

## 基准选择（用户 2026-10-16 指示，适用于集群交付包的多层实验计划）
- 原则：结果必须能与官方、被广泛认可的 benchmark 上的基线和相关工作直接对比（参考 FAST 用 OCBench 的做法；OCBench 本身是机器人
  BC/RL 基准，不适用于本课题，取其原则）。
- 头条层（官方 harness，任务定义不改）：RULER 官方实现（13 任务；16k/32k/64k/128k）为主；LongBench v2（及 Quest 可比的 LongBench
  子集）与 InfiniteBench 为辅；可选 HELMET 的 recall/RAG 类。模型取相关工作报告过的 Llama-3.1-8B-Instruct、Qwen2.5-7B-Instruct
  （本机 1.5B 只作开发）。
- 基线在同一 harness、同一读取预算与写明的摘要内存核算下跑：dense、Quest（页 16/32）、COBS 式、RBS 质心∪半径（改 decode）、KVFetch
  式 +ptr、LycheeCluster 式分隔符块、SnapKV 式邻域扩张（+nb）；有公开代码者优先用其官方实现核对数值。
- 机制/诊断层：本地受控 NIAH（网格平移、强制块、逐数字诊断、当前 prereg v3 确认）——只支撑机制主张，不作头条数字。
- 效率层：需真实 block-sparse kernel（当前实现为掩码模拟，只测准确率），延迟/吞吐另立计划。
- 硬件分层（参考 FAST_UR_PLANS 三层）：单卡 24GB（7B/8B ≤32k）、单卡 80GB（≤128k）、多卡（全矩阵 + 多种子）。
- 近期可做：本机用官方 RULER 跑 Qwen2.5-1.5B@16k 的小规模试点，打通官方 harness 与我们的块选择接口。
- 官方 RULER 已克隆到 /home/kooki3/tools/sources/ruler（NVIDIA/RULER，--depth 1，未运行）。接入方案：官方 data/prepare.py 生成
  13 任务样本、官方 eval/evaluate.py 计分，均不改；只替换 pred 步——新写 src/ruler_pred.py 用我们的 blocksparse 方法逐条生成，
  按官方 prediction jsonl 字段 {index, pred, input, outputs, others, truncation, length} 输出，生成长度用官方 tokens_to_generate。
  官方模板无 Qwen：在 data/template.py 增加 Qwen2.5 chat 模板（唯一改动，记录 diff）。依赖：essay 数据（官方下载脚本）、
  squad/hotpotqa（QA 任务）。本机先做 Qwen2.5-1.5B@16k、每任务少量样本的打通试点，排在当前 GPU 队列之后。

## 用户分工决定（2026-10-16 16:1x）
- 选集群与层级：不由本会话决定；实验计划须写到集群上的 agent 能自行判断选择（docs/EXPERIMENT_PLANS_ZH.md §1/§8）。
- 建 GitHub 仓库：用户完成。
- 项目命名、确认实验结论判断：授权 Claude 与 Codex 共同完成（不需再请用户裁决；仍不含对外发布/投稿）。

## 19:5x 更新（确认实验运行中，job ef183d15）
- 冻结文件（确认运行与分析完成前不得修改）：src/sparse_attn.py、src/run_e2e.py、src/tasks.py、src/phase2_*.py、src/newcombe.py、
  src/make_phase2_spec.py、theory/prereg_g2f_v3.md。新工作只新增文件。
- 理论 v8（theory/theory_v8_draft.md）：F 机制的精确界与两个反例；R 机制；描述性证据矩阵 src/mech_matrix.py（sha256 9c5fb760…，
  完整诊断前已固定）；§6 k 网格覆盖引理（尖锐）与等摘要内存下"双网格 B vs 单网格 B/2"的比较、预测 P-grid。
- P-grid 开发运行（quest --block 16 vs quest+g2f@32，phase-1 种子）需先把 sink 大小与块长解耦（改 sparse_attn.py），故排在确认分析之后。
- 实验计划 v1（docs/EXPERIMENT_PLANS_ZH.md，1122150）：主格 Llama-3.1-8B/RULER16k/预算1024/100 条冻结；主比较 Quest-16+g2f vs Quest-16。
- ptr 开发：32/80 时 CUDA unknown error（根因未定位），同配置续跑排在最后。
- GPU 队列：确认 → 63 提示完整诊断 → F3 → ptr 续跑。

## 22:0x 更新
- 登记预测 P-kv（src/kv_prediction.py sha256 0b80dc34…，4122939，登记时确认 76/300、未看结果）：检索单元 = 关键词→数值段；
  开发数据上 COBS 式 g2f 增益全部来自该段跨界的题。确认冻结分析之后原样运行。
- 机制矩阵 v4（ffb91312…）+ EXPECTED 身份（results/mech_full_expected.json，在完整诊断运行前生成）。完整诊断结束后：
  make_mech_manifest.py → 提交 → mech_matrix.py MECH results/mech_prompts_full.json IDS --manifest M --expected results/mech_full_expected.json。
- RULER：数据管线 + 准备清单 + 预测适配器（R1–R4 已修）；GPU 空闲后做相同输入对照与顺序无关性验收。
- 确认实验 86/300（job ef183d15）。

## 确认实验完成与定名（2026-10-16 18:48–19:3x）
- 确认 job ef183d15 succeeded；冻结门控 PASSED；主比较 cobs+g2f vs cobs +12.3 pts [8.0,16.9]，p=5.73e-8（43/6）；nf_pt1 +9.7（Holm 显著）；
  Quest +1.0 [−1.3,3.4] 不显著；P-kv P1–P3 达到登记规则。结论 C1–C3 与 Codex 共同定稿（claude-response.txt 19:1x 段，措辞以其为准）。
- 项目定名 TWIN-GRID（Codex 审查：SEAM 与 CVPR 2020 SEAM 注意力重名而放弃）；TWIN-GRID = g2_fill_v1 = +g2f；旧收据/SPEC/源码标识不改。
- 交付包进行中：cluster/plan_v1.json、cluster/make_jobs.py、cluster/aggregate.py；数据准备脚本已参数化（模型/模板/环境）。
- GPU 队列：完整机制诊断（job 1f56da28）→ F3 → ptr 续跑；之后做 RULER 适配验收与适配基线。

## v1.4 冻结前稿（2026-10-16 20:0x–21:xx；第三方审查 = Claude 会话 "TWIN-GRID审查"，Codex 已由用户暂停）
- 审查批次 1（T1–T7）与批次 2（U1–U3）全部与审查方逐条对齐；处置见 /home/kooki3/docs/projects/twin-grid-review/2026-10-16/researcher-response.md。
- 计划 v1.4（docs/EXPERIMENT_PLANS_ZH.md、cluster/plan_v1.json）：
  - 主比较改为 cobs+g2f vs cobs（B32）；Holm 恰好两项：nf_pt1±g2f，以及可覆盖子集上的 cobs±g2f；
  - 登记的描述性分析：P-gain（增益律 κ）、P-cov、P-asym、P-grid；
  - 功效表；层 3 改为跑完整主格的 7 个臂。
- 新代码：
  - cluster/unit_lengths.py v3：运行前冻结单元长度与类别；
  - aggregate.py v4：--units / --subset；
  - gain_law.py：P-gain 与 P-asym；holm.py 加入 T6 修复；
  - make_jobs.py 写出单元长度文件；
  - 测试 theory/check_cluster_aggregate.py、check_gain_law.py 均 rc 0。
- 统一理论 docs/THEORY_ZH.md：合并 v1–v8，以增益律 Δ = π·G·R + 段内项为中心命题；可检验的内容只有三条（G 的顺序、范围内 R≈1、范围外 R≈0）。
- 事后诊断（src/kv_residual.py，只用于生成假设）：
  - T4：确认数据中 99 个跨界单元全是 repair 类，约 8/99 的残差处于本底错误水平；
  - T5：切点位置不对称，cobs 在 m≤3 时为 7/34 vs 24/44，支持 R。
- 适配验收（本机 1.5B）：
  - 顺序 / 缓存检查 4/4 一致；相同输入 14/15 一致；
  - 不一致的 1 条（cwe:4）诊断为 bf16 平局（results/accept_diag_cwe4.out）；
  - 严格判定 NOT ACCEPTED，平局规则待审查方确认。
- 依赖：项目 venv 增加 requests（官方 HF wrapper 需要）；RULER venv 增加 scipy（fwe 需要）。
- 提交 69cbf51；无 remote（用户之后自行建 GitHub 仓库）。
- 恢复点：
  1. 读审查方对 v1.4 冻结前稿的回复（findings.md 或 SendMessage），处理后冻结 v1.4；
  2. 平局规则确认后写入 accept_adapter.py；
  3. G3 布局匹配（sink 与块长解耦，作为可选开关，不改变默认行为），然后在本机开发运行 cobs16 / nf_pt1@16；
  4. F3 与 ptr-resume：代码变更后不能续跑，待决定是全新重跑还是放弃，并记录理由；
  5. 显存探针、LongBench 入口、两个适配基线；最后终审，然后交付可推送的仓库。
- 21:xx 冻结（审查方批次 3）：v1.4 冻结在 8e55275，文件哈希见 results/freeze_v14_sha256.txt（d65fca7）。
  - 平局规则 a–d 由 cluster/accept_tie_diag.py 和 accept_ties.py 实现。本机验收在平局规则下仍为 NOT ACCEPTED（15 行时平局上限为 0），集群按 65 行重新验收。
  - G3 布局匹配：`--layout-unit 32`，theory/check_g3_layout.py 通过，默认路径逐位一致；cobs16 / nf_pt1@16 已就绪。
  - E5 / P-asym 改写为“不是 R 的检验”；最小可检出 g 主比较约 0.15、子集约 0.14。检查集 scripts/check_all.sh 的 rc_all=0。
  - 恢复点：等审查方完成冻结身份核对 → 本机开发运行 P-grid（cobs16、nf_pt1@16 vs cobs+g2f）→ 决定 F3 / ptr → 显存探针、LongBench 入口、适配基线 → 交付。
- 22:xx 审查方批次 4：冻结身份核对通过（13/13）。
  - 补记：README / 指南写明 C1–C3 只能在 e194213 的 sparse_attn（c9b56b7f）上复现；“逐位一致”只指选择、预算、常驻区与候选数。
  - check_all.sh 加入 e194213 与 HEAD 的对照，rc_all=0。
  - 显存探针 cluster/probe_memory.py 完成（本机 1.5B @16k 峰值 3.9 GiB）。
  - **F3 与 ptr 收尾：未完成，已搁置，不在主线。**两者都不服务于任何 v1.4 登记预测，不再占用 GPU。
    - F3（job 944b8afd）为了让位给适配验收，被 work-handoff stop 中断，部分结果见 runs/f3_ch2048_r8_shared.jsonl。
    - ptr 开发运行（job 3e27e7ee）因 “CUDA unknown error” 失败，原因未定位，部分结果见 runs/ptr_dev.jsonl。
    - 代码已变（布局单位 U），两者都不能续跑；如以后需要，必须全新重跑。
  - 下一步：本机 P-grid 开发运行（cobs16、nf_pt1@16 对 cobs+g2f，开发/探索性质）。
- P-grid 开发运行（scripts/queue_pgrid.sh；分析 src/pgrid_dev.py，在结果出来前提交，之后不改）预期方向（开发/探索，无决策规则）：在被 16 网格切开的单元上 twin@32（cobs+g2f、nf_pt1+g2f）≥ fine@16（cobs16、nf_pt1@16），依据为 L4 与审查方的玩具模拟；在未切开的单元上两者接近。报告时给出单元数与以样本为单位的任务内 bootstrap 区间，不据此下结论。
  - 时间更正：queue_pgrid.sh 在 20:37:35 启动（第一个预测文件创建），pgrid_dev.py 在 20:37:53 写成，2f61f07 在 20:38:25 提交。提交前只看过 ls 和 wc -l（1 行），没有看任何预测内容；准确表述为“在查看任何预测内容之前固定”，而不是“在任何预测产生之前”。
- P-grid 运行偏离记录：原队列 14 个作业中 10 个成功；multivalue 和 multiquery 的 4 个作业在 0.6 上限下 OOM。
  - 补跑 1（queue_pgrid_rerun.sh，加 expandable_segments）：成功 3 个。b32-multivalue 仍然 OOM，4 个方法同进程，已分配 4.66 GiB / 上限 4.8 GiB。
  - 补跑 2（queue_pgrid_mv.sh）：只针对 b32 的 multivalue，拆成两个进程。跑完后由 pgrid_mv_merge.sh 把 1 行部分文件存档到 b32/_oom_partial/，再把新结果复制到分析脚本读取的路径。分析脚本不改。
  - 显存探针（3.9 GiB）低估了真实峰值（> 4.8 GiB）；指南已补充 expandable_segments 和“按探针峰值 1.5 倍估算”。

## 推送授权与公开发布（2026-10-16 23:5x – 2026-10-17）
- 用户授权原话（2026-10-16 23:5x）：“我准备睡觉了，请继续全速推进各项工作，各项工作均完成且检查无误可以送去集群开始训练后，推送到git@github.com:Koooki3/TWIN-GRID.git”；
  随后“你可以先测试一下能不能并推送”，并在添加 deploy key 后回复“已添加，明天见”。
- 仓库为用户自建、公开（GitHub API 200）。推送方式：scripts/public_snapshot.sh 生成快照提交（去掉 runs/ruler/ 的第三方文本，加入 PROVENANCE.md），
  作者使用 noreply 地址（Koooki3@users.noreply.github.com）——个人邮箱是否公开由用户决定，本地 git 配置未改；完整本地历史以 git bundle 存档到
  /home/kooki3/archives/twin-grid/2026-10-17/（不推送）。
- 推送专用 key：~/.ssh/id_ed25519_twingrid（用户已添加为 TWIN-GRID 的可写 deploy key；~/.ssh/config 未改）。
- 用户入睡后按其 GPU 规则（人不在时全速）：nf 补跑 3 与变体改为 --vram-frac 0.75 --throttle 0（只影响资源，不影响计算；审查方同意）。

## 夜间结果（2026-10-17 00:3x–03:3x，全速；开发/探索性质，不进确认结论）
- P-grid（等内存，G3 匹配）：预期 twin@32 ≥ fine@16 **未成立**；cobs@16 0.668 > cobs+g2f 0.609；nf_pt1 在被切开的单元上 −0.131 [−0.181, −0.078]。
- delim：0.541（≈ cobs 0.547，< cobs+g2f 0.609），覆盖率 100%（含 vt；“vt 覆盖差”的预期被证伪）。
- 变体：quest32 0.850 ≫ mean+g2f 0.591（等内存，−0.259）≈ cobs+g2f 0.609；G 的顺序 mean > cobs > quest 成立；nb/nbr/nbh 都比不扩展更差。
- box 摘要 P-grid（E4）：quest16 0.888 ≈ quest32+g2f 0.865（−0.024 [−0.059, +0.009]）→ 矩摘要上细网格的优势主要来自单元占比效应。
- 叙述已按审查方批次 10 收窄（README、附录 A4 §8.1–8.5、RESULT_SCENARIOS 必报项）；冻结的 v1.4 与集群包未改（13/13）。
- 审查方会话在变体结果出来前离线（ListAgents 只剩 M2A 会话）；变体与 E4 的结果写入了 researcher-response.md，并用不依赖项目代码的脚本独立重算
  （results/recompute_dev_independent.log，8 个臂全部一致），作为审查方离线期间的自检。
- 恢复点：审查方回来后复核变体与 E4（2d1e27f、ac44080、2571fc3）；用户醒来后汇报：TWIN-GRID 在等内存下不如细网格、Quest 远强于矩摘要，
  项目价值收窄为“块长三效应 + 切分度量”；是否继续把集群主比较作为头条，由用户决定。
- 03:4x 已推送：公开仓库 github.com/Koooki3/TWIN-GRID 的 main = 快照 dc22980（来自本地 e644c87；去掉 runs/ruler/ 共 275 个文件，加入 PROVENANCE.md；作者 noreply）。
  推送前检查：check_all rc 0、check_delim / check_nbh ALL OK、冻结 13/13、快照树检查、公开内容扫描（无凭据/邮箱/长文本）；推送后在 clone 上冻结 13/13、check_all rc 0。
  完整本地历史：/home/kooki3/archives/twin-grid/2026-10-17/twin-grid-local-e644c87.bundle。审查方离线：变体与 E4 由独立重算自检，待审查方复核；后续修正以快进推送（父节点 = 公开 main）。

## RECAP GPU 窗口（方案 A；2026-10-17 23:58 → 2026-10-18 04:00）
- 23:58 停止 queue_rep2.sh 与补跑链；当前作业 9ec0867d 经 work-handoff stop 记为 interrupted；GPU 释放（约 1.5 GB 为桌面）；check_jsonl_tail rc 0（全部可解析，无重复）。
- 重启：RECAP 通知结束后，先 check_jsonl_tail，再用同一命令重启 queue_rep2.sh（同 vram-frac/throttle）及补跑→价值附录→补跑→门控链。
- 03:56 RECAP 交还 GPU（无 RECAP 进程，681 MiB）；check_jsonl_tail rc 0；用 scripts/queue_rep2_resume.sh 重启（已开始的 (臂, 任务) 用 manifest 中记录的 0.6/0.3，其余全速 0.75/0），之后 R7、价值附录、门控依次进行。
- 2026-10-18 11:5x 用户开始使用本机：停止全速队列，作业 5c7ce618 记为 interrupted；check_jsonl_tail rc 0；以 scripts/queue_rep2_resume_low.sh 重启（未开始的任务用 0.6/0.3；已开始的沿用 manifest 记录的参数）。已转告 RECAP。
- 2026-10-18 13:2x 存储清理（Codex 转达用户授权）：删除作业临时目录中的 unitdev_qwen_16k（23 MB，与 runs/ruler/pgrid_qwen1.5b_16k 中 13 个 jsonl 与 prep_manifest 逐一哈希一致）以及 __pycache__（可重建），共约 24 MB。保留：runs/（原始实验结果与续跑依赖）、.venv、results/docs/代码（冻结/登记/审查证据）、archives 中的 bundle、llama31_tok 与 unitdev_llama_16k（功效估计复现所需）、git 对象（共享仓库内部）。共享 ~/.cache/uv 未动（磁盘约 7%，不紧张；已告知 RECAP，避免双方同时操作）。

## rep2 预算 512 提前判定（2026-10-18 14:19，已登记规则 R1–R6；R7 待预算 1024）
- scripts/rep2_early.sh final rc=0：门控 --b512-only 通过（63 个 jsonl = 9 臂 × 7 任务，每格 0..99 行齐全）；mean+g2f/niah_multikey_2 崩溃前 14/14、续跑 6/6 与第 1 轮一致；
  8 个稀疏臂 0..19 行与第 1 轮逐条一致（140/140）；防偷看断言成立（此刻 R7 的 5 个臂中仅 1 个有 b1024 输出）。日志 results/rep2_early.log，收据 results/rep2_b512_receipts.txt（列表哈希 57cfeae4…e4b1）。
- 新样本（20..99 行，560 样本 / 1360 单元）：dense .956；cobs .535；cobs+g2f .593；mean .548；mean+g2f .622；quest32 .846；quest32+g2f .855；cobs16 .659；quest16 .890。
- R1 cobs16 − cobs+g2f = +0.065 [+0.037, +0.096] 复现；R2 quest32 − mean+g2f = +0.224 [+0.196, +0.253] 复现；R3 G 顺序 mean > cobs > quest 完全复现；
  R4 quest32+g2f − quest16 = −0.035，90% [−0.048, −0.022] 等价（但 95% [−0.050, −0.019] 不含 0：box 摘要上 TWIN-GRID 仍略低于细网格）；R5 两个 +g2f 增益复现；R6 pooled κ 1.060 [0.684, 1.600] 相符（受勘误 E1 限制，仅同样本描述性校准）。
- 价值口径（docs/VALUE_CRITERIA_ZH.md）：A1 不成立（等内存下 TWIN-GRID 输给细网格，与开发结果同向）；C1 = R1 & R2 & R3 & R4 成立（认识层面结论在新样本上复现）；C2 相符（证据强度按 E1 降级）。
- 偏差：未改规则；资源参数混用（0.6/0.3 与 0.75/0，只影响资源）已在日志逐格列出。
- 下一步：队列继续 R7（b1024）→ 价值附录（em、quest16+g2f、quest8）→ 全门控 → R7、A4、A5 最终判定；提前判定已交审查方与 Codex 复核。
- Codex 独立复核 70d0779（twin-review-70d0779.txt）：R1–R6 全部点估计和区间独立重算吻合，63 个哈希、UID 集合、冻结文件 13 个与登记文件 8 个、6300 行绑定、5600 个预算计数、8 臂 × 140 确定性均通过。
  处理：T4 已修正——rep2_gate.py 的成功提示只列实际检查的范围（--b512-only 时写明 b1024 未检查、R7 未判定），检查逻辑未改；70d0779 的原始 rep2_early.log 保留原样。
  报告口径：
  - R4 的叫法是“登记的配对、按任务分层的 percentile-bootstrap 90% 区间落在 ±0.06 等价界限内”，不是另行做的经典参数 TOST，也没有 p 值；±6pp 等价不等于效果相等，同时报告 −3.5pp 差值及 95% 区间不含 0。
  - 本批 out 类为 0，R_out 不可估计（日志中的 All-NaN 警告即由此而来），不能写成零效果。
  - R6 quest 不确定；C2 遵守 E1。
  - 全套没有做多重比较控制，不升级为集群确证。
  - 来源边界：63 份 manifest 的 model revision 未记录，权重只有 size/mtime。已补 results/model_sha256_20261018.txt（当前内容哈希），只作为之后运行的佐证，不回填，也不证明历史权重身份。

## RECAP GPU 窗口二（2026-10-19 00:5x；用户入睡）
- 停止 queue_rep2_resume_low.sh；当前作业 6e73a78b 经 work-handoff stop 记为 interrupted（无残留 ruler_pred 进程）；GPU 1497 MiB（桌面）；check_jsonl_tail 通过（全部可解析、无重复）。
  停止时 R7（b1024）已完成 1920/2100 行；价值附录尚未开始。已向 RECAP 发出“可以开始”（mpre 约 75 分钟）。
- 恢复：RECAP 交还后，先跑 check_jsonl_tail，再用 scripts/queue_rep2_resume.sh 全速恢复（已开始的 (臂, 任务) 沿用 manifest 记录的参数，新任务用 0.75/0），依次跑完 R7、价值附录和全门控（--value）。
- 03:57 RECAP 停止 mpre 并交还 GPU（作业 3c8aa151，01:40:41 启动，进度 5620/7456，剩余约 45 分钟改日续跑，届时先与 TWIN-GRID 确认）。
  03:58 复核：无 run_rh05 / ruler_pred 进程，GPU 1330 MiB；check_jsonl_tail 通过。已用 scripts/queue_rep2_resume.sh 全速恢复（后台任务 bcjz1z6au）。

## rep2 最终判定（2026-10-19 09:55；全门控 --value 通过；规则均为事先登记）
- 队列 09:54:55 结束；全门控 rc 0（b512 rep2 0..99、b1024 0..59、价值附录 0..99）。rep2_final.sh（7b6e404，提交时门控尚未运行，也没有读过价值附录的任何输出）独立重跑门控 rc 0，
  两个登记分析脚本 rc 0，R1..R7、A4、A5 的判定行齐全，final rc=0。日志：results/rep2_final.log、results/rep2_value.log、results/rep2_gate.log；
  收据 results/rep2_final_receipts.txt（119 个 jsonl，列表哈希 9b11ef77…2947）；b512 rep2 的 63 个文件哈希仍为 57cfeae4…，与提前判定一致。
- R1–R6 与提前判定相同（复现；R4 为登记的 bootstrap 90% 等价；R6 quest 不确定；R_out 不可估计）。
- R7（b1024，样本 20..59，680 单元）：R1 点估计 +0.059（512 时 +0.065），R2 点估计 +0.097（512 时 +0.224），方向均一致。R2 的差距在预算翻倍后缩小一半以上（描述性）。
- 价值附录（b512）：em .847、quest16 .890、quest16+g2f .888、quest8 .912。
  A4 em − quest16 = −0.043 [−0.059, −0.027]（规则：下界 > −0.02）→ 不成立；
  A5 quest16+g2f − quest8 = −0.024 [−0.035, −0.014]（等内存，规则：下界 > 0）→ 不成立；quest16+g2f − quest16 = −0.001 [−0.012, +0.010]（描述性）。
- 价值口径汇总：
  - A1、A4、A5 均不成立；A2/A3 的前提是实测真实 kernel 延迟，本项目未测，不能声称成立；B（速度）未知。所以不存在可声明的性能或内存收益。
  - C1 成立（R1–R4 复现，R7 方向一致）；C2 相符，但按勘误 E1 仅为描述性证据。
  - 认识层面的一致结论：在两类块摘要（矩类、box 类）上，同等内存下更细的网格都胜过 TWIN-GRID（cobs16 > cobs+g2f；quest16 > quest32+g2f；quest8 > quest16+g2f）；
    摘要类型的差距远大于网格的差距；TWIN-GRID 只能部分修复被块边界切开的情况。
- 偏差：资源参数混用（逐格列在 rep2_final.log 中）；RECAP 窗口二期间作业 6e73a78b 被中断后续跑；价值附录比预计慢约 2 小时，不影响计算。
- 下一步：交 Codex 与审查方复核 → 按价值口径决定：不做性能类集群实验；是否设计认识层面的三层计划 v2，等用户决定 → 重写 README 与全部资料。
- Codex 复核 7d669b4（twin-final-review-7d669b4.txt）通过：119 个哈希、UID、manifest 一致，全门控快照 rc 0，10500 行绑定、9800 个预算计数通过，R7/A4/A5 独立重算吻合。
  - T5（低优先，本次未触发）已修正：rep2_final.sh 与 rep2_early.sh 在检查完判定行后，会把最终 rc 回写到 .rc 文件；scripts/check_t5_rc.sh 用末尾段落做故障注入测试，齐全时 0/0/0，缺行时 5/5/5。原始 log、.rc 与收据保持原样。
  - 对外结论的限定：本次 Qwen2.5-1.5B、16k、7 个 RULER 任务、所测预算与配置下；B（速度）未知，不等于零收益；C1、C2 是描述性复现，C2 受 E1 限制；R7 只看方向，不是跨预算的显著性检验，也不代表 1024 达到 R2 的 0.15 门槛；没有做多重比较控制，不升级为集群确证。
  - v2 只作为待评审材料；该复核不授权启动集群或新实验。v2 须写明可证伪的增量、数据分离、主比较与多重比较、停止规则，以及强模型身份。

## 资料整理与 v2 草案（2026-10-19 上午）
- README 首屏改为 rep2 的最终结论：方法价值不成立；认识层面复现；写明范围与边界，并给出全部方法的准确率。更早的预注册 C1–C3 保留，并注明是块长固定下的比较，不是等内存比较。
- 各文档已更新：
  - VALUE_CRITERIA 增加“判定结果”；THEORY_ADDENDUM 增加 §8.6（rep2）；
  - RESULT_SCENARIOS、CLUSTER_GUIDE、PLAN_V14_APPENDIX 加了状态说明：v1.4 不再执行，当前没有获准的集群运行；
  - 冻结的 13 个文件未改（13/13）；check_all rc_all=0。
- 新增 docs/PLAN_V2_COGNITIVE_ZH.md（认识层面三层计划，**待评审草案**，未冻结、未授权运行）：
  - Holm 三项：H1 cobs16 − cobs+g2f、H2 quest8 − quest16+g2f、H3 G_cobs − G_quest；
  - H4：在独立校准样本（行 100..199）上做增益律预测，修正 E1；
  - 强模型身份（revision + 内容哈希）；停止规则；
  - 用 rep2 标准误换算的近似功效表；实现缺口清单。
- 下一步：交审查方与 Codex 评审 v2 和整理后的资料；用户据 v2 §0 判断是否值得使用集群算力。
- Codex v2 评审（v2-review-0654202.txt）：需修订后再评审，不建议冻结。已按 V1–V7 修订为草案 r1（不运行任何研究实验）：
  - Q2 收窄为 8B 内部排序；
  - H4 改为固定 R = 1 的结构律预测，对比常数基线（加权 MAE），定位为探索性；拟合 R 的版本改称迁移；
  - 共用主读数只用 cobs；S_unit 写出 9 个任务的名单（含 UUID）与权重；
  - 置换与 Holm 的参数写死；
  - 功效改由 theory/power_v2.py 从 rep2 重采样得到（results/power_v2.log；效应减半时 H1/H2 的功效约 0.4–0.66）；
  - README 中 R7 的指代、附录 §8.6 的机制归因已改正。
  冻结 13/13。
- Codex r1 复核（v2-r1-review-504ab33）接受主设计，功效数值独立重算吻合。草案 r2 补冻结前定义：R1-a（C 的空/单格状态；几何随样本重采样；无效抽取整体作废，超过 1% 记不可判定；H3 bootstrap 的空类别；κ 分母规则）、R1-b（3pp 改称 C 上的加权平均误差上界，分三种状态，与相对标签分开）；注明 H3 功效未模拟 UUID 类别变化；功效输入哈希写在 results/power_v2_inputs_sha256.txt（49 个数据与预测文件）。
- v2 实现（只用 CPU，未运行实验）：cluster/plan_v2.json、cluster/make_jobs_v2.py（分 prep / predict 两个阶段，--revision 必填）、cluster/aggregate_v2.py（门控 v5：强模型身份、主样本与校准样本不相交、n=200 单元文件；复用冻结的 aggregate.py）、src/v2_stats.py；测试 check_v2_stats（27 项）、check_aggregate_v2（1 个正例、10 个反例）、check_v2_on_rep2（与 rep2 已报告的点估计一致）；check_all rc_all=0，冻结 13/13。
- Codex 实现复核（v2-impl-review-83e0d68）判为不通过（I1–I5）。已修复（只用 CPU，未运行实验）：新增 cluster/v2_run_descriptor.py（register/verify/bind：完整权重分片哈希、40 位 revision、适配 ACCEPTED、显存余量、计划/数据/单元文件/代码哈希；预测前 verify，预测后 bind）；make_jobs_v2 的 predict 阶段需要 --rd，先 verify、每条预测后 bind；aggregate_v2 的期望值全部取自运行描述，并输出 registered_parameters / formal_v2_result 两个标志；κ 恢复 r2 的求和口径（独立构造的例子 .75，恰好 .005 处有边界测试）；H4 补覆盖、逐格缺类别计数和 B0。端到端测试 28 项（含 I1–I3 全部反例，并限定失败原因），统计测试全部通过；check_all rc_all=0。
- Codex 复核 57267a2（v2-fix-review-57267a2）：I4、I5 关闭，I1、I3 判为已修复，I2 仍有两处阻断。已修复：I2-a 由结构化报告重新推出验收结论（覆盖、平局证据、结论文本一致性，以及验收 manifest 与模型和代码的绑定）；I2-b 检查显存探测的阶段覆盖与峰值一致性，并用 probe-receipt 收据绑定权重、代码和参数。端到端测试 44 项（上次交付说 28 项，实际是 27 项，系我计数有误），统计测试 36 项，全部通过；check_all rc_all=0；冻结 13/13；只用 CPU，未运行任何实验。
- Codex 复核 d6ad9a4（v2-fix2-review-d6ad9a4）：I2 剩三处残留。已修复：(a) 新增 accept-receipt，在验收后立即写收据，记录完整输入（accept_adapter 目录布局、13 个任务的 dense 运行及两组方法的 ab/ba/single）、报告中的 uid 与各运行一一对应、独立的验收数据、完整的模型与代码哈希，以及报告/结论/诊断的哈希；登记时逐项复核并重新推出对应关系；验收顺序任务固定在 plan_v2.json（niah_single_1）。(b) 平局诊断要求字段完整，并按 input_equal、swap、margin ≤ 2 ulp 重算是否为平局（同时核对 margin = l1 − l2、ulp = ulp_bf16）。(c) 探测收据保留完整 sha256。端到端测试 57 项、统计测试 36 项，全部通过；check_all rc_all=0；冻结 13/13；只用 CPU。
- Codex 复核 627c62d（v2-fix3-review-627c62d）：I2-b 关闭；I2-a.2 已修，另有零 logit 的兼容问题；I2-a.1 仍有阻断（顺序运行的预测与覆盖未核对，且收据可以被覆盖重写）。已修复：order_problems 按 accept_adapter 的方法，由绑定的行重新推出 identical 与 covered_rows，再与报告比对；收据不得覆盖（以 open 'x' 方式写入，已存在即拒绝）；新增 eval-run，在评价边界记录来源：评价前后输入哈希不变，输出由本次运行生成，并记录完整模型与代码哈希；报告、平局诊断、结论都必须带来源记录，accept-receipt 与 register 都会核对；prep 中的验收命令都经过 eval-run；ulp_bf16(0) = 2^-133，与冻结的生成程序一致。端到端测试 67 项（假评价程序，只用 CPU）、统计测试 36 项，全部通过；check_all rc_all=0；冻结 13/13。
- Codex 复核 4ff254b（v2-fix4-review-4ff254b）：顺序证据、收据覆盖和零 logit 三项关闭；I2-a.1 剩“来源快照不要求必需输入”一处。已修复：evidence_problems 由验收布局独立推出各角色的必需输入（报告与平局诊断 = 全部验收运行的 manifest 与预测文件 + 验收数据；结论 = 报告 + 全部平局诊断），每份来源快照都必须包含全部必需键，且哈希与当前相同；布局推不出时拒绝。新增反例：报告评价只监视数据（不论之后有无改动）、平局诊断评价只监视数据、结论评价漏掉诊断——全部拒绝；完整监视后再改动的对照组也被拒绝。端到端测试 72 项、统计测试 36 项全部通过；check_all rc_all=0；冻结 13/13；只用 CPU。
- 2026-10-19 Codex 复核闭环（v2-closure-297db75）：自 83e0d68 起的 I1–I5 实现问题全部关闭（独立复跑统计 36 项、端到端 72 项全部 OK，另加 7 个独立反例）。这只是实现层面的 CPU 合成验收，不代表模型结果，也不是冻结或运行授权；§4 的正式功效、真实显存、8B tokenizer / UUID 类别以及 §8 的记录，仍要等真实证据。用户未授权研究实验。
- 2026-10-19 公开仓库快进推送：github.com/Koooki3/TWIN-GRID main 6129374 → b0264da（来自本地 e868ed5，去掉 runs/ruler/ 的 521 个文件，加入 PROVENANCE.md，作者使用 noreply 地址）。推送前：快照树与冻结文件检查通过，内容扫描未发现个人邮箱或凭据；推送后在克隆上冻结文件 13/13 一致，check_all rc_all=0。完整本地历史的 bundle 存档在 /home/kooki3/archives/twin-grid/2026-10-17/。

## v2 冻结并交集群（2026-10-19）
- 用户：“先做整理和收尾、然后 polish 认知层面价值和完善实验计划，然后推送更新，我会去集群做实验”。本机已没有要做的实验（rep2 已完成；v2 全部在集群上跑）。
- 冻结前的最后修订（写入计划修订记录）：
  - §0 补写“这对谁有用”与三条可证伪的操作性结论；
  - 验收的顺序/缓存隔离任务定为 cwe、--local 16（与 v1.4 一致；原写的 niah_single_1 生成太短，覆盖可能为 0）；
  - prep 阶段写全验收流程（验收数据、dense 与顺序运行、经 eval-run 的报告与平局诊断、tie-plan、结论、验收收据）；
  - §4 改为运行时证据；§8 改为在 docs/PLAN_V2_SELECTION_RECORD_ZH.md 中填写，冻结计划保持字节不变。
- 新增文档：docs/RESULT_SCENARIOS_V2_ZH.md（结果的预先叙述）、docs/CLUSTER_GUIDE_V2_ZH.md（运行指南）、docs/PLAN_V2_SELECTION_RECORD_ZH.md。
- plan_v2.json 的 status 改为 FROZEN；冻结哈希写入 results/freeze_v2_sha256.txt（20 个文件），v1.4 的冻结文件 13/13 不变。
- 测试：统计 36 项、端到端 73 项全部通过（新增 tie-plan 测试，测试中的 RULER 路径可由 RULER_ROOT 指定）；check_all rc_all=0。
- 收尾：rep2 的数据准备日志入库；work-handoff 中本项目没有未结束的作业；GPU 空闲（已告知 RECAP）。
- 恢复点：集群上按 docs/CLUSTER_GUIDE_V2_ZH.md 执行；结果回来后按 RESULT_SCENARIOS_V2_ZH.md 解读，交审查方与 Codex 复核。
- 公开仓库快进推送：main b0264da → 8ab70c1（来自本地 29b882e）。推送前检查：内容扫描干净，v2 冻结文件 20/20、v1.4 冻结文件 13/13 在快照上一致；推送后在克隆上 check_all rc_all=0。
- 2026-10-19 推送后的本地存储清理（用户：“推送更新到远程 github 仓库后可以完整清理一下本地存储，把冗余/过期/缓存的文件等都删除”）。日志：results/cleanup_20261019.log。
  - 已删除或压缩：
    - git gc 把 2105 个松散对象（487 MiB）打包为 43 MiB，.git 从 489M 降到 44M；fsck --connectivity-only rc 0，所有可达历史不变；
    - 3 个被取代的历史 bundle（e644c87、0f29fe5、e868ed5；它们的 HEAD 已验证都是 29b882e 的祖先），只保留 twin-grid-local-29b882e.bundle（完整历史），存档从 113M 降到 44M；
    - 382 个 __pycache__ 目录（约 70M，可重建）；pip 下载缓存（85.2 MB）；本会话 /tmp 下的后台任务输出（1.6M）；
    - 作业临时目录（45M：临时脚本、测试夹具，以及已过期的 v1.4 功效复现输入 llama31_tok、unitdev_*，其结论已在 results/ 中）。
  - 保留及原因：
    - runs/（2.1G：实验原始结果，含 gitignore 的机制诊断原始数据 mech_full/mech_check，只能用 GPU 重跑）；
    - results、docs 与代码（冻结、登记与审查证据）；.venv（6.8G，CPU 检查需要）；
    - 模型（qwen2.5-1.5b，rep2 复现需要；pythia 属于其他项目）；RULER 源码（测试需要）；work-handoff 日志；
    - 共享的 ~/.cache/uv、huggingface、playwright（其他项目可能在用，未动）；/tmp 下 bash-edit-diff（工具链共享，未动）。
  - 清理后检查：check_all rc_all=0；冻结文件 v2 20/20、v1.4 13/13 一致。
- 2026-10-19 Codex 冻结复核（v2-freeze-review-29b882e）：哈希一致，复跑通过；但要求在预测前修订。已在任何 v2 prep 或预测之前完成冻结修订 a1：
  - F1：验收数据改为主冻结文件的保留行 200..204（accept-data），在生成与 register 时都按 input + answer_prefix 检查与主样本、校准样本不重叠，并核对派生来源。
    原写法会重复主样本前 5 行，但从未执行过，所以没有任何主样本被提前看到。
  - F2、F3：措辞收窄与改正（§0、§4、RESULT_SCENARIOS_V2）。
  - 冻结哈希重算；旧清单保存为 results/freeze_v2_sha256_29b882e.txt。端到端测试 75 项（新增：同提示不同文件哈希、来自另一份主数据，均拒绝），统计测试 36 项，check_all rc_all=0。
