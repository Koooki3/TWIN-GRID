# TWIN-GRID

**Two offset candidate grids for budget-matched sparse attention**（错位双网格候选选择）

TWIN-GRID 是推理时块稀疏注意力的一种候选块划分方式。它同时使用两套错开半块（B/2）的网格，从中选取候选块，每次选择的候选 token 预算保持不变（实现：`g2_fill_v1`，实验臂后缀 `+g2f`）。它本身不是一个打分器，可以和不同的块摘要打分器组合（COBS 式投影方差、nf_pt1、Quest 等）。

## 结论（2026-10-19；rep2 新样本复现，Codex 独立复核通过，并经 Claude 审查方事后复核（批次 16），提交 7d669b4）

范围：Qwen2.5-1.5B-Instruct、上下文 16k、官方 RULER 的 7 个 NIAH/VT 任务（每任务 80 条未用过的样本，共 560 样本 / 1360 个答案单元）、读取预算 512（R7 另测 1024）。
所有判定规则都在运行前登记（src/rep2_analysis.py、src/rep2_value_analysis.py、docs/VALUE_CRITERIA_ZH.md）。结论只适用于这一范围。

**1. 方法价值：不成立。** 在所测配置下，TWIN-GRID 与 EM 打分器没有可证实的性能、内存或增益收益。
- 等内存下 TWIN-GRID 输给更细的单网格：cobs16 − cobs+g2f = +0.065 [+0.037, +0.096]（A1 不成立）；quest8 − quest16+g2f = +0.024 [+0.014, +0.035]（A5 不成立）。
- EM（摘要内存为 quest16 的 75%）比 quest16 低 0.043 [−0.059, −0.027]（A4 不成立）。
- 速度与真实 kernel 成本没有测（掩码模拟），记为“未知”，**不等于零收益**；依赖它的 A2/A3 不能主张。

**2. 认识层面：在新样本上复现（描述性复现，C1 成立）。**
- **摘要类型主导**：quest32 − mean+g2f = +0.224 [+0.196, +0.253]；
- **切分惩罚随摘要类型不同**：G = acc(段内) − acc(跨界)，G_mean .302 > G_cobs .231 > G_quest .031（两个差的下界都 > 0）；
- **TWIN-GRID 只部分修复切分**：固定块长下 cobs +0.058、mean +0.074（均复现），但不及等内存的细网格；
- **box 摘要上 TWIN-GRID 与细网格接近但略低**：quest32+g2f − quest16 = −0.035，90% 区间在登记的 ±0.06 等价界限内（95% 区间 [−0.050, −0.019] 不含 0）；
- 预算 1024（R7，行 20..59，只看方向）：cobs16 − cobs+g2f = +0.059、quest32 − mean+g2f = +0.097，与预算 512 同向；R7 没有检验 G 排序，也不是跨预算的显著性检验；
- 增益律 Δ ≈ π·G·R 合计 κ = 1.06 [0.68, 1.60]，但交叉拟合不能去掉合计 κ 中的共享噪声（勘误 E1），只算同样本的描述性校准；R_out 本批不可估计。

各方法准确率（单元）：dense .956；quest8 .912；quest16 .890；quest16+g2f .888；quest32+g2f .855；em .847；quest32 .846；cobs16 .659；mean+g2f .622；cobs+g2f .593；mean .548；cobs .535。

**边界**：box 摘要有效是 Quest 的结论；更细的块更好属常识方向；错开半块的布局在 NSA 压缩分支中已有（l = 32、d = 16，端到端训练；reports/related_work_verified_ZH.md）。
本项目增加的是同一协议下的定量分解（切分惩罚 G、增益律）和一个负结果：在免训练块选择中，错位双网格在等内存下不如直接细分。没有做多重比较控制；不推广到其它模型、长度或任务。

**下一步：v2 已冻结，交集群执行**（2026-10-19，用户决定使用集群算力）。认识层面的三层计划 docs/PLAN_V2_COGNITIVE_ZH.md 检验三件事：8B（Llama-3.1-8B-Instruct）上等内存的细网格是否仍胜过错位双网格（矩类 H1、box 类 H2）；矩类摘要的切分惩罚是否仍大于 box 类（H3）；以及只用单网格臂测得的 G 能否在独立校准样本上预测布局修复的收益（H4，探索性）。结果的措辞事先写定在 docs/RESULT_SCENARIOS_V2_ZH.md，运行步骤见 docs/CLUSTER_GUIDE_V2_ZH.md，冻结哈希在 results/freeze_v2_sha256.txt。
v1.4 计划（docs/EXPERIMENT_PLANS_ZH.md）保留冻结原样，不再执行其主比较。

完整记录：PROGRESS.md；判定日志 results/rep2_early.log、results/rep2_final.log、results/rep2_value.log；收据 results/rep2_final_receipts.txt；
Codex 复核报告在审查目录（twin-review-70d0779.txt、twin-final-review-7d669b4.txt）。

## 更早的确认结果（本机受控 NIAH，预注册；块长固定，不是等内存比较）

范围：冻结的 Qwen2.5-1.5B-Instruct，本地 RULER 式 NIAH（multikey / single_essay），v2 固定语料，上下文 16k，块长 32，每次候选读取预算 512，300 题新种子。

- **C1（确认性）** TWIN-GRID 把 COBS 式打分器的严格准确率从 244/300（81.3%）提高到 281/300（93.7%）。差 12.3 个百分点，配对 95% 区间 [8.0, 16.9]；43 题改善、6 题退化；精确 McNemar p = 5.73×10⁻⁸。
  对照是同块长的单网格（摘要内存 2× 对 1×）；等内存的比较见上文开发结果。
- **C2（确认性次要族）** nf_pt1 提高 9.7 个百分点（单项 95% 区间 [5.5, 14.1]），通过 Holm 校正；Quest 变化 +1.0 个百分点（[−1.3, 3.4]），未检出改善。
- **C3（数据收集中登记的描述性预测，三条规则均达到）** COBS 式的收益集中在 key→value 单元跨界的题目上：跨界组净改善 +39/99，段内组 −2/201。这支持收益在不同题目间分布不均的描述，尚不证明切分就是错误的原因。

**统一的读法（增益律，docs/THEORY_ZH.md §4）**：Δ = π·G·R + 段内项，其中 π 是单元跨界的比例，G 是单网格下的跨界惩罚，R 是双网格的修复比。
确认数据上（事后计算）：COBS 式 G = 0.43、R = 0.92；nf_pt1 G = 0.28、R = 0.96；Quest G = 0.04，所以 Quest 增益近零是这条规律的预测。集群实验把它作为登记的描述性预测（P-gain）首次检验。

**代码版本：** C1–C3 只能在确认时的代码上复现：`git show e194213:src/sparse_attn.py`（sha256 c9b56b7f…，与 theory/phase2_spec.json 一致）；HEAD 的 sparse_attn.py（1fd33e6b…）已加入布局单位 U。冻结文件（计划 v1.4、THEORY、sparse_attn 的 docstring）中“逐位一致 / bit for bit”只指 U = B 时块选择、读取预算、常驻区与候选数在 check_g2f、check_fixed_layout、check_invariant 中与旧版相同，不是模型输出逐比特相同（审查方批次 4）。

**绝对准确率（同一批 300 题，严格计分）：** dense 298（99.3%）、Quest 284（94.7%）、Quest+TWIN-GRID 287（95.7%）、COBS 式+TWIN-GRID 281（93.7%）、
nf_pt1+TWIN-GRID 280（93.3%）、nf_pt1 251（83.7%）、COBS 式 244（81.3%）。TWIN-GRID 让矩类摘要接近、但没有超过 Quest；本机设定下最好的单一打分器仍是 Quest。
摘要内存（每 token，理论最小值）：COBS 式+TWIN-GRID 是 Quest-32 的 1.28 倍、Quest-16 的 0.64 倍（theory/summary_memory.py）。

**真实任务上的泛化尚未建立。** 集群结果的预先叙述见 docs/RESULT_SCENARIOS_ZH.md（请在集群运行前推送仓库，以获得外部时间戳）。

**不主张：** 优于 Quest；速度或内存收益（掩码模拟只测准确率；双网格的摘要存储约为单网格的 2 倍）；推广到其它模型、任务或长度。

完整的推导见 `docs/THEORY_ZH.md`（统一版；旧版 theory/theory_v*.md 保留供追溯），证据链见 `results/` 和研究文档。结论先后经独立审查方 Codex 与 Claude 审查方逐项核验。

## 目录

| 路径 | 内容 |
|---|---|
| `src/sparse_attn.py` | 块稀疏 decode 注意力（HF AttentionInterface "blocksparse"）、各打分器、`+g2f`（TWIN-GRID）、逐调用预算断言 |
| `src/run_e2e.py`、`src/tasks.py` | 本地受控 NIAH 实验与生成器（v1 / v2 haystack） |
| `src/phase2_*.py`、`theory/prereg_g2f_v3.md`、`theory/phase2_spec.json` | 预注册确认实验：冻结规格、门控、分析 |
| `src/ruler_pred.py`、`src/ruler_prep_manifest.py`、`scripts/ruler_prepare.sh` | 官方 RULER 接入：数据准备清单、预测适配器 |
| `cluster/` | 集群交付：`plan_v1.json`（机器可读计划）、`make_jobs.py`、`unit_lengths.py`、`aggregate.py`、`holm.py`、`gain_law.py`、`accept_adapter.py` |
| `docs/PLAN_V2_COGNITIVE_ZH.md` | 认识层面三层计划 v2（已冻结，2026-10-19） |
| `docs/RESULT_SCENARIOS_V2_ZH.md`、`docs/CLUSTER_GUIDE_V2_ZH.md`、`docs/PLAN_V2_SELECTION_RECORD_ZH.md` | v2 结果的预先叙述、运行指南、§8 选择记录（运行时填写） |
| `cluster/plan_v2.json`、`make_jobs_v2.py`、`v2_run_descriptor.py`、`aggregate_v2.py`、`src/v2_stats.py` | v2 机器计划、作业生成（prep / predict）、运行描述与收据、门控与汇总、统计 |
| `docs/EXPERIMENT_PLANS_ZH.md` | v1.4 多层实验计划（冻结原样保留；主比较已被 rep2 结论取代，不再执行） |
| `docs/VALUE_CRITERIA_ZH.md` | 价值判定口径（运行前写定） |
| `src/rep2_analysis.py`、`src/rep2_value_analysis.py`、`src/rep2_gate.py` | rep2 新样本复现：登记分析、价值附录、完整性门控 |
| `docs/CLUSTER_GUIDE_ZH.md` | 集群运行指南（含已验证 / 未验证清单） |
| `docs/THEORY_ZH.md` | 统一理论：记号、两个误差源、L1–L4、F 模型、增益律、证据、未解决问题 |
| `theory/` | 旧版理论（v1–v8）、功效计算与各项数值 / 门控检查脚本 |
| `results/`、`runs/` | 本机实验输出、分析与收据 |
| `reports/`、`research_notes/` | 相关工作调研 |

## 在集群上运行

**v2 请按 `docs/CLUSTER_GUIDE_V2_ZH.md` 运行。** 下文与 `docs/CLUSTER_GUIDE_ZH.md` 是 v1.4 的流程，保留供追溯，不再执行。指南第 0 节列出了尚未验证的组件，下表是摘要：

| 组件 | 状态 |
|---|---|
| 预测适配器的 GPU 验收 | 本机 1.5B：严格判定 NOT ACCEPTED（1/15 诊断与 bf16 平局相符，计分相同）；集群上按计划 §3.2 的平局规则以 65 行重新验收 |
| cobs16 / quest16 / quest8 等内存对照 | 本机 1.5B 已在 GPU 上运行（rep2）；8B 未测 |
| Llama-3.1-8B | 未测 |
| 适配基线 RBS-decode | 未实现，层 1 跳过（属于误差源 (i)，不在主线上） |
| 分隔符分块（主要竞争方案） | 理论比较见 docs/THEORY_ADDENDUM_ZH.md；本机开发对照（src/delim_attn.py，探索性质）；集群上至多作为 v1.4 的描述性附录 |
| 显存上限 | 未实测 |

**公开仓库未包含 `runs/ruler/`**：其中是 RULER 准备的输入与带完整 prompt 的预测文件，含第三方文本（如 Paul Graham essays），可用
`scripts/ruler_prepare.sh` 按准备清单重新生成（文件哈希见各 prep_manifest 与 results/）。公开仓库是本地开发分支的快照
（`scripts/public_snapshot.sh`），文档中引用的本地提交号（如冻结提交 8e55275）不在公开历史中；冻结以文件 sha256 核对（results/freeze_v14_sha256.txt）。

依赖见 `requirements.txt`；官方 RULER 固定在 commit `c3f5e3b`，只打 `docs/ruler_template_qwen.diff` 这一个补丁，需要执行 `git lfs pull`，其 venv 需要 scipy。

## 许可与发布

代码与文档由项目所有者授权公开在 github.com/Koooki3/TWIN-GRID（见 PROGRESS.md 中的授权原话）；未授权投稿；尚未选择许可证（默认保留所有权利）。
