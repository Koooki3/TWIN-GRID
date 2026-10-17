# TWIN-GRID

**Two offset candidate grids for budget-matched sparse attention**（错位双网格候选选择）

TWIN-GRID 是推理时块稀疏注意力的一种候选块划分方式。它同时使用两套错开半块（B/2）的网格，从中选取候选块，每次选择的候选 token 预算保持不变（实现：`g2_fill_v1`，实验臂后缀 `+g2f`）。它本身不是一个打分器，可以和不同的块摘要打分器组合（COBS 式投影方差、nf_pt1、Quest 等）。

**主要发现（本机开发数据，探索性质；附录 A4）**
1. **摘要类型主导**：box 摘要（Quest）远强于一阶、二阶矩摘要，等内存下差约 25 分（Quest-32 0.850 vs 均值池化+TWIN-GRID 0.591）。
2. **块长固定时，切分损害矩摘要**，TWIN-GRID 用 2× 摘要内存修复其中一部分（约 +0.06）。
3. **等内存下更细的单网格更好**；开发数据上最好的设计是块长 16 的 Quest（0.888）。

TWIN-GRID 因此是一个研究工具和降级的实现，**不是推荐方法**。

**错开半块的重叠布局并非新设计**：NSA 的压缩分支（块长 l = 32、步长 d = 16，d < l 用于防止碎片化，端到端训练）已经采用同样的布局（reports/related_work_verified_ZH.md）。
我们的定量分析与 NSA 的 d < l 一致，但 NSA 是端到端训练的，我们没有检验它。

开发结果的完整数字（P-grid、分隔符分块、读邻块变体、均值池化、box 摘要 P-grid）见 docs/THEORY_ADDENDUM_ZH.md 附录 A4；集群上的 cobs16 与 quest16 是检验这些结论最重要的描述性对照。

据此，本项目的贡献收窄为：
1. **块长三效应分解**（切分已度量；box 摘要的 P-grid 对照中，等内存的细网格与 TWIN-GRID 接近（−0.024 [−0.059, +0.009]，四目标任务 −0.013 [−0.062, +0.037]），
   所以可选位置数效应在 box 摘要上至多约 6 分（区间上限），小于矩摘要上 10–26 分的差距；剩余部分是与摘要类型有关的块长效应，与 F 模型的单元占比一致。
   这是开发数据上的解释，不是确认）：块长同时控制切分（∝ 1/B，小块不利）、
   单元占比（∝ 1/B，小块利于矩摘要）和可选位置数（∝ 1/B，小块利于多目标）；切分的惩罚 G 随摘要类型不同（矩摘要大，box 摘要小）；
2. **几何命题**：覆盖引理 L1、内存–读取对偶 L1′；开发结果与 L3（细网格的可达集合包含错位网格的可达集合；它允许、但不预测细网格更好）一致，L4（单次选择占优）的前提不成立；
3. **增益律**：限定为“固定块长下加不加 TWIN-GRID”，其三条预测（G 的顺序、范围内 R ≈ 1、范围外 R ≈ 0）在集群上做样本外检验；
4. （降级）TWIN-GRID 作为免训练、预算精确不变的最小实现；它剩下可能有用的场景只是“读取或 kernel 粒度被固定为 B”，这一点未经检验。

段内项以外的增益以跨界比例 π ≈ (ℓ−1)/B 为上界（对任何摘要类型成立），块长增大时必然趋于 0，所以只对块长为数十 token 的选择方法有意义；
大块方法（如 MoBA 的 512–4096 token 块，π ≈ 0.3%）几乎不受切分影响。

## 已确认的结果（本机，预注册）

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
| `docs/EXPERIMENT_PLANS_ZH.md` | 多层实验计划（官方基准 × 三个硬件层） |
| `docs/CLUSTER_GUIDE_ZH.md` | 集群运行指南（含已验证 / 未验证清单） |
| `docs/THEORY_ZH.md` | 统一理论：记号、两个误差源、L1–L4、F 模型、增益律、证据、未解决问题 |
| `theory/` | 旧版理论（v1–v8）、功效计算与各项数值 / 门控检查脚本 |
| `results/`、`runs/` | 本机实验输出、分析与收据 |
| `reports/`、`research_notes/` | 相关工作调研 |

## 在集群上运行

先读 `docs/CLUSTER_GUIDE_ZH.md`，选层由集群上的 agent 判断。指南第 0 节列出了尚未验证的组件，下表是摘要：

| 组件 | 状态 |
|---|---|
| 预测适配器的 GPU 验收 | 本机 1.5B：严格判定 NOT ACCEPTED（1/15 诊断与 bf16 平局相符，计分相同）；集群上按计划 §3.2 的平局规则以 65 行重新验收 |
| cobs16 / nf_pt1@16（等摘要内存对照） | G3 布局已匹配（CPU 检查通过），未在 GPU 上运行 |
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
