# TWIN-GRID 多层实验计划（v1.4，2026-10-16 冻结）：官方基准 × 三个硬件层

> 状态：**v1.4 已冻结（审查方批次 3 确认）；计划未启动，尚无任何集群数据。**项目名 **TWIN-GRID**（错位双网格候选选择；Two offset candidate grids for
> budget-matched sparse attention）；TWIN-GRID = 当前实现 g2_fill_v1 = 实验臂后缀 +g2f。理论见 docs/THEORY_ZH.md（统一版）。
> 本文规定在集群上跑什么、怎么比、怎么记；机器可读副本为 cluster/plan_v1.json（两者不一致时以本文为准）。
> 所有层共用 §3 协议。**由集群上的 agent 按本机算力自行选层**（依据 §1“适合”行与实测显存），选择在启动前写入 §8，看到结果后不能更换。
> v1.4 的修订依据第三方审查（Claude 审查方）批次 1（T1–T7）与批次 2（U1–U3），见文末修订记录。

## 0. 主张—证据矩阵

| 主张 | 证据层 | 官方版本 | 模型 | 任务 / 样本 | 读取 / 内存口径 | 状态 |
|---|---|---|---|---|---|---|
| A. 同读取预算下，TWIN-GRID 提高官方任务得分（主比较：COBS 式+g2f vs COBS 式，B = 32） | 头条 | RULER v1 旧 scripts，`c3f5e3b` | 主格：Llama-3.1-8B-Instruct | RULER 13 任务 @16k，每任务固定 100 条 | 读取 token 相同；摘要量约 2×（单列） | 计划 |
| B. 增益律 Δ ≈ π·G（P-gain）：增益大小可由运行前的 π 与基线臂的 G 预测 | 登记描述 | 同 A | 同 A | 同 A，打分器 cobs / nf_pt1 / quest32 | 同 A | 计划；本机确认数据上事后符合（THEORY E3） |
| C. 同一规律跨打分器与范围成立（Holm 两项） | 头条次要族 | 同 A | 同 A | 同 A | 同 A | 计划 |
| D. 本机确认（受控 NIAH，Qwen2.5-1.5B） | 机制 | 本地生成器 + v2 haystack | Qwen2.5-1.5B-Instruct | prereg v3：300 题 | 同 A | **已完成**：C1 +12.3 分 [8.0, 16.9]；C2；P-kv 三条达到 |
| E. 与 Quest 论文数字直接对照 | 复现 | Quest 官方 repo 的 LongBench v1 / passkey 入口 | 与所选 Quest 表相同的 checkpoint | 所选表的任务子集 | 与该表相同 | 计划（先选定表） |
| F. 速度 / 吞吐 | 效率 | — | — | — | 需真实 block-sparse kernel | 未开始（当前为掩码模拟，不作速度主张） |

块命中率、质量捕获率等只在机制诊断中用于解释，不作头条指标。D 的成败不能替代 A 的官方基准结论，反之亦然。
“直接复现”只用于模型全名/revision、上下文扩展、dtype、模板、生成长度、停止条件、预算单位、全注意力层规则全部匹配的情形；否则写“同设置重评”或“适配”。

## 1. 三个层（每层都包含完整主格）

| | **层 1：完整** | **层 2：单卡 80GB** | **层 3：单卡 24–48GB** |
|---|---|---|---|
| 适合 | ≥ 4×80GB | 1×80GB | 1×24GB 或 48GB |
| 模型 | Llama-3.1-8B-Instruct；Qwen2.5-7B-Instruct | Llama-3.1-8B-Instruct | Llama-3.1-8B-Instruct |
| RULER 长度 | 16k（主格）、4k、8k、32k、64k、128k | 16k（主格）、32k、64k | 16k（主格）；48GB 加 32k |
| 每任务样本 | 100（仅主格：Llama-3.1-8B / 16k / 预算 1024）；其余格 500 | 100 | 100 |
| 读取预算（token） | 1024（主格）、2048、4096 | 1024（主格）、2048 | 1024（主格） |
| 臂 | §2 全部 | §2 中 ★ 与 ☆ | §2 中 ★（= 完整主格） |
| 额外基准 | LongBench v1（Quest 子集）、InfiniteBench 检索类 | LongBench v1（Quest 子集） | 无 |

**复用规则**：只有模型 revision、数据样本 ID、模板、长度、预算、实现身份（代码 + 模型 + 协议）全部相同的格子才能跨层复用。三层的主格完全相同；
不同层或主机对同一样本 ID 的重复运行只作可复现性信息，分析时按样本 ID 去重。样本扩充（如 100 → 500）须沿用同一冻结 ID 序列向后续接。

## 2. 臂（逐项写明块长；★ = 主格，层 3 起必跑；☆ = 层 2 增加）

| 臂 | 块长 | 网格 | 角色 | 层 |
|---|---|---|---|---|
| dense | — | — | 上限参照 | ★ |
| cobs、cobs+g2f | 32 | 单 / 双（错开 16） | 主比较；Holm 成员 2（可覆盖子集） | ★ |
| nf_pt1、nf_pt1+g2f | 32 | 单 / 双 | Holm 成员 1 | ★ |
| quest32、quest32+g2f | 32 | 单 / 双 | P-gain 的 box 端（预期增益接近 0，按大小预测报告，不作检出检验） | ★ |
| em | 32 | 单 | 描述（误差源 (i)：em vs quest32） | ☆ |
| quest16、quest16+g2f | 16 | 单 / 双 | 描述（B = 16 时只有数字 value 本身 ≤ 9 token，kv 单元不可覆盖） | ☆ |
| cobs+ptr、quest16+ptr | 32 / 16 | 单 | 描述（KVFetch 启发的适配） | ☆ |
| cobs16、nf_pt1@16 | 16 | 单 | 描述（P-grid，等摘要内存；G3 匹配：所有作业 `--layout-unit 32`，常驻区与候选读取数同 B32） | ☆ |
| cobs+nb；RBS-decode；分隔符分块 | 32 / 变长 | — | 描述 / 适配基线（后两者未实现） | 层 1 |

共同：local 64、前 2 层稠密、sink 1 块、固定尾窗；读取预算 = sink + 尾窗 + 候选 token，逐调用断言（`--strict-budget`）。
作业生成器只展开本表列出的组合；同一块长的臂在一个进程内共享 prefill。

## 3. 共同协议

### 3.1 官方 harness
RULER v1 旧 scripts 流程（README 已标 deprecated；为与历史结果可比而固定），commit `c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a`；
`data/prepare.py` 与 `eval/evaluate.py` 不改；只替换预测步，并只增加 Qwen 模板（diff 入库）。生成长度、停止条件按官方
`tokens_to_generate`；贪心解码。样本：用官方 prepare 生成 500 条/任务/长度并冻结（文件哈希入准备清单）；“100 条”指冻结文件的行号 0–99
（sample_uid = 文件哈希:任务:行号）。RULER 的 venv 须含 scipy（fwe 依赖），项目 venv 须含 requests（官方 HF wrapper 依赖）。

### 3.2 适配验收（每个模型首次运行前，在相同输入上对照）
选每任务固定样本，用官方参考 dense 预测入口（HF 后端）与本项目 dense 预测器分别运行，逐条比较最终输入 token、生成参数与停止、输出文本、官方计分；
另做同族臂的顺序与缓存隔离检查（A→B、B→A、单臂，生成须进入候选区）。cluster/accept_adapter.py 按严格文本一致判定。
集群验收用 13 任务 × 每任务 5 条固定样本（65 行）。
**数值平局规则（v1.4，审查方批次 3 确认）**：严格判定之外，每条不一致的输出用 cluster/accept_tie_diag.py 在官方 HF 回退路径（bf16、sdpa、一次性 prefill）上诊断，
再由 cluster/accept_ties.py 判定。记为 bf16 平局须同时满足：
(a) 输入 token 完全相同，官方计分相同；
(b) 首个分歧步官方 top-2 logit 差距 ≤ 2·ulp_bf16(max(|l₁|, |l₂|))，ulp_bf16(x) = 2^(⌊log₂x⌋−7)；
(c) 该步是 top-2 互换：适配器的 token 正好是官方路径的第 2 名；
(d) 每任务最多 1 条，平局总数 ≤ ⌊0.02 × 相同输入行数⌋；超出即 NOT ACCEPTED；每条平局保留诊断 JSON。
满足时判为 ACCEPTED_WITH_TIES，报告逐条列出。不调参追数。
本机 1.5B 的结果（规则是看到结果后增设的，只如实记录）：顺序/缓存检查 4/4 一致；相同输入 14/15 一致，严格判定 NOT ACCEPTED；
不一致的 cwe:4 满足 (a)–(c)（logit 18.75 / 18.5，差距 0.25 = 2 ulp，top-2 互换），但 15 行时上限为 0，按规则仍为 NOT ACCEPTED。

### 3.3 资源口径（每臂单列）
驻留 KV、每步实际读取 token、摘要字节（含网格数与 dtype）、峰值显存与工作区（实测）。

### 3.4 估计量与检验（启动前冻结，结果前不改）

**冻结步骤（在任何预测之前）**：每个 (模型, 长度) 的数据准备完成后，立即用该模型自己的 tokenizer 运行
`cluster/unit_lengths.py`（v3），写出 `<out-root>/data/<model>/<length>/unit_lengths_b<B>.json`，并把它的 sha256 填进 §8。
该文件逐样本、逐单元记录 ℓ、X_0、X_1、类别（fixed / in0 / repair / out）、value 长度与切点后的 value token 数。
汇总时 aggregate.py 会核对它与冻结数据、预测清单中的 tokenizer、块长和 local 是否一致，否则拒绝。

- **主比较**：主格上 COBS 式+g2f vs COBS 式（B = 32）。估计量为 13 任务等权的宏平均得分差 Δ̄ = (1/13)Σ_t mean_i(d_ti)。
  检验：任务内分层的配对符号翻转置换检验，双侧 α = 0.05，10 万次置换；区间：任务内分层的样本 bootstrap 95% 区间。
  依据：v1.3 的主比较 Quest-16±g2f 落在 L1 不适用的格子（B = 16 时 kv 单元 > 9 token），且本机证据最弱（C2-quest 不显著）。
  这一修订发生在冻结前、尚无任何集群数据时，依据是理论（L1 的适用条件）和本机确认（C1 对 C2-quest），不属于事后挑选。
- **次要族（Holm，α = 0.05，恰好两项）**：(1) nf_pt1+g2f vs nf_pt1；(2) COBS 式+g2f vs COBS 式在**可覆盖任务子集**上（同一批运行）。
  可覆盖：该任务所有被评估样本的所有单元满足 ℓ ≤ B/2 + 1，由冻结的单元长度文件机械判定；无单元任务为 cwe、fwe、qa_1、qa_2。
  按本机 tokenizer 实测（开发数据，仅供参考，正式归组以主格 tokenizer 为准）：可覆盖 7 个（niah_single_1/2、niah_multikey_1/2、niah_multivalue、niah_multiquery、vt），
  不可覆盖 2 个（niah_single_3、niah_multikey_3，UUID）。Holm 只在两项都在主格跑完时计算（cluster/holm.py；缺项、重复、身份不一致时拒绝）。
- **登记的描述性预测**（报告估计与区间，不作显著性结论）：
  - **P-gain（增益律）**：对 cobs、nf_pt1、quest32 与每个有单元的任务，预测 Δ = π·Ĝ_cf；主读数为合计校准比 κ（交叉拟合，预测 ≈ 1），
    辅助读数为校准斜率、R̂_repair（≈ 1）、R̂_out（≈ 0）、段内项（≈ 0）；以样本为单位的任务内 bootstrap（cluster/gain_law.py；THEORY §4）；
  - **P-cov**：COBS 式 ± g2f 的增益按任务组（可覆盖 / 不可覆盖 / 无单元）：可覆盖组 > 0，其余 ≈ 0；交互只作描述；
  - **quest32**：quest32+g2f vs quest32 的增益大小与 π·Ĝ 对照（预期接近 0），不作检出检验；
  - **P-asym**：基线打分器在 repair 单元上，“value 被切”（切点落在 value 内部）vs “只有 key 被切”（切点在 value 之前）的失败率；只作描述性读数，
    不作为 R 的检验（两组内容不同，非均匀 F 也预测不对称；THEORY E5）；
  - **P-grid**：cobs16 vs cobs+g2f（等摘要内存，G3 已匹配）：被 16 网格切开的单元上预测 twin ≥ fine；检验选择规则，不区分机制；
  - 其余：em vs quest32、cobs+ptr vs cobs、quest16 vs quest16+g2f、逐任务、逐长度、逐预算、其它模型与 LongBench / InfiniteBench 的全部结果。
- 所列区间都是单项、未作多重性调整的区间。若某层把主格作为探索而非正式头条，须在 §8 写明。

### 3.5 功效（冻结前计算；theory/power_v14.py，results/power_v14_dev.log）
情景：“增益只出现在有单元的任务上”。单元类别用 Llama-3.1 tokenizer（unsloth 镜像，仅用于功效估计，未绑定身份）在 13 任务 × 30 条开发数据上实测：
kv 单元 8–13 token（Qwen 为 12–17），repair 占比 π 为 0.13–0.33。逐单元模型参数取自本机 C3：repair 单元 +1 的概率为 g、−1 为 0.01；in0 单元 +0.015 / −0.025；
无单元任务只贡献对称噪声。检验用分层符号翻转的正态近似。

| 跨界修复概率 g | 主比较 E[Δ̄]（13 任务） | 主比较功效 α = 0.05 | 可覆盖子集 E[Δ] | 子集功效 α = 0.025（Holm 最严一步） |
|---|---|---|---|---|
| 0.40（本机 1.5B 的值） | +5.3 分 | 1.00 | +9.1 分 | 1.00 |
| 0.20 | +2.4 分 | 0.97 | +4.1 分 | 0.99 |
| 0.10 | +0.9 分 | 0.36 | +1.5 分 | 0.43 |

**最小可检出 g（功效 0.8；results/power_v14_mdg.log）**：主比较（α = 0.05）约 0.15（宏平均约 +1.7 分）；可覆盖子集（α = 0.025）约 0.14（约 +2.4 分）。
8B 模型在跨界单元上的本底失败若显著低于 1.5B（G 小），主比较的功效会明显不足（g = 0.10 时约 0.36）。这一点在此写明；**不在看到结果后增加样本**。
即便主比较不显著，P-gain 仍按登记报告 κ：G 小导致 Δ 小，这正是增益律的预测。

### 3.6 失败
中断按同一命令续跑；不删样本、不替补；缺样本则该格报告不完整，主比较要求主格 13×100 条全部有效。

## 4. 分析与汇总（集群上，仓库根目录）

1. `cluster/aggregate.py`（v4）：先过门控，依次核对 RULER 源码哈希、冻结数据与 uid、预测清单与协议、逐行答案/输入/token 哈希、预算计数；
   传 `--units` 时再核对单元长度文件。门控通过后，用官方公式算未舍入的逐题得分，标注角色（PRIMARY / SECONDARY / DESCRIPTIVE）并做统计。
   主比较：`--arm-a cobs+g2f --arm-b cobs --units FILE --json primary.json`（同时打印 P-cov 分组）；
   Holm 成员：`--arm-a nf_pt1+g2f --arm-b nf_pt1 --json s1.json`；`--arm-a cobs+g2f --arm-b cobs --units FILE --subset coverable --json s2.json`。
2. `cluster/holm.py s1.json s2.json`：两项齐全、身份一致时给出 Holm 决策。
3. `cluster/gain_law.py --units FILE --agg <cobs、nf_pt1、quest32 三对的 aggregate JSON> ...`：P-gain 与 P-asym。
4. 所有对比按同一张表报告：读取 token、摘要量（网格数 × 向量数 / 块长）、π、G、R、Δ。

## 5. 运行命令
见 docs/CLUSTER_GUIDE_ZH.md：`cluster/make_jobs.py --tier N ... --dry_run` 查看作业，生成的脚本依次完成数据准备、单元长度冻结与预测。

## 6. 与本机工作的关系
本机只做开发与机制层（1.5B、16k、受控 NIAH、prereg v3、机制诊断、事后诊断）。头条数字只来自本计划在集群上的官方基准运行。

## 7. 交付包缺口（推送到 GitHub 前）
- [x] `src/ruler_pred.py` 官方 RULER 预测适配器；本机 1.5B 适配验收已运行（严格判定差 1 条 bf16 平局，§3.2）
- [x] §3.2 的平局规则（cluster/accept_tie_diag.py、accept_ties.py；审查方批次 3 确认）
- [x] G3 布局匹配（`--layout-unit 32`；theory/check_g3_layout.py：B16/U32 与 B32 的常驻区、候选区、读取数逐一相同；默认路径与旧版逐位一致）
- [x] 作业生成器、续跑、单元长度冻结；汇总与门控（v4）、Holm（两项）、P-gain（gain_law）及其测试
- [ ] 分隔符分块、RBS-decode 两个适配基线；LongBench v1 / Quest 官方入口的对照脚本
- [ ] 各层显存的实测核算（24GB / 80GB 上限目前只是目标）

## 8. 选择记录（集群 agent 启动前填写，结果出来后不得修改）

| 集群 | 层 | 模型 revision | 长度 | 每任务样本 | 预算 | 单元长度文件 sha256 | 主格用途（正式 / 探索） | 登记时间 |
|---|---|---|---|---|---|---|---|---|

## 修订记录

- v1.4（2026-10-16）：依第三方审查批次 1（T1–T7）与批次 2（U1–U3），冻结前、无任何集群数据时修订：
  - 主比较改为 COBS 式+g2f vs COBS 式（B = 32），依据见 §3.4；
  - Holm 族改为两项：nf_pt1±g2f 与可覆盖子集上的 COBS 式±g2f；em vs quest32、cobs+ptr 移到描述层；quest32±g2f 改作 P-gain 的大小预测；
  - 新增单元长度冻结（cluster/unit_lengths.py v3）、P-gain / P-cov / P-asym / P-grid 登记、功效表与最小可检出 g（§3.5）、数值平局规则（§3.2）；
  - G3 布局匹配（`--layout-unit 32`）；增加臂 cobs16、nf_pt1@16（P-grid）；层 3 改为跑完整主格 7 个臂；
  - holm.py 要求成员带非空身份与数据指纹（T6）；依赖补充 requests / scipy。
- v1.3（2026-10-16）：依 Codex 交付包第一批审查（A1–A5、D1）：主格样本数按完整 (模型, 长度, 预算) 识别；次要族改为同块长的 EM vs Quest-32；
  Holm 只在完整家族时计算；汇总门控与角色标注；输出目录带层与样本数标签。
- v1.2（2026-10-16）：项目定名 TWIN-GRID；机器可读副本 cluster/plan_v1.json 与作业生成器 cluster/make_jobs.py。
- v1.1（2026-10-16）：依 Codex G3/P5：跨块长比较的匹配条件与探索性限定；次要族统一为完整家族；重复运行按 ID 去重。
- v1（2026-10-16）：依 Codex P1–P4：A 改为官方任务得分；定主比较与次要族；三层共用同一主格；臂表写明块长与网格；适配验收改为相同输入对照。
- v0（2026-10-16）：初稿。
