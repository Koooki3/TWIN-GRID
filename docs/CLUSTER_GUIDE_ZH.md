# TWIN-GRID 集群运行指南（v1，2026-10-16，对应计划 v1.4）

> **状态更新（2026-10-19）**：v1.4 的主比较已被 rep2 结论取代（见 README“结论”），不再执行。**v2 已冻结，请按 docs/CLUSTER_GUIDE_V2_ZH.md 运行。** 下文保留 v1.4 的流程供追溯；v2 指南引用了本文 §1（环境）与 §3（RULER）。

> 面向在显卡集群上执行实验的 agent 或研究者。**项目名 TWIN-GRID**（错位双网格候选选择；Two offset candidate grids for
> budget-matched sparse attention）= 当前实现 g2_fill_v1 = 实验臂后缀 `+g2f`。实验矩阵、主比较与统计规则以
> [EXPERIMENT_PLANS_ZH.md](EXPERIMENT_PLANS_ZH.md) 为准，机器可读副本是 `cluster/plan_v1.json`。
> **本指南不授权投稿。**

## 0. 已验证与未验证（先读）

| 组件 | 状态 |
|---|---|
| 块选择与 TWIN-GRID 实现（`src/sparse_attn.py`） | 本机 Qwen2.5-1.5B、16k、本地 NIAH 上经预注册确认（C1：COBS 式 81.3% → 93.7%） |
| 官方 RULER 数据准备（`scripts/ruler_prepare.sh` + 准备清单） | 本机 16k 全部 13 任务通过（Qwen 与 Llama-3.1 两种 tokenizer，各 30 条） |
| 单元长度冻结（`cluster/unit_lengths.py` v3） | 本机两种 tokenizer 上运行通过；正式文件须用主格模型自身的 tokenizer 生成 |
| 预测适配器（`src/ruler_pred.py`） | 本机 1.5B GPU 验收：顺序 / 缓存隔离 4/4 一致；相同输入 14/15 一致，严格判定 NOT ACCEPTED；不一致的 1 条诊断与 bf16 平局相符，但 15 行时平局上限为 0，按平局规则仍为 NOT ACCEPTED（计划 §3.2）。集群上须按 65 行重新验收 |
| G3 布局匹配（`--layout-unit 32`） | CPU 检查通过（theory/check_g3_layout.py）；cobs16 / nf_pt1@16 未在 GPU 上运行 |
| 作业生成器（`cluster/make_jobs.py`） | 三层 dry-run 通过；未在集群执行 |
| 汇总与统计（`cluster/aggregate.py` v4、`holm.py`、`gain_law.py`） | 合成数据端到端检查通过（官方计分函数、门控负例、子集与单元文件绑定、增益律自检） |
| Llama-3.1-8B 上的块选择 | **未测**（本机只测过 Qwen2.5-1.5B） |
| 适配基线 RBS-decode | **未实现**，层 1 跳过（生成器提示） |
| 分隔符分块（主要竞争方案） | 开发实现 src/delim_attn.py（不在冻结的作业生成器里）；理论比较见 docs/THEORY_ADDENDUM_ZH.md |
| 显存上限（24GB / 80GB） | 目标值，**未实测** |

在第 0 节里标为未验证的项目全部完成之前，集群结果只能作为探索，不能当作头条结论。

## 1. 环境

本机开发环境（照此复现，版本不同须在 §8 记录）：Python 3.12；torch 2.11.0+cu128；transformers 5.19.0；numpy 2.5.4；
scipy 1.18.1；pyyaml 6.0.3。

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install transformers==5.19.0 numpy==2.5.4 scipy==1.18.1 pyyaml==6.0.3 requests   # requests：官方 HF wrapper 需要
```

RULER 数据准备用单独的轻量环境（只需分词器，不需要 torch）：

```bash
python3 -m venv ruler-venv
ruler-venv/bin/pip install transformers tokenizers nltk wonderwords pyyaml html2text beautifulsoup4 tqdm numpy tenacity scipy
# scipy 不能省：fwe 的生成脚本 import scipy.special；缺了时 prepare.py 不报错，只是 fwe 文件为空（准备脚本的行数检查会拦下）
```

## 2. 模型

| 键 | Hugging Face ID | 模板 |
|---|---|---|
| `llama3.1-8b` | `meta-llama/Llama-3.1-8B-Instruct`（需在 HF 申请访问） | `meta-llama3` |
| `qwen2.5-7b` | `Qwen/Qwen2.5-7B-Instruct` | `qwen2.5-chat` |

下载到 `$MODELS/<键>/`（目录名必须等于键，生成器据此找模型）。记录每个模型的 revision（HF commit）到 §8。

## 3. 官方 RULER（固定版本 + 唯一补丁）

```bash
git clone https://github.com/NVIDIA/RULER.git $RULER && cd $RULER
git checkout c3f5e3b4f87f97e048793bb510a3a6b19a46bf3a
git apply <本仓库>/docs/ruler_template_qwen.diff      # 只给 template.py 增加 qwen2.5-chat
git lfs install && git lfs pull                        # 必须：json 文件由 Git LFS 管理（cwe 用的 english_words.json）
cd scripts/data/synthetic/json
<ruler-venv>/bin/python download_paulgraham_essay.py
bash download_qa_dataset.sh       # HotpotQA 的 CMU 地址若不可达，脚本会用官方写好的 Hugging Face 备用地址
```

检查 `scripts/data/synthetic/json/english_words.json` 约 8.6MB、sha256 为
`affcd6d45fdf3cc843d585c99c97ad615094e760e6c4756b654bab6c73bc2eca`。如果只有 132 字节，说明拿到的是 LFS 指针，cwe 的数据准备会失败。本机就遇到过这种情况：官方 prepare.py 吞掉了子进程的报错，是我们的准备脚本靠行数检查发现的。

为了与历史结果可比，我们固定使用 RULER v1 的旧 scripts 流程（README 已将其标为 deprecated）。不要把 v1 和 v2 的结果放在同一张表里。

## 4. 选层（由集群 agent 判断）

1. 看 `cluster/plan_v1.json` 各层的 `fits`：层 3 需单卡 24–48GB，层 2 需单卡 80GB，层 1 需 4 张以上 80GB。
2. 先在所选层的最长长度上运行显存探针，确认显存够用（留 ≥ 10% 余量；cwe/fwe 生成 120 token，实际可能略高）：
   `python cluster/probe_memory.py --model $MODELS/llama3.1-8b --length 16384 --budget 1024 --methods dense,cobs,cobs+g2f,nf_pt1,quest --json probe.json`
   （本机 Qwen2.5-1.5B @16k：峰值 reserved 约 3.9 GiB，其中权重 2.9 GiB；各方法差别 < 0.1 GiB，results/probe_memory_qwen1.5b_16k.json。
   粗估 Llama-3.1-8B：权重约 15 GiB，KV 约 2 GiB/16k token，128k 时 KV 约 16 GiB。）
   **探针会低估实际峰值**：本机同一模型在真实 RULER 数据上（niah_multivalue / multiquery，4 个方法臂同进程）超过了 4.8 GiB 上限而 OOM，
   探针只报 3.9 GiB（探针只解码几步，摘要缓存与真实提示不同）。建议所有作业设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，
   并按探针峰值的 1.5 倍估算所需显存。
   实测例子：Qwen2.5-1.5B、16k、读取预算 512、niah_multivalue 上，nf_pt1+g2f 即使只与 nf_pt1 同进程，也已分配 ≥ 4.65 GiB（探针 3.9 GiB，约 1.2 倍），
   在 4.8 GiB 上限下 OOM。规避 OOM 只允许不改变语义的调整（单方法单进程、内存分配器设置），不改协议字段。
3. 不确定时选低一层。三层共享同一主格（Llama-3.1-8B、16k、读取预算 1024、每任务 100 条），低层结果可以并入高层。
4. 启动前把选择写进计划文档 §8（集群、层、模型 revision、长度、样本数、预算、主格用途、时间），结果出来后不得修改。

## 5. 运行

```bash
python cluster/make_jobs.py --tier 3 --models-dir $MODELS --out-root $OUT \
   --project-py $(which python) --ruler-root $RULER --ruler-venv <ruler-venv> --dry_run     # 先看规模
python cluster/make_jobs.py --tier 3 ... > jobs_tier3.sh                                       # 审阅后执行
bash jobs_tier3.sh
```

每个 (模型, 长度) 先准备 500 条/任务并冻结（准备清单记录配置与文件哈希，同目录换配置会被拒绝），紧接着用该模型的 tokenizer 写出
单元长度文件 `unit_lengths_b<B>.json`（在任何预测之前；把它的 sha256 填进计划 §8）。预测只用冻结文件的行号 0..n−1
（`sample_uid` = 文件哈希:任务:行号）。同一进程内跑共用块长的所有方法臂，以复用 prefill；每个方法臂开始时清空摘要缓存。

**适配验收（每个模型首次运行前）**：在 13 个任务、每任务 5 条固定样本上运行 `cluster/accept_adapter.py`（流程参照 `scripts/queue_accept.sh`）。
对每条不一致的输出运行 `cluster/accept_tie_diag.py MODEL DATA_JSONL LINE ADAPTER_PRED_JSONL MAX_NEW OUT.json`，
再运行 `cluster/accept_ties.py REPORT.json OUT*.json`：只有 ACCEPTED 或 ACCEPTED_WITH_TIES 才可继续（计划 §3.2 条件 a–d）。
一次性 prefill 16k 时，若显存紧张，可设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

**冒烟（必须先做）**：`--tier 3` 生成的第一个任务上加 `--limit 5`，跑 dense、cobs、cobs+g2f，确认：
输出行的 `tokens` 计数无违规；dense 的答案在合理范围；`manifest.<task>.json` 生成。

## 6. 汇总与主比较

```bash
U=$OUT/data/llama3.1-8b/16384/unit_lengths_b32.json
A="--ruler $RULER --out-root $OUT --model llama3.1-8b --length 16384 --budget 1024 --n 100 --run-tag t3_n100"
python cluster/aggregate.py $A --arm-a cobs+g2f --arm-b cobs --units $U --json primary.json           # 主比较 + P-cov 分组
python cluster/aggregate.py $A --arm-a nf_pt1+g2f --arm-b nf_pt1 --json s1.json                        # Holm 成员 1
python cluster/aggregate.py $A --arm-a cobs+g2f --arm-b cobs --units $U --subset coverable --json s2.json   # Holm 成员 2
python cluster/holm.py s1.json s2.json                                   # 缺项、重复或身份不一致时退出码 3、不计算 Holm
python cluster/aggregate.py $A --arm-a quest32+g2f --arm-b quest32 --json q32.json                     # 描述（P-gain 用）
python cluster/gain_law.py --out-root $OUT --model llama3.1-8b --length 16384 --budget 1024 --n 100 \
   --run-tag t3_n100 --units $U --agg primary.json --agg s1.json --agg q32.json --json gain_law.json  # P-gain、P-asym
```

脚本先过门控：13 个任务都要齐全，样本 ID 恰为冻结的 0..n−1，两个方法臂的输入哈希一致，预算计数没有违规；任何一项不满足就以退出码 2 结束。门控通过后再计算：
- 宏平均差 D（13 个任务等权）；
- 任务内分层的符号翻转置换检验 p（10 万次）；
- 分层 bootstrap 95% 区间。

主比较固定为 `cobs+g2f` vs `cobs`（B = 32）。次要族（Holm）恰好两项：`nf_pt1+g2f vs nf_pt1`，以及可覆盖任务子集上的
`cobs+g2f vs cobs`（子集由单元长度文件机械判定，aggregate 会核对它与数据、tokenizer、块长是否绑定）。脚本会给每个比较标注角色：
- **PRIMARY / SECONDARY：** 只在主格上，且两个方法臂块长相同时给出。
- **DESCRIPTIVE：** 其余情况，只给估计值和区间，不给 p 值。

**v1.4 附录（描述性）**：主格上的 mean、mean+g2f、delim_cobs 对照见 docs/PLAN_V14_APPENDIX_ZH.md（`cluster/make_appendix_jobs.py` 生成作业，
`cluster/aggregate_appendix.py` 汇总，全部 DESCRIPTIVE），在主作业的数据准备之后运行。

逐任务结果、P-cov 分组、`gain_law.py` 的 κ / R̂ / P-asym 都只作描述（登记的描述性预测，计划 §3.4）。`--run-tag` 对应作业生成器的
输出目录标签 `t<层>_n<样本数>`。所有对比按同一张表报告：读取 token、摘要量、π、G、R、Δ。

## 7. 记录与续跑

- 每个作业保留原生日志与退出码；中断后用同一命令续跑，因为样本 ID、清单、代码哈希不一致时会拒绝。
- 不删样本、不替补；某格缺样本就报告不完整。主比较要求主格 13×100 条全部有效。
- `--strict-budget` 下逐调用预算断言失败属于实现缺陷：停止运行并报告，不要重跑。
- 汇报时附上：`plan_v1.json` 与本仓库提交号、模型 revision、RULER commit、数据准备清单哈希、汇总脚本输出。

## 8. 结果解读边界（与本机结论一致）

- C1–C3 只能在确认时的代码上复现：`git show e194213:src/sparse_attn.py`（sha256 c9b56b7f…，与 theory/phase2_spec.json 一致）；HEAD 的 sparse_attn.py（1fd33e6b…）已加入布局单位 U。冻结文件（计划 v1.4、THEORY、sparse_attn 的 docstring）中“逐位一致 / bit for bit”只指 U = B 时块选择、读取预算、常驻区与候选数在 check_g2f、check_fixed_layout、check_invariant 中与旧版相同，不是模型输出逐比特相同（审查方批次 4）。

- 每次候选读取预算相同，不等于全程读取量或显存相同；双网格的摘要存储约为单网格 2 倍（结构性口径，非实测）。
- 掩码模拟只测准确率，不作速度主张。
- 本机已经确认的是 C1、C2，它们是预注册的确认性结果。C3 只是数据收集过程中登记的描述性预测，不是对因果机制的确认（见研究文档第七轮）。
- 增益律 Δ = π·G·R + … 本身是恒等式；可检验的只是三条：G 的顺序、范围内 R ≈ 1、范围外 R ≈ 0（docs/THEORY_ZH.md §4）。G 小导致的小增益是预测，不是失败。
- 集群结果是在官方基准和其它模型上做的新检验，不论正负都按预注册规则如实报告。`plan_v1.json` 里方法臂的 `ready` 只表示代码已实现，不表示已在目标模型或官方基线上验收过。

## 9. 待补（推送前）

- [ ] 分隔符分块的集群描述性附录（独立生成脚本；本机开发对照之后）
- [ ] LongBench v1 入口（先选定要对照的 Quest 表；排在集群主格之后）
- [ ] LongBench v1（Quest 子集）对照入口
