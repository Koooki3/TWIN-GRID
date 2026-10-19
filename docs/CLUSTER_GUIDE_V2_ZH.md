# TWIN-GRID 集群运行指南 v2（对应冻结计划 v2，2026-10-19）

> 面向在显卡集群上执行 v2 的 agent 或研究者。实验矩阵、检验与判定以冻结计划 [PLAN_V2_COGNITIVE_ZH.md](PLAN_V2_COGNITIVE_ZH.md) 为准，
> 机器可读副本是 `cluster/plan_v2.json`；结果的措辞以 [RESULT_SCENARIOS_V2_ZH.md](RESULT_SCENARIOS_V2_ZH.md) 为准。
> v1.4 的指南（CLUSTER_GUIDE_ZH.md）保留供追溯，其主比较不再执行。**本指南不授权投稿或对外发布结果。**

## 0. 先读：已验证与运行时才能验证的

| 组件 | 状态 |
|---|---|
| 块选择实现与 ★ 臂（cobs、cobs+g2f、cobs16、quest32、quest16+g2f、quest8、dense） | 本机 Qwen2.5-1.5B 在 rep2 中全部 GPU 运行过；8B 上未测 |
| 统计（src/v2_stats.py）、门控与汇总（cluster/aggregate_v2.py）、运行描述（cluster/v2_run_descriptor.py）、作业生成器（cluster/make_jobs_v2.py） | CPU 合成测试通过（统计 36 项、端到端 73 项），Codex 独立复核 I1–I5 全部关闭 |
| 显存是否够用 | **运行时验证**：prep 阶段的探测与收据，余量不足时 register 拒绝 |
| 8B tokenizer 下的单元类别（含 UUID 任务） | **运行时验证**：prep 阶段冻结 n = 200 的单元文件 |
| 适配器在 8B 上的验收 | **运行时验证**：prep 阶段在主冻结文件的保留行 200..204 上做（每任务 5 行，按内容与主样本、校准样本不重叠），结论必须是 ACCEPTED 或 ACCEPTED_WITH_TIES |

## 1. 环境、模型与 RULER

- 环境与依赖：同 CLUSTER_GUIDE_ZH.md §1（Python 3.12，torch 2.11.0+cu128，transformers 5.19.0，numpy 2.5.4，scipy 1.18.1，pyyaml 6.0.3，requests；
  RULER 数据准备另建轻量的 ruler-venv，scipy 不能省）。版本不同须写进偏差记录。
- 仓库：使用公开仓库 main 的固定提交，先核对冻结文件：
  `sha256sum -c results/freeze_v2_sha256.txt && sha256sum -c results/freeze_v14_sha256.txt`（都必须全部 OK），再跑 CPU 检查：
  `RULER_ROOT=$RULER bash scripts/check_all.sh $(which python)`（rc_all 必须为 0；端到端测试用 `RULER_ROOT` 指向固定版本的 RULER checkout）。
- 模型：下载到 `$MODELS/<键>/`（目录名必须等于键），并记下 **40 位 HF commit**（例如 `huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --revision <commit>`）。
  键与模板见 plan_v2.json 的 models。
- RULER：同 CLUSTER_GUIDE_ZH.md §3（commit `c3f5e3b`，只打 docs/ruler_template_qwen.diff，`git lfs pull`，下载 essay 与 QA 数据，核对 english_words.json 的 sha256）。

## 2. 选层（只依据硬件与 prep 结果，不依据任何主格结果）

1. 按 plan_v2.json tiers 的 `fits` 初选：层 3 单卡 24–48GB，层 2 单卡 80GB，层 1 ≥ 4×80GB。不确定时选低一层。
2. 跑该层的 prep（§3）。若 register 因显存余量、适配验收或任何收据拒绝，**不要改文件重试**：换一个新的 `$OUT` 目录作为新的尝试；
   显存不够就降一层。这些都发生在任何主格预测计分之前，所以允许。
3. register 成功后，把运行描述的 sha256 与摘要填进 docs/PLAN_V2_SELECTION_RECORD_ZH.md（不要改冻结计划本身）。之后不得换层。

## 3. prep（每个模型一次）

```bash
python cluster/make_jobs_v2.py --phase prep --tier 3 --model llama3.1-8b --revision <40位commit> \
   --models-dir $MODELS --out-root $OUT --project-py $(which python) --ruler-root $RULER --ruler-venv <ruler-venv> > prep.sh
less prep.sh      # 审阅
bash prep.sh
```

prep.sh 依次完成（任何一步失败都会停下）：
1. 每个长度：准备 500 条/任务并冻结（准备清单记录配置与文件哈希）；用该模型自己的 tokenizer 写 n = 200 的单元文件 `unit_lengths_b32_n200.json`。
2. 主格长度下，每种块长（8、16、32）运行显存探测，并立刻写探测收据（完整权重哈希、代码与参数）。
3. 适配验收：
   - `accept-data` 从主冻结文件逐字取出每个任务的保留行 200..204 作为验收数据（不在主样本 0..99 与校准样本 100..199 中，按内容检查不重叠，不计入任何研究分母）；
   - 跑 13 个任务的 dense，以及 cwe 上两组同族方法（cobs/cobs+g2f、quest/quest+g2f）的 ab、ba、single 运行（`--local 16`）；
   - 经 `eval-run` 运行 accept_adapter.py 生成报告；
   - `tie-plan` 列出每条不一致的输出，逐条经 `eval-run` 运行 accept_tie_diag.py；
   - 经 `eval-run` 运行 accept_ties.py 得到结论；
   - `accept-receipt` 写验收收据。收据不会被覆盖：要重新验收，只能换一个新的 `$OUT`。
4. `register`：重新推出验收结论、核对全部收据与完整哈希，写出 `$OUT/run_descriptor_t<层>_<模型>.json`。

适配验收只有 ACCEPTED 或 ACCEPTED_WITH_TIES 才能继续（v1.4 §3.2 的平局规则：每任务至多 1 条，总数不超过 2%）。
NOT ACCEPTED 时停止该模型，如实报告，**不调参追数**。

## 4. predict

```bash
python cluster/make_jobs_v2.py --phase predict --tier 3 --rd $OUT/run_descriptor_t3_llama3.1-8b.json \
   --models-dir $MODELS --out-root $OUT --project-py $(which python) --ruler-root $RULER > predict.sh
bash predict.sh
```

- 第一条命令 `verify` 会重新哈希全部权重、数据、单元文件、计划与代码，有任何差异就退出；每条预测之后都有 `bind`，绑定 manifest 与预测文件。
- 规模：层 3 合计 10900 次 16k 预测（7 臂 × 13 任务 × 100 + 2 个校准臂 × 9 任务 × 100）；层 2、层 1 的规模见计划 §1。
- 中断：重新运行同一个 predict.sh（ruler_pred 按 sample_uid 续跑，verify 与 bind 会重新执行）。不删样本、不替补。
- OOM：只允许不改语义的调整（`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`、单方法单进程），并记入偏差；不改协议字段，不换层。
- `--strict-budget` 下的预算断言失败属于实现缺陷：停下并报告，不要重跑。

## 5. 汇总

```bash
python cluster/aggregate_v2.py --ruler $RULER --out-root $OUT --tier 3 \
   --units $OUT/data/llama3.1-8b/16384/unit_lengths_b32_n200.json \
   --run-descriptor $OUT/run_descriptor_t3_llama3.1-8b.json --json v2_result.json
```

- 不要加 `--test-params`（那只用于合成测试，会把输出标为非登记参数）。
- 门控失败时退出码为 2，不会计算任何统计量，按“无法判定（不完整）”报告。
- 输出里 `formal_v2_result` 必须为 true，才是正式 v2 结果（计划为 FROZEN、运行描述在 FROZEN 计划下登记、使用登记参数）。
- 本脚本只汇总主格（Llama-3.1-8B / 16k / 预算 1024）。辅助格（其它长度、预算 2048、Qwen2.5-7B、mean 两臂）的预测与绑定照常保留，
  描述性汇总在主格结论之后另做，不影响任何判定。

## 5.1 规划风险：功效按预算 512 估算，主格是预算 1024（审查方批次 16，K1；不改冻结计划）

冻结计划 §3.5 的功效近似来自 rep2 的预算 512。审查方用 rep2 的 R7 数据（1.5B、预算 1024、第 20..59 行，其 checks/g_at_1024.py）核对：
- G_cobs 从 +0.205 降到 +0.111；
- H3 的类比量 G_cobs − G_quest32 从 +0.178 降到 +0.141 [+0.066, +0.213]；
- H1 的类比量从 +0.088 降到 +0.059。

即预算翻倍后效应缩小约 20–33%，仍在 §3.5 “效应减半”的敏感性情景之内，不阻碍运行。H2 没有预算 1024 的数据；它在 512 时只有 +0.024，在 1024 下可能更难检出。
这些是 1.5B 上的事后描述，**不是对 8B 的预测**，也不改任何检验、阈值或判定；报告结果时把它作为功效风险写明，不显著时不推出“效应不存在”。

## 6. 交付

按 RESULT_SCENARIOS_V2_ZH.md 的措辞与必报项写结果，附上：
- v2_result.json、运行描述、PLAN_V2_SELECTION_RECORD_ZH.md 的记录行；
- prep.sh、predict.sh 及其日志与退出码；验收报告、结论、平局诊断与各类收据；
- 偏差记录（中断、续跑、OOM、环境版本差异）。
不论结果正负，都按冻结规则如实报告。
