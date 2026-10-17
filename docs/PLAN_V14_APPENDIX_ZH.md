# 计划 v1.4 附录（描述性，不改检验定义；2026-10-17；审查方批次 11，F2）

> 冻结的计划 v1.4（docs/EXPERIMENT_PLANS_ZH.md、cluster/plan_v1.json、cluster/make_jobs.py 等 13 个文件）**不改**。本附录只为主格增加几组描述性对照，
> 让集群也能复现本机开发结论（docs/THEORY_ADDENDUM_ZH.md 附录 A4）。附录的任何读数都不进入主比较或 Holm 族，也不改变它们的判定。
> 冻结计划 §2 把分隔符分块列为“未实现”，那是冻结时的状态；本附录中它以 delim_cobs 的形式实现（src/delim_attn.py，开发实现）。

## 附录臂（只在主格：Llama-3.1-8B-Instruct、16k、预算 1024、每任务 100 条；B = 32，U = 32，协议旗标同冻结计划）

| 组 | 运行器 | 臂 | 用途 |
|---|---|---|---|
| sum | src/ruler_pred.py | mean、mean+g2f、quest | 摘要类型对比：mean+g2f vs quest32（每 token 8 个 float，等内存）；mean vs mean+g2f（固定块长下 TWIN-GRID 对一阶摘要） |
| delim | src/delim_dev.py | cobs、cobs+g2f、delim_cobs | 分隔符分块 vs 单网格、vs TWIN-GRID |

已在冻结计划中的对照（层 2 描述臂，按冻结的 make_jobs 生成）：quest16、quest32、quest32+g2f、cobs16、nf_pt1@16。
不进集群：nb、nbr、nbh（开发数据上在所有分组都比不扩展更差）。

## 运行与汇总
```bash
python cluster/make_appendix_jobs.py --tier 3 --models-dir $MODELS --out-root $OUT --project-py $(which python) \
   --ruler-root $RULER --dry_run          # 26 个作业（13 任务 × 2 组）
python cluster/make_appendix_jobs.py --tier 3 ... > jobs_appendix.sh && bash jobs_appendix.sh   # 主作业的数据准备之后运行
A="--ruler $RULER --out-root $OUT --model llama3.1-8b --length 16384 --budget 1024 --n 100"
python cluster/aggregate_appendix.py $A --arm-a mean+g2f --arm-b quest32 --run-tag t3_n100_appendix-sum
python cluster/aggregate_appendix.py $A --arm-a mean+g2f --arm-b mean --run-tag t3_n100_appendix-sum
python cluster/aggregate_appendix.py $A --arm-a delim_cobs --arm-b cobs+g2f --run-tag t3_n100_appendix-delim
python cluster/aggregate_appendix.py $A --arm-a delim_cobs --arm-b cobs --run-tag t3_n100_appendix-delim
```
aggregate_appendix.py 原样调用冻结的 aggregate.py（同样的门控与官方计分），只在内存中补上附录臂、清空主比较与 Holm 族，所以全部标为 DESCRIPTIVE、没有检验；
plan_v1.json 不被修改（theory/check_appendix.py 验证）。比较只在同一组（同一 run tag）内进行。

## 报告
附录结果与冻结计划的描述臂放在同一张统一对比表（读取 token、摘要量、π、G、R、Δ），并按 docs/RESULT_SCENARIOS_ZH.md 的必报内容报告。
