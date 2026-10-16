# 稀疏注意力块/页/token 选择与 KV 检索方法（2024–2026，侧重 2025–2026）

> 核验说明：标注【全文】表示已打开 arXiv HTML（或 PDF 文本）的方法/实验部分；【摘要】表示只通过 arXiv API 读到摘要，方法细节来自摘要原文，未逐节复核；【全文·部分】表示方法节已读，但个别数字由抽取模型转述，标了"需复核"。检索日期 2026-10-16。
>
> 我方发现（对照用）：冻结 Qwen2.5-1.5B，本地 RULER 式 needle，16k/32k，读预算 512/1024 token。(a) 矩型摘要（均值/投影协方差，COBS-style）与盒型/极值摘要（Quest 逐通道 min/max）错误互补，取并集有益；(b) 7 位答案值跨越固定 32-token 块边界时矩型打分失败，盒型较少失败；(c) 两套相差半块的块网格联合选择（总预算精确不变）能救回矩型打分的这些失败；(d) 等摘要内存下，16-token 页（Quest 默认）胜过更复杂的 32-token 摘要；(e) 离线选择质量指标（attention mass 捕获率、needle 块命中）不能预测端到端准确率。

## 问题 1：各方法的核心选择机制（摘要内容、打分方式、粒度、训练与否、prefill/decode、结果）

### Takeaway
2024–2026 的块选择方法大体分成四类：(1) 一阶/均值摘要（MoBA、ShadowKV landmark、InfLLM-V2 mean-pool、Tactic 聚类质心、CEDAR 均值）；(2) 极值/包围体摘要（Quest 的 min/max 盒、ArkVale 的球/长方体）；(3) 二阶或"均值+离散度"摘要（COBS 的低秩协方差、RBS-Attention 的质心+最大半径、CEDAR 的均值+半径误差界）；(4) 非摘要式（直接估计注意力：XAttention 反对角、FlexPrefill 采样、Loki/DoubleSparsity 降维 key、RetrievalAttention/MagicPIG/HashAttention 向量检索或 LSH、Twilight/Tactic/vAttention 的自适应预算）。2026 年的新工作（COBS、RBS-Attention、CEDAR）开始从理论上解释"均值摘要为什么漏检"，三者都把块内离散度（协方差或半径）加进打分。

### Cited Findings

**A. 与我方最相关的 2026 新工作**

- **COBS（Cumulant Order Block Sparse Attention），arXiv 2607.09052，2026-07【全文】**
  - 摘要内容：每块存 bf16 块均值 key $\bar k_b$，再加一个低秩（主配置 r=4）块内协方差 $\Sigma_b$；协方差特征向量投影到 query 子空间（s≈85 维）后用 FP4 存储。— [arXiv HTML](https://arxiv.org/html/2607.09052v1)
  - 打分：$\text{score}_b=\sum_g \frac{1}{\hat Z}\exp(q^\top\bar k_b+\tfrac12 q^\top\Sigma_b q)$，即对块 attention mass 做二阶 cumulant 展开。论文认为已有选择器（mean-pool、NSA MLP、Quest）"只到一阶近似"。— [arXiv HTML](https://arxiv.org/html/2607.09052v1)
  - 粒度与设置：在 NSA 框架内替换选择分支；选择块长 L=32，decode 时取 top-16 块，滑窗 256；模型约 1.2B（16 层、4 个 KV head），用 LongCrawl64 约 20B token 在 4k 上预训练，再用 YaRN 扩到 32k，并在生成的 RULER 任务上做 SFT。注意：这是自训练的 NSA 模型，不是冻结的现成模型。子空间投影和量化不更新权重，属于后处理。仅用于 decode。— [arXiv HTML](https://arxiv.org/html/2607.09052v1)
  - 结果（32k RULER 平均分）：Dense 0.9040；mass oracle（OSA）0.9010；COBS 0.8195（缩小约 86% 差距）；NSA+Quest(min/max) 0.5765；NSA mean-pool 0.5554；NSA MLP 基线 0.2999。KV 读流量是 NSA 的 1.21 倍，比 dense 少 15.15 倍。秩扫描：r=8 最好（0.8539），r=16 退化到 0.8006，论文归因于无符号方差在多 key 任务中累积高秩噪声。— [arXiv](https://arxiv.org/abs/2607.09052)、[HTML](https://arxiv.org/html/2607.09052v1)
  - 边界、并集与误差界：块不重叠、连续；没有讨论 needle 跨块边界，也没有把多种摘要做并集（NSA 的压缩分支本身使用重叠窗口，但论文研究的选择分支不重叠）。论文给出了"丢弃块 → 输出误差"的分解式，但没有给二阶截断误差的界。— [arXiv HTML](https://arxiv.org/html/2607.09052v1)

- **RBS-Attention（Radius-Bounded Sparse Prefill），arXiv 2609.20971，2026-09【全文】**
  - 摘要内容：每块存质心 $c_b$ 和最大半径 $r_b=\max_k\|k-c_b\|_2$。论文把"质心掩盖块内单个高相关 token"的失败模式命名为 **mean dilution**。— [arXiv HTML](https://arxiv.org/html/2609.20971v1)
  - 打分：有两个分支。base 分支为 $q^\top c_b$；rescue 分支为 $q^\top c_b+\|q\|_2 r_b\beta_b$，其中 $\beta_b\in[0,1]$ 按每层每头的半径分位数（q0.5 到 q0.9）自适应。理论依据是 Cauchy–Schwarz 上界 $s_b(q)\le q^\top c_b+\|q\|r_b$，但实现中把 β 衰减，避免上界过于保守。两个分支各自做相对阈值（α_base=0.22，α_rescue=0.18），最终 mask 取**并集**，再并上 sink、window 和 last。— [arXiv HTML](https://arxiv.org/html/2609.20971v1)
  - 粒度：默认块长 B=128。块大小消融在约 5.34% 密度下：B=64 为 71.86，B=128 为 80.28，B=256 为 70.73，非单调。训练无关，只用于 prefill。— [arXiv HTML](https://arxiv.org/html/2609.20971v1)
  - 结果：Qwen3-30B-A3B 上 128K RULER 达 80.28%（密度 5.34%），MInference 76.87%，FlashPrefill 74.73%；Qwen3-32B 上 RULER 88.65，dense 为 89.52。prefill attention 独立加速 20.65 倍，TTFT 加速 5.97 倍。消融（等 5.34% 密度，130 例）：仅质心 73.63，完整 L2 上界 72.41，Quest-style 73.71，RBS 自适应并集 76.67。— [arXiv HTML](https://arxiv.org/html/2609.20971v1)
  - 统计观察：仅用均值排序造成的低估随半径增大；半径最大的五分位（Q5）里包含 39.5% 的 top-5% 注意力块。没有讨论跨块边界的 token 或重叠块。— [arXiv HTML](https://arxiv.org/html/2609.20971v1)

- **CEDAR（Error-Bounded Residual Routing），arXiv 2609.07237，2026-09【全文】**
  - 摘要内容：每个语义 chunk 存均值 key、均值 value，以及块内 key/value 离散度（到均值的最大 L2 距离 $r^k_c, r^v_c$，或 RMS×α=2.0）。chunk 边界沿用 DHSA 的边界预测器，变长（目标 64，范围 16–256）。— [arXiv HTML](https://arxiv.org/html/2609.07237v1)
  - 打分与路由：未展开的 chunk 以摘要形式进入残差注意力路径，与精确 token 共享同一个 softmax 归一化，因此 coarse 证据不会被完全丢弃。展开优先级为 $\pi=e^{\ell^{sum}}[(e^{\epsilon}-1)(\|\bar v\|+\rho)+e^{\epsilon}r^v]$，其中 $\epsilon=\|q\|r^k/\tau$。— [arXiv HTML](https://arxiv.org/html/2609.07237v1)
  - 误差界：Proposition 1 给出输出 L2 误差上界，随摘要 mass 和 key/value 离散度增加，并可按未展开的 chunk 逐项相加。训练无关，结果只报告 prefill（decode 只作为次要目标提及）。— [arXiv HTML](https://arxiv.org/html/2609.07237v1)
  - 结果：Llama-3.1-8B-Instruct（4-bit），128K，精确密度约 6.25%。RULER 上 dense 89.3、DHSA(hard) 79.3、CEDAR 87.1，约 3 倍 kernel 加速。消融：每个 chunk 存"两个摘要"时 RULER 升到 87.6，但速度从 3.02 倍降到 2.71 倍；去掉离散度优先级后 RULER 从 87.1 降到 84.9。— [arXiv HTML](https://arxiv.org/html/2609.07237v1)
  - 指标论述：§2.2 "attention recall is not enough"：被省略的 value 与已选部分的均值差异很大时，即使只漏掉少量 mass 也会出问题。没有讨论 chunk 边界切断事实的问题。— [arXiv HTML](https://arxiv.org/html/2609.07237v1)

**B. 极值/包围体摘要**

- **Quest，arXiv 2406.10774（ICML 2024）【全文】**：每页存逐通道 key 的 min $m$ 与 max $M$；打分为 $\sum_i\max(q_im_i,q_iM_i)$。论文明确指出这是页内任意 $q\cdot k$ 的**上界**。默认页大小 16，论文没有页大小消融；预算 256–4096；前两层保持 dense；只用于 decode。passkey 100k 在预算 256 到 4096 下为 88% 到 100%；self-attention 加速 7.03 倍（摘要写的是 2.23 倍 self-attn 加速和 7.03 倍延迟降低，抽取结果与摘要表述不同，以摘要为准）。— [arXiv](https://arxiv.org/abs/2406.10774)、[HTML](https://arxiv.org/html/2406.10774v2)
- **ArkVale（NeurIPS 2024）【全文·PDF 文本抽取】**：页满后把 KV 备份到 CPU，并压缩成"digest"包围体：bounding-sphere（中心取 AABB 中心，加半径标量）或 bounding-cuboid（存 max/min 向量），每种各有 r_max、r_center、r_mean 三种"半径"变体，共 6 种。只用质心的基线连 top-1 页召回都不到 5%，top-4 也不到 50%；6 种包围体方法召回至少 60%；cuboid-mean 的 top-1 召回达 95%，其他 k 下也超过 80%。cuboid 普遍优于 sphere（存两个向量，sphere 只存一个向量加一个标量）；"mean"变体更好，作者认为它避免了高估。默认页大小 32；在紧预算下 ArkVale-16 常优于 ArkVale-32，原因是同预算下能选更多页，更容易命中关键 token。注意：这是**等 token 预算**的比较，不是等摘要内存。— [NeurIPS 2024 PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/cd4b49379efac6e84186a3ffce108c37-Paper-Conference.pdf)、[mlanthology](https://mlanthology.org/neurips/2024/chen2024neurips-arkvale)

**C. 均值/landmark/代表 token 摘要**

- **MoBA，arXiv 2502.13189（Moonshot/Kimi）【全文】**：每块的门控分数为 $q\cdot\text{meanpool}(K_{block})$，取 top-k 块，块大小例如 512 或 4096，因果掩码禁止看未来块。需要训练（可以在训练中途在 full 和 sparse 之间切换，也可以做逐层混合）。128K RULER：MoBA 0.7818，full 0.7849。细粒度块划分明显有利，最粗粒度和较细粒度之间 loss 相差约 1e-2。— [arXiv](https://arxiv.org/abs/2502.13189)、[HTML](https://arxiv.org/html/2502.13189v1)
- **InfLLM，arXiv 2402.04617【全文】**：记忆单元为 128-token 块，每块选 4 个代表 token，依据是该 token 在局部窗口中得到的注意力。相关度为当前 token 与代表 key 点积之和；局部窗口 4K；检索 96 个单元（Mistral）或 32 个单元（Llama-3）；训练无关。消融：代表 token 数从 1 到 4 逐步变好，到 8 略降；块大小 {32,64,128,256} 的最优值随任务而变，作者提出未来做 dynamic segmentation。— [arXiv](https://arxiv.org/abs/2402.04617)、[HTML](https://arxiv.org/html/2402.04617v2)
- **InfLLM-V2，arXiv 2509.24663（MiniCPM4.1）【全文】**：块表示分三阶段：(1) mean-pool，块长 B/2、步长 B/4，**有重叠**；(2) 在 GQA 组内对头求和；(3) max-pool 得到选择分数。选择块 B=64，共 96 块（1 个初始块 + 63 个 top-k + 32 个局部块，约 6k token 可见）。可训练，但复用 dense 的 KV 参数，不增加参数；短输入走 dense，长输入切换到 sparse。32k RULER：sparse 82.6，full 84.3。— [arXiv](https://arxiv.org/abs/2509.24663)、[HTML](https://arxiv.org/html/2509.24663v1)
- **NSA，arXiv 2502.11089（DeepSeek，ACL 2025）【全文】**：压缩分支块长 l=32、步长 d=16，**有重叠**，用带块内位置编码的 MLP 压缩出一个 key。选择分支块长 l′=64，重要性来自压缩 token 的注意力分数；当压缩块和选择块划分不同时，按空间重叠关系聚合；GQA 组内共享；选 16 块（含 1 个初始块和 2 个局部块）。滑窗 512。端到端训练。64k NIAH 全位置满分。— [arXiv](https://arxiv.org/abs/2502.11089)、[HTML](https://arxiv.org/html/2502.11089v2)
- **ShadowKV，arXiv 2410.21465【全文·部分】**：post-RoPE key 按 chunk=8 取均值作为 landmark；prefill 时用 landmark 与块内 key 的余弦相似度找出低相似的"outlier chunk"（约占 token 的 0.2–0.3%），常驻 GPU。pre-RoPE key 做 SVD 低秩（rank 160），value 卸载到 CPU；稀疏预算 1.56%；只用于 decode，训练无关。抽取结果称 Llama-3.1-8B 128K RULER 上 ShadowKV 83.57、Quest 79.72（**需复核**，抽取只覆盖了前 100k 字符）。— [arXiv](https://arxiv.org/abs/2410.21465)、[HTML](https://arxiv.org/html/2410.21465v3)
- **SeerAttention，arXiv 2410.13276【全文】**：可学习的 AttnGate。Q 用 avg-pool；**K 同时做 max、min、avg 池化，三者在隐藏维拼接后再经线性层**，理由是 K 的离群值较多；这在 PG19 上是经验最优配置。块大小 64；用 2D max-pool 的 full attention 图做 KL 自蒸馏，约 0.5B token。只用于 prefill。128k RULER 73.37。— [arXiv](https://arxiv.org/abs/2410.13276)、[HTML](https://arxiv.org/html/2410.13276v4)；decode 版 **SeerAttention-R**（arXiv 2506.08889）去掉 query pooling，块 64/128，AIME 上 4K 预算近无损【摘要】— [arXiv](https://arxiv.org/abs/2506.08889)

**D. 聚类、向量检索与 token 级方法**

- **Tactic，arXiv 2502.12216【全文】**：prefill 时对 key 做 K-means（平均簇大小 32，最多 10 次迭代）；decode 时用 q 与簇质心的点积给 token 排序，再用 $y=a/x+b$ 拟合尾部注意力分布，按**累计注意力比例**而不是固定预算取 token。给出界 $\epsilon\le 2(1-p)\max\|v\|$。论文图 6 指出连续 token 的 key 在空间中分散，以此批评 Quest 的按位置分页。decode attention 加速 7.29 倍，端到端 1.58 倍。— [arXiv](https://arxiv.org/abs/2502.12216)、[HTML](https://arxiv.org/html/2502.12216v1)
- **RetrievalAttention，arXiv 2409.10516【摘要】**：在 CPU 上为 KV 建 ANNS 索引，用注意力感知的向量检索处理 query/key 分布不一致（OOD）问题，只访问 1–3% 的数据；token 级，decode，训练无关。— [arXiv](https://arxiv.org/abs/2409.10516)
- **RetroInfer，arXiv 2505.02922【摘要】**：wave index 包括 tripartite attention approximation、accuracy-bound attention estimation 和 segmented clustering；120K 下解码吞吐为 full 的 4.4 倍。— [arXiv](https://arxiv.org/abs/2505.02922)
- **MagicPIG，arXiv 2410.16179【摘要】**：论文认为 TopK 本身在注意力不够稀疏时会退化，于是改用基于 LSH 的**采样**估计注意力输出（有理论保证），在 CPU 上计算；token 级，decode。— [arXiv](https://arxiv.org/abs/2410.16179)
- **HashAttention，arXiv 2412.14468【摘要】**：用学习到的映射把 q/k 编码到 Hamming 空间，每 token 32 bit 辅助内存，稀疏度可达 16–32 倍；需要训练映射。— [arXiv](https://arxiv.org/abs/2412.14468)
- **Double Sparsity，arXiv 2408.07092【摘要】**：离线校准出重要通道（channel sparsity 较静态），用少数通道近似 $q\cdot k$ 选 token；1/16 token 和通道稀疏；后训练、token 级、decode。— [arXiv](https://arxiv.org/abs/2408.07092)
- **Loki，arXiv 2406.02542【摘要】**：key 位于低维空间，用 PCA 低维下的注意力分数给 token 排序选 top-k；token 级、decode、训练无关。— [arXiv](https://arxiv.org/abs/2406.02542)
- **RocketKV，arXiv 2502.14051【摘要】**：两阶段：先做粗粒度永久驱逐，再做混合稀疏 top-k，同时在 head 维和序列维降维来近似分数。— [arXiv](https://arxiv.org/abs/2502.14051)

**E. 自适应预算与误差保证**

- **Twilight，arXiv 2502.02770【全文】**：先用 Quest 或 DoubleSparsity 以保守预算（约 1/4）选 token，再用 INT4 量化的 K 估计注意力并做 top-p（p=0.85–0.95）剪枝，最多剪掉 98% 的 token；self-attn 比 SOTA 稀疏方法快 1.4 倍。误差界 $(1-p)\|V\|_F$。— [arXiv](https://arxiv.org/abs/2502.02770)、[HTML](https://arxiv.org/html/2502.02770v5)
- **vAttention，arXiv 2510.05688【全文·部分】**：确定性部分（sink + local + 近似 top-k，例如 HashAttention）加上对剩余 token 的随机采样，并用 CLT 按用户给定的 $(\epsilon,\delta)$ 确定样本量。论文观察到 top-k 和采样互补：top-k 擅长少数 token 主导的情况，采样擅长注意力较均匀的情况。RULER-HARD 上约 +4.5 pp（Llama-3.1-8B），在最高 20 倍稀疏下达到 full 质量。token 级、decode。— [arXiv](https://arxiv.org/abs/2510.05688)、[HTML](https://arxiv.org/html/2510.05688v2)

**F. prefill 块稀疏（不存跨步摘要，直接估计块分数）**

- **XAttention，arXiv 2503.16428【摘要】**：用块内注意力的反对角线元素之和作为块重要性代理；块稀疏 prefill，attention 加速 13.5 倍。— [arXiv](https://arxiv.org/abs/2503.16428)
- **FlexPrefill，arXiv 2502.20766【摘要】**：用 JS 散度在 query-specific 模式和 vertical-slash 模式之间切换；按累计注意力阈值选索引；prefill。— [arXiv](https://arxiv.org/abs/2502.20766)
- **Block Sparse Flash Attention（BSFA），arXiv 2512.07011【摘要】**：先算精确 QK 分数，再用每块最大分数与按层/头校准的阈值比较，跳过 V 块；约跳过 50% 计算。— [arXiv](https://arxiv.org/abs/2512.07011)
- **CompactAttention，arXiv 2605.16839【摘要】**：面向 chunked prefill，把 2D 块稀疏 mask 转为 GQA 感知的 KV 块表（Q-block union 加组内 union）。— [arXiv](https://arxiv.org/abs/2605.16839)
- **LServe，arXiv 2502.14866【摘要】**：一半的头改成 streaming head；按 query-centric 相似度做分层 KV 页选择；论文发现只需常数个 KV 页就能保持长上下文能力。— [arXiv](https://arxiv.org/abs/2502.14866)
- **AsyncTLS（arXiv 2604.07815）**、**Fluxion（arXiv 2605.07719）**、**FreeKV（arXiv 2505.13109）**、**LouisKV（arXiv 2510.11292）**【均为摘要】：分别做粗块过滤加细 token 选择、按头选择粒度和预算、推测式检索、在语义边界触发检索。属于系统侧改进。— [AsyncTLS](https://arxiv.org/abs/2604.07815)、[Fluxion](https://arxiv.org/abs/2605.07719)、[FreeKV](https://arxiv.org/abs/2505.13109)、[LouisKV](https://arxiv.org/abs/2510.11292)

**G. 评测型工作**

- **The Sparse Frontier，arXiv 2504.17768【全文·部分】**：在 Qwen2.5（7B–72B）等模型上比较 Vertical-Slash、FlexPrefill、Block-Sparse（经消融选定 16×16）、SnapKV、Ada-SnapKV、Quest（经消融选定 16-token 页）。decode 阶段 Quest 能承受更高压缩；但 Quest 在合成检索（RULER NIAH）上表现差，在自然语言检索（Story Retrieval）上表现好。高 scope 或高分散度的任务在 0.5–0.67 的稀疏度下就开始退化。— [arXiv HTML](https://arxiv.org/html/2504.17768v3)

### Inferences
- 在"摘要几何"这条线上，2024 年的 ArkVale（包围体优于质心）和 2026 年的 COBS（二阶矩优于一阶与 min/max）、RBS（质心加半径）形成一条清晰的演进线。我方的"矩型 vs 盒型"对比正处在这条线上，而且 COBS 自己就把 Quest(min/max) 作为基线（0.5765 vs 0.8195），所以我方结论需要与 COBS 的设置差异对齐：COBS 用的是自训练的 NSA 模型、L=32、top-16 块；我方是冻结的 Qwen2.5-1.5B、512/1024 预算。
- RBS-Attention 本质上就是"均值分支 ∪ 极值（半径上界）分支"，与我方发现 (a) 的结构几乎同构。区别在于 RBS 的极值分支是各向同性的 L2 半径，Quest 的是逐通道的盒，而且 RBS 只用于 prefill、块长 128。

### Gaps
- ArkVale 的方法细节来自 PDF 文本抽取（公式部分抽取后字符错乱），数字已核对，公式没有逐字核对。
- RetrievalAttention、MagicPIG、DoubleSparsity、Loki、XAttention、FlexPrefill、HashAttention、RocketKV、LServe 只读了摘要，块大小等实现参数没有从正文核实。
- ShadowKV 与 Quest 的 RULER 对比数字（83.57 vs 79.72）需复核。
- 没有找到 MoBA 或 NSA 的冻结模型、训练无关版本在 RULER 7 位数值 needle 上的细粒度分析。

## 问题 2：边界、多摘要组合、误差界：各方法是否处理？

### Takeaway
没有找到任何方法专门讨论"事实或 needle 跨越固定块边界"这一失败模式，也没有找到"两套半块错位网格"的选择方案。重叠只出现在**训练式压缩分支**（NSA 压缩 l=32/d=16，InfLLM-V2 第一阶段 B/2、步长 B/4），而且这些方法的动机不是救回跨边界事实。多摘要组合有 RBS（质心 ∪ 半径救援）、SeerAttention（K 的 max/min/avg 拼接后由可学习门控使用）、CEDAR（每块存两个摘要的消融）、vAttention（top-k 加采样）。误差界有 Quest（逐页上界）、RBS（Cauchy–Schwarz 上界）、CEDAR（输出误差界）、Twilight/Tactic（$(1-p)$ 型界）、vAttention（$(\epsilon,\delta)$ 概率保证）、COBS（丢块误差分解，但没有截断界）。

### Cited Findings
- 跨边界：COBS 的选择块不重叠，没有讨论 needle 跨边界 — [COBS HTML](https://arxiv.org/html/2607.09052v1)；RBS 没有讨论跨块 token 或重叠块 — [RBS HTML](https://arxiv.org/html/2609.20971v1)；CEDAR 使用变长语义边界，但没有分析是否切断事实 — [CEDAR HTML](https://arxiv.org/html/2609.07237v1)；SeerAttention 没有讨论块边界 — [SeerAttention HTML](https://arxiv.org/html/2410.13276v4)；InfLLM 承认块边界可能切碎内容，建议未来做 dynamic segmentation — [InfLLM HTML](https://arxiv.org/html/2402.04617v2)；LouisKV 在"语义边界"触发检索（指时间维的检索时机，不是块划分）【摘要】— [arXiv](https://arxiv.org/abs/2510.11292)。
- 重叠块：NSA 压缩分支 l=32、d=16，重叠；选择块 l′=64，按空间重叠聚合分数 — [NSA HTML](https://arxiv.org/html/2502.11089v2)；InfLLM-V2 第一阶段 mean-pool 块长 B/2、步长 B/4 — [InfLLM-V2 HTML](https://arxiv.org/html/2509.24663v1)。
- 多摘要：RBS 消融中，并集 76.67 高于质心 73.63、Quest-style 73.71 和完整 L2 上界 72.41 — [RBS HTML](https://arxiv.org/html/2609.20971v1)；SeerAttention 的 K 池化为 max+min+avg 拼接 — [SeerAttention HTML](https://arxiv.org/html/2410.13276v4)；CEDAR 每块两个摘要时 RULER 87.1→87.6，速度 3.02→2.71 倍 — [CEDAR HTML](https://arxiv.org/html/2609.07237v1)；vAttention 观察到 top-k 与采样互补 — [vAttention](https://arxiv.org/abs/2510.05688)；ArkVale 比较了 6 种包围体，但每次只用一种，没有组合 — [ArkVale PDF](https://proceedings.neurips.cc/paper_files/paper/2024/file/cd4b49379efac6e84186a3ffce108c37-Paper-Conference.pdf)。
- 误差界：Quest 上界 — [Quest HTML](https://arxiv.org/html/2406.10774v2)；CEDAR Proposition 1 — [CEDAR HTML](https://arxiv.org/html/2609.07237v1)；Twilight $(1-p)\|V\|_F$ — [Twilight HTML](https://arxiv.org/html/2502.02770v5)；Tactic $2(1-p)\max\|v\|$ — [Tactic HTML](https://arxiv.org/html/2502.12216v1)；vAttention $(\epsilon,\delta)$ — [vAttention HTML](https://arxiv.org/html/2510.05688v2)。

### Inferences
- 边界问题在文献里只是"块大小或分割方式"的间接议题（InfLLM 的块大小消融、RBS 的 B 消融呈非单调、MoBA 认为细粒度更好、CEDAR 用语义变长 chunk），没有人把"答案 token 被块边界一分为二"当作独立失败模式来测量。我方的 (b)(c) 看起来是新的经验证据。
- 我方 (b) 的机制解释可以借用 RBS 的 "mean dilution"：答案值被切成两半后，每块里只有一部分高相关 token，块均值被进一步稀释；而 max 型摘要（Quest 的盒）只要块内有一个通道极值高就能命中，所以对切分不敏感。这只是推断，尚未验证。
- 我方 (c) 的半块错位双网格在形式上类似 NSA 的"重叠压缩块"（步长等于半块长），但 NSA 是训练式的，用于压缩打分后再聚合到不重叠的选择块，而且没有固定总预算的说法。可以把 (c) 定位为"训练无关、预算精确的重叠选择"。

### Gaps
- 没有检索到专门研究跨块边界 needle 或数值切分的论文；搜索 "shifted/misaligned/overlapping block selection" 没有命中相关的 LLM 推理工作（命中的多为视频生成）。不能排除存在未被 arXiv 关键词覆盖的工作。
- SeerAttention 没有报告"只用 max/min" vs "只用 avg"的分项数字（抽取结果只给出了最优配置）。

## 问题 3：我方发现 (a)–(e) 与已有工作的对应

### Takeaway
(a) **已被部分报告**，RBS-Attention 2026 的"质心 ∪ 半径救援"最接近，SeerAttention 用 K 的多池化拼接学习门控；但"COBS 式二阶矩 vs Quest 逐通道盒、错误互补、取并集"这一具体组合尚未见到。(b)(c) **未见报告**。(d) **部分报告**：Quest 默认 16，Sparse Frontier 消融也选 16，ArkVale 在紧预算下 16 优于 32，但都不是"等摘要内存"的比较。(e) **部分报告**：CEDAR 明确论述"attention recall 不够"；COBS 却显示 mass oracle 接近 dense，与 (e) 存在张力。

### Cited Findings
- (a) 互补与并集：RBS 两分支并集优于单分支（76.67 vs 73.63/73.71）— [RBS](https://arxiv.org/html/2609.20971v1)；ArkVale 中包围体（含 cuboid=min/max）召回远高于质心 — [ArkVale](https://proceedings.neurips.cc/paper_files/paper/2024/file/cd4b49379efac6e84186a3ffce108c37-Paper-Conference.pdf)；COBS 中 Quest 0.5765 ≈ mean-pool 0.5554 ≪ COBS 0.8195，但没有测并集 — [COBS](https://arxiv.org/html/2607.09052v1)；vAttention 认为 top-k 与采样互补 — [vAttention](https://arxiv.org/abs/2510.05688)。
- (b)(c)：上述论文都没有讨论（见问题 2）。
- (d) 页粒度：Quest 默认 16 页，无消融 — [Quest](https://arxiv.org/html/2406.10774v2)；Sparse Frontier 经消融选 Quest 16-token 页、Block-Sparse 16×16 — [Sparse Frontier](https://arxiv.org/html/2504.17768v3)；ArkVale-16 在紧预算下常优于 -32（等 token 预算）— [ArkVale](https://proceedings.neurips.cc/paper_files/paper/2024/file/cd4b49379efac6e84186a3ffce108c37-Paper-Conference.pdf)；MoBA 细粒度更好 — [MoBA](https://arxiv.org/html/2502.13189v1)；ShadowKV 用更细的 chunk=8 — [ShadowKV](https://arxiv.org/html/2410.21465v3)；RBS 中 B=128 优于 64 和 256（prefill，非单调）— [RBS](https://arxiv.org/html/2609.20971v1)；COBS 的摘要内存/读流量是 NSA 的 1.21 倍，没有与"更小页的 Quest"在等内存下比较 — [COBS](https://arxiv.org/html/2607.09052v1)。
- (e) 代理指标：CEDAR §2.2 "attention recall is not enough" — [CEDAR](https://arxiv.org/html/2609.07237v1)；COBS 的 OSA（精确 mass 选择）0.9010 ≈ dense 0.9040，说明"按 mass 选"这一准则足够 — [COBS](https://arxiv.org/html/2607.09052v1)；ArkVale 以 top-k 页召回率作为主要中间指标 — [ArkVale](https://proceedings.neurips.cc/paper_files/paper/2024/file/cd4b49379efac6e84186a3ffce108c37-Paper-Conference.pdf)；Sparse Frontier 发现 Quest 在合成 NIAH 与自然语言检索上表现相反 — [Sparse Frontier](https://arxiv.org/html/2504.17768v3)。

### Inferences
- (e) 与 COBS 的张力可能来自两点：COBS 的 oracle 是"按精确 mass 选"，对应的是**选择准则**正确；我方 (e) 说的是近似选择器之间，"mass 捕获率"或"needle 块命中"的**差异**不能预测准确率。两者并不矛盾，但写作时需要区分这两层。另外，COBS 是在 NSA 自训练模型上做的，冻结的 Qwen2.5-1.5B 可能对局部漏读更敏感（例如 7 位数字只读到一半），而"needle 块命中"这一指标会把"只命中半个答案"也算作命中。这是推断。
- (d) 的"等摘要内存"维度在文献中是空白：COBS 只报告读流量，ArkVale 和 Sparse Frontier 都是在等 token 预算下比较页大小。我方 (d) 是一个新的、对 COBS 不利的对照：在 Quest 16 页与 COBS 32 块等内存的条件下 Quest 更好，而 COBS 论文中 Quest 也是 32 块。这一点需要明确报告双方的块长设置。
- 与 RBS 的区别是：RBS 用 prefill、块长 128、阈值加并集（预算不固定）；我方是 decode、固定精确预算、比较二阶矩与逐通道盒。投稿时应把 RBS 列为最直接的相关工作。

### Gaps
- 没有找到在冻结小模型（约 1.5B）上做块选择对比的论文；大多数工作使用 7B–32B 模型或自训练模型。
- 没有找到报告"选择质量指标与端到端准确率相关性"（例如 Spearman 系数）的定量研究；CEDAR 只做了论述并给出消融证据。
