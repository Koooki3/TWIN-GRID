# 由紧凑摘要估计块注意力质量 / log-partition 的理论：相关工作定位

（检索日期 2026-10-16。由调研子任务以文本交回、协调者原样保存。标注：【已开原文】= 本轮读了 HTML/全文；【仅摘要】= 只看了摘要或搜索结果摘录；【未打开，凭既有知识】= 本轮没打开，需要核对。）

## Q1：极值界 (1) 是否已知？

### Takeaway
(1) 的数学内容是已知结果，不是新定理。
- 上界：Samuelson（Laguerre–Samuelson）不等式给出 x_i − m ≤ s√(n−1)。把它代入 Bennett(1962)/Hoeffding(1963) 的两点 MGF 界（条件 X ≤ y、方差固定），取 y = s√(n−1)、σ = s，结果与 m + φ_n(s) 完全相同。极值分布是单尖峰：1/n 的质量在 s√(n−1)，其余在 −s/√(n−1)。
- 下界：来自 Chebyshev 系统 / 矩问题的经典结论（f''' > 0 时，均值、方差固定、支撑在区间上，E f 的最小值由含下端点的两点分布取到）；也可看作 Cîrtoaje 等变量（EV）定理的特例（k=2，f=exp）。
- 写法建议：表述为"经典两点极值结果的 log-partition 形式"；贡献在于把它解读为均值+方差摘要的最坏情况不可辨识性。

### Cited Findings
- Samuelson 1968 "How Deviant Can You Be?"（JASA）：任一点与均值的距离不超过 √(n−1) 个未校正标准差；与 Laguerre 1880 等价；n−1 个点相等时取等号。来源：https://en.wikipedia.org/wiki/Samuelson%27s_inequality ；原文 PDF：https://www.Gwern.net/doc/statistics/probability/1968-samuelson.pdf
- Bennett–Hoeffding 设定（X ≤ y、EX=0、EX² ≤ σ²）的 MGF 界由 {−a,b} 上的两点分布取到；Pinelis 的复述引用 Bennett 1962 与 Hoeffding 1963。来源：Pinelis, AIHP 2014, https://ar5iv.labs.arxiv.org/html/0902.4058 【已开原文，前半部分】
- 尖锐两点形式 E e^{tX} ≤ (y² e^{−tσ²/y} + σ² e^{ty})/(y²+σ²)【凭既有知识；Bennett JASA 57:33–45 与 Hoeffding JASA 58:13–30 的式号未核对】。
- 本地数值核对（非文献）：n=7、s=1.3 时，n 乘以 Bennett 界后取 log 等于 φ_n(s)=3.3207224459，到机器精度相同；2 万组随机/重尾样本中 (1) 的上下界均未被违反。
- Karlin & Studden, Tchebycheff Systems（Wiley 1966）：https://ds.mainlib.upd.edu.ph/Record/UP-99796217607157079 【未打开，章节号需核对】
- Cîrtoaje, "The Equal Variable Method", JIPAM 8(1) Art.15 (2007)：https://emis.dsd.sztaki.hu/journals/JIPAM/article828.html 【仅摘要】
- Cîrtoaje, Creative Math. Inform. 24(2) (2015) 145–154：https://www.creative-mathematics.cunbm.utcluj.ro/article/on-the-equal-variables-method-applied-to-real-variables/ 【仅摘要】
- 有界支撑矩约束下极值分布限于少量支撑点：https://arxiv.org/pdf/0808.1669 【仅搜索摘要】

### Inferences
- 区间宽度约 s√(n−1) − log n：给定均值方差后，真实 log 质量可落在这样宽的区间内任何位置，直接说明 COBS 类二阶摘要在"大海捞针"块上不充分。
- 可补充 Bhatia–Davis：Var ≤ (M−μ)(μ−m)（Amer. Math. Monthly 107(4) 2000）【未打开】。
- 可补充 Edmundson–Madansky：E e^X ≤ ((b−μ)e^a + (μ−a)e^b)/(b−a)（Madansky, Ann. Math. Statist. 30 (1959) 743–746, DOI 10.1214/aoms/1177706203）【未打开】；在只有均值+区间信息时最紧，可直接收紧 (2) 中只用 box 最大值的上界——候选改进，尚未实验。

### Gaps
- Bennett/Hoeffding 原文式号、Karlin–Studden 对应定理章节未核对。
- 未见文献直接以 log Σe^{x_i}、固定 (m, s²) 的形式写出 φ_n(±s)。

## Q2：注意力领域的块质量 / log-partition 估计及其保证

### Takeaway
2025–2026 年已有一批工作把块选择归结为估计块注意力质量，但都没有给出均值+方差摘要的最坏情况不可辨识定理；也未见"异常值精确 + Jensen 下界 + box 上界"的块 LSE 认证区间，或白化（Mahalanobis）质心+半径界。

### Cited Findings
- COBS, arXiv:2607.09052（2026-07）【已开原文】 https://arxiv.org/html/2607.09052v1 ：按块质量 m_b 排序；质量 oracle 在 32k RULER 上 0.9010（dense 0.9040）。核心近似 ln m_b ≈ ln L + qᵀk̄_b + ½ qᵀΣ_b q（累积量展开），论证一阶选择器表示不了二次项。Σ 为秩 r 谱分解、投影到查询二阶矩子空间、FP4 量化。NSA 基线 0.2999 → 0.8195，KV 读流量为 NSA 的 1.21×。引用 CCQ；把 Quest 的 min/max 描述为"轴对齐展布代理"。读到的部分无最坏情况保证。
- CEDAR, arXiv:2609.07237（2026-09）【已开原文】 https://arxiv.org/html/2609.07237 ：每 chunk 存 k̄、v̄、n_c 与离散度（最大范数或 RMS×α）；摘要 logit qᵀk̄/τ + log n_c（相当于 Jensen 点估计）；Proposition 1 给确定性输出误差界 ‖o−ô‖ ≤ (1/Z) Σ_unrefined n_c e^{z̄}[e^{ε} r^v + (e^{ε}−1)(‖v̄‖ + M)]，ε 为 logit 半径界。
- RBS-Attention, arXiv:2609.20971（2026-09）【已开原文】 https://arxiv.org/html/2609.20971 ：s_b(q) ≤ q·c_b + ‖q‖₂ r_b（欧氏、非白化 Cauchy–Schwarz），约束块内最大单 token logit 而非块质量；实际用衰减版 β_b<1，可违反该界（作者称"风险信号"），无定理；128K 下 TTFT 加速 5.97×。
- Top-k TV 理论, arXiv:2512.07647（2025-12）【仅摘要】 https://arxiv.org/abs/2512.07647 ：TV = 丢弃的尾部质量 = 1 − e^{−KL}；输出误差 = τ‖μ_tail − μ_head‖；只依赖有序 logit 的间隙界与分块界；高斯分数模型下有闭式尾部质量与 k_ε/n ≈ Φ_c(σ + Φ^{-1}(ε))。
- vAttention, arXiv:2510.05688（ICLR 2026）【仅摘要】 https://arxiv.org/html/2510.05688v1 ：top-k 重击者 + 长尾均匀采样，以概率 1−δ 保证输出相对误差 ≤ ε。
- Tactic, arXiv:2502.12216（ICLR 2026）【仅摘要】 https://arxiv.org/abs/2502.12216v1 ：按累计注意力比例选 token，聚类 + 分布拟合；无认证界。
- Twilight（NeurIPS 2025, arXiv:2502.02770）【仅摘要】：层级 top-p 剪枝，经验方法。
- Runtime-Certified Bounded-Error Quantized Attention, arXiv:2605.20868【仅摘要】：每 head 每步分布失真界，运行时可算，超界回退 FP16。
- LOCKS, arXiv:2607.24555【仅摘要】：每页存秩 r 谱摘要，按 LSE 质量选页；摘要中无形式化保证。
- Uncertainty-gated selection, arXiv:2607.07724【仅摘要】：只看第 k 与第 k+1 块的分数间隙，不估计质量区间。
- 未打开、凭既有知识：Quest（2406.10774，逐通道 min/max 是页内最大 logit 的严格上界，(2) 中 box 上界的前身）；MagicPIG（2410.16179，LSH 重要性采样，概率层面分析）；HyperAttention（2310.05869）；Scatterbrain（2110.15343）、Reformer（2001.04451）、Fast Multipole Attention（2310.11960）：无块质量区间。

### Gaps
- COBS 引用的 CCQ 书目未核实；vAttention 等定理细节未读；较早方法未打开原文。

## Q3：REM 截断与 (3)

### Takeaway
(3) 的"最大项 + 截断其余项"是 REM 截断分析的有限 n 版本，数学上已知；新意只在于用作块质量估计器的工程近似并刻画其误差。

### Cited Findings
- Bovier–Kurkova–Löwe, Ann. Probab. 30(2) 2002，arXiv:cond-mat/0007175【仅摘要】；Project Euclid 开放全文：https://projecteuclid.org/journals/annals-of-probability/volume-30/issue-2/Fluctuations-of-the-free-energy-in-the-REM-and-the/10.1214/aop/1023481004.full 。三区间：β < √(ln2/2) 高斯波动；√(ln2/2) < β < √(2ln2) 非高斯、极值 Poisson 过程支配；β > √(2ln2) 时 log Z 波动 O(1)、由极值决定；临界点 Z/EZ → 1/2。
- Ben Arous–Bogachev–Molchanov, PTRF 132(4) 2005 579–612，https://eprints.whiterose.ac.uk/id/eprint/86345/ 【仅摘要】：S_N(t)=Σe^{tX_i}，N ~ e^{λH_0(t)}；两临界点 λ₁<λ₂ 处 LLN 与 CLT 分别失效。
- Bovier–Schertzer 后续：https://arxiv.org/pdf/2205.15080
- 注意力侧最接近：top-k TV 论文的高斯闭式尾部质量。

### Inferences
- 相变点 σ ≈ √(2 ln n)；σ 小时 log n + m + σ²/2（annealed，即 COBS 二阶形式）准确；σ > √(2 ln n) 时 log Z ≈ m + σ√(2 ln n)，σ²/2 严重高估（常数需按 BKL 全文核对）。
- 建议补引对数正态和近似（Fenton–Wilkinson 1960、Schwartz–Yeh 1982），本轮未检索。

## Q4：(1)–(4) 新还是旧

| 结论 | 判断 | 依据 |
|---|---|---|
| (1) 极值界 | 数学已知 | Samuelson + Bennett/Hoeffding / Chebyshev 系统 / Cîrtoaje EV 定理 |
| (2) 认证区间 | 最可能主张新颖 | 组件已知（Jensen ≈ CEDAR/NSA 均值 logit + log n、Quest box），组合成块 LSE 认证区间未见 |
| (3) 高斯 E LSE 近似 | 数学已知 | REM / 指数和渐近的有限 n 近似；作估计器有一定新意 |
| (4) 质心+半径界 | 新颖性中等 | 欧氏版已有（RBS、CEDAR），白化/leverage 版未见 |

- (1) 解读为对 COBS 类二阶摘要的最坏情况反例（误差 Θ(s√n)），未见先例。
- 建议定位措辞："我们把经典两点极值（Samuelson–Bennett）重述为 log-partition 的紧界，据此说明只用 (均值, 方差) 的摘要在最坏情况下不可辨识；再提出由精确异常值 + Jensen 下界 + box/白化半径上界构成的认证区间。"
- Edmundson–Madansky 的"均值 + 区间"上界在相同摘要下严格比只用 box 最大值的上界更紧，值得实验验证。
- 剩余缺口：2026 年 7–10 月相关论文增长很快，投稿前再检索"certified block log-sum-exp""Mahalanobis key radius"等。
