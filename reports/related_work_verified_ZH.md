# 相关工作原文核对（2026-10-17；审查方批次 7 要求引用前核对原文）

核对方式：arXiv HTML 全文，只摘录与本项目定位直接相关的定义与超参数。引号内为原文。

## MoBA（arXiv 2502.13189）
- 块打分：“the score s_i is computed by the inner product between q and the mean pooling of K[I_i] along the sequence dimension: s_i = ⟨q, mean_pool(K[I_i])⟩”，即一阶均值池化。
- 块划分：I_i = [(i−1)×B+1, i×B]，**不重叠**。
- 块长：scaling 实验 512；Llama-8B-1M-MoBA 为 4096。
- 对本项目的含义：MoBA 用的正是均值摘要（对应 mean 臂），但块长为 512–4096。按增益律，跨界概率 π ≈ (ℓ−1)/B，ℓ ≈ 12 时只有约 0.3–2%，所以在 MoBA 原生设定下 TWIN-GRID 的预期增益接近 0。切分问题只在小块（数十 token）时才重要。

## InfLLM（arXiv 2402.04617）
- 记忆单元 l_bs = 128 个 token；每个单元取代表分数最高的 r_k = 4 个 token 作为代表：“r_k tokens with the highest representative scores are selected as representative tokens”。
- 相关度：查询与这些代表 token 的点积之和。
- 对本项目的含义：InfLLM 用的是代表 token，**不是**均值摘要（审查方的记忆略有出入，以此为准）。它属于另一类摘要（少数精确 token），与 P3 中的“精确异常值”思路相近。

## NSA（arXiv 2502.11089）
- 压缩分支：K̃ = {φ(k_{id+1 : id+l}) | 0 ≤ i ≤ ⌊(t−l)/d⌋}，块长 l、滑动步长 d，“d < l”用于防止信息碎片化（原文措辞为 prevent information fragmentation）。
- 默认超参数：l = 32，d = 16，选择块 l′ = 64，选中块数 n = 16，滑动窗口 w = 512。
- 选择分数来自压缩注意力分数 p_cmp = Softmax(qᵀ K̃_cmp)，再按块对齐方式直接赋值或空间聚合到选择块。
- 对本项目的含义：
  - NSA 压缩分支的布局（l = 32、d = 16）**正是两套错开半块的 32-token 网格**，与 TWIN-GRID 的布局相同；它明确以“防止碎片化”为目的。因此“错位半块网格”本身不是新想法，不能作为贡献；
  - 区别在于：NSA 的压缩函数 φ 是端到端训练的，选择块 l′ = 64 不重叠；TWIN-GRID 用于冻结模型、免训练的候选选择，在固定候选 token 预算内去重并精确填满（g2_fill_v1），并给出覆盖引理 L1 与增益律的量化刻画；
  - TWIN-GRID 的贡献表述须收窄为：在免训练的块稀疏 decode 中，用与内容无关的错位候选网格修复“边界切断”这一误差源，给出适用范围（L1）与增益的分解及预测（增益律），并证明预算不增加。

## 对叙述的总体影响
1. “谁需要 TWIN-GRID”：块长为数十 token 的免训练选择方法（Quest 类页面、NSA 类小选择块）；大块均值池化（MoBA）几乎不受切分影响。
2. 新颖性：布局本身已有先例（NSA 压缩分支），贡献在于免训练选择中的误差源刻画、覆盖保证与预测性分解。
