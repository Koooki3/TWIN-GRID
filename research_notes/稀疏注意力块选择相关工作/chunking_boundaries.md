# 上下文切分单元、块边界与块摘要类型：KV 压缩 / 稀疏注意力 / KV 检索相关工作（2024–2026）

说明：下列每篇都打开过 arXiv HTML/PDF 的方法和实验部分（经 WebFetch 摘要），没有只看摘要的条目。例外是 Swin Transformer（2021，属于经典背景，本次没有打开），只在推断里提到，不当作证据。工具层面的限制：LycheeCluster 的 PDF 解析失败，改读 arXiv HTML v1；所有数字都转述自 WebFetch 返回的内容，引用前最好对照原文表格复核。

## Q1：各工作如何形成选择/压缩单元（固定、句子、聚类、动态）

### Takeaway
2025–2026 年这条线的主流叙事是“固定块会切断语义单元，所以改用句子、分隔符或聚类切分”。单元形式包括：固定 chunk（ChunkKV）、标点句子（SentenceKV）、分隔符分层 + 8–16 token 长度约束（LycheeCluster）、动态分隔符搜索（DynSplit-KV）、分隔符切分 + key 聚类合并（SemantiCache）、全局 k-means 聚类（ClusterKV）、重叠滑窗压缩（NSA）。块摘要几乎都是 mean key 或 Quest 式 min/max；COBS（2026-07）加入了块内协方差。

### Cited Findings
- **ChunkKV**（arXiv 2502.00299，2025-02）：使用固定大小、不重叠的 chunk，C=⌈T/c⌉；chunk 分数是观察窗口内 attention 分数在块内求和；chunk size 扫了 {3,5,10,20,30}，10 最好。LLaMA-3-8B 在 LongBench 10% 压缩下的得分为 40.49/40.47/40.51/40.05/39.57。论文把 size=3 的退化解释为 “context is fragmented”，把 size=30 的退化解释为 “granularity too coarse”。它属于 token eviction，不是逐步检索 — [ChunkKV](https://arxiv.org/html/2502.00299v5)
- **SentenceKV**（arXiv 2504.00970，COLM 2025）：按标点切成变长 “sentence buckets”；对每个句子内保留的 token 求 per-head mean key 作为语义向量，留在 GPU 上，KV 卸载到 CPU；解码时用 q̄ᵀk̄ 排序并按 token 预算取句。在 Table 3 的消融中，等长 chunk 比句子差，例如 NrtvQA 上 27.02 vs 29.59 — [SentenceKV](https://arxiv.org/html/2504.00970v2)
- **LycheeCluster**（arXiv 2603.08453；ACL 2026 Findings）：分隔符分四级优先（段落/结构 > 句末 > 逗号等 > 空白），贪心切分，每块最少 8、最多 16 token。块表示是块内 key 的 mean pooling 加 L2 归一化，再建层次聚类索引（三角不等式剪枝） — [LycheeCluster HTML](https://arxiv.org/html/2603.08453v1)；[ACL Anthology 页](https://preview.aclanthology.org/ingest-acl/2026.findings-acl.376/)
- **DynSplit-KV**（arXiv 2602.03184，2026-02）：在理想终点 sc+C 附近的窗口 [se−Δ, se+Δ] 里，用 α·分隔符权重 + (1−α)·距离项挑选切点；分隔符权重随模型和语料变化。主方法的块摘要是 element-wise min/max（与 Quest 一致）；再用 V2F 把变长块映射成定长格式，以降低开销 — [DynSplit-KV](https://arxiv.org/html/2602.03184v1)
- **SemantiCache**（arXiv 2603.14303，2026-03）：按分隔符集合 [. , ? ! ; : 空格 \t \n] 切块，分隔符 token 作为 “structural anchors” 单独保留；块内用贪心种子聚类（cosine 阈值 τ），每个聚类 mean-pool 合并，并在 logit 上加 log s（proportional attention） — [SemantiCache](https://arxiv.org/html/2603.14303)
- **ClusterKV**（arXiv 2412.03213，2024-12）：不按位置切块，而是用 cosine 距离对 key 做 k-means，prefill 时取 C₀=L/80 个簇（32k 上下文对应 400 簇）；簇中心用 qμᵀ 打分；解码阶段每 320 步对新 token 增量聚类 — [ClusterKV](https://arxiv.org/html/2412.03213v1)
- **NSA**（arXiv 2502.11089，ACL 2025）：压缩分支的块长 l=32、步长 d=16，原文理由是 “d < l to mitigate information fragmentation”；选择块 l'=64，n=16，窗口 w=512。选择分数通过对重叠的压缩块分数求和得到，即 p_slc[j]=Σ_mΣ_n p_cmp[(l'/d)j−m−n] — [NSA](https://arxiv.org/html/2502.11089)
- **Quest**（ICML 2024）：页内存 key 的逐维 min/max，按 Σ_j max(q_j·min_j, q_j·max_j) 得到注意力上界，默认 16 token/页 — [Quest HTML](https://arxiv.org/html/2406.10774)；[Quest 项目页](https://hanlab.mit.edu/projects/quest)
- **SnapKV**（arXiv 2404.14469，2024）：对观察窗口得到的重要性做 1D max pooling（kernel 5/7/13）后再选 token，相当于一种“邻域扩张”式的软切分 — [SnapKV](https://arxiv.org/html/2404.14469)
- **COBS**（arXiv 2607.09052，2026-07，MatX）：块长 L=32，每块存 mean key 和压缩后的块内协方差，近似 ln m_b ≈ ln L + qᵀk̄_b + ½qᵀΣ_b q；top-16 块，256 token 滑窗 — [COBS](https://arxiv.org/html/2607.09052)

### Inferences
- 在“块摘要”这个维度上，现有方法可以粗分为三类：一阶矩（mean key：SentenceKV、LycheeCluster、SemantiCache、ClusterKV 的中心）、盒/极值（Quest、DynSplit-KV 的 min/max）、二阶矩（COBS）。我们的 moment-based 与 box/extreme-value 的二分正好对应这三类，而且 32-token 块和 top-k 块的配置与 COBS、NSA 很接近。

### Gaps
- Swin 式 shifted window 在 LLM KV 选择中的直接对应物，我没有找到。NSA 的重叠压缩窗口是最近的对应，但它用于训练过的压缩分支，不是在冻结模型上用两套网格做选择。

## Q2：是否测量碎片化 / 边界切割？是否有“输入固定、只移动边界”的实验？

### Takeaway
多数论文从定性角度论证固定块会切断语义，证据形式是“固定块 vs 语义块”的端到端对比。没有一篇在输入完全不变的前提下只平移网格（offset）做受控实验，也没有一篇统计“目标信息跨块”的频率与失败率之间的关系。NSA 用重叠窗口缓解碎片化，但没有做消融。

### Cited Findings
- LycheeCluster 的 pilot（Sec 3.1，Fig 2）在 StrucText-Eval 上用**相同打分**比较 Quest 固定页（S=16）和结构感知分块：JSON 抽取 +15.0%，平均 +10.6%。原文结论是 “merely aligning retrieval units with semantic boundaries yielded substantial performance boost”。消融（Fig 6）中去掉边界感知后，Long Structured Data 任务下降 3.03%。论文没有报告平移边界的实验 — [LycheeCluster](https://arxiv.org/html/2603.08453v1)
- DynSplit-KV 报告，rigid splitting（固定间隔或预定义分隔符）相对 full KV 的精度下降在 5.5%–55.1% 之间（Fig 3a）；把分隔符重要性顺序反转会掉点最多 28.6%（Fig 3d）；分隔符重要性在代码与自然语言之间相差 45%，跨模型相差 24%。论文没有固定输入、只移动边界的实验，也没有碎片化指标；变长块会带来 73.1% 的额外推理开销 — [DynSplit-KV](https://arxiv.org/html/2602.03184v1)
- ClusterKV（Fig 3b）给出了一种定性的“内部碎片”证据：8k 上下文、16-token 页的 attention 热图上，每页只有 1–2 个重要 token，有时要 2 页（32 token）才能覆盖两个相邻的重要 token。论文没有扫页大小或边界 — [ClusterKV](https://arxiv.org/html/2412.03213v1)
- NSA 用重叠压缩窗口 d<l 来“mitigate information fragmentation”，但没有针对 stride、重叠或边界的消融 — [NSA](https://arxiv.org/html/2502.11089)
- ChunkKV 只做了 chunk size 消融，没有分析 offset 或重叠 — [ChunkKV](https://arxiv.org/html/2502.00299v5)
- SemantiCache Table 4（Llama-3-8B，LongBench，20% 预算）：语义分块 30.01，固定 64-token 分块 28.22，不分块 26.94 — [SemantiCache](https://arxiv.org/html/2603.14303)
- Quest 原文没有页大小的系统消融，16 token/页是实现细节；论文也没有讨论关键 token 跨页的问题 — [Quest](https://arxiv.org/html/2406.10774)
- COBS 没有分析碎片化、needle 跨块或平移/重叠块 — [COBS](https://arxiv.org/html/2607.09052)

### Inferences
- 现有的“边界效应”证据都混入了块长分布、块数和预算的变化：语义块是变长的，固定块是定长的。我们的实验在输入、块长和预算都不变的情况下只平移网格，并且把答案值放在边界的已知位置上，这是这些论文都没有做过的受控设计。
- “两套网格偏移半块、精确 token 预算”在思路上接近 NSA 的 stride<window，以及 Swin 的 shifted window。但我没有找到在冻结模型的检索式选择中做这种双网格选择、并严格对齐 token 预算的工作。

### Gaps
- DynSplit-KV Fig 3 中具体的固定间隔大小和分隔符集合，我没有拿到（WebFetch 未提取到）。
- 我没有逐一核查 InfLLM、MoBA、ShadowKV、RetroInfer 等系统是否有页或块大小、offset 的消融，这些仍需查。

## Q3：有没有工作显示边界敏感性取决于块摘要/打分类型？

### Takeaway
没有找到。确实有工作比较过摘要类型（mean vs max/min-max vs 协方差），但都在**单一切分方式**下比较，没有做“切分方式 × 摘要类型”的交叉实验。而且已有两个比较结果方向相反（DynSplit-KV 认为差别不大，LycheeCluster 认为 mean 明显更好），这说明结论可能受切分方式影响，但没有人检验过这一点。

### Cited Findings
- DynSplit-KV Table 6（Appendix A.2）固定使用其**动态语义切分**，比较 min/max 和 mean pooling：LongBench 9 个数据集、0.1 KV 使用率下分别为 42.64 和 42.46（full KV 42.72），作者结论是 “nearly unchanged”。论文没有在固定间隔切分下比较摘要类型 — [DynSplit-KV](https://arxiv.org/html/2602.03184v1)
- LycheeCluster Table 3 在其**结构感知切分**下，mean pooling 的检索 recall 为 40.37%，max pooling 为 33.60%（LongBench V2）。它的 pilot 则是在“相同打分”下只换切分方式。两者没有交叉 — [LycheeCluster](https://arxiv.org/html/2603.08453v1)
- COBS 在固定 32-token 块的 32k RULER（11 任务）上：mean-pool 0.5554，Quest min/max 0.5765，COBS（mean+协方差）0.8195，dense 0.9040，NSA baseline 0.2999。没有边界相关分析 — [COBS](https://arxiv.org/html/2607.09052)
- SentenceKV 比较的是查询端聚合（当前 token query vs 句内 mean query），结论是后者更好；它没有比较 key 端摘要类型与切分方式的交互 — [SentenceKV](https://arxiv.org/html/2504.00970v2)

### Inferences
- 我们的发现（矩类打分器在值跨边界时更脆弱，盒类较稳，但盒类在值被从中间切开时同样受损）在已核查的文献中没有对应。一个可能的解释与 DynSplit-KV 和 LycheeCluster 结论不一致的情况相符：在边界对齐的语义块下，mean 和 min/max 差别小或 mean 更好；在固定网格下，一个被切开的值只占块内少数 token，被块内均值稀释，而 min/max 盒仍能保留单个 token 的极值方向。这只是推断，需要用我们的数据支持。
- COBS 中 mean-pool < Quest 的排序（固定 32-token 块）与我们“固定网格下矩类弱于盒类”的方向一致，但 COBS 没有把差异归因于边界。

### Gaps
- “边界敏感性 × 摘要类型”的交叉实验，我在 2024–2026 文献中没有找到。由于检索次数有限，不能排除 2026 年下半年还有未覆盖的预印本。

## Q4：数字按位分词与切分的交互有没有证据？

### Takeaway
没有找到专门研究“逐位分词的数字跨块”与 KV 选择交互的工作。最接近的是 SnapKV 的轶事例子：只保留注意力峰值会“只取回电话号码的国家代码”，因此用 max pooling 保留邻域。这是邻域完整性的动机，不是对块边界的受控研究。

### Cited Findings
- SnapKV 指出 “naively selecting the top features results in retaining only portions of details”，并以只取回电话号码国家代码的幻觉为例。在 LongEval-Lines（Mistral-7B）上，max pooling（kernel 5）在 16K 以内能正确检索值，不加 pooling 则显著退化。作者推测原因是 “initial portions of critical token clusters are weighted higher” — [SnapKV](https://arxiv.org/html/2404.14469)
- ChunkKV 的动机例子是离散 token 选择丢掉主语、chunk 保留主语，属于语义完整性论证，不涉及数字分词 — [ChunkKV](https://arxiv.org/html/2502.00299v5)
- 一般的 needle 评测常用 5–7 位随机数作为 value（例如 RULER 和 KV-Fold 的描述），但这些工作没有分析数字的各位 token 落在不同块时的失败率 — [KV-Fold](https://arxiv.org/pdf/2605.12471)（只从搜索摘要看到其 5 位数 needle 设计，正文没有打开，算作 abstract/snippet-only）

### Inferences
- Qwen2.5 按位分词，7 位值占 7 个 token，在 32-token 块中跨边界的概率不低（随机位置时约 6/32≈19%，这是我们自己的推算）。相关文献普遍用 RULER 风格的数字 needle 做评测，却没有按“是否跨块”分层报告结果，所以这一混杂因素在已有结果里大概率没有被注意到。

### Gaps
- 我没有找到关于数字分词（逐位 vs 多位 token）对 KV 检索或块稀疏选择影响的论文。没有专门检索 tokenization 方向的文献（例如 Llama-3 三位一组分词与 Qwen 逐位分词的对比），这可以作为后续检索方向。
