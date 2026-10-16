# 长上下文检索与复制的机制研究（与块稀疏注意力 near-miss 现象相关）

> 背景：冻结的 Qwen2.5-1.5B 做块稀疏解码（32 token/块），7 位数字答案（Qwen 每个数字一个 token）跨块边界时，部分块打分器出现 near-miss：找对了 needle，但个别数字抄错、输出被截断，或尾部数字被换成干扰 needle 的数字。把答案尾部所在块强制加入选择后，许多失败能被救回。两种竞争假设：H1“尾部片段是一个 spike，基于矩（均值等）的块摘要看不到它”；H2“数字跟着包含关键短语的块一起被选中”。
>
> 来源标注：**[全文/HTML 已读]** 表示本次读过 HTML 或全文的主要部分；**[仅摘要]** 表示本次只核对了 arXiv 摘要页；**[既有知识，未在本次核对]** 表示是经典文献，本次没有重新抓取，引用前请自行核对细节。2026 年的几篇都是单作者或小团队的预印本，未经同行评审。

## Q1：哪些 head 负责 needle 检索和复制，它们的注意力有多稀疏，稀疏注意力方法怎样和它们相互作用

### Takeaway
多篇工作都发现，长上下文的“取回并复制”集中在一小部分 head 上（retrieval heads，通常不到 5%）。它们在复制时逐 token 读取源序列，attention 峰值随生成步单调前移，机制上接近 induction head 的“前缀匹配 + 拷贝下一个”。对我们最直接的证据是 2026 年的 KVFetch：基于内容打分的 KV 选择会保留答案片段的开头（它和 query 语义相关），却丢掉中段和尾部的后继 token（它们单独打分时普遍偏低）。结果是逐字复制在中途突然断掉。这和我们观察到的“检索成功、尾部出错”是同一类现象。

### Cited Findings
- **Retrieval heads（Wu et al., 2024）**：不到 5% 的 head 是 retrieval head；性质为 universal、sparse、intrinsic（短上下文预训练阶段就已存在，扩展上下文后保留）、dynamically activated（Llama-2-7B 中有 12 个 head 始终关注所需信息，其余按上下文激活）、causal（剪掉它们会导致检索失败和幻觉，随机剪掉非检索 head 几乎没有影响）— [arXiv 2404.15574](https://arxiv.org/abs/2404.15574) [仅摘要]
- Wu et al. 的 retrieval score 以 NIAH 的 copy-paste 行为定义：生成的 token 属于 needle，并且该 head attention 的 top-1 位置正好是被复制的那个 needle token，按复制频率计分。这个定义来自正文，本次只读了摘要，摘要页没有给出公式 — [arXiv 2404.15574](https://arxiv.org/abs/2404.15574) [既有知识，未在本次核对]
- 有综述性描述把 retrieval head 和 induction head 区分开：induction head 依靠局部 token 级重复来预测下一个 token，retrieval head 则为回答 query 从远处复制语义上需要的片段 — 见检索结果中对 [arXiv 2404.15574](https://arxiv.org/pdf/2404.15574) 的转述 [二手转述，需核对]
- **Induction heads（Olsson et al., 2022）**：机制是“prefix matching + copying”。当前 token A 在前文出现过时，head 关注前文 A 后面的那个 token B，再把 B 复制出来，由 previous-token head 和 induction head 两层组合实现 — [Transformer Circuits](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) [既有知识，未在本次核对]
- **KVFetch（Dong, 2026-09）**：作者把这个失败模式命名为 “sequential forgetting”。原文写道 content-based eviction “retains the span's head tokens, which carry the query-relevant semantics, but discards mid-span successors whose standalone scores are uniformly low”。模型先正确生成前 j 个 token，然后在第 j+1 个位置“fails irreversibly”。另一处原文：“When the model copies a span verbatim, its attention peak advances monotonically in the original position coordinate” — [arXiv 2610.08811 HTML](https://arxiv.org/html/2610.08811) [全文/HTML 已读，单作者预印本]
- KVFetch 的 Sustained Copy Length 诊断：full KV 平均连续复制 31.25 个 token，H2O 为 5.43，OBCache+SnapKV 为 10.84。作者认为是突然断裂，而不是逐渐退化 — [arXiv 2610.08811 HTML](https://arxiv.org/html/2610.08811)
- KVFetch 在 RULER-16K（Qwen2.5-7B）的逐字复制子任务 s3 上：H2O、SnapKV 和 OBCache+SnapKV 都是 0.8，Quest 8.8，MomentKV（摘要补偿类）4.8，KVFetch 78.4，full KV 99.2。作者把 Quest 的分数稍高归因于页粒度加载 “incidentally co-loads some successors”，并且没有系统研究粒度这个设计维度。数字由 HTML 页面抽取，引用前请核对原表 — [arXiv 2610.08811 HTML](https://arxiv.org/html/2610.08811)
- KVFetch 的修复方法是一条“时间/位置召回通道”：取每步 attention 峰值位置作为读指针；步长 0<Δ≤4 视为前进，−4≤Δ≤0 视为停顿，|Δ|>4 视为跳转并重置；8 步内前进不少于 2 次即进入 copy mode，然后预取指针之后的位置后继进入热层 — [arXiv 2610.08811 HTML](https://arxiv.org/html/2610.08811)
- **RazorAttention（Tang et al., ICLR 2025）**：只为 retrieval heads 保留完整 KV cache，其余 head 压缩，并用一个 “compensation token” 汇总被丢弃 token 的信息；KV cache 减少 70% 以上，性能无明显影响。作者批评 token-dropping 方法 “irreversibly erases critical information that might be needed for future queries” — [arXiv 2407.15891](https://arxiv.org/abs/2407.15891) [仅摘要]；[ICLR 2025 poster](https://iclr.cc/virtual/2025/poster/28028)
- 据我们记忆，RazorAttention 正文把 echo heads 和 induction heads 都算作需要完整 cache 的 retrieval head。摘要页没有出现这些名称，需核对正文 — [arXiv 2407.15891](https://arxiv.org/abs/2407.15891) [既有知识，未在本次核对]
- **DuoAttention（Xiao et al., 2024/ICLR 2025）**：把 head 分为 retrieval heads（完整 KV）和 streaming heads（只保留 attention sink 和最近 token），用基于合成数据（passkey 类）的轻量优化来识别 retrieval heads。显存最多节省 2.55×（MHA）/1.67×（GQA），解码最多加速 2.18×/1.50× — [arXiv 2410.10819](https://arxiv.org/abs/2410.10819) [仅摘要]
- **QRHead（Zhang, Yen, Chen 等，EMNLP 2025）**：指出 NIAH copy-paste 频率只能找出一部分 head；改为用真实任务中对 query 的 attention 聚合来识别 query-focused retrieval heads，并把它们的 attention mass 当作检索分数（QRRetriever） — [arXiv 2506.09944](https://arxiv.org/abs/2506.09944) [仅摘要/检索摘要]
- **LOCOS（Gema, Alex, Minervini, 2026-07）**：认为只看 attention 的检测器 “reward heads whose attended token matches the generated token”，因此漏掉那些通过 OV 电路合成答案的非字面 retrieval heads。他们把 head 输出投影到答案 token 的 unembedding 方向上来打分。在 Qwen3-8B 上消融前 50 个 LOCOS head 后，ROUGE-L 从 0.401 降到 0.000（对照为 0.292） — [arXiv 2607.01002](https://arxiv.org/abs/2607.01002) [仅摘要]
- **RoPE 与 retrieval heads（Bayram, 2026-06）**：在 OLMo-2 中遮蔽 87 个 retrieval head 后，recall 从 1.00 降到 0.00；把 retrieval head 最低频的 RoPE 维度置零会造成剂量依赖的退化（置零 128 维中的 32 维时，recall 从 1.00 降到 0.18）。验证模型包括 Qwen2.5 — [arXiv 2606.21249](https://arxiv.org/abs/2606.21249) [仅摘要，单作者预印本]
- **Optimizing Retrieval Heads（Ma & Okazaki, 2026-01）**：以 retrieval heads 为对象改进长上下文能力，在 citation 生成和段落重排上有提升 — [arXiv 2601.11020](https://arxiv.org/abs/2601.11020) [仅检索摘要]
- **Copy suppression（McDougall et al., 2023）**：GPT-2 small 的 L10H7 是负向 head，当某个 token 已在上下文中且被前层预测时会压低它的 logit，用来校准复制 — [arXiv 2310.04625](https://arxiv.org/abs/2310.04625) [既有知识，未在本次核对]
- **Successor heads（Gould et al., ICLR 2024）**：一些 head 对有序 token（数字、月份等）执行“+1”映射，作者在多个模型族中发现了这种 head，并用 mod-10 特征来解释 — [arXiv 2312.09230](https://arxiv.org/abs/2312.09230) [既有知识，未在本次核对]
- **HeadKV（Fu et al., 2024）**：按 head 的 retrieval-reasoning 重要性分配 KV 预算，而不是在各 head 间平均分配 — [arXiv 2410.19258](https://arxiv.org/abs/2410.19258) [既有知识，未在本次核对]
- **Repeat After Me（Jelassi et al., 2024）**：从理论和实验两方面说明，长串复制依赖能精确寻址的 attention，固定状态的 SSM 在长字符串复制上失败 — [arXiv 2402.01032](https://arxiv.org/abs/2402.01032) [既有知识，未在本次核对]

### Inferences
- 机制上，复制 7 位数字是一个“读指针”过程：生成第 i 位时，复制 head（induction/retrieval 类）需要关注源序列中第 i+1 位所在的位置。块稀疏选择相当于每一步都要求包含指针目标的那个块被选中。答案跨块边界时，尾部块中“与 query 语义相关”的 token 很少，这正是 KVFetch 描述的 “mid-span successors standalone scores uniformly low” 场景。救回实验（强制加入尾部块）与这一机制一致。
- KVFetch 的现象本身不能区分 H1 和 H2。原因有两种可能：一是 prefill 时这些 token 的打分低（这比 H1 更接近“语义低相关”而不是“spike 被平均掉”）；二是解码时块摘要无法反映单个位置的高 q·k（这才是 H1）。H2 的说法是：被选中的是含关键短语的块，尾部数字只有恰好在同一块里才会被带进来。Quest 页粒度“顺带”加载后继 token 的现象，正是 H2 那种“搭便车”机制的实例。
- 干扰 needle 的数字替换尾部，可以用 induction 机制做出可检验的预测。正确的后继块不在选择中时，前缀匹配会找到干扰 needle 里同样的前缀（相同模板、相同的前一位数字），于是复制它的后继。预测：被替换的数字应等于干扰 needle 中“与当前已生成前缀/当前数字匹配的位置”之后的那个数字，而不是随机数字。
- “截断”可能对应 copy-mode 指针失去目标后，模型转而生成结束符或引号等格式 token。个别数字抄错可能是 successor/数字表示的“就近”猜测（见 Q2 中 Levy & Geva 关于逐位误差的发现），但这一点只是推测。
- 稀疏方法给 retrieval head 保留全注意力（DuoAttention/RazorAttention 的思路）是一种直接缓解手段，但 Qwen2.5-1.5B 使用 GQA，KV 在 head 组间共享，需要在 KV-head 粒度上判断。

### Gaps
- 没有找到在 Qwen2.5-1.5B 上公开的 retrieval head 列表或分数，需要自己用 Wu et al. 的方法（[官方仓库](https://github.com/nightdessert/Retrieval_Head)，未在本次核对）计算。
- KVFetch 只研究了 eviction 类方法和 Quest，没有专门研究 32-token 块的边界效应，也没有给出 token 级错误类型（替换/截断）的分布。
- 没有找到直接比较“均值/矩摘要 vs 最大值/逐通道上下界摘要”对复制 head 目标块召回的机制研究。

## Q2：数字 tokenization 与长数字复制错误

### Takeaway
现有工作主要讨论算术中多位 token（如 GPT 系的 3 位一组）带来的位值错位。Qwen 的单数字 token 消除了这种错位，但也让 7 位答案变成 7 个连续的复制步，每一步都依赖指针正确前移。LLM 对数字是逐位表示的，错误倾向于落在个别数位上，而不是在数值上“接近”。这与我们看到的个别数字抄错一致。

### Cited Findings
- **Tokenization counts（Singh & Strouse, 2024）**：在 frontier LLM 的算术任务上，从左到右的 3 位分组 tokenization 会产生有规律的错误（集中在特定数位）；答案位数多于加数时表现明显更差；改为从右到左（如用逗号分隔）能显著改善 — [arXiv 2402.14903](https://arxiv.org/abs/2402.14903) [既有知识 + 检索摘要，未读全文]
- **Levy & Geva（NAACL 2025 short）**：LLM 内部用 base-10 逐位的圆形表示来编码数字，而不是直接编码数值；数值任务中的错误分布在各个数位上，而不是围绕真实数值呈正态分布 — [arXiv 2410.11781](https://arxiv.org/abs/2410.11781)；[ACL Anthology](https://aclanthology.org/2025.naacl-short.33) [仅检索摘要]
- **The Readout Shortcut（Liu, 2026-05）**：在三个 1–3B 指令模型（Qwen、Llama、Gemma 系列）上发现，最终答案往往直接复制 CoT 中最后一个数字；错误样本中最终答案与最后一个 CoT 数字一致的比例为 95–96%，替换这个数字会使准确率接近归零。作者用 head 消融定位到各架构特定的 head 集合。摘要没有说明逐位复制的细节 — [arXiv 2605.22870](https://arxiv.org/abs/2605.22870) [仅摘要，单作者预印本]
- **Benford's Curse（2025）**：把数字生成中的偏差追溯到预训练语料的数字频率分布（摘要层面） — [arXiv 2506.01734](https://arxiv.org/abs/2506.01734) [仅检索标题/摘要，未核对]
- **Lost in the Middle（Liu et al., 2023/TACL 2024）**：长上下文检索准确率随相关信息的位置呈 U 形，中间位置最差 — [arXiv 2307.03172](https://arxiv.org/abs/2307.03172) [既有知识，未在本次核对]

### Inferences
- 单数字 token 下，7 位答案需要 7 次连续的指针前移。只要中间任何一步目标块缺失，后续步就容易出错或被干扰项接管（KVFetch 的“irreversibly”）。因此错误应集中在跨块之后的数位上。可以按“距块边界的偏移”对错误数位做直方图来检验。
- 根据 Levy & Geva 的结论，抄错的数字未必在数值上接近正确答案。区分“干扰 needle 数字”“successor 偏移（±1）”和“高频数字（Benford 偏差）”三类错误，有助于判断失败来自检索、复制还是先验。

### Gaps
- 没有找到专门研究“上下文中长数字串逐字复制”错误机制的论文；现有数字研究以算术为主。
- 没有找到 Qwen 单数字 tokenization 对复制鲁棒性的直接对照研究。

## Q3：用 teacher forcing 与自由生成、activation patching、attention knockout 定位检索/复制失败；方法学注意事项

### Takeaway
标准工具有：attention knockout（阻断特定位置之间的信息流）、activation/path patching，以及按 head 消融。这些工具配合 teacher forcing 可以逐个数位定位“哪一步、哪个 head、哪个块”导致失败。方法学上要注意三点：attention 权重不等于因果贡献；patching 结果对度量和腐化方式敏感；只看“attended token = generated token”的检测器会漏掉非字面的 head。

### Cited Findings
- **Dissecting Recall（Geva et al., EMNLP 2023）**：用 attention knockout（阻断从某些位置到最后位置的注意力边）定位事实回忆中“主语 → 最后位置”的信息流和属性抽取 head — [arXiv 2304.14767](https://arxiv.org/abs/2304.14767) [既有知识，未在本次核对]
- **Activation patching 最佳实践（Zhang & Nanda, ICLR 2024）**：patching 结论对度量（logit difference 还是概率）和腐化方式（高斯噪声还是对称 token 替换）敏感，建议用语义对称的替换和 logit difference — [arXiv 2309.16042](https://arxiv.org/abs/2309.16042) [既有知识，未在本次核对]
- **Attention is not Explanation（Jain & Wallace, 2019）/ Attention is not not Explanation（Wiegreffe & Pinter, 2019）**：attention 权重与基于梯度的重要性往往不一致，存在不同的 attention 分布却给出相同的预测；反驳文章认为在一定条件下 attention 仍有解释价值 — [arXiv 1902.10186](https://arxiv.org/abs/1902.10186)；[arXiv 1908.04626](https://arxiv.org/abs/1908.04626) [既有知识，未在本次核对]
- **LOCOS**：只看 attention 的 retrieval-head 检测只是“read-only”度量，会漏掉写入答案方向的 head；应同时看 OV 输出在答案 unembedding 上的投影（write-aware） — [arXiv 2607.01002](https://arxiv.org/abs/2607.01002) [仅摘要]
- **Stream（2025-10）**：用稀疏注意力把机制可解释性扩展到长上下文 — [arXiv 2510.19875](https://arxiv.org/html/2510.19875v1) [仅检索标题，未读]
- KVFetch 的 Sustained Copy Length 和 attention 峰值读指针本身可以作为诊断量（连续复制长度、峰值位置是否单调前移） — [arXiv 2610.08811 HTML](https://arxiv.org/html/2610.08811)

### Inferences（给我们的诊断建议）
- **teacher-forced 逐位 oracle**：用正确前缀做 teacher forcing，在每个数位步 t 下：(a) 用全注意力取复制 head（先用 Wu et al. 的打分找出 Qwen2.5-1.5B 的 top retrieval heads）的 argmax 源位置 p_t，检查 p_t 是否落在稀疏选择的块中；(b) 记录打分器给 p_t 所在块的排名和分数。如果 teacher forcing 下仍然失败，说明是块选择问题；如果 teacher forcing 下正确、自由生成失败，说明错误在传播（前一位出错使指针偏移）。
- **区分 H1 与 H2**：对 p_t 所在块，比较三种情况：块内单 token 的 q·k 最大值；块均值/矩摘要的打分；含关键短语那个块的打分。H1 的预测是 max(q·k) 很高但均值摘要很低（spike 被稀释），并且换成 max-pooling 或逐通道上下界摘要（Quest 式）就能召回。H2 的预测是，即使打分器用 max 也不会选尾部块，尾部只有与关键短语同块时才被带入；把答案整体移到不跨边界的位置会消除失败，而只改尾部块的表示不会。
- **attention knockout**：在全注意力下只阻断复制 head 对尾部块的注意力，看能否复现 near-miss（截断或干扰替换）。如果能复现，说明尾部块对这些 head 是因果必需的；如果需要阻断所有 head 才能复现，说明信息由多条路径传递。
- **干扰替换的检验**：检查被替换的数字是否等于干扰 needle 中匹配前缀之后的后继（induction 预测），并用 patching 把干扰 needle 的数字换掉，看替换数字是否随之改变。
- **注意**：只用 attention 图下结论是不够的，需要用消融、patching 或 logit 归因（LOCOS 式）互相印证。

### Gaps
- 没有找到在稀疏或块选择解码设定下系统使用 teacher forcing 与自由生成对比的已发表工作。

## Q4：是否有工作明确表明稀疏/近似注意力会导致部分复制（near-miss）错误

### Takeaway
有。最直接的是 KVFetch（2026）：基于内容打分的 KV 选择会让逐字复制在中途断裂（保留头部、丢失后继），页粒度的 Quest 也只部分缓解。KVDiagnosis（2026）报告，大量压缩失败对应证据覆盖“低或部分”。我们没有找到专门研究块边界上数字复制，或系统统计替换与截断比例的工作。

### Cited Findings
- **KVFetch**：失败模式是“生成前 j 个 token 正确，然后在 p_{j+1} 不可逆失败”。RULER-16K 逐字复制子任务上，score-based 方法得 0.8，Quest 8.8，full KV 99.2；KVFetch 恢复到 78.4，13 个任务平均提升 +8.4 — [arXiv 2610.08811](https://arxiv.org/abs/2610.08811) [全文/HTML 已读]
- **KVDiagnosis（2026-08）**：分析了 59,800 次运行中的 12,520 个失败，其中 63.2% 的证据覆盖低或部分（40.3% 低，22.9% 部分） — [arXiv 2608.09412](https://arxiv.org/abs/2608.09412) [仅检索摘要]
- **Divergence Timing and Cumulative Disagreement under KV-Cache Eviction（2026-09）**：研究 eviction 后生成在哪一步开始与全 KV 分歧 — [arXiv 2609.16617](https://arxiv.org/pdf/2609.16617) [仅检索标题，未读]
- RazorAttention 的论点：token-dropping 会不可逆地抹掉后续 query 需要的信息 — [arXiv 2407.15891](https://arxiv.org/abs/2407.15891) [仅摘要]

### Inferences
- KVFetch 的“读指针 + 预取后继”思路可以直接迁移为一种块选择启发式：检测到 copy mode（attention 峰值单调前移）时，自动把峰值所在块的下一块加入选择。这与我们“强制加入尾部块即可救回”的结果在机制上一致，可作为一个低成本的对照方法。
- 和 KVFetch 不同，我们的设置是解码期按 query 动态选块，而不是 prefill 期一次性淘汰。所以如果 H1 成立，失败点应是“解码期块摘要与单位置峰值不匹配”，而不只是“prefill 打分低”。这是两者的关键区别，值得在报告中明确写出。

### Gaps
- 没有找到在块稀疏动态选择（Quest、NSA、MoBA 类）下按块边界统计 near-miss 错误类型的论文。
- KVDiagnosis 和 Divergence Timing 两篇本次未读全文，具体与“部分复制”的关系有待核对。
