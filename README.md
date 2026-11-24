# 从约束推导 AI Infra

**在线阅读：<https://wikieden.github.io/ai-infra-from-constraints/>** · [讲义目录](https://wikieden.github.io/ai-infra-from-constraints/lessons/index.html) · [全部动画](https://wikieden.github.io/ai-infra-from-constraints/animations/index.html) · [复习卡片](https://wikieden.github.io/ai-infra-from-constraints/review.html) · [全部讲义书](https://wikieden.github.io/ai-books/)

> 30 / 67 章已写（覆盖 AII 第 1–5 章）· 31 个动画（GIF + 可交互页面）· 104 张静态图 · 118 张复习卡片 · 主线《深入理解 AI Infra》（AII）12 章，AISPE 配读

## 这本书讲什么

从一个"帮我找代码里的 bug"的请求讲起：它在系统里穿过哪几层、要存多少、读多少、算多少、在卡之间传多少，最后在哪块硬件上等。之后沿原书顺序，从模型架构、负载、单卡加速器和算子，一路讲到超节点、数据中心网络、推理与训练系统、资源调度和端边云。

- **适合谁**：会一点 Transformer（知道 q、k、v 和 attention），对 GPU 和分布式系统没有基础也可以；用到的硬件、数学和其他书的概念，在那一章现场补，或链接到对应的书。
- **读完能做什么**：拿到一个（模型、请求负载、硬件）组合，先用数量级估算定下显存、带宽、通信和卡数，再判断实测离物理极限差在哪一项（study-plan 的目标，对应里程碑 AI-M0–M11）。
- **怎么读**：按章顺序读。每章先看讲解和动画，再自己做练习，最后用复习卡片隔几天考自己。

全书反复回答五个问题：**搬什么、搬多少、搬几次、经过哪里、谁在等待**；主线是原书的一句话：**数据搬移塑造了 AI Infra 的架构**。

## 讲解的特点

- **一次一小步**：每个概念按"白话类比 → 公式（每个符号说清楚怎么读）→ 小数字亲手算一遍"讲。
- **先算极限，再看差距**：每个估算都先用峰值算力、带宽、容量算出下界，再和原书实测对比，分清差距来自模型漏项还是系统开销。
- **关键机制都有图和动画**：结构和数据流画静态图，数字之间的关系画图表，要拖参数才看得出规律的做成可交互动画（batch、tile 边长、链路速率……）。
- **对照原文**：讲到原文的地方标本地行号（如 AII §1.3 L274），数字都能在原书或它的计算脚本里查到；编的数会写明"编的"。
- **保留真实的学习过程**：练习与作答记录里是学习时的原话、对错和点评。
- **术语不设门槛**：缩写第一次出现就给英文全称并逐词解释，全书缩写见 [缩写与术语表](https://wikieden.github.io/ai-infra-from-constraints/lessons/appendix-glossary.html)。
- **每章都有代码对照**：`solutions/chapter-code/` 一章一个纯 Python 模块，配测试核对讲义里的数字和一本 notebook；只算讲过的例子，不提前给练习答案。
- **每章收尾有口诀**：全部口诀汇总在本页末尾。

## 示例动画

| | |
|---|---|
| ![六层与 KV 缓存](https://wikieden.github.io/ai-infra-from-constraints/animations/01-six-layers.gif)<br>**第 01 章 · 六层**：一次请求从上往下穿过六层；KV 缓存像便签越积越厚<br>[讲义](https://wikieden.github.io/ai-infra-from-constraints/lessons/01-six-layers.html) · [▶ 交互](https://wikieden.github.io/ai-infra-from-constraints/animations/01-six-layers.html) | ![一次生成请求](https://wikieden.github.io/ai-infra-from-constraints/animations/02-request-lifecycle.gif)<br>**第 02 章 · 请求处理过程**：路由到实例，prefill 一次、decode 一步步；权重加载 1 次、读 N 次<br>[讲义](https://wikieden.github.io/ai-infra-from-constraints/lessons/02-request-lifecycle.html) · [▶ 交互](https://wikieden.github.io/ai-infra-from-constraints/animations/02-request-lifecycle.html) |
| ![分块](https://wikieden.github.io/ai-infra-from-constraints/animations/03-dataflow-hierarchy.gif)<br>**第 03 章 · 分块**：一个数从显存读一次，在近处被用 t 次<br>[讲义](https://wikieden.github.io/ai-infra-from-constraints/lessons/03-dataflow-hierarchy.html) · [▶ 交互](https://wikieden.github.io/ai-infra-from-constraints/animations/03-dataflow-hierarchy.html) | ![Amdahl 定律](https://wikieden.github.io/ai-infra-from-constraints/animations/04-key-numbers.gif)<br>**第 04 章 · Amdahl 定律**：只加速一部分，整体最多快 1/(1-f)<br>[讲义](https://wikieden.github.io/ai-infra-from-constraints/lessons/04-key-numbers.html) · [▶ 交互](https://wikieden.github.io/ai-infra-from-constraints/animations/04-key-numbers.html) |
| ![批内复用与转折点](https://wikieden.github.io/ai-infra-from-constraints/animations/05-estimate.gif)<br>**第 05 章 · 批内复用**：读权重不变、计算随 batch 增长，转折点 B* ≈ 148<br>[讲义](https://wikieden.github.io/ai-infra-from-constraints/lessons/05-estimate.html) · [▶ 交互](https://wikieden.github.io/ai-infra-from-constraints/animations/05-estimate.html) | ![SmartNIC](https://wikieden.github.io/ai-infra-from-constraints/animations/06-architecture.gif)<br>**第 06 章 · SmartNIC**：网速越高，光转发要越多 CPU 核<br>[讲义](https://wikieden.github.io/ai-infra-from-constraints/lessons/06-architecture.html) · [▶ 交互](https://wikieden.github.io/ai-infra-from-constraints/animations/06-architecture.html) |

全部动画和本地打开方法见 [animations/README.md](https://wikieden.github.io/ai-infra-from-constraints/animations/index.html)。

## 目录

| 章 | 主题 | 一句话精要 | 对应原文 | 动画 | 卡片 | 状态 |
|---|---|---|---|---|---|---|
| [01](https://wikieden.github.io/ai-infra-from-constraints/lessons/01-six-layers.html) | 一次请求走过的六层 | 任务从上往下变成"算什么、搬什么"，硬件从下往上限制能做什么；KV 缓存变小，下面各层跟着变 | AII §1.1.2 L29–L53、§1.1.3 L65 | [▶](https://wikieden.github.io/ai-infra-from-constraints/animations/01-six-layers.html) | I01–I02 | 🟡 进行中（练习 2 待答） |
| [02](https://wikieden.github.io/ai-infra-from-constraints/lessons/02-request-lifecycle.html) | 一次生成请求的处理过程 | 路由器管"去哪家"，调度器管"怎么做"；权重加载 1 次，输出 N 个 token 就读 N 遍 | AII §1.1.3 L55–L95 | [▶](https://wikieden.github.io/ai-infra-from-constraints/animations/02-request-lifecycle.html) | I03–I06 | 🟡 正文已写，练习 0 / 3 |
| [03](https://wikieden.github.io/ai-infra-from-constraints/lessons/03-dataflow-hierarchy.html) | 从存储程序到数据流执行 | 存储层次在"近"和"大"之间折中；分块把"读一次、用多次"写进执行结构 | AII §1.1.4 L97–L103 | [▶](https://wikieden.github.io/ai-infra-from-constraints/animations/03-dataflow-hierarchy.html) | I07–I09 | 🟡 正文已写，练习 0 / 3 |
| [04](https://wikieden.github.io/ai-infra-from-constraints/lessons/04-key-numbers.html) | 系统设计的关键数字 | 先看数量级；Amdahl 定律给出上限；容量、算力、带宽、延迟分开算 | AII §1.2 L105–L231 | [▶](https://wikieden.github.io/ai-infra-from-constraints/animations/04-key-numbers.html) | I10–I13 | 🟡 正文已写，练习 0 / 3 |
| [05](https://wikieden.github.io/ai-infra-from-constraints/lessons/05-estimate.html) | 用几个数字估算一次模型执行 | 读权重比计算慢 148 倍；批内复用涨吞吐不缩等待；差距不是漏项就是开销 | AII §1.3 L233–L365 | [▶](https://wikieden.github.io/ai-infra-from-constraints/animations/05-estimate.html) | I14–I18 | 🟡 正文已写，练习 0 / 3 |
| [06](https://wikieden.github.io/ai-infra-from-constraints/lessons/06-architecture.html) | 需求如何推动架构 | 比例一变，分工重排：TPU、SmartNIC、UB；老办法在新比例下又合算 | AII §1.4 L367–L454 | [▶](https://wikieden.github.io/ai-infra-from-constraints/animations/06-architecture.html) | I19–I22 | 🟡 正文已写，练习 0 / 3 |
| [07](https://wikieden.github.io/ai-infra-from-constraints/lessons/07-compute-graph.html) | 计算图与张量结构 | 投影和 FFN 按行数 m 算，注意力按查询—键配对数算；8K prefill 133.6 TFLOPs，一步 decode 20 GFLOPs | AII §2.1 L15–L213 | [▶](https://wikieden.github.io/ai-infra-from-constraints/animations/07-compute-graph.html) | I23–I26 | 🟡 正文已写，练习 0 / 3 |
| [08](https://wikieden.github.io/ai-infra-from-constraints/lessons/08-generation-budget.html) | 自回归生成的资源账 | 缓存 K、V 不缓存 Q；batch 超过 R_W/(c_KV·H) 读 KV 就多于读权重；累计读取是保存的近千倍 | AII §2.2 L214–L301 | [▶](https://wikieden.github.io/ai-infra-from-constraints/animations/08-generation-budget.html) | I27–I30 | 🟡 正文已写，练习 0 / 3 |
| [09](https://wikieden.github.io/ai-infra-from-constraints/lessons/09-kv-sharing.html) | KV 怎么变小 ①：头共享与潜变量压缩 | GQA / MQA 减 KV 份数不减查询头；MLA 用结合律只存低维潜变量 | AII §2.3.1–§2.3.2 L302–L356 | [▶](https://wikieden.github.io/ai-infra-from-constraints/animations/09-kv-sharing.html) | I31–I33 | 🟡 正文已写，练习 0 / 3 |
| [10](https://wikieden.github.io/ai-infra-from-constraints/lessons/10-kv-sparse-linear.html) | KV 怎么变小 ②：窗口、稀疏、线性与混合注意力 | 窗口管近期、压缩管远期、索引管挑选；递推状态固定大小；混合模型看交点 H* | AII §2.3.3–§2.3.6 L358–L523 | [▶](https://wikieden.github.io/ai-infra-from-constraints/animations/10-kv-sparse-linear.html) | I34–I37 | 🟡 正文已写，练习 0 / 3 |
| [11](https://wikieden.github.io/ai-infra-from-constraints/lessons/11-moe.html) | MoE：专家路由与批内权重复用 | 存 E·N_e、用 k·N_e；运算按分派行数、读取按不同专家数，同样计算读取可差 32 倍 | AII §2.4 L524–L586 | [▶](https://wikieden.github.io/ai-infra-from-constraints/animations/11-moe.html) | I38–I40 | 🟡 正文已写，练习 0 / 3 |
| [12](https://wikieden.github.io/ai-infra-from-constraints/lessons/12-network-shape.html) | 网络形状：深宽、残差与多 token 预测 | 深宽与专家粒度在不变量下改形状；残差方式决定要留什么；推测解码要 ḡ > (T_s+T_v)/T_d | AII §2.5 L587–L636 | [▶](https://wikieden.github.io/ai-infra-from-constraints/animations/12-network-shape.html) | I41–I43 | 🟡 正文已写，练习 0 / 3 |
| [13](https://wikieden.github.io/ai-infra-from-constraints/lessons/13-model-comparison.html) | 整个模型的资源需求与比较 | 权重、单次运算、每请求状态三个维度分开比；B_max = ⌊剩余显存 ÷ 每请求状态⌋ | AII §2.6 L637–L909 | [▶](https://wikieden.github.io/ai-infra-from-constraints/animations/13-model-comparison.html) | I44–I48 | 🟡 正文已写，练习 0 / 3 |
| [14](https://wikieden.github.io/ai-infra-from-constraints/lessons/14-workload-arrival.html) | 从单次请求到持续到达的负载 | TTFT、输出间隔、完整请求各量一段；等待也按 GiB·s 占容量；平均够用、突发照样积压 | AII §3.1 L1–L146 | [▶](https://wikieden.github.io/ai-infra-from-constraints/animations/14-workload-arrival.html) | I49–I52 | 🟡 正文已写，练习 0 / 3 |
| [15](https://wikieden.github.io/ai-infra-from-constraints/lessons/15-agent-tasks.html) | 多轮交互与 Agent 任务 | 每轮重算 I−K，改在哪比改多少要紧；成功任务成本 c/p；工具并行缩关键路径不减工作 | AII §3.2 L147–L279 | [▶](https://wikieden.github.io/ai-infra-from-constraints/animations/15-agent-tasks.html) | I53–I56 | 🟡 正文已写，练习 0 / 3 |
| [16](https://wikieden.github.io/ai-infra-from-constraints/lessons/16-multimodal-realtime.html) | 多模态与实时交互（略讲） | 一张图在原文件、EC、语言 KV 三站差两个数量级；音频缓冲吸收抖动但推迟首播 | AII §3.3 L280–L379 | [▶](https://wikieden.github.io/ai-infra-from-constraints/animations/16-multimodal-realtime.html) | I57–I59 | 🟡 正文已写，练习 0 / 3 |
| [17](https://wikieden.github.io/ai-infra-from-constraints/lessons/17-training-rl.html) | 训练与强化学习的计算和状态 | 训练是前向的三倍（6ND），每参数 18 字节；6ND 漏注意力、多算词嵌入；RL 筛得越严生成越多 | AII §3.4–§3.5 L380–L548 | [▶](https://wikieden.github.io/ai-infra-from-constraints/animations/17-training-rl.html) | I60–I63 | 🟡 正文已写，练习 0 / 3 |
| [18](https://wikieden.github.io/ai-infra-from-constraints/lessons/18-scaling-lifecycle.html) | Scaling law 与全生命周期成本 | 固定预算按指数分参数和数据；实际训练远超计算最优，因为要算服务成本；回本点 ΔC0/Δc | AII §3.6 L549–L707 | [▶](https://wikieden.github.io/ai-infra-from-constraints/animations/18-scaling-lifecycle.html) | I64–I68 | 🟡 正文已写，练习 0 / 3 |
| [19](https://wikieden.github.io/ai-infra-from-constraints/lessons/19-accelerator-basics.html) | 从矩阵乘法认识加速器 | 投影强度封顶 KN/(K+N)；面积给缓冲还是乘加；搬得越远越耗电，单请求 decode 九成能量在读权重；HBM 堆与 TDP 封顶 | AII §4.1 L1–L222 | [▶](https://wikieden.github.io/ai-infra-from-constraints/animations/19-accelerator-basics.html) | I69–I73 | 🟡 正文已写，练习 0 / 3 |
| [20](https://wikieden.github.io/ai-infra-from-constraints/lessons/20-compute-units.html) | 计算单元：矩阵、向量、流水与低精度 | 矩阵单元按块复用、行少就补零；归约要一轮轮等；注意力流水 τ 由最慢资源定；低精度省字节要算转换账 | AII §4.2 L223–L370 | [▶](https://wikieden.github.io/ai-infra-from-constraints/animations/20-compute-units.html) | I74–I77 | 🟡 正文已写，练习 0 / 3 |
| [21](https://wikieden.github.io/ai-infra-from-constraints/lessons/21-memory-movement.html) | 存储层次与数据搬移 | 工作台放两块操作数加部分和；Little 定律定在途事务；步长寻址与描述符；缓冲槽数随计算变快而增；跨 SM 只能经 L2 | AII §4.3–§4.4 L371–L649 | [▶](https://wikieden.github.io/ai-infra-from-constraints/animations/21-memory-movement.html) | I78–I82 | 🟡 正文已写，练习 0 / 3 |
| [22](https://wikieden.github.io/ai-infra-from-constraints/lessons/22-package-interconnect.html) | 多芯片封装与互联 | 链路 / 本地带宽比决定计算放哪；H2D 要复用 32 次以上；卡间 α + V/R，转折点 αR | AII §4.5 L650–L730 | [▶](https://wikieden.github.io/ai-infra-from-constraints/animations/22-package-interconnect.html) | I83–I85 | 🟡 正文已写，练习 0 / 3 |
| [23](https://wikieden.github.io/ai-infra-from-constraints/lessons/23-architecture-evolution.html) | 架构演进与专用架构 | Tensor Core 只帮计算主导；精度省字节 × 带宽省时间；统一内存容量 vs 带宽；片上 SRAM 与固定权重 ROM 的账 | AII §4.6–§4.7 L731–L959 | [▶](https://wikieden.github.io/ai-infra-from-constraints/animations/23-architecture-evolution.html) | I86–I91 | 🟡 正文已写，练习 0 / 4 |
| [24](https://wikieden.github.io/ai-infra-from-constraints/lessons/24-choose-accelerator.html) | 根据架构选择加速器 | 同资源相加、异资源取最大、阶段串联；decode 看带宽、prefill 看矩阵；实测先修口径再去开销；成本与回本 | AII §4.8 L960–L1259 | [▶](https://wikieden.github.io/ai-infra-from-constraints/animations/24-choose-accelerator.html) | I92–I96 | 🟡 正文已写，练习 0 / 4 |
| [25](https://wikieden.github.io/ai-infra-from-constraints/lessons/25-submit-sync.html) | 提交、搬移与同步 | launch 异步，返回不等于算完；stream 内按序、跨流靠 event；kernel 快 4 倍整体只快 1.75 倍；驻留块数由最紧的资源定，在途字节要够带宽 × 延迟 | AII §5.1 L1–L175 | [▶](https://wikieden.github.io/ai-infra-from-constraints/animations/25-submit-sync.html) | I97–I100 | 🟡 正文已写，练习 0 / 3 |
| [26](https://wikieden.github.io/ai-infra-from-constraints/lessons/26-tiling-layout.html) | 单个算子：分块、布局与切分 | 128 MiB 的矩阵乘能读出 3 GiB；输出块越大读得越少但驻留越少；padding / swizzle 分散 bank；拆归约维换并行，代价是重读和浮点次序 | AII §5.2 L176–L338 | [▶](https://wikieden.github.io/ai-infra-from-constraints/animations/26-tiling-layout.html) | I101–I104 | 🟡 正文已写，练习 0 / 3 |
| [27](https://wikieden.github.io/ai-infra-from-constraints/lessons/27-fusion-flash.html) | 算子融合与 FlashAttention | 融合省中间量的一写一读；双缓冲只藏住较短的一段；在线 Softmax 带 m、ℓ、u 逐块更新；K/V 块越小读得越少但更新越多 | AII §5.3 L339–L480 | [▶](https://wikieden.github.io/ai-infra-from-constraints/animations/27-fusion-flash.html) | I105–I108 | 🟡 正文已写，练习 0 / 3 |
| [28](https://wikieden.github.io/ai-infra-from-constraints/lessons/28-compiler.html) | 编译器：表达、变换与选择（部分跳过） | split / reorder / compute_at 定生命周期；归约合并与舍入位置不能动；先排除、再估计、按调用分布实测 | AII §5.4 L481–L767（跳过 AKG、TileLang-Ascend） | [▶](https://wikieden.github.io/ai-infra-from-constraints/animations/28-compiler.html) | I109–I111 | 🟡 正文已写，练习 0 / 3 |
| [29](https://wikieden.github.io/ai-infra-from-constraints/lessons/29-runtime.html) | 运行时：重叠、CUDA Graph、动态形状与 persistent kernel | 加速器快过主机后节拍由主机定；图重放省提交但要付输入复制；通用 / 分桶 / 特化按 P + rT 换手；persistent kernel 按块交接 | AII §5.5 L768–L905 | [▶](https://wikieden.github.io/ai-infra-from-constraints/animations/29-runtime.html) | I112–I115 | 🟡 正文已写，练习 0 / 3 |
| [30](https://wikieden.github.io/ai-infra-from-constraints/lessons/30-request-view.html) | 从局部优化到完整请求 | Amdahl 封顶 1/(1−f)；并行分支的关键路径会换人；SwiGLU 快 2.6 倍，请求只快 0.16% | AII §5.6 L906–L1118 | [▶](https://wikieden.github.io/ai-infra-from-constraints/animations/30-request-view.html) | I116–I118 | 🟡 正文已写，练习 0 / 3 |

**第 1 部分待补**：AII §1.1.1 L9–L27（编程抽象上移：模型成了 LLM 时代的"操作系统"）还没讲，讲第 01 章时并进去，作为第零步；前言（`00-前言.md`）不单独成章，要点已写进本页"这本书讲什么"。

---

本仓库由私有学习仓库里的脚本 `tools/publish-transformer/build.py` 自动生成，内容摘自讲义目录，不要直接修改这里的文件。
