"""第 17 章 · 训练与强化学习的计算和状态：为什么训练要 6ND、要存 18 字节。

讲义：ai-infra/notes/lessons/17-training-rl.md
运行：uv run python ch17_training_rl.py
例题 3-3（Qwen3-8B 8,190,735,360 个参数）、长度分组、8192 token 训练矩阵运算、例题 3-4 的各阶段运算量取自 AII 第 3 章 §3.4–§3.5；
m = 2、k = 3、n = 4 是编的。
"""

N_QWEN = 8_190_735_360
# 例题 3-4：32 条回答时各阶段（TFLOPs），保留 16 条
F_PER_RESP = (465.143 + 129.056 + 634.944) / 32  # 一条回答的 prefill + decode + 参考模型前向
F_UPDATE = 952.416  # 16 条保留样本的一次策略更新


def linear_train_flops(m: int, k: int, n: int) -> tuple[int, int, int]:
    """(前向, 输入梯度, 权重梯度)，各 2mkn。"""
    f = 2 * m * k * n
    return f, f, f


def train_state_gb(n_params: int, bytes_per: int = 18) -> tuple[float, float]:
    """参数相关的长期训练状态：(GB, GiB)。默认 BF16 权重 2 + FP32 梯度 / 主权重 / Adam 两矩各 4。"""
    b = n_params * bytes_per
    return b / 1e9, b / 2**30


def doc_pairs(lengths: list[int]) -> int:
    """若干篇独立文档的有效因果配对数：每篇一个三角形 P(P+1)/2。"""
    return sum(p * (p + 1) // 2 for p in lengths)


def rl_cycle(n_generate: int) -> float:
    """RL 一轮的矩阵运算（TFLOPs）= 生成数 × (生成 + 参考模型) + 固定更新。"""
    return n_generate * F_PER_RESP + F_UPDATE


def sync_gb(replicas: int, weight_gb: float = 16.38147072) -> float:
    """每个生成副本独立收一份完整权重的总传输量。"""
    return replicas * weight_gb


def main() -> None:
    print("\n第一步：线性层训练（编的 m=2、k=3、n=4）")
    print(f"  前向 / 输入梯度 / 权重梯度 = {linear_train_flops(2, 3, 4)}，合计 {sum(linear_train_flops(2, 3, 4))}")

    print("\n第二步：Qwen3-8B 全参数训练的参数相关状态")
    gb, gib = train_state_gb(N_QWEN)
    print(f"  18 B × 参数 = {gb:.3f} GB = {gib:.3f} GiB（推理 BF16 权重的 9 倍）")

    print("\n第三步：同样 8192 个 token，分组不同")
    for g in ([4096, 4096], [7168, 1024]):
        pairs = doc_pairs(g)
        print(f"  {g}：配对 {pairs:,}，一层注意力前向 {16384 * pairs / 1e12:.3f} TFLOPs")
    print(f"  逐项 431.368 vs 6ND 402.591：高 {431.368 / 402.591 - 1:.2%}")

    print("\n第四步：RL 一轮（保留 16 条）")
    for n in (32, 64):
        c = rl_cycle(n)
        print(f"  生成 {n} 条：{c:,.3f} TFLOPs，每个保留样本 {c / 16:.3f}")
    print(f"  4 个副本同步权重：{sync_gb(4):.3f} GB")


if __name__ == "__main__":
    main()
