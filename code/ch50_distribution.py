"""第 50 章 · 计算与状态的分布方式。

讲义：ai-infra/notes/lessons/50-distribution.md
运行：uv run python ch50_distribution.py
四张 A100 + 四张 H20、每秒 3.5 个 8K 输入 / 1025 输出请求、Qwen3-8B BF16，取自 AII 第 9 章 §9.1。
"""

GiB = 2**30
K_TOKEN = 2 * 36 * 8 * 128 * 2  # Qwen3-8B 每 token KV 147,456 B = 144 KiB
K_235B = 2 * 94 * 4 * 128 * 2   # Qwen3-235B-A22B 每 token KV 192,512 B = 188 KiB

# 每请求的 GPU 秒（第 51 章从峰值推出，这里先拿来用）
D_P = {"A100": 0.8564, "H20": 1.8054}
D_D = {"A100": 1.7634, "H20": 0.8782}


def gpus_needed(lam: float, d: float) -> float:
    """λ d：持续工作所需的资源单元数，需要 < n_r。"""
    return lam * d


def colocated_rate(cards: dict[str, int]) -> float:
    """共置副本的请求率上限：Σ N / (d_P + d_D)。"""
    return sum(n / (D_P[c] + D_D[c]) for c, n in cards.items())


def kv_gib(tokens: int, copies: int = 1, per_token: int = K_TOKEN) -> float:
    return tokens * per_token * copies / GiB


def residency(gib: float, seconds: float) -> float:
    """容量 × 时间（GiB·s）。"""
    return gib * seconds


def main() -> None:
    print("\n第一步：资源约束 λ d < n")
    print(f"  每秒 4 个请求、每个 0.5 GPU 秒 → 需要 {gpus_needed(4, 0.5):.1f} 张持续工作的卡")
    for c in D_P:
        print(f"  {c}：P {D_P[c]:.3f} + D {D_D[c]:.3f} = {D_P[c] + D_D[c]:.2f} GPU 秒/请求")
    mu = colocated_rate({"A100": 4, "H20": 4})
    print(f"  八个完整副本：{mu:.2f} 请求/s，到达 3.5 → 每秒积压 {3.5 - mu:.2f}")

    print("\n第二步：TP2×EP4 的一个实例")
    print(f"  Qwen3-235B 注意力与 KV 在四个 EP 组各复制一份：8K 上下文 {kv_gib(8192, 1, K_235B):.3f} GiB × 4 = {kv_gib(8192, 4, K_235B):.3f} GiB")

    print("\n第三步：状态驻留")
    one = kv_gib(8192)
    print(f"  8K 上下文 {one:.3f} GiB，工具等 10 s → {residency(one, 10):.2f} GiB·s；十个会话同时等 → {10 * one:.2f} GiB")
    print(f"  返回 1025 个 token 后 KV 覆盖 {8192 + 1024} 个 token（最后一个还没送回模型）")


if __name__ == "__main__":
    main()
