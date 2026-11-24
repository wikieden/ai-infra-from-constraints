"""第 18 章 · Scaling law 与全生命周期成本：训练多大、训练多久、服务多少次。

讲义：ai-infra/notes/lessons/18-scaling-lifecycle.md
运行：uv run python ch18_scaling_lifecycle.py
例题 3-5 的留出点、Llama / Qwen 训练投入表、GPU 小时、回本例子取自 AII 第 3 章 §3.6。
"""

import math

# (模型, 参数, 训练 token)
HISTORY = [
    ("Llama 1 6.7B", 6.7e9, 1e12),
    ("Llama 2 7B", 7e9, 2e12),
    ("Llama 3.1 8B", 8e9, 15e12),
    ("Qwen2.5 7B", 7e9, 18e12),
    ("Qwen3 8B", 8e9, 36e12),
]


def alloc_exponents(alpha: float, beta: float) -> tuple[float, float]:
    """最优分配的指数：N_opt ∝ C^(β/(α+β))，D_opt ∝ C^(α/(α+β))。"""
    return beta / (alpha + beta), alpha / (alpha + beta)


def rmse(pred: list[float], actual: list[float]) -> float:
    """均方根误差。"""
    return math.sqrt(sum((p - a) ** 2 for p, a in zip(pred, actual)) / len(pred))


def history() -> list[tuple[str, float, float]]:
    """(模型, D/N, 6ND)。"""
    return [(name, d / n, 6 * n * d) for name, n, d in HISTORY]


def expert_train_flops(n_layers: int, h: int, f: int, k_routed: int, k_shared: int) -> float:
    """MoE 主干专家每 token 的训练运算：18 · n_L · h · f · (k_r + k_s)。"""
    return 18 * n_layers * h * f * (k_routed + k_shared)


def calendar_days(gpu_hours: float, n_gpu: int) -> float:
    """工期（天）= GPU 小时 ÷ 卡数 ÷ 24。"""
    return gpu_hours / n_gpu / 24


def breakeven_tasks(extra_train_gpu_s: float, saving_per_task_gpu_s: float) -> float:
    """多花的训练要多少次任务才回本：ΔC0 / Δc。"""
    return extra_train_gpu_s / saving_per_task_gpu_s


def main() -> None:
    print("\n第一步：α = β 时的分配指数")
    print(f"  N、D 各 ∝ C^{alloc_exponents(0.3, 0.3)[0]}：预算 ×4 → 各 ×{4 ** 0.5:.0f}")

    print("\n第二步：例题 3-5 的两个留出点")
    print(f"  均方根误差 {rmse([2.583, 2.363], [2.574, 2.337]):.4f}（原文用未取整的值得 0.0193）")

    print("\n第三步：相近规模模型的训练投入")
    for name, dn, c in history():
        print(f"  {name:13s} D/N ≈ {dn:7,.0f}   6ND ≈ {c:.3g}")
    print(f"  V4-Flash 主干专家：{expert_train_flops(43, 4096, 2048, 6, 1) / 1e9:.2f} GFLOPs/token")

    print("\n第四步：GPU 小时 → 工期")
    for name, h, n in (("Llama 1 65B", 1_022_362, 2048), ("DeepSeek-V3 预训练", 2.664e6, 2048), ("Llama 3.1 405B", 30.84e6, 16384)):
        print(f"  {name}：{calendar_days(h, n):.2f} 天")
    print(f"  H100 / A100 峰值比 {989.4 / 312:.2f}")

    print("\n第五步：回本")
    print(f"  多训 10 万 GPU 小时、每次省 0.72 GPU 秒 → {breakeven_tasks(1e5 * 3600, 0.72):.0e} 次")


if __name__ == "__main__":
    main()
