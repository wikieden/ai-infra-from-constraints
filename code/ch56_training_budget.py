"""第 56 章 · 训练的容量与时间估算。

讲义：ai-infra/notes/lessons/56-training-budget.md
运行：uv run python ch56_training_budget.py
Qwen3-8B 继续训练 100B token、30 天期限；临界 batch、每参数 16 字节、MFU 与卡数下界，取自 AII 第 10 章 §10.1。
"""

import math

N = 8_190_735_360               # Qwen3-8B 参数数
D = 100e9                        # 有效 token
SEQ, BATCH = 8192, 384
TOK_STEP = SEQ * BATCH           # 3,145,728
FLOP_TOKEN = 4.31e14 / 8192      # 前向 + 反向每 token 约 52.7 GFLOPs
DAY = 86400


def steps(d: float = D, tok_step: int = TOK_STEP) -> int:
    return math.ceil(d / tok_step)


def state_bytes(n: int = N, per_param: int = 16) -> float:
    """混合精度 Adam：BF16 权重 2 + BF16 梯度 2 + FP32 主权重 4 + 两份矩 8 = 16 字节。"""
    return n * per_param


def cards_for_compute(peak: float, eta: float = 0.4, days: float = 30, d: float = D) -> int:
    """p = ⌈F / (P η T)⌉。"""
    return math.ceil(d * FLOP_TOKEN / (peak * eta * days * DAY))


def step_compute_s(cards: int, peak: float = 165.2e12, eta: float = 0.4) -> float:
    return TOK_STEP * FLOP_TOKEN / (cards * peak * eta)


def critical_fit(S: int = 31790, B: float = TOK_STEP, b_noise: float = 2e6) -> tuple[float, float]:
    """由 (S/S_min − 1)(E/E_min − 1) = 1、B_crit = E_min/S_min = b_noise 反推 S_min、E_min。"""
    k = B / b_noise                    # E/E_min = k · S/S_min
    x = (k + 1) / k                    # (x − 1)(kx − 1) = 1 的正根
    s_min = S / x
    return s_min, s_min * b_noise


def steps_at(B: float, s_min: float, e_min: float) -> float:
    """给定 batch，达到同一损失所需的步数。"""
    k = B * s_min / e_min
    return (k + 1) / k * s_min


def main() -> None:
    n = steps()
    print("\n第一步：任务与时间预算")
    print(f"  {n:,} 步；30 天平均要 {D / (30 * DAY):,.0f} token/s，留 5 天后 {D / (25 * DAY):,.0f} token/s；每步 {25 * DAY / n:.1f} s")
    s_min, e_min = critical_fit()
    print(f"  B_noise = 2M token：S_min {s_min:,.0f} 步、E_min {e_min / 1e9:.1f}B token；当前 batch 是 B_noise 的 {TOK_STEP / 2e6:.2f} 倍")
    for cards in (48, 96, 192, 384, 1536):
        k = cards // 48
        weak = steps_at(TOK_STEP * k, s_min, e_min) * 52.8 / DAY
        strong = n * (52.2 / k + 0.58) / DAY
        print(f"  {cards:>5} 卡：弱扩展 {weak:.1f} 天，强扩展 {strong:.1f} 天")
    print(f"  弱扩展上限 {n / s_min:.2f} 倍；B_noise 换成 2000 万 → {n / critical_fit(b_noise=2e7)[0]:.2f} 倍")

    print("\n第二步：状态容量")
    for name, b in (("BF16 权重", 2), ("BF16 梯度", 2), ("FP32 主权重", 4), ("Adam 一阶矩", 4), ("Adam 二阶矩", 4)):
        print(f"  {name}：{state_bytes(per_param=b) / 1e9:.1f} GB")
    print(f"  合计 {state_bytes() / 1e9:.1f} GB = {state_bytes() / 2**30:.1f} GiB；梯度 FP32 时 {state_bytes(per_param=18) / 1e9:.1f} GB")
    print(f"  FP8 操作数副本每参数 {1 + 4 / 16384:.4f} B，读权重 {N * (1 + 4 / 16384) / 1e9:.1f} GB（常驻状态不变）")

    print("\n第三步：计算量与卡数")
    print(f"  每 token {FLOP_TOKEN / 1e9:.1f} GFLOPs，100B token 共 {D * FLOP_TOKEN:.2e} FLOPs")
    print(f"  H100 40%：每卡 {989e12 * 0.4 / FLOP_TOKEN:,.0f} token/s，至少 {cards_for_compute(989e12)} 张；容量只要 2 张 80 GB，但要 {D * FLOP_TOKEN / (2 * 989e12 * 0.4) / DAY:.0f} 天")
    print(f"  RTX 4090 165.2 TFLOP/s：至少 {cards_for_compute(165.2e12)} 张；每步 {TOK_STEP * FLOP_TOKEN:.2e} FLOPs → 32 卡 {step_compute_s(32):.1f} s、48 卡 {step_compute_s(48):.1f} s")


if __name__ == "__main__":
    main()
