"""第 61 章 · 从系统方案到期限与硬件选择。

讲义：ai-infra/notes/lessons/61-training-choice.md
运行：uv run python ch61_training_choice.py
32 / 48 卡 RTX 4090 方案的完成时间、每步预算、共用网卡、Amdahl 式硬件替换与模型规模的卡数边界，取自 AII 第 10 章 §10.6。
"""

import math

STEPS = 31790
DAY = 86400


def finish_days(ts: float, L: float, T0: float = 5, steps: int = STEPS) -> float:
    """T_finish = T0 + ⌈D/b⌉ t_s (1 + L)。"""
    return T0 + steps * ts * (1 + L) / DAY


def step_budget(days: float = 25, L: float = 0.0108, steps: int = STEPS) -> float:
    return days * DAY / (steps * (1 + L))


def amdahl(f: float, r: float) -> float:
    """T'/T = 1 − f + f/r。"""
    return 1 - f + f / r


def cards_needed(params: float, tokens: float, peak: float, eta: float, days: float) -> float:
    """p ≥ 6ND / (P η T)，返回连续值。"""
    return 6 * params * tokens / (peak * eta * days * DAY)


def days_needed(params: float, tokens: float, peak: float, eta: float, cards: int) -> float:
    return 6 * params * tokens / (peak * eta * cards * DAY)


def main() -> None:
    print("\n第一步：汇总")
    for name, ts, L in (("32 卡", 78.3 + 0.08 + 0.5, 0.0102), ("48 卡", 52.2 + 0.08 + 0.5, 0.0108)):
        print(f"  {name}：每步 {ts:.1f} s，基础 {finish_days(ts, 0, 0):.1f} 天，加保存恢复和 5 天 → {finish_days(ts, L):.1f} 天")

    print("\n第二步：可行范围")
    b = step_budget()
    print(f"  每步上限 {b:.1f} s；48 卡余量 {b - 52.78:.1f} s；通信最多 {b - 52.2 - 0.5:.1f} s")
    print(f"  预取全失效 12.0 s 全暴露：{52.2 + 12.0 + 0.5:.1f} s，仍可行")
    left = step_budget(L=0.0102) - 0.08 - 0.5
    print(f"  32 卡留给计算 {left:.1f} s，要效率 {0.4 * 78.3 / left * 100:.0f}%")
    v = 385e9
    print(f"  八卡共用一张 200 Gbit/s：{v / 25e9:.1f} s > {b - 52.7:.1f} s；预取失效时每步 {52.2 + 0.5 + v / 25e9:.1f} s → 超期")

    print("\n第三步：换硬件")
    print(f"  100 ms 里通信 5 ms，带宽减半 → {100 * amdahl(0.05, 0.5):.0f} ms；通信 50 ms → {100 * amdahl(0.5, 0.5):.0f} ms")
    ts30 = 52.2 * 0.4 / 0.3
    print(f"  48 卡效率 40% → 30%：单卡 {ts30:.1f} s，每步 {ts30 + 0.58:.1f} s > {b:.1f}")

    print("\n第四步：模型规模（20T token、16,384 张卡）")
    for name, peak in (("A100", 312e12), ("H100", 989e12), ("B200", 2.25e15)):
        print(f"  1T 参数 {name} 40%：{days_needed(1e12, 20e12, peak, 0.4, 16384):.0f} 天")
    print(f"  B200 90 天要 {math.ceil(cards_needed(1e12, 20e12, 2.25e15, 0.4, 90)):,} 张；50% 时 {days_needed(1e12, 20e12, 2.25e15, 0.5, 16384):.0f} 天")
    print("  D = 20N 时 F = 120N²：参数 5 倍，计算 25 倍")


if __name__ == "__main__":
    main()
