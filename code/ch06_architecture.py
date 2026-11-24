"""第 6 章 · 需求如何推动架构：TPU、SmartNIC、Unified Bus。

讲义：ai-infra/notes/lessons/06-architecture.md
运行：uv run python ch06_architecture.py
40 Gbit/s、最小帧 84 字节、单核 1000 万–2000 万包/秒取自 AII 第 1 章 §1.4.2 与脚注；
专用芯片回本的 5000 万元、每 1000 次省 0.01 元、每天 1e10 次请求是编的。
"""

import math


def breakeven(upfront: float, saving_per_request: float) -> float:
    """回本请求数 = 前期投入 ÷ 每次请求省下的钱。"""
    return upfront / saving_per_request


def days_to_breakeven(upfront: float, saving_per_request: float, requests_per_day: float) -> float:
    """按每天的请求量，多少天回本。"""
    return breakeven(upfront, saving_per_request) / requests_per_day


def packet_rate(gbps: float, wire_bytes: int = 84) -> float:
    """最小帧条件下每秒到达多少个包：比特率 ÷ (每包线路字节 × 8)。"""
    return gbps * 1e9 / (wire_bytes * 8)


def cores(rate: float, per_core: float) -> int:
    """光转发要几个核：λ / μ，向上取整。"""
    return math.ceil(rate / per_core)


def main() -> None:
    print("\n第二步：专用芯片什么时候回本（编的数）")
    n = breakeven(5e7, 0.01 / 1000)
    print(f"  5000 万元 ÷ 每次省 1e-5 元 = {n:.0e} 次请求")
    for per_day in (1e10, 1e11):
        print(f"  每天 {per_day:.0e} 次 → {days_to_breakeven(5e7, 0.01 / 1000, per_day):.0f} 天")

    print("\n第三步：40 Gbit/s、最小帧 84 字节")
    lam = packet_rate(40)
    print(f"  包率 = 40e9 ÷ 672 ≈ {lam / 1e6:.1f} 百万包/秒")
    print(f"  单核 2000 万包/秒 → {cores(lam, 2e7)} 个核；单核 1000 万包/秒 → {cores(lam, 1e7)} 个核")


if __name__ == "__main__":
    main()
