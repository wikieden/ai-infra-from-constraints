"""第 42 章 · 共享网络中的拥塞与可靠性。

讲义：ai-infra/notes/lessons/42-congestion.md
运行：uv run python ch42_congestion.py
例题 7.5、反馈时延与缓冲、incast 与 PFC、多路径与重传、ECMP 哈希冲突、逐包喷洒取自 AII 第 7 章 §7.5。
"""

from fractions import Fraction
from functools import lru_cache
from math import comb, ceil

KiB, MiB = 2**10, 2**20


def overlap_backlog_gb(overlap_ms: float, peak: float = 50, out: float = 50) -> float:
    return (2 * peak - out) * overlap_ms / 1e3


def tolerable_delay_us(free: float, arrive: float, out: float) -> float:
    return free / (arrive - out) * 1e6


def incast(N: int, free: float = MiB, B: float = 50e9, hop_s: float = 0.33e-6, rtt: float = 20e-6) -> tuple[float, float, float]:
    """(允许反馈时延 μs, PFC headroom B, 端到端 RTT 内超额 B)。"""
    excess = (N - 2) * B
    return free / excess * 1e6, excess * hop_s, excess * rtt


def pfc_hop_us(cable_m: float = 30, mtu: int = 1500, B: float = 50e9) -> float:
    return (2 * cable_m * 5e-9 + mtu / B) * 1e6


def multipath_us(pkts: int = 8, size: int = 4096, delays: tuple[float, ...] = (1, 1), B: float = 50e9) -> float:
    per = size / B * 1e6
    n = len(delays)
    k = ceil(pkts / n)
    return max(d + k * per for d in delays)


def max_load_dist(n: int, m: int) -> dict[int, Fraction]:
    """n 个球独立均匀放进 m 个盒子，最大负载的精确分布。"""
    @lru_cache(None)
    def ways(balls: int, boxes: int, cap: int) -> int:
        if boxes == 0:
            return 1 if balls == 0 else 0
        return sum(comb(balls, j) * ways(balls - j, boxes - 1, cap) for j in range(0, min(cap, balls) + 1))
    total = m**n
    cdf = {c: Fraction(ways(n, m, c), total) for c in range(0, n + 1)}
    return {c: cdf[c] - (cdf[c - 1] if c > 0 else 0) for c in range(1, n + 1)}


def ecmp(n: int, m: int) -> tuple[float, int, float, float]:
    """(E[Lmax], p99, 无冲突概率, 相对无冲突速度)。"""
    d = max_load_dist(n, m)
    e = sum(c * p for c, p in d.items())
    acc, p99 = Fraction(0), None
    for c in sorted(d):
        acc += d[c]
        if p99 is None and acc >= Fraction(99, 100):
            p99 = c
    ideal = ceil(n / m)
    return float(e), p99, float(d.get(ideal, 0)), ideal / float(e)


def main() -> None:
    print("\n第一步：例题 7.5")
    print(f"  平均 {2 * 50 * 0.2:.0f} GB/s；重叠 20 ms 积压 {overlap_backlog_gb(20):.1f} GB，排空 {overlap_backlog_gb(20) / 50 * 1e3:.0f} ms；重叠 5 ms {overlap_backlog_gb(5) * 1e3:.0f} MB")
    print(f"  512 KiB 缓冲容忍的重叠 {512 * KiB / 50e9 * 1e6:.1f} μs")

    print("\n第二步：反馈时延")
    print(f"  剩 256 KiB、超额 50 GB/s：{tolerable_delay_us(256 * KiB, 100e9, 50e9):.1f} μs；20 μs 才降速丢 {(50e9 * 20e-6 - 256 * KiB) / 1e3:.0f} KB")
    print(f"  降到 40 GB/s 排空 512 KiB {512 * KiB / 10e9 * 1e6:.0f} μs")
    print(f"  PFC 一跳 {pfc_hop_us():.2f} μs")
    for N in (8, 16, 64):
        t, h, e = incast(N)
        print(f"  incast N = {N:2d}：允许 {t:.2f} μs，headroom {h / 1e3:.0f} KB，20 μs 内超额 {e / 1e6:.1f} MB")

    print("\n第三步：多路径与重传（8 × 4 KiB）")
    print(f"  两条 1 μs：{multipath_us():.2f}；单路径：{multipath_us(delays=(1,)):.2f}；第二条 9 μs：{multipath_us(delays=(1, 9)):.2f} μs")
    print(f"  首包丢失 20 μs 后重传：{4096 / 50e9 * 1e6 + 20 + 4096 / 50e9 * 1e6 + 1:.1f} μs")
    for n, m in ((8, 32), (8, 16), (32, 32), (32, 16), (128, 16)):
        e, p99, p0, sp = ecmp(n, m)
        print(f"  ECMP {n:3d} 流 → {m} 上联：E[Lmax] {e:.2f}，p99 {p99}，无冲突 {p0:.1%}，速度 {sp:.1%}")
    spray = 1 * MiB / 50e9 * 1e6 + 8
    print(f"  8 MiB 喷洒到 8 条路径（1–8 μs）：{spray:.1f} μs；单路径 {8 * MiB / 50e9 * 1e6 + 1:.1f} μs")


if __name__ == "__main__":
    main()
