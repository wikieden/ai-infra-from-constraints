"""第 35 章 · 集合通信的实现与代价。

讲义：ai-infra/notes/lessons/35-collectives.md
运行：uv run python ch35_collectives.py
环形 / 树形 AllReduce 时间模型、Qwen3-32B 的 TP 单步时间、交点、busbw、实验 6-5 的重叠取自 AII 第 6 章 §6.4。
"""

import math

KiB, MiB, GiB = 2**10, 2**20, 2**30
ALPHA, B = 0.822e-6, 450e9  # MSCCL++ 测得的 H100 NVLink 单向延迟、每方向带宽
T_LOCAL = 29.35e-3


def ring_ar(n: int, M: float, alpha: float = ALPHA, bw: float = B) -> float:
    """式 (6-9)。"""
    return 2 * (n - 1) * alpha + 2 * (n - 1) * M / (n * bw) if n > 1 else 0.0


def tree_ar(n: int, M: float, alpha: float = ALPHA, bw: float = B) -> float:
    """式 (6-10)，n 为 2 的幂。"""
    return 2 * math.log2(n) * (alpha + M / bw)


def tp_step(p: int, ar=ring_ar, M: float = 10 * KiB, **kw) -> float:
    """式 (6-5)。"""
    return T_LOCAL / p + 128 * ar(p, M, **kw)


def ring_tree_crossover(n: int = 8, alpha: float = ALPHA, bw: float = B) -> float:
    """两式相等的数据量（字节）。"""
    a = 2 * (n - 1) - 2 * math.log2(n)
    b = 2 * math.log2(n) - 2 * (n - 1) / n
    return a * alpha * bw / b


def busbw(algbw: float, n: int) -> float:
    return algbw * 2 * (n - 1) / n


def main() -> None:
    print("\n第一步：环形 AllReduce，Qwen3-32B 128K，M = 10 KiB")
    for p in (1, 2, 4, 8):
        print(f"  TP{p}：本地 {T_LOCAL / p * 1e3:.2f} ms，128 次归约 {128 * ring_ar(p, 10 * KiB) * 1e3:.2f} ms，单步 {tp_step(p) * 1e3:.2f} ms")
    print(f"  TP8 单次 {ring_ar(8, 10 * KiB) * 1e6:.2f} μs；带宽翻倍省 {(tp_step(8) - tp_step(8, bw=2 * B)) * 1e6:.1f} μs，α 减半省 {(tp_step(8) - tp_step(8, alpha=ALPHA / 2)) * 1e3:.2f} ms")

    print("\n第二步：环 vs 树（八卡）")
    for M in (10 * KiB, 80 * MiB):
        print(f"  {M / KiB:,.0f} KiB：环 {ring_ar(8, M) * 1e6:,.2f} μs，树 {tree_ar(8, M) * 1e6:,.2f} μs")
    print(f"  交点约 {ring_tree_crossover() / KiB:.0f} KiB；α 减半后 {ring_tree_crossover(alpha=ALPHA / 2) / KiB:.0f} KiB")

    print("\n第三步：测量与重叠")
    print(f"  16 rank、64 MiB：algbw 143.99 GB/s → busbw {busbw(143.99, 16):.2f} GB/s")
    for name, c, k, cc, kk, both in (("4 MiB", 2.67, 11.53, 6.14, 11.52, 11.58), ("64 MiB", 36.92, 11.03, 45.21, 12.89, 45.24)):
        print(f"  {name}：分开 {c + k:.2f} ms，同时 {both:.2f} ms，省 {c + k - both:.2f} ms")


if __name__ == "__main__":
    main()
