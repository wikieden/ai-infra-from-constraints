"""第 28 章 · 编译器：表达、变换与选择。

讲义：ai-infra/notes/lessons/28-compiler.md
运行：uv run python ch28_compiler.py
融合划分数、SiLU 与归约的次序、例 5-7 的 E4M3 整行 scale、双缓冲驻留、例 5-9 的调用比例取自 AII 第 5 章 §5.4。
"""

import math


def partitions(n_ops: int) -> int:
    """一条 n 个算子的链，每条相邻边界可切可不切：2^(n-1) 种。"""
    return 2 ** (n_ops - 1)


def silu(x: float) -> float:
    return x / (1 + math.exp(-x))


def e4m3_round(x: float) -> float:
    """舍入到 E4M3FN（3 位尾数，最大 448），就近取偶，只处理正数。"""
    if x >= 448:
        return 448.0
    e = max(math.floor(math.log2(x)), -6) if x > 0 else -6
    step = 2.0 ** (e - 3)
    q = x / step
    lo = math.floor(q)
    frac = q - lo
    r = lo + (1 if frac > 0.5 or (frac == 0.5 and lo % 2) else 0)
    return r * step


def quantize_first(first: float, row_max: float) -> float:
    """按整行最大绝对值把 first 映射到 448，舍入后反量化。"""
    scale = 448 / row_max
    return e4m3_round(first * scale) / scale


def resident(work_kib: int, smem: int = 102_400, reserve: int = 1024) -> int:
    return smem // (work_kib * 1024 + reserve)


def mix_time(p: float, a: float = 5, b: float = 20) -> float:
    """形状 A 占比 p 时新实现的平均时间（μs）。"""
    return a * p + b * (1 - p)


def main() -> None:
    print("\n第一步：候选有多少")
    print(f"  3 个算子 {partitions(3)} 种划分，10 个算子 {partitions(10)} 种（还没算块大小、布局、流水深度）")

    print("\n第二步：依赖与舍入")
    print(f"  先加后激活 SiLU(1 − 1) = {silu(0):.4f}；先激活后加 SiLU(1) + SiLU(−1) = {silu(1) + silu(-1):.4f}")
    full, early = quantize_first(1, 10), quantize_first(1, 1)
    print(f"  例 5-7：读完整行再量化 {full:.6f}（= 55/56），读完第一块就量化 {early:.6f}，差 {(early - full) * 100:.1f}%")

    print("\n第三步：成本估计与实测选择")
    print(f"  RTX PRO 6000 100 KB：单缓冲 24 / 80 KiB 驻留 {resident(24)} / {resident(80)} 块；双缓冲 32 / 96 KiB 驻留 {resident(32)} / {resident(96)} 块")
    print(f"  例 5-9：各一次 20 → {5 + 20} μs；A 100 次、B 1 次 {100 * 10 + 10} → {100 * 5 + 20} μs（{1010 / 520:.1f} 倍）")
    print(f"  平均时间 20 − 15p < 10 ⇔ p > {2 / 3:.3f}；p = 0.9 时 {mix_time(0.9):.1f} μs")


if __name__ == "__main__":
    main()
