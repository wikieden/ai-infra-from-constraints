"""第 27 章 · 算子融合与 FlashAttention：中间结果不落地。

讲义：ai-infra/notes/lessons/27-fusion-flash.md
运行：uv run python ch27_fusion_flash.py
例 5-4、双缓冲例 5-5、背压、例 5-6 在线 Softmax、注意力块大小的取舍取自 AII 第 5 章 §5.3。
"""

import math

MiB = 2**20


def activation_chain_mib(fuse_silu_mul: bool, fuse_quant: bool | None = None, t: int = 24) -> int:
    """G、U、T、Z 各 t MiB（BF16），量化输出 t/2。fuse_quant=None 表示没有量化这一步。"""
    total = 0
    if fuse_silu_mul:
        total += 3 * t  # 读 G、U，写 Z
    else:
        total += 2 * t + 3 * t  # SiLU：读 G 写 T；乘：读 T、U 写 Z
    if fuse_quant is None:
        return total
    if fuse_quant and fuse_silu_mul:
        return 2 * t + t // 2  # 读 G、U，写量化结果
    return total + t + t // 2  # 独立量化：读 Z，写 1 字节结果


def pipe_time(n: int, t_m: float, t_c: float) -> tuple[float, float]:
    """(串行, 双缓冲) 的总时间。"""
    return n * (t_m + t_c), t_m + (n - 1) * max(t_m, t_c) + t_c


def online_softmax(scores: list[float], values: list[float], block: int = 1) -> tuple[float, float, float, float]:
    """逐块更新 (m, ℓ, u)，返回 (m, ℓ, u, u/ℓ)。"""
    m, l, u = -math.inf, 0.0, 0.0
    for s0 in range(0, len(scores), block):
        s, v = scores[s0:s0 + block], values[s0:s0 + block]
        m_new = max(m, max(s))
        r = math.exp(m - m_new) if m != -math.inf else 0.0
        p = [math.exp(x - m_new) for x in s]
        l = r * l + sum(p)
        u = r * u + sum(pi * vi for pi, vi in zip(p, v))
        m = m_new
    return m, l, u, u / l


def full_softmax(scores: list[float], values: list[float]) -> float:
    m = max(scores)
    w = [math.exp(x - m) for x in scores]
    return sum(wi * vi for wi, vi in zip(w, values)) / sum(w)


def flash_tile(b: int, d: int = 128, L: int = 8192, smem: int = 101_376) -> tuple[int, int, int, int]:
    """K/V 块 b 行时：(可容纳的 Q 行 a, 完整扫描次数, 访问 MiB, 状态更新次数)。"""
    # 2ad + 2bd + 4ab + 4ad + 12a ≤ smem
    a = (smem - 2 * b * d) // (2 * d + 4 * b + 4 * d + 12)
    scans = math.ceil(L / a)
    kv = 2 * L * d * 2 / MiB  # K、V 各 2 MiB
    traffic = scans * kv + 4  # 再加 Q 读、O 写
    return a, scans, int(traffic), scans * (L // b)


def main() -> None:
    print("\n第一步：例 5-4，G、U 各 24 MiB")
    print(f"  分开 {activation_chain_mib(False)} MiB，融合 {activation_chain_mib(True)} MiB，时间比 {72 / 120:.1f}")
    print(f"  加量化：全分开 {activation_chain_mib(False, False)}，融前两步 {activation_chain_mib(True, False)}，三步融合 {activation_chain_mib(True, True)} MiB")
    print("  显存峰值：独立 SiLU 时 G、U、T 共 72 MiB；三步融合 G、U、输出 60 MiB")

    print("\n第二步：例 5-5，每块搬 1.29 μs、算 0.28 μs，4 块")
    s, p = pipe_time(4, 1.29, 0.28)
    print(f"  串行 {s:.2f} μs，双缓冲 {p:.2f} μs（−{(1 - p / s) * 100:.0f}%），计算单元忙 {4 * 0.28 / p:.0%}")
    print(f"  两块同驻 RTX PRO 6000 一个 SM，每块最多 {(102_400 - 2048) // 2 // 1024} KiB")

    print("\n第三步：例 5-6 在线 Softmax")
    m, l, u, o = online_softmax([0, math.log(2)], [1, 3])
    print(f"  m = {m:.3f}，ℓ = {l}，u = {u}，输出 {o:.4f}（一次处理整行 {full_softmax([0, math.log(2)], [1, 3]):.4f} = 7/3）")
    print(f"  L = 8192、d = 128：完整 S、P 各 {8192 * 8192 * 4 / MiB:.0f} MiB，写读共 {4 * 8192 * 8192 * 4 / 2**30:.0f} GiB")
    for b in (1, 64, 128):
        a, scans, mib, upd = flash_tile(b)
        print(f"  b = {b:3d}：a = {a:3d}，扫描 {scans} 次，访问 {mib} MiB，更新 {upd:,} 次")


if __name__ == "__main__":
    main()
