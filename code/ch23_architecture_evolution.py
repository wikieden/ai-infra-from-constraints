"""第 23 章 · 架构演进与专用架构：每一代新增的东西，省掉了哪一笔。

讲义：ai-infra/notes/lessons/23-architecture-evolution.md
运行：uv run python ch23_architecture_evolution.py
V100、MX 格式、产品线表、昇腾交接、Apple 统一内存、Groq SRAM、固定权重 ROM 的参数取自 AII 第 4 章 §4.6–§4.7。
"""

import math

MiB = 2**20
GiB = 2**30


def v100_paths(m: int, d: int = 4096) -> tuple[float, float, float]:
    """V100 上 [m, d] × [d, d] 投影：(普通 FP32 μs, Tensor Core μs, FP16 读写 μs)。"""
    flops = 2 * m * d * d
    io = (m * d + d * d + m * d) * 2
    return flops / 15.7e12 * 1e6, flops / 125e12 * 1e6, io / 900e9 * 1e6


def weight_mib(bits: int, d: int = 4096, group: int = 32) -> float:
    """d×d 权重的存储量（MiB）：BF16 无 scale，低精度每 group 个值共享 1 字节 scale。"""
    n = d * d
    scale = 0 if bits == 16 else n / group
    return (n * bits / 8 + scale) / MiB


def read_us(mib: float, bw: float) -> float:
    return mib * MiB / bw * 1e6


def handoff_gbps(kib: float, budget_us: float) -> float:
    """在 budget_us 内传完 kib 所需带宽（GB/s）。"""
    return kib * 1024 / (budget_us * 1e-6) / 1e9


def chips_needed(gib: float, sram_mib: float = 220) -> int:
    return math.ceil(gib * 1024 / sram_mib)


def rom_ratio(b: int, w: float = 15.1, k: float = 1.21) -> float:
    """权重与 KV 共用 HBM 时的读取量 / 权重移到 ROM 后的 HBM 读取量 = 1 + W/(BK)。"""
    return 1 + w / (b * k)


def lanes_needed(ops: float, budget_s: float, util: float = 1.0, hz: float = 1e9) -> int:
    """每条通道每周期一次 FMA（2 次运算），在 budget_s 内做完 ops 需要的通道数。"""
    return math.ceil(ops / (2 * hz * budget_s * util))


def main() -> None:
    print("\n第一步：V100 上的 Q 投影")
    for m in (4096, 1):
        a, b, io = v100_paths(m)
        print(f"  M = {m:4d}：普通 {a:,.2f} μs，Tensor Core {b:,.3f} μs，读写 {io:,.1f} μs")

    print("\n第二步：精度与带宽")
    for bits, name in ((16, "BF16"), (8, "MXFP8"), (4, "MXFP4")):
        print(f"  {name}：{weight_mib(bits):.1f} MiB")
    t4090, t5090, t5090_fp4 = read_us(32, 1008e9), read_us(32, 1792e9), read_us(8.5, 1792e9)
    print(f"  4090 BF16 {t4090:.2f} μs → 5090 BF16 {t5090:.2f} μs → 5090 MXFP4 {t5090_fp4:.2f} μs，共 {t4090 / t5090_fp4:.2f} 倍")
    print(f"  3090→4090 矩阵 {165.2 / 71.0:.2f} 倍、带宽 +{(1008 / 936 - 1) * 100:.1f}%；A100→H100 矩阵 {989.4 / 312:.2f} 倍、带宽 {3350 / 2039:.2f} 倍")
    print(f"  64 MiB 过 NVLink：A100 {64 * MiB / 300e9 * 1e6:.0f} μs，H100 {64 * MiB / 450e9 * 1e6:.0f} μs")

    print("\n第三步：昇腾的两次改造")
    unrolled = 3136 * 576 * 2 / MiB
    print(f"  img2col 展开 {unrolled:.3f} MiB，HBM 写读 {2 * unrolled:.3f} MiB，{2 * unrolled * MiB / 1.2e12 * 1e6:.2f} μs")
    print(f"  注意力块交接 256 KiB：94 GB/s 要 {256 * 1024 / 94e9 * 1e6:.2f} μs；1.024 μs 内完成要 {handoff_gbps(256, 1.024):.0f} GB/s；CV 直通 {handoff_gbps(128, 1.024):.0f} GB/s")

    print("\n第四步：Apple 统一内存")
    for name, bw in (("M1/M2 Ultra", 800e9), ("M3 Ultra", 819e9), ("M4 Max", 546e9)):
        print(f"  Qwen3-8B 每步 16.345 GB：{name} {16.345e9 / bw * 1e3:.2f} ms")
    print(f"  235B-A22B 每步 13.697 GB：M3 Ultra {13.697e9 / 819e9 * 1e3:.2f} ms；八请求 {8 / (24.737e9 / 819e9):.0f} / {8 / (75.967e9 / 819e9):.0f} token/s")

    print("\n第五步：Groq 每芯片 220 MiB SRAM")
    print(f"  权重 16.4 GB → {chips_needed(16.4e9 / GiB)} 颗；KV 1 条 {chips_needed(1.125)}，8 条 {chips_needed(9)}，8 条 32K {chips_needed(36)}")

    print("\n第六步：固定权重放 ROM")
    print(f"  HBM 读取缩小 B=1 {rom_ratio(1):.1f} 倍，B=16 {rom_ratio(16):.1f} 倍；交点 B = {15.1 / 1.21:.1f}，32K 时 {15.1 / (4 * 1.21):.2f}")
    print(f"  100 μs/token 读 1.21 GB KV 要 {1.21e9 / 100e-6 / 1e12:.1f} TB/s")
    ops = 15_134_641_792
    print(f"  通道：100 μs {lanes_needed(ops, 100e-6):,}；40 μs、60% {lanes_needed(ops, 40e-6, 0.6):,}；256 条要 {ops / (2 * 256 * 1e9) * 1e3:.1f} ms")


if __name__ == "__main__":
    main()
