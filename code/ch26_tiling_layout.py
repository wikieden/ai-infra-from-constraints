"""第 26 章 · 单个算子：分块、布局与切分。

讲义：ai-infra/notes/lessons/26-tiling-layout.md
运行：uv run python ch26_tiling_layout.py
Qwen3-8B FFN 一支投影 [1024,4096]×[4096,12288]、例 5-3、RTX PRO 6000 的共享内存驻留、bank 冲突、RMSNorm 拆分与 FP32 加法次序取自 AII 第 5 章 §5.2。
"""

import struct

MiB = 2**20
M, K, N = 1024, 4096, 12288


def f32(x: float) -> float:
    """舍入到 FP32。"""
    return struct.unpack("f", struct.pack("f", x))[0]


def tile_local_kib(m: int, n: int, k: int = 32) -> float:
    """一组输入缓冲 + FP32 累加器：2mk + 2kn + 4mn 字节。"""
    return (2 * m * k + 2 * k * n + 4 * m * n) / 1024


def tile_traffic_mib(m: int, n: int) -> tuple[float, float, float]:
    """各输出块独立读输入：(A 重复读取, W 重复读取, C 写出)，单位 MiB。"""
    return 2 * M * K * (N / n) / MiB, 2 * K * N * (M / m) / MiB, 2 * M * N / MiB


def resident(kib: float, smem: int = 102_400, reserve: int = 1024) -> int:
    return smem // (int(kib * 1024) + reserve)


def bank_rounds(stride_words: int, lanes: int = 32, banks: int = 32, swizzle: bool = False) -> int:
    """32 个 lane 读同一列（第 0 列）时，同一个 bank 最多被请求几次 = 要几轮。"""
    hits: dict[int, int] = {}
    for r in range(lanes):
        col = (0 ^ r) if swizzle else 0
        b = (r * stride_words + col) % banks
        hits[b] = hits.get(b, 0) + 1
    return max(hits.values())


def rmsnorm_traffic_mib(rows: int = 1024, width: int = 4096, parts: int = 8) -> tuple[float, float]:
    """(一组一行, 拆成 parts 段) 的总访问量：读输入 + 读 gamma + 写输出；拆分后多读一遍输入，再加局部和与缩放因子。"""
    x = rows * width * 2 / MiB
    one = 3 * x
    extra = (rows * parts * 4 * 2 + rows * 4 * 2) / MiB
    return one, one + x + extra


def main() -> None:
    print(f"\n第一步：{M}×{K} × {K}×{N}，BF16")
    print(f"  运算 {2 * M * K * N / 1e9:.0f} GFLOPs；三份矩阵各一次 {2 * (M * K + K * N + M * N) / MiB:.0f} MiB")

    print("\n第二步：例 5-3，k = 32、单组输入缓冲")
    for t in (32, 64, 128):
        a, w, c = tile_traffic_mib(t, t)
        tot = a + w + c
        print(f"  {t:3d}×{t:<3d}：局部 {tile_local_kib(t, t):3.0f} KiB，访问 {a:.0f} + {w:.0f} + {c:.0f} = {tot:.0f} MiB，"
              f"强度 {2 * M * K * N / (tot * MiB):.1f}，100 KB 里驻留 {resident(tile_local_kib(t, t))} 块")
    print(f"  大块只拿到四成带宽：时间比 {1560 / 3096 / 0.4:.2f}")

    print("\n第三步：32×32 FP32 暂存块，32 个 lane 读一列")
    print(f"  行跨度 32：{bank_rounds(32)} 轮；补成 33：{bank_rounds(33)} 轮（空间 +{(33 / 32 - 1) * 100:.1f}%）；swizzle：{bank_rounds(32, swizzle=True)} 轮")

    print("\n第四步：RMSNorm 拆分与加法次序")
    one, split = rmsnorm_traffic_mib()
    print(f"  一组一行 {one:.0f} MiB；拆成 8 段 {split:.1f} MiB，第一阶段任务 1024 → 8192")
    e = 2.0**-24
    left, right = f32(f32(1 + e) + e), f32(1 + f32(e + e))
    print(f"  FP32：(1 + 2^-24) + 2^-24 = {left!r}，1 + (2^-24 + 2^-24) = {right!r}，差 {(right - left) / 2**-23:.0f} ULP")


if __name__ == "__main__":
    main()
