"""第 20 章 · 计算单元：矩阵、向量、流水与低精度。

讲义：ai-infra/notes/lessons/20-compute-units.md
运行：uv run python ch20_compute_units.py
16³ 矩阵块、MoE 补零、Softmax 归约、Qwen3-8B 的 RoPE、FlashAttention-4 的 B200 单 SM 分析、分组量化存储取自 AII 第 4 章 §4.2。
"""

import math


def padded_rows(rows_per_expert: list[int], tile: int = 16) -> int:
    """每个专家的行数向上补齐到 tile，返回实际执行的总行数。"""
    return sum(math.ceil(r / tile) * tile for r in rows_per_expert if r > 0)


def reduce_rounds(n: int) -> int:
    """理想二叉归约要几轮。"""
    return math.ceil(math.log2(n))


def rope_flops(n_q: int, n_k: int, d_h: int) -> int:
    """每 token 每层 RoPE：(n_q + n_k) × d_h/2 对，每对 4 乘 2 加。"""
    return (n_q + n_k) * (d_h // 2) * 6


def rope_table_kib(positions: int, d_h: int = 128, b: int = 2) -> float:
    """每个位置存 d_h/2 个 cos 和 d_h/2 个 sin。"""
    return positions * d_h * b / 1024


def attn_block_cycles(mat_x: float = 1, exp_x: float = 1, smem_x: float = 1, tile: int = 128, d: int = 128) -> tuple[float, float, float, float]:
    """FlashAttention-4 的 B200 单 SM 分析：(矩阵, 共享内存, 指数, 下界 τ) 周期。"""
    matrix = 2 * 2 * tile * tile * d / (8192 * mat_x)  # QK + PV 的 FLOPs ÷ 每周期矩阵速率
    exp = tile * tile / (16 * exp_x)  # 每个分数一次指数
    smem = 96 * 1024 / (128 * smem_x)  # 操作数从共享内存读 96 KiB
    return matrix, smem, exp, max(matrix, smem, exp)  # 流水下界取最慢的


def quant_mib(n: int, bits: int, group: int, scale_bytes: int) -> float:
    """分组量化的存储量（MiB）= n·b/8 + (n/g)·s。"""
    return (n * bits / 8 + n / group * scale_bytes) / 2**20


def main() -> None:
    print("\n第一步：矩阵单元")
    print(f"  16×16×16 块：{2 * 16**3} FLOPs")
    uni, con = padded_rows([2] * 256), padded_rows([64] * 8)
    print(f"  MoE 512 个有效行：均匀 {uni} 行（有效 {512 / uni:.1%}），集中 {con} 行")

    print("\n第二步：向量工作")
    print(f"  128 个数的归约 {reduce_rounds(128)} 轮")
    print(f"  Qwen3-8B 一层 RoPE {rope_flops(32, 8, 128):,} FLOPs/token；1024 个位置的表 {rope_table_kib(1024):.0f} KiB")

    print("\n第三步：注意力流水（例 4-1）")
    for name, kw in (("原样", {}), ("矩阵 ×2", {"mat_x": 2}), ("矩阵、指数 ×2", {"mat_x": 2, "exp_x": 2}), ("三项都 ×2", {"mat_x": 2, "exp_x": 2, "smem_x": 2})):
        m, s, e, tau = attn_block_cycles(**kw)
        print(f"  {name:10s} 矩阵 {m:.0f}，共享内存 {s:.0f}，指数 {e:.0f} → τ ≥ {tau:.0f}")

    print("\n第四步：4096×4096 权重的分组量化")
    print(f"  4 bit、每 32 个一个 1 字节 scale：{quant_mib(4096 * 4096, 4, 32, 1):.1f} MiB；每 16 个：{quant_mib(4096 * 4096, 4, 16, 1):.1f} MiB；BF16 32 MiB")


if __name__ == "__main__":
    main()
