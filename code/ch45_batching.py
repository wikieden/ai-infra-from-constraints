"""第 45 章 · 批处理与请求调度。

讲义：ai-infra/notes/lessons/45-batching.md
运行：uv run python ch45_batching.py
每输出 token 的读取量、b_KV、LoRA 容量、连续批处理与分块的步时间、注意力配对取自 AII 第 8 章 §8.2。
"""

import math

KiB, MiB, GiB = 2**10, 2**20, 2**30
DW = 15_136_811_008
K = 147_456


def per_token_read_gib(b: int, L: int) -> tuple[float, float]:
    """(分摊权重, 本请求 KV) GiB。"""
    return DW / b / GiB, L * K / GiB


def b_kv(L: int) -> int:
    return math.ceil(DW / (L * K))


def lora_mib(rank: int = 16, layers: int = 36, h: int = 4096, kv: int = 1024) -> float:
    """Q（h→h）与 V（h→kv）各一对低秩矩阵，BF16。"""
    q = h * rank + rank * h
    v = h * rank + rank * kv
    return (q + v) * layers * 2 / MiB


def step_ms(new_tokens: int, pairs: int) -> float:
    """实验 8-1 拟合：26.22 ms + 30.4 μs × 新 token + 3 ns × 因果配对。"""
    return 26.22 + 30.4e-3 * new_tokens + 3e-6 * pairs


def chunk_pairs(c: int, h: int) -> int:
    return c * h + c * (c + 1) // 2


def main() -> None:
    print("\n第一步：每输出 token 的读取")
    for L in (2048, 8192):
        for b in (1, 16, 64):
            w, kv = per_token_read_gib(b, L)
            print(f"  L = {L}, b = {b:2d}：权重 {w:.2f} + KV {kv:.2f} = {w + kv:.2f} GiB")
        print(f"  L = {L}：b_KV = {b_kv(L)}")
    print(f"  2K 下 b 1 → 64 降为 1/{(DW / GiB + 2048 * K / GiB) / (DW / 64 / GiB + 2048 * K / GiB):.0f}")
    print(f"  LoRA r = 16（Q、V）：一套 {lora_mib():.1f} MiB，100 套 {100 * lora_mib() / 1024:.2f} GiB")

    print("\n第二步：调度算例的步时间")
    print(f"  2K prefill {step_ms(2048, chunk_pairs(2048, 0)):.1f} ms；decode {step_ms(1, 2048):.2f} ms")
    print(f"  r2 8K 一次 prefill（+ r0 decode）{step_ms(8192 + 1, chunk_pairs(8192, 0)):.0f} ms")
    print(f"  分块 2048：首块 {step_ms(2048, chunk_pairs(2048, 0)):.1f} ms，末块（h = 6144）{step_ms(2048, chunk_pairs(2048, 6144)):.1f} ms")

    print("\n第三步：块内注意力")
    a, b = chunk_pairs(512, 0), chunk_pairs(512, 7680)
    print(f"  c = 512：首块 {a:,} 对，末块 {b:,} 对（{b / a:.0f} 倍）")


if __name__ == "__main__":
    main()
