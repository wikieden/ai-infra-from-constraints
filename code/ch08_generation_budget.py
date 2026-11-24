"""第 8 章 · 自回归生成的资源账：存一次的、每步读的、用完就扔的。

讲义：ai-infra/notes/lessons/08-generation-budget.md
运行：uv run python ch08_generation_budget.py
Qwen3-8B 的 144 KiB / token、R_W = 15,136,811,008 字节、例题 2-2 与 2-3 取自 AII 第 2 章 §2.2；H = 4、n_d = 4 是编的小例子。
"""

import math

R_W = 15_136_811_008  # Qwen3-8B 一步 decode 的主要权重读取（字节）


def kv_bytes_per_token(L: int, n_kv: int, d_h: int, b: int) -> int:
    """每个上下文 token 的 KV：2 × L × n_kv × d_h × b（K、V 各一份）。"""
    return 2 * L * n_kv * d_h * b


C_KV = kv_bytes_per_token(36, 8, 128, 2)  # Qwen3-8B、BF16


def b_star_kv(H: int, r_w: int = R_W, c_kv: int = C_KV) -> int:
    """KV 读取追上权重读取的最小整数 batch：ceil(R_W / (c_KV × H))。"""
    return math.ceil(r_w / (c_kv * H))


def old_reads_tokens(H: int, n_d: int) -> int:
    """生成期间累计读取的旧 KV（token·次）：n_d·H + n_d(n_d-1)/2。"""
    return n_d * H + n_d * (n_d - 1) // 2


def old_reads_bruteforce(H: int, n_d: int) -> int:
    """一步一步加：第 j 次 decode 读 H + j 份。"""
    return sum(H + j for j in range(n_d))


def main() -> None:
    print("\n第一步：Qwen3-8B 每个 token 的 KV")
    print(f"  2 × 36 × 8 × 128 × 2 = {C_KV:,} 字节 = {C_KV / 1024:.0f} KiB")
    print(f"  8192 个上下文 token = {C_KV * 8192 / 2**30:.3f} GiB")

    print("\n第三步：batch 多大时读 KV 多于读权重")
    for H in (8192, 2048):
        print(f"  H = {H}：R_W / (c × H) = {R_W / (C_KV * H):.2f} → 最小 batch {b_star_kv(H)}")

    print("\n第四步：H = 4、生成 4 步（编的）")
    print(f"  公式 {old_reads_tokens(4, 4)}，逐步加 {old_reads_bruteforce(4, 4)}；最终保存 {4 + 4} 份")

    print("\n第四步：例题 2-3（S = 0、P = 8192、G = 1025）")
    n_d = 1024
    reads = old_reads_tokens(8192, n_d)
    print(f"  累计读取 {reads:,} token·次 = {reads * C_KV / 2**30:.0f} GiB")
    print(f"  生成期间新增 {n_d * C_KV / 2**20:.0f} MiB；结束时保存 {(8192 + n_d) * C_KV / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
