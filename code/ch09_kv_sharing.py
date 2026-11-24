"""第 9 章 · KV 怎么变小 ①：头共享与潜变量压缩。

讲义：ai-infra/notes/lessons/09-kv-sharing.md
运行：uv run python ch09_kv_sharing.py
Qwen3-8B（36 层、头宽 128、32 个查询头）与 Kimi K3（24 个 MLA 层、512 + 64 维、展开后 1,474,560 B/token）取自 AII 第 2 章 §2.3.1–§2.3.2；
c、U、q 的小矩阵是编的。
"""


def kv_gib(n_kv: int, L: int = 36, d_h: int = 128, H: int = 8192, b: int = 2) -> float:
    """H 个 token 的 KV（GiB）= 2 × L × n_kv × d_h × b × H。"""
    return 2 * L * n_kv * d_h * b * H / 2**30


def attn_flops_per_pair(n_q: int = 32, d_h: int = 128) -> int:
    """每个查询—键配对的注意力运算：4 × n_q × d_h，与 KV 头数无关。"""
    return 4 * n_q * d_h


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """普通矩阵乘法（列表实现，方便看清每一步）。"""
    return [[sum(a[i][t] * b[t][j] for t in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def transpose(a: list[list[float]]) -> list[list[float]]:
    """转置。"""
    return [list(r) for r in zip(*a)]


def scores_expanded(c: list[list[float]], U: list[list[float]], q: list[float]) -> list[float]:
    """展开路径：K = cU，每个上下文 token 都升维，再算 qKᵀ。"""
    K = matmul(c, U)
    return matmul([q], transpose(K))[0]


def scores_compact(c: list[list[float]], U: list[list[float]], q: list[float]) -> list[float]:
    """紧凑路径：先算 qUᵀ（只变换当前查询），再和潜变量 c 点积。"""
    qU = matmul([q], transpose(U))[0]
    return matmul([qU], transpose(c))[0]


def mla_compact_mib(H: int, layers: int = 24, d_c: int = 512, d_r: int = 64, b: int = 2) -> float:
    """MLA 紧凑缓存（MiB）= 层数 × H × (d_c + d_r) × b。"""
    return layers * H * (d_c + d_r) * b / 2**20


def mla_expanded_gib(H: int, per_token: int = 1_474_560) -> float:
    """MLA 展开缓存（GiB）= H × 每 token 展开后的字节数。"""
    return H * per_token / 2**30


def main() -> None:
    print("\n第二步：Qwen3-8B 只改 KV 头数，8192 个 token 的 KV")
    for name, n_kv in (("MHA", 32), ("GQA", 8), ("MQA", 1)):
        print(f"  {name}：{n_kv} 个 KV 头 → {kv_gib(n_kv):.6g} GiB")
    print(f"  注意力每个配对 {attn_flops_per_pair()} FLOPs，三种结构都一样")

    print("\n第三步：结合律小例子（编的）")
    c, U, q = [[1, 0], [2, 1]], [[1, 0, 2], [0, 1, 1]], [1, 1, 1]
    print(f"  展开 {scores_expanded(c, U, q)}，紧凑 {scores_compact(c, U, q)}")

    print("\n第三步：Kimi K3 的 24 个 MLA 层，8192 个 token，BF16")
    print(f"  紧凑 {mla_compact_mib(8192):.0f} MiB，展开 {mla_expanded_gib(8192):.2f} GiB")


if __name__ == "__main__":
    main()
