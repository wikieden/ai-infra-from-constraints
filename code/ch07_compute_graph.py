"""第 7 章 · 计算图与张量结构：一个 token 经过一层要算多少。

讲义：ai-infra/notes/lessons/07-compute-graph.md
运行：uv run python ch07_compute_graph.py
Qwen3-8B 的配置与 133.6 TFLOPs / 20.0 GFLOPs 取自 AII 第 2 章 §2.1；m = 3、k = 4、n = 2 和 S = 2、P = 3 是编的小例子。
"""

QWEN3_8B = {"L": 36, "d": 4096, "f": 12288, "n_q": 32, "n_kv": 8, "d_h": 128, "vocab": 151936}


def matmul_cost(m: int, k: int, n: int, b_w: int = 2) -> tuple[int, int, int]:
    """[m,k] × [k,n]：(参数数 kn, 运算量 2mkn, 权重字节 b_w·kn)。"""
    return k * n, 2 * m * k * n, b_w * k * n


def layer_params(c: dict) -> tuple[int, int]:
    """一层的 (注意力四个投影, SwiGLU 三个矩阵) 参数数。"""
    kv_w = c["n_kv"] * c["d_h"]
    proj = 2 * c["d"] * c["d"] + 2 * c["d"] * kv_w  # W_q、W_o；W_k、W_v
    ffn = 3 * c["d"] * c["f"]  # gate、up、down
    return proj, ffn


def n_pair(B: int, S: int, P: int) -> int:
    """有效查询—键配对数：长方形 PS + 三角形 P(P+1)/2，再乘请求数。"""
    return B * (P * S + P * (P + 1) // 2)


def n_pair_bruteforce(B: int, S: int, P: int) -> int:
    """逐个数：第 i 个新查询能看到 S + i 个 token（i 从 1 开始）。"""
    return B * sum(1 for i in range(1, P + 1) for j in range(1, S + P + 1) if j <= S + i)


def forward_flops(c: dict, B: int, S: int, P: int) -> float:
    """一次前向的主要矩阵运算：L × (投影与 FFN + 注意力) + 词表头。"""
    m = B * P  # 本次参与投影的 token 数
    linear = 2 * m * sum(layer_params(c))  # 投影 + FFN：按行数
    attn = 4 * c["n_q"] * c["d_h"] * n_pair(B, S, P)  # 注意力：按配对数
    head = 2 * B * c["d"] * c["vocab"]  # 词表头：每条请求只算最后一个 token
    return c["L"] * (linear + attn) + head


def main() -> None:
    print("\n第二步：[3,4] × [4,2]（编的小例子，BF16）")
    n_w, f, mw = matmul_cost(3, 4, 2)
    print(f"  参数 {n_w} 个，运算 {f} FLOPs，权重 {mw} 字节")

    print("\n第三步：Qwen3-8B 一层的参数")
    proj, ffn = layer_params(QWEN3_8B)
    print(f"  注意力投影 {proj:,}，FFN {ffn:,}，FFN 占 {ffn / (proj + ffn):.0%}")
    print(f"  一层 FFN 的 BF16 权重 = {ffn * 2 / 2**20:.0f} MiB")
    print(f"  投影 + FFN 运算量 = {2 * (proj + ffn):,} × m")

    print("\n第四步：S = 2、P = 3 的配对数（编的）")
    print(f"  公式 {n_pair(1, 2, 3)}，逐个数 {n_pair_bruteforce(1, 2, 3)}")

    print("\n第五步：一次前向")
    pre = forward_flops(QWEN3_8B, 1, 0, 8192)
    dec = forward_flops(QWEN3_8B, 1, 8192, 1)
    print(f"  prefill 8192 个 token：{pre / 1e12:.1f} TFLOPs")
    print(f"  已有 8192 个上下文再补 1 个：{dec / 1e9:.1f} GFLOPs")


if __name__ == "__main__":
    main()
