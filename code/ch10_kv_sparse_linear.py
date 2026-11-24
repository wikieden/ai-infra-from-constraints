"""第 10 章 · KV 怎么变小 ②：窗口、稀疏、线性与混合注意力。

讲义：ai-infra/notes/lessons/10-kv-sparse-linear.md
运行：uv run python ch10_kv_sparse_linear.py
DeepSeek V4-Flash / V4.1 Flash、Kimi K3、Qwen3.6 的结构与字节数取自 AII 第 2 章 §2.3.3–§2.3.6；线性注意力的 2 维小例子是编的。
"""


def v4_flash_state(S: int, win: int = 128, d: int = 512, d_idx: int = 128, b: int = 2) -> tuple[float, float]:
    """V4-Flash（2 窗口层 + 21 CSA + 20 HCA，BF16 参考格式）：(保存 MiB, 最后一次查询读取 MiB)。"""
    csa, hca, w = S // 4, S // 128, min(S, win)
    keep = 43 * w * d * b + 21 * csa * (d + d_idx) * b + 20 * hca * d * b
    read = 43 * w * d * b + 21 * (csa * d_idx + min(512, csa) * d) * b + 20 * hca * d * b
    return keep / 2**20, read / 2**20


def linear_attention(tokens: list[tuple[list[float], list[float]]], q: list[float]) -> tuple[list[list[float]], list[float]]:
    """S_t = S_{t-1} + k vᵀ，o = qᵀ S。返回 (最终状态, 输出)；状态大小始终 d_k × d_v。"""
    d_k, d_v = len(tokens[0][0]), len(tokens[0][1])
    state = [[0.0] * d_v for _ in range(d_k)]
    for k, v in tokens:
        for i in range(d_k):
            for j in range(d_v):
                state[i][j] += k[i] * v[j]  # 外积累加
    out = [sum(q[i] * state[i][j] for i in range(d_k)) for j in range(d_v)]
    return state, out


def recurrent_mib(layers: int, heads: int, d: int, b: int = 4) -> float:
    """递推状态（MiB）= 层数 × 头数 × d × d × b，不随上下文变化。"""
    return layers * heads * d * d * b / 2**20


def h_star(fixed_bytes: float, per_token_bytes: float) -> float:
    """混合模型固定状态与逐 token 部分相等的上下文长度 H* = M_fixed / c_state。"""
    return fixed_bytes / per_token_bytes


def v41_bytes_per_token(main: int = 288, idx: int = 68) -> float:
    """V4.1 Flash：3 层每 2 token 一条 + 1 层每 token 一条，每条主记录 + 索引。"""
    return (3 / 2 + 1) * (main + idx)


def v4_bytes_per_token(main: int = 584, idx: int = 68) -> float:
    """V4-Flash：21 层 4:1（主 + 索引）+ 20 层 128:1（只有主记录）。"""
    return 21 * (main + idx) / 4 + 20 * main / 128


def main() -> None:
    print("\n第一步：V4-Flash，8192 个 token，BF16 参考格式")
    keep, read = v4_flash_state(8192)
    print(f"  保存 {keep:.3f} MiB，最后一次查询读 {read:.3f} MiB（Qwen3-8B 同样 8K 各 1152 MiB）")

    print("\n第二步：线性注意力小例子（编的）")
    state, out = linear_attention([([1, 0], [2, 3]), ([0, 1], [1, 1])], [1, 0])
    print(f"  S_2 = {state}，q = [1, 0] 时 o = {out}")
    kda = recurrent_mib(69, 96, 128)
    print(f"  Kimi K3 的 KDA：{kda:.0f} MiB，每步读写 {2 * kda:.0f} MiB")

    print("\n第三步：混合模型的交点")
    print(f"  Qwen3.6：61.875 MiB ÷ 20 KiB = {h_star(61.875 * 2**20, 20 * 1024):.0f}")
    print(f"  Kimi K3：433.406 MiB ÷ 27 KiB ≈ {h_star(433.406 * 2**20, 27 * 1024):.0f}")

    print("\n第四步：每 token 的全局 KV 增量")
    print(f"  V4.1 Flash {v41_bytes_per_token():.0f} B，V4-Flash {v4_bytes_per_token():.2f} B，相差 {v4_bytes_per_token() / v41_bytes_per_token():.2f} 倍")


if __name__ == "__main__":
    main()
