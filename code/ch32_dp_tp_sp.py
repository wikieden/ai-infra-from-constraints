"""第 32 章 · 数据、张量、序列并行。

讲义：ai-infra/notes/lessons/32-dp-tp-sp.md
运行：uv run python ch32_dp_tp_sp.py
Qwen3-32B 的 128K 会话容量、TP 的列 / 行切分、式 (6-4)、二维网格的收集量取自 AII 第 6 章 §6.2.1–§6.2.3。
"""

KiB, MiB, GiB = 2**10, 2**20, 2**30
HBM = 3350e9
WORK = 2 * GiB / 1e9


def matvec(x: list[float], W: list[list[float]]) -> list[float]:
    return [sum(x[i] * W[i][j] for i in range(len(x))) for j in range(len(W[0]))]


def col_split(x, W):
    """按列切：每卡用完整输入，各得不同输出元素，拼接。"""
    return [matvec(x, [[row[j]] for row in W])[0] for j in range(len(W[0]))]


def row_split(x, W):
    """按行切：每卡用一个输入特征，各得一份部分和，相加。"""
    parts = [[x[i] * w for w in W[i]] for i in range(len(x))]
    return parts, [sum(p[j] for p in parts) for j in range(len(W[0]))]


def sessions_fit(p: int, weights_gb: float = 65.52, sess_gb: float = 34.36, cap: float = 80) -> int:
    """TP p 卡时每卡能放下的 128K 会话数（p ≤ 8，KV 按头均分）。"""
    return int((cap - WORK - weights_gb / p) // (sess_gb / p))


def step_bytes(s: int) -> float:
    """式 (6-4)：Qwen3-32B 一步的 HBM 访问量。"""
    return 64 * 930 * MiB + 256 * KiB * (s + 1) + 151936 * 5120 * 2


def grid_send_mib(rows: int, cols: int, inp: float = 4, wt: float = 6) -> tuple[float, float]:
    """r×c 网格：横向收集同行输入、纵向收集同列权重，每卡发送 (MiB, MiB)。"""
    return (cols - 1) * inp, (rows - 1) * wt


def main() -> None:
    print("\n第一步：TP 的两种切法，x = [2, 3]，W = [[1, 4], [2, 5]]")
    x, W = [2, 3], [[1, 4], [2, 5]]
    parts, total = row_split(x, W)
    print(f"  按列：{col_split(x, W)}（拼接）；按行：部分和 {parts} → {total}（相加）")

    print("\n第二步：Qwen3-32B 一个 128K 会话 34.36 GB")
    print(f"  每 token KV {2 * 64 * 8 * 128 * 2 // KiB} KiB；单卡需要 {65.52 + 34.36 + WORK:.2f} GB")
    for p in (2, 4, 8):
        print(f"  TP{p}：每卡 {(65.52 + 34.36) / p + WORK:.2f} GB，最多 {sessions_fit(p)} 个会话")
    f = (2 * 5120 * 25600 + 25600 * 5120) * 2
    print(f"  FFN 三矩阵 {f // MiB} MiB → TP4 每卡 {f // 4 / MiB:.1f} MiB，{f / HBM * 1e6:.1f} → {f / 4 / HBM * 1e6:.1f} μs")
    v = step_bytes(131064)
    print(f"  式 (6-4)：一步 {v / 1e9:.2f} GB，{v / HBM * 1e3:.2f} ms")

    print("\n第三步：SP 与二维网格")
    print(f"  [8192, 4096] 激活 {8192 * 4096 * 2 // MiB} MiB：四卡各存完整 {4 * 64} MiB，按位置分片共 64 MiB")
    for r, c in ((4, 4), (2, 8), (8, 2)):
        a, b = grid_send_mib(r, c)
        print(f"  {r}×{c} 网格：输入 {a:.0f} MiB、权重 {b:.0f} MiB，较大的 {max(a, b):.0f}")


if __name__ == "__main__":
    main()
