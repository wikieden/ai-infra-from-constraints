"""第 12 章 · 网络形状：深宽、专家粒度、残差与多 token 预测。

讲义：ai-infra/notes/lessons/12-network-shape.md
运行：uv run python ch12_network_shape.py
Qwen3-235B-A22B（94 层、d = 4096、128 个专家宽 1536、选 8 个）与推测解码的 20 ms / 30 ms 取自 AII 第 2 章 §2.5；
深宽比例、Sinkhorn 的 2×2 矩阵是编的。
"""


def dense_scale(layer_factor: float, width_factor: float) -> float:
    """稠密主干参数 ∝ L·d²：层数乘 a、宽度乘 b，参数变成 a·b² 倍。"""
    return layer_factor * width_factor**2


def expert_totals(E: int, f: int, k: int, d: int = 4096) -> tuple[int, int]:
    """单层路由专家 (总参数 3Edf, 单 token 运算 6kdf)。"""
    return 3 * E * d * f, 6 * k * d * f


def sinkhorn(m: list[list[float]], rounds: int) -> list[list[float]]:
    """Sinkhorn 迭代：交替把每行、每列除以各自的和。"""
    for _ in range(rounds):
        m = [[v / sum(r) for v in r] for r in m]  # 每行除以行和
        cols = [sum(r[j] for r in m) for j in range(len(m[0]))]
        m = [[r[j] / cols[j] for j in range(len(r))] for r in m]  # 每列除以列和
    return m


def spec_per_output(t_round: float, g: float) -> float:
    """推测解码每个输出分摊的时间：(T_s + T_v) / 平均有效输出数。"""
    return t_round / g


def main() -> None:
    print("\n第一步：层数 ×4、宽度 ×1/2（编的）")
    print(f"  参数变成 {dense_scale(4, 0.5):.2f} 倍，依赖链变成 4 倍")

    print("\n第二步：Qwen3-235B-A22B 的三种专家粒度")
    for E, f, k in ((128, 1536, 8), (64, 3072, 4), (256, 768, 16)):
        p, op = expert_totals(E, f, k)
        print(f"  E = {E:3d}，f = {f}，选 {k:2d}：每层参数 {p:,}，单 token 运算 {op:,}")
    print(f"  16 个 token × 94 层 = {16 * 94 * expert_totals(128, 1536, 8)[1] / 1e9:.0f} GFLOPs")

    print("\n第三步：Sinkhorn（编的 2×2）")
    m1 = sinkhorn([[1, 3], [1, 1]], 1)
    print(f"  一轮后 {[[round(v, 3) for v in r] for r in m1]}，行和 {[round(sum(r), 3) for r in m1]}")
    m50 = sinkhorn([[1, 3], [1, 1]], 50)
    print(f"  50 轮后行和 {[round(sum(r), 4) for r in m50]}，列和 {[round(sum(r[j] for r in m50), 4) for j in range(2)]}")

    print("\n第四步：推测解码（普通 decode 20 ms，一轮 30 ms）")
    for g in (2, 1.5, 1):
        print(f"  平均有效输出 {g} → 每输出 {spec_per_output(30, g):.0f} ms")
    print(f"  打平点 g = 30 / 20 = {30 / 20}")


if __name__ == "__main__":
    main()
