"""第 11 章 · MoE：专家路由与批内权重复用。

讲义：ai-infra/notes/lessons/11-moe.md
运行：uv run python ch11_moe.py
Qwen3.6-35B-A3B（d = 2048、256 个专家、选 8 个、专家宽 512、40 层）与 V4-Flash（d = 4096、专家宽 2048）取自 AII 第 2 章 §2.4。
"""


def expert_params(d: int, f: int) -> int:
    """一个 SwiGLU 专家的参数：gate、up、down 共 3 × d × f。"""
    return 3 * d * f


def route_uniform(B: int, k: int, E: int) -> list[int]:
    """把 B·k 次分派轮流摊到 E 个专家：返回每个专家收到的 token 数 t_e。"""
    te = [0] * E
    for i in range(B * k):
        te[i % E] += 1
    return te


def route_concentrated(B: int, k: int, E: int = 256) -> list[int]:
    """所有 token 都选同一组 k 个专家。"""
    return [B if e < k else 0 for e in range(E)]


def expert_cost(te: list[int], n_e: int, b: int = 2) -> tuple[int, int]:
    """(专家运算 2·N_e·Σt_e, 权重读取 b·N_e·U)，U = 实际访问的不同专家数。"""
    flops = 2 * n_e * sum(te)  # 按分派的总行数
    u = sum(1 for t in te if t > 0)  # 实际访问的不同专家数
    return flops, b * n_e * u  # 读取按不同专家数


def main() -> None:
    print("\n第一步：Qwen3.6 的路由专家")
    n_e = expert_params(2048, 512)
    print(f"  一个专家 {n_e:,} 个参数 = {n_e * 2 / 2**20:.0f} MiB（BF16）")
    print(f"  每层 256 个 = {256 * n_e * 2 / 2**30:.1f} GiB；一个 token 选中的 8 个 = {8 * n_e * 2 / 2**20:.0f} MiB")

    print("\n第二步：B = 64，每 token 选 8 个，共 512 次分派")
    for name, te in (("均匀", route_uniform(64, 8, 256)), ("集中", route_concentrated(64, 8))):
        flops, read = expert_cost(te, n_e)
        u = sum(1 for t in te if t > 0)
        print(f"  {name}：U = {u}，每专家 {max(te)} 个 token，运算 {flops / 1e9:.2f} GFLOPs，读 {read / 2**20:.0f} MiB")
    print(f"  40 层：全部存 {40 * 256 * n_e * 2 / 2**30:.0f} GiB，集中时每批读 {40 * 8 * n_e * 2 / 2**30:.3f} GiB")

    print("\n第三步：V4-Flash 的一个路由专家")
    print(f"  3 × 4096 × 2048 = {expert_params(4096, 2048):,}")


if __name__ == "__main__":
    main()
