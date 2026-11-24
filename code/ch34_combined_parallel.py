"""第 34 章 · 组合并行与模型规模。

讲义：ai-infra/notes/lessons/34-combined-parallel.md
运行：uv run python ch34_combined_parallel.py
TP2×EP4 的归约发送量、批内专家复用、式 (6-8) 的期望活跃专家数取自 AII 第 6 章 §6.3。
"""

KiB, MiB, GiB = 2**10, 2**20, 2**30
EXPERT = 3 * 4096 * 1536 * 2  # 36 MiB


def ring_send_factor(n: int) -> float:
    """环形 AllReduce 每卡发送量是完整向量的 2(n-1)/n 倍。"""
    return 2 * (n - 1) / n


def tp2ep4_send_kib(hidden: int = 4096, b: int = 4) -> float:
    """每层三次汇合（注意力组内、专家组内、四组之间）每卡发送 KiB。"""
    v = hidden * b / KiB
    return v * ring_send_factor(2) + v * ring_send_factor(2) + v * ring_send_factor(4)


def expected_active(E: int, k: int, m: int) -> float:
    """式 (6-8)：m 个 token 各选 k 个，期望活跃专家数。"""
    return E * (1 - (1 - k / E) ** m)


def group_load(assign: list[int], groups: int = 4, per_group: int = 32) -> list[int]:
    """assign[e] = 专家 e 收到的 token 数；返回各 EP 组的任务数。"""
    load = [0] * groups
    for e, n in enumerate(assign):
        load[e // per_group] += n
    return load


def main() -> None:
    print("\n第一步：TP2×EP4，FP32 归约")
    s = tp2ep4_send_kib()
    print(f"  每 token 每卡 {s:.0f} KiB（16 + 16 + 24），八卡 {8 * s:.0f} KiB；8192 token 每卡 {s * 8192 / 1024:.0f} MiB")

    print("\n第二步：64 个 token × 8 个专家 = 512 次分派")
    uniform = [4] * 128
    hot = [64 if e < 8 else 0 for e in range(128)]
    spread = [64 if e in (0, 1, 32, 33, 64, 65, 96, 97) else 0 for e in range(128)]
    for name, a in (("均匀", uniform), ("集中在组 0", hot), ("热点分散到四组", spread)):
        act = sum(1 for n in a if n)
        print(f"  {name}：读 {act * EXPERT / MiB:,.0f} MiB，各组任务 {group_load(a)}")
    for m in (1, 8, 64, 512):
        ea = expected_active(128, 8, m)
        print(f"  随机路由 m = {m:3d}：期望活跃 {ea:5.1f} 个，读 {ea * EXPERT / GiB:.2f} GiB")


if __name__ == "__main__":
    main()
