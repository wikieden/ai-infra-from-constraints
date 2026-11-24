"""第 33 章 · 上下文、流水线、专家并行。

讲义：ai-infra/notes/lessons/33-cp-pp-ep.md
运行：uv run python ch33_cp_pp_ep.py
因果配对的负载、跨卡合并注意力统计量、式 (6-7) 的流水、专家 dispatch 字节取自 AII 第 6 章 §6.2.4–§6.2.6。
"""

import math

KiB, MiB, GiB = 2**10, 2**20, 2**30


def causal_pairs(positions: list[int]) -> int:
    """因果注意力：位置 i（从 1 起）的查询看 i 个键。"""
    return sum(positions)


def merge_stats(parts: list[tuple[float, float, float]]) -> float:
    """各卡 (m_r, l_r, u_r) 合并成最终输出 u / l（标量值版本）。"""
    m = max(p[0] for p in parts)
    l = sum(math.exp(mr - m) * lr for mr, lr, _ in parts)
    u = sum(math.exp(mr - m) * ur for mr, _, ur in parts)
    return u / l


def local_stats(scores: list[float], values: list[float]) -> tuple[float, float, float]:
    m = max(scores)
    w = [math.exp(s - m) for s in scores]
    return m, sum(w), sum(wi * v for wi, v in zip(w, values))


def pipeline(q: int, b: int, t: float = 1.0) -> tuple[float, float]:
    """式 (6-7)：(总时间, 利用率)。"""
    return (q + b - 1) * t, b / (q + b - 1)


def pipeline_uneven(stages: list[float], b: int) -> list[float]:
    """各阶段耗时不等时每个 micro-batch 的完成时刻（无限缓冲）。"""
    free = [0.0] * len(stages)
    done = []
    for _ in range(b):
        t = 0.0
        for s, d in enumerate(stages):
            t = max(t, free[s]) + d
            free[s] = t
        done.append(t)
    return done


def main() -> None:
    print("\n第一步：CP，8 个位置的因果注意力")
    print(f"  共 {causal_pairs(list(range(1, 9)))} 对；连续切半 {causal_pairs([1, 2, 3, 4])} / {causal_pairs([5, 6, 7, 8])}；"
          f"首尾搭配 {causal_pairs([1, 2, 7, 8])} / {causal_pairs([3, 4, 5, 6])}")
    s, v = [0.5, 2.0, -1.0, 1.5], [1, 2, 3, 4]
    two = merge_stats([local_stats(s[:2], v[:2]), local_stats(s[2:], v[2:])])
    print(f"  两卡各算统计量再合并 {two:.4f}；一次算整行 {merge_stats([local_stats(s, v)]):.4f}")

    print("\n第二步：PP，式 (6-7)")
    for b in (4, 16):
        T, eta = pipeline(4, b)
        print(f"  4 阶段、{b} 个 micro-batch：{T:.0f} ms，利用率 {eta:.0%}")
    print(f"  阶段 1、1、2、1 ms，4 个 micro-batch 完成于 {pipeline_uneven([1, 1, 2, 1], 4)} ms")

    print("\n第三步：EP")
    print(f"  8 个专家两两在 4 张远端卡：按专家发 {8 * 8} KiB，按目的卡发 {4 * 8} KiB")
    e = 3 * 4096 * 1536 * 2
    print(f"  每专家 {e // MiB} MiB，单层 128 个 {128 * e / GiB:.1f} GiB，94 层 {94 * 128 * e / GiB:.0f} GiB，八卡每卡 {94 * 128 * e / GiB / 8:.1f} GiB")


if __name__ == "__main__":
    main()
