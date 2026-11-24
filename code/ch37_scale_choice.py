"""第 37 章 · 从并行方案到超节点规模。

讲义：ai-infra/notes/lessons/37-scale-choice.md
运行：uv run python ch37_scale_choice.py
四个 128K 会话续写 8 个 token 的部署比较、按时会话成本、故障、32K 对照取自 AII 第 6 章 §6.7.1–§6.7.3。
"""

KiB, MiB = 2**10, 2**20
HBM = 3350e9


def step_bytes(s: int) -> float:
    """式 (6-4)（同第 32 章）：Qwen3-32B 一步的 HBM 访问量。"""
    return 64 * 930 * MiB + 256 * KiB * (s + 1) + 151936 * 5120 * 2


def ring_ar(n: int, M: float, alpha: float = 0.822e-6, bw: float = 450e9) -> float:
    """式 (6-9)（同第 35 章）：环形 AllReduce。"""
    return 2 * (n - 1) * alpha + 2 * (n - 1) * M / (n * bw) if n > 1 else 0.0


def eight_steps_ms(p: int, s0: int = 131064) -> float:
    """一个会话续写 8 个 token 的时间（式 6-4、6-5）。"""
    return sum(step_bytes(s) / HBM / p + 128 * ring_ar(p, 10 * KiB) for s in range(s0, s0 + 8)) * 1e3


def finish_times(p: int, sessions: int = 4, cards: int = 8, **kw) -> list[float]:
    t = eight_steps_ms(p, **kw)
    inst = cards // p
    return [t * (i // inst + 1) for i in range(sessions)]


def cost_per_on_time(finish: list[float], deadline: float, cards: int = 8) -> tuple[float, int, float]:
    """(占用 GPU·s, 按时会话数, 每个按时会话的 GPU·s)。"""
    occ = cards * max(finish) / 1e3
    n = sum(1 for f in finish if f <= deadline)
    return occ, n, occ / n if n else float("inf")


def main() -> None:
    print("\n第一步：四个 128K 会话各续写 8 个 token，八张 H100")
    for p in (2, 4, 8):
        f = finish_times(p)
        occ, n, c = cost_per_on_time(f, 130)
        print(f"  {8 // p} 个 TP{p}：单会话 {eight_steps_ms(p):.1f} ms，完成 {[round(x, 1) for x in f]}，"
              f"占用 {occ:.3f} GPU·s，按时 {n}，每个 {c:.3f}")

    print("\n第二步：20 ms 卡 0 故障，停 40 ms 后从头重做")
    t2 = eight_steps_ms(2)
    f = [60 + t2] + [t2] * 3
    occ, n, c = cost_per_on_time(f, 130)
    print(f"  4 个 TP2：完成 {[round(x, 1) for x in f]}，占用 {occ:.3f}，每个按时会话 {c:.3f} GPU·s")
    print(f"  能耗（10.2 kW）：TP2 {10.2 * t2 / 1e3:.2f} kJ")

    print("\n第三步：上下文改成 32K")
    for p in (1, 2):
        print(f"  TP{p}：8 步 {eight_steps_ms(p, s0=32760):.1f} ms")
    f = finish_times(2, s0=32760)
    print(f"  4 个 TP2 每个按时会话 {cost_per_on_time(f, 130)[2]:.3f} GPU·s")


if __name__ == "__main__":
    main()
