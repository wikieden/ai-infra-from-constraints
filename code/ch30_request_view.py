"""第 30 章 · 从局部优化到完整请求。

讲义：ai-infra/notes/lessons/30-request-view.md
运行：uv run python ch30_request_view.py
Amdahl、两分支的关键路径、Qwen3-8B 一次请求的 SwiGLU 替换取自 AII 第 5 章 §5.6。
"""


def amdahl(f: float, s: float) -> float:
    """热点占 f、加速 s 倍时的整体加速比。"""
    return 1 / ((1 - f) + f / s)


def dag_time(prep: float, a: float, b: float, tail: float) -> float:
    """准备后 A、B 并行，收尾等两者都完成。"""
    return prep + max(a, b) + tail


def main() -> None:
    print("\n第一步：热点占 20%")
    for s in (2, 10, float("inf")):
        print(f"  加速 {s} 倍：整体 {amdahl(0.2, s):.2f} 倍（新时间 {1 / amdahl(0.2, s):.2f}）")

    print("\n第二步：两条分支")
    print(f"  原 {dag_time(10, 60, 40, 10):.0f} μs；A 快 4 倍 {dag_time(10, 15, 40, 10):.0f} μs；A 再快也停在 {dag_time(10, 0, 40, 10):.0f} μs")
    print(f"  A 抢带宽让 B 变 90 μs：{dag_time(10, 15, 90, 10):.0f} μs")

    print("\n第三步：Qwen3-8B，7239 入 32 出，36 层")
    print(f"  SwiGLU 调用 {36 * 32}：prefill {36} 次共 {36 * 7239:,} 行，decode {36 * 31} 次各 1 行")
    print(f"  原激活合计 13.6 ms，占 851 ms 的 {13.6 / 851:.1%}；decode 2.4 → 0.92 ms（{2.4 / 0.92:.1f} 倍）")
    print(f"  预期省 ≈ 1.4 ms，实测配对中位数 1.3 ms（{1.3 / 817:.2%}）")


if __name__ == "__main__":
    main()
