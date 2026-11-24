"""第 21 章 · 存储层次与数据搬移：放得下、取得到、来得及。

讲义：ai-infra/notes/lessons/21-memory-movement.md
运行：uv run python ch21_memory_movement.py
乘加块、例 4-2、子矩阵寻址、双缓冲、B200 单 SM 的输入槽时序取自 AII 第 4 章 §4.3–§4.4。
"""

import math


def tile_kib(m: int, n: int, k: int, b_in: int = 2, b_acc: int = 4) -> tuple[float, float]:
    """一套乘加块的 (输入 + 权重 KiB, 累加器 KiB)。"""
    return (m * k + k * n) * b_in / 1024, m * n * b_acc / 1024


def inflight_needed(rate: float, latency_s: float, s: int = 128) -> int:
    """维持目标带宽需要的在途事务数 ⌈R·L/s⌉。"""
    return math.ceil(rate * latency_s / s)


def effective_bw(n_o: int, latency_s: float, interface: float, s: int = 128) -> float:
    """有效带宽 = min(接口带宽, N_o·s/L)。"""
    return min(interface, n_o * s / latency_s)


def submatrix_span(rows: int, cols: int, row_width: int, b: int = 2) -> tuple[int, int, int]:
    """从行宽 row_width 的矩阵取 rows×cols：(有效字节, 行步长, 首行起点到末行终点的跨度)。"""
    stride = row_width * b
    return rows * cols * b, stride, (rows - 1) * stride + cols * b


def pipe_time(n: int, t_l: float, t_c: float) -> tuple[float, float]:
    """(串行, 双缓冲流水) 的总时间。"""
    return n * (t_l + t_c), t_l + t_c + (n - 1) * max(t_l, t_c)


def simulate_slots(slots: int, t_c: int, n: int = 4, xfer: int = 64, wait: int = 128) -> int:
    """输入槽逐块模拟：传输串行，发起后 xfer+wait 就绪，算完才释放槽。返回完成时刻（tick）。"""
    free_at = [0] * slots
    issue_free = 0
    comp_end = 0
    for _ in range(n):
        s = min(range(slots), key=lambda i: free_at[i])  # 最早空出来的槽
        issue = max(free_at[s], issue_free)
        issue_free = issue + xfer  # 传输串行
        start = max(issue + xfer + wait, comp_end)  # 就绪了、前一块也算完了
        comp_end = start + t_c
        free_at[s] = comp_end  # 算完才释放槽
    return comp_end


def main() -> None:
    print("\n第一步：m = n = 128、k = 64 的乘加块")
    inp, acc = tile_kib(128, 128, 64)
    print(f"  输入 + 权重 {inp:.0f} KiB，累加器 {acc:.0f} KiB，合计 {inp + acc:.0f} KiB；双缓冲 {2 * inp + acc:.0f} KiB")

    print("\n第二步：例 4-2，读 1.125 GiB 的 KV")
    kv = 1.125 * 2**30
    print(f"  用满 1,008 GB/s 需要 {inflight_needed(1.008e12, 500e-9)} 个在途事务")
    for n_o, iface in ((128, 1.008e12), (4096, 1.008e12), (4096, 1.792e12)):
        bw = effective_bw(n_o, 500e-9, iface)
        print(f"  {n_o:5d} 个，接口 {iface / 1e9:,.0f} GB/s → {bw / 1e9:,.1f} GB/s，{kv / bw * 1e3:.2f} ms")
    print(f"  用满 1,792 GB/s：{inflight_needed(1.792e12, 500e-9)} 个；延迟 800 ns：{inflight_needed(1.792e12, 800e-9)} 个")

    print("\n第三步：从行宽 4096 的 BF16 矩阵取 128×128")
    valid, stride, span = submatrix_span(128, 128, 4096)
    print(f"  有效 {valid // 1024} KiB，行步长 {stride} B，跨度 {span:,} B")

    print("\n第四步：双缓冲与输入槽")
    print(f"  8 块、加载 2 μs、计算 1 μs：串行 {pipe_time(8, 2, 1)[0]:.0f} μs，流水 {pipe_time(8, 2, 1)[1]:.0f} μs")
    print(f"  输入槽 1–4：{[simulate_slots(s, 128) for s in (1, 2, 3, 4)]} tick；最早 {192 + 4 * 128}")
    print(f"  矩阵 ×2：三槽 {simulate_slots(3, 64)}，四槽 {simulate_slots(4, 64)}（目标 {192 + 4 * 64}）")


if __name__ == "__main__":
    main()
