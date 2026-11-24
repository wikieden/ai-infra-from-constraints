"""第 38 章 · 从超节点走向集群：容量、割集与 Clos 网络。

讲义：ai-infra/notes/lessons/38-cluster.md
运行：uv run python ch38_cluster.py
训练状态容量、例题 7.1 的强扩展、Clos 网络的端点数与半分带宽、超售比取自 AII 第 7 章 §7.1。
"""

import math

MiB = 2**20


def strong_scaling(x: float, comp_ms: float = 20, comm_ms: float = 360 * MiB / 50e9 * 1e3) -> tuple[float, float]:
    """(串行, 完全重叠) 的完成时间。"""
    c = comp_ms / x
    return c + comm_ms, max(c, comm_ms)


def clos(k: int, layers: int) -> tuple[int, int]:
    """无阻塞 Clos：(端点数, 交换机数)。"""
    if layers == 2:
        return k * k // 2, k + k // 2
    return k**3 // 4, 5 * k * k // 4


def bisection_tbps(endpoints: int, port_gbps: float = 50) -> float:
    return endpoints / 2 * port_gbps / 1e3


def oversub_two_layer(k: int = 64, down: int = 48, up: int = 16) -> tuple[int, int, float]:
    """超售两层：(端点数, 交换机数, 半分带宽 TB/s)。"""
    leaves = k
    spines = leaves * up // k
    return leaves * down, leaves + spines, leaves * up / 2 * 50 / 1e3


def job_cut(cards: int, down: int, up: int, port_gbps: float = 50) -> tuple[int, int, float, float]:
    """一个作业占整数台叶交换机：(叶数, 割集链路, 割集 TB/s, 每卡 GB/s)。"""
    leaves = math.ceil(cards / down)
    links = leaves * up
    return leaves, links, links * port_gbps / 1e3, links * port_gbps / cards


def main() -> None:
    print("\n第一步：容量")
    print(f"  8B 混合精度 Adam 16 B/参数：{8e9 * 16 / 1e9:.0f} GB；284B：{284e9 * 16 / 1e12:.2f} TB")
    print(f"  推理 0.5 B/参数：1.6T {1.6e12 * 0.5 / 1e9:.0f} GB → {math.ceil(800 / 640)} 台；2.8T {2.8e12 * 0.5 / 1e12:.1f} TB → {math.ceil(1400 / 640)} 台")

    print("\n第二步：例题 7.1，跨服务器 360 MiB 走一张 50 GB/s 网卡")
    t = 360 * MiB / 50e9 * 1e3
    for x in (1, 2, 4):
        s, o = strong_scaling(x)
        print(f"  卡数 ×{x}：串行 {s:.1f} ms，重叠 {o:.1f} ms")
    print(f"  通信 {t:.2f} ms；计算 = 通信时 x = {20 / t:.2f}")

    print("\n第三步：Clos，k = 64")
    for L in (2, 3):
        e, s = clos(64, L)
        print(f"  {L} 层无阻塞：{e:,} 个端点、{s:,} 台交换机，半分 {bisection_tbps(e):,.1f} TB/s")
    e, s, b = oversub_two_layer()
    print(f"  3:1 超售两层：{e} 个端点、{s} 台交换机，半分 {b} TB/s（无阻塞同规模 {bisection_tbps(e):.1f}）")
    for name, d, u in (("1:1", 32, 32), ("3:1", 48, 16)):
        print(f"  1024 卡 {name}：{job_cut(1024, d, u)}")


if __name__ == "__main__":
    main()
