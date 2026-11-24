"""第 36 章 · 超节点的物理组织与内存池。

讲义：ai-infra/notes/lessons/36-topology-pool.md
运行：uv run python ch36_topology_pool.py
跨服务器 TP16、NVSwitch 端口分配、环面与交换网络、供电上限、UB 连接状态、内存池的在途事务取自 AII 第 6 章 §6.5–§6.6。
"""

import math

KiB, MiB = 2**10, 2**20


def tp16_cross_server(local_ms: float = 2.48, ar_us: float = 32.74) -> float:
    return local_ms + 128 * ar_us / 1e3


def uplink_gbps(up_ports: int, port_gbps: float = 25) -> float:
    return up_ports * port_gbps


def torus_alltoall_us(M: float, k: int = 4, ports: int = 6, port_bw: float = 50e9) -> tuple[float, float]:
    """均匀 All-to-All：(环面每条有向链路负载对应的时间, 交换网络每端口时间)。平均跳数 = 维数 × k/4。"""
    n = k**3
    hops = 3 * k / 4
    links = n * ports
    torus = hops * n * M / links / port_bw
    switch = M / ports / port_bw
    return torus * 1e6, switch * 1e6


def max_cards(budget_kw: float, server_kw: float = 10.2, g: int = 8, fixed_kw: float = 0) -> int:
    """式 (6-11)。"""
    return g * math.floor((budget_kw - fixed_kw) / server_kw)


def nic_state(H: int, A: int = 8) -> tuple[int, int]:
    """(逐对连接, UB 端点加通道) 的每网卡状态字节。"""
    return 512 * A * A * (H - 1) + 32 * A, 52 * A + 56 * (H - 1)


def max_hosts(cache: int = 256 * KiB, A: int = 8) -> tuple[int, int]:
    pair = (cache - 32 * A) // (512 * A * A) + 1
    ub = (cache - 52 * A) // 56 + 1
    return pair, ub


def setup_s(N: int, M: int, syscall: float = 5e-6, rtt: float = 500e-6, cores: int = 32) -> tuple[float, float]:
    pair = N * M * (4 * syscall + rtt)
    ub = N * syscall + M * (syscall + rtt)
    return pair / cores, ub / cores


def window_bw(u: int, q: int = 256, L: float = 7.52e-6, path: float = 50e9) -> float:
    """式 (6-12)。"""
    return min(path, u * q / L)


def main() -> None:
    print("\n第一步：再加一台服务器")
    print(f"  TP16 跨服务器 {tp16_cross_server():.2f} ms（TP8 机内 5.15 ms）；同一 NVLink 域 {2.48 + 128 * 30 * 0.822e-3:.2f} ms")
    print(f"  NVSwitch 64 端口：32 / 32 上联 {uplink_gbps(32):.0f} GB/s，48 / 16 上联 {uplink_gbps(16):.0f} GB/s")

    print("\n第二步：64 卡环面 vs 交换网络（每卡 6 × 50 GB/s，每卡发 32 MiB）")
    t, s = torus_alltoall_us(32 * MiB)
    print(f"  All-to-All：环面 {t:.0f} μs，交换 {s:.0f} μs；512 卡环面 {torus_alltoall_us(32 * MiB, k=8)[0]:.0f} μs")
    print(f"  环面最远多 4 颗中间芯片 × 200 ns = 0.8 μs，128 次 {128 * 0.8 / 1e3:.2f} ms")
    for k in (4, 8):
        print(f"  {k}³ 环面：{k**3} 颗芯片，二分链路 {2 * k * k} 条")
    print(f"  Swing vs 递归 前三轮瓶颈：{12 * MiB / 50e9 * 1e6:.0f} vs {8 * MiB / 50e9 * 1e6:.0f} μs")

    print("\n第三步：供电")
    for n in (64, 72, 96):
        print(f"  {n} 卡 {n / 8 * 10.2:.1f} kW")
    print(f"  120 kW 最多 {max_cards(120)} 张卡")

    print("\n第四步：UB 的连接状态")
    pr, ub = nic_state(192)
    print(f"  192 台主机：逐对 {pr / MiB:.1f} MiB，端点加通道 {ub / KiB:.1f} KiB；256 KiB 缓存最多 {max_hosts()} 台")
    a, b = setup_s(1024, 1024)
    print(f"  N = M = 1024 建连（32 核）：逐对 {a:.0f} s，UB {b * 1e3:.0f} ms")

    print("\n第五步：内存池")
    print(f"  128 个在途 × 256 B / 7.52 μs = {window_bw(128) / 1e9:.2f} GB/s；用满 50 GB/s 要 {math.ceil(50e9 * 7.52e-6 / 256)} 个；4096 个 → {4096 * 256 / 7.52e-6 / 1e9:.0f} GB/s")
    print(f"  20 GB：50 GB/s {20e9 / 50e9:.2f} s，4.36 GB/s {20e9 / window_bw(128):.2f} s；读 10 次：远端 {10 * 20e9 / 50e9:.1f} s，取回 + 本地 {20e9 / 50e9 + 10 * 20e9 / 3350e9:.2f} s")


if __name__ == "__main__":
    main()
