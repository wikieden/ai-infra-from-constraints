"""第 39 章 · 跨节点流量与物理路径。

讲义：ai-infra/notes/lessons/39-cross-node.md
运行：uv run python ch39_cross_node.py
例题 7.2 三种归约、对照 PCIe 配置、在网归约、层间交接、EP 接收、多网卡中继取自 AII 第 7 章 §7.2。
"""

MiB, GiB = 2**20, 2**30
NIC, NVL, A = 50e9, 450e9, 0.83e-6


def ring_continuous_ms(M: float = 192 * MiB, p: int = 16, bw: float = NIC) -> float:
    """连续环：每轮等跨服务器那条边。"""
    return 2 * (p - 1) * (A + M / p / bw) * 1e3


def hierarchical_ms(M: float = 192 * MiB, local: int = 8, servers: int = 2, BL: float = NVL, BN: float = NIC) -> float:
    """分层：本地 RS + AG 共 2(local-1) 轮，跨服务器 2(servers-1) 轮。"""
    shard = M / local
    t_local = 2 * (local - 1) * shard / BL
    t_remote = 2 * (servers - 1) * shard / servers / BN
    return (t_local + t_remote + (2 * (local - 1) + 2 * (servers - 1)) * A) * 1e3


def cross_bytes_mib() -> tuple[float, float, float]:
    return 2 * 30 * 12, 16 * 30 * 12, 8 * 48


def in_network_ms(S: int, shard: float = 24 * MiB) -> tuple[float, float]:
    """跨服务器阶段：(环, 在网归约)。"""
    ring = 2 * (S - 1) * A + 2 * (S - 1) / S * shard / NIC
    inn = A + shard / NIC
    return ring * 1e3, inn * 1e3


def msg_us(M: float, alpha: float = 5e-6, bw: float = NIC) -> float:
    return (alpha + M / bw) * 1e6


def relay_gbps(BD: float) -> float:
    return min(BD, 50 + min(450, 100))


def main() -> None:
    print("\n第一步：例题 7.2，192 MiB 梯度，两台 HGX H100")
    print(f"  跨服务器字节：连续 / 交错 / 分层 = {cross_bytes_mib()} MiB")
    print(f"  连续环 {ring_continuous_ms():.1f} ms；分层 {hierarchical_ms():.2f} ms")
    print(f"  分层合算的本地带宽下限 ≈ {336 * MiB / ((ring_continuous_ms() - 0.503) * 1e-3) / 1e9:.0f} GB/s")
    ring8 = 14 * 24 * MiB / 32e9 * 1e3
    loc, rem = 6 * 48 * MiB / 32e9 * 1e3, 2 * 96 * MiB / 32e9 * 1e3
    print(f"  对照 PCIe 配置：连续环 {ring8:.1f} ms，分层 {loc:.1f} + {rem:.1f} = {loc + rem:.1f} ms；本地 300 GB/s 时 {6 * 48 * MiB / 300e9 * 1e3 + rem:.1f} ms")
    print(f"  持平的本地带宽 ≈ {6 * 48 * MiB / ((ring8 - rem) * 1e-3) / 1e9:.0f} GB/s")
    for S in (2, 8, 32):
        r, i = in_network_ms(S)
        print(f"  在网归约 {S:2d} 台：环 {r:.3f} ms，在网 {i:.3f} ms")

    print("\n第二步：TP / PP 交接，α = 5 μs")
    print(f"  64 MiB {msg_us(64 * MiB) / 1e3:.2f} ms，8 KiB {msg_us(8192):.2f} μs；转折 {5e-6 * NIC / 1024:.0f} KiB")
    print(f"  4 阶段 8 micro-batch：{8 + 3} ms，利用率 {8 / 11:.0%}；90% 要 {next(m for m in range(1, 100) if m / (m + 3) >= 0.9)} 个")

    print("\n第三步：EP，1024 token × 8 专家 × 8 KiB，一半跨服务器")
    v = 1024 * 8 * 8 * 1024 * 0.5
    print(f"  dispatch {v / MiB:.0f} MiB：一张网卡 {v / NIC * 1e3:.2f} ms，八张 {v / 8 / NIC * 1e3:.3f} ms，共享 32 GB/s 插槽 {v / 32e9 * 1e3:.2f} ms")

    print("\n第四步：多网卡")
    print(f"  对照配置单端口 {14 * 24 * MiB / 25e9 * 1e3:.1f} ms，双端口受插槽限制 {14 * 24 * MiB / 32e9 * 1e3:.1f} ms")
    for BD in (50, 100, 150, 200):
        print(f"  下游 {BD} GB/s：可用 {relay_gbps(BD)} GB/s，1.125 GiB {1.125 * GiB / (relay_gbps(BD) * 1e9) * 1e3:.1f} ms")
    print(f"  δ = 18.6 ns 时速率与带宽交叉的消息大小 {NIC * 18.6e-9:.0f} B；192 MiB 一张网卡 {192 * MiB / NIC * 1e3:.2f} ms")


if __name__ == "__main__":
    main()
