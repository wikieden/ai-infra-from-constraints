"""第 43 章 · 从通信改进到任务完成。

讲义：ai-infra/notes/lessons/43-task-completion.md
运行：uv run python ch43_task_completion.py
就绪偏差、例题 7.6 的六种安排、训练与推理的启动 / 带宽、1024 卡超节点表、故障开销取自 AII 第 7 章 §7.6。
"""

MiB = 2**20
A = 0.83e-6
NIC, NVL = 50e9, 450e9


def ring_continuous_ms(M: float = 192 * MiB, p: int = 16, bw: float = NIC) -> float:
    """连续环（同第 39 章）：每轮等跨服务器那条边。"""
    return 2 * (p - 1) * (A + M / p / bw) * 1e3


def hierarchical_ms(M: float = 192 * MiB, local: int = 8, servers: int = 2, BL: float = NVL, BN: float = NIC) -> float:
    """分层归约（同第 39 章）。"""
    shard = M / local
    t_local = 2 * (local - 1) * shard / BL
    t_remote = 2 * (servers - 1) * shard / servers / BN
    return (t_local + t_remote + (2 * (local - 1) + 2 * (servers - 1)) * A) * 1e3


def step_ms(comm: float, ready: float | None = None, comp: float = 20, upd: float = 2) -> float:
    """ready=None 时计算、通信、更新串行；否则通信从 ready 开始与剩余计算重叠。"""
    if ready is None:
        return comp + comm + upd
    return max(comp, ready + comm) + upd


def ar8_us(M: float, alpha: float = 5e-6, B: float = 50e9) -> float:
    return (14 * alpha + 1.75 * M / B) * 1e6


def t_grad(S: int, exit_bw: float | None = None, G: float = 8e9, BL: float = 450e9, BN: float = 50e9) -> float:
    q, H = S // 8, 1024 // S
    v_out = 2 * (H - 1) / H * 8 * G
    b_out = exit_bw if exit_bw else S * BN
    local = 2 * (q - 1) * A + 2 * (q - 1) * G / (q * BL)
    remote = 2 * (H - 1) * A + max(2 * (H - 1) * G / (H * q * BN), v_out / b_out)
    hier = local + remote
    ring = 2 * 127 * (A + G / 128 / BN)  # 不分层的连续 DP 环
    return min(hier, ring)


def step_1024(S: int, exit_bw: float | None = None) -> tuple[float, float]:
    t = 0.485 + 0.086 + t_grad(S, exit_bw) + 0.050
    return t, 2**20 / t


def fault_overhead(S: int, C: float = 10, I: float = 600, mtbf_h: float = 7.9) -> float:
    lam = 1 / (mtbf_h * 3600)
    R = 60 + S * 64e9 / 64e9
    return C / I + lam * (I / 2 + R)


def main() -> None:
    print("\n第一步：就绪偏差：0、0、0、2 ms 就绪，交换 0.4 ms")
    print(f"  {2 + 0.4} ms；交换减半 {2 + 0.2} ms；同时就绪 0.4 ms")

    print("\n第二步：例题 7.6（192 MiB 梯度）")
    ring, hier = ring_continuous_ms(), hierarchical_ms()
    hier_lowslot = hier - 2 * 12 * MiB / 50e9 * 1e3 + 24 * MiB / 16.384e9 * 1e3
    one_nic = hier - 2 * 12 * MiB / 50e9 * 1e3 + 2 * 96 * MiB / 50e9 * 1e3
    for name, c, r in (("连续环串行", ring, None), ("分层串行", hier, None), ("分层 128 槽位", hier_lowslot, None),
                       ("连续环 17 ms 就绪", ring, 17), ("分层 17 ms 就绪", hier, 17), ("只剩一张网卡", one_nic, 17)):
        print(f"  {name}：通信 {c:.1f} ms，整步 {step_ms(c, r):.1f} ms")
    print(f"  分层通信最晚 {20 - hier:.1f} ms 就绪仍能藏住")

    print("\n第三步：训练与推理")
    for M in (8 * MiB, 8 * 1024):
        print(f"  {M / 1024:,.0f} KiB：50 GB/s {ar8_us(M):.1f} μs，150 GB/s {ar8_us(M, B=150e9):.1f} μs")
    print(f"  72 次 8 KiB：α 5 → 2 μs 省 {72 * 14 * 3 / 1e3:.2f} ms；带宽 ×3 省 {72 * (ar8_us(8192) - ar8_us(8192, B=150e9)):.0f} μs")
    print(f"  启动 = 传输的数据量 {14 * 5e-6 * 50e9 / 1.75 / 1e6:.0f} MB")
    print(f"  V4.1 迁移 128K 全局 KV（25 GB/s）：V4-Flash {439.281 * MiB / 25e9 * 1e3:.3f} ms，V4.1 {111.25 * MiB / 25e9 * 1e3:.3f} ms")

    print("\n第四步：1024 卡 TP8×DP128")
    for S in (8, 64, 128, 256):
        t1, th1 = step_1024(S)
        t2, th2 = step_1024(S, 400e9)
        print(f"  {S:3d} 卡超节点：出口随卡数 {t1:.3f} s / {th1 / 1e4:.1f} 万 token/s；出口 400 GB/s {t2:.3f} s / {th2 / 1e4:.1f} 万")
    for S in (128, 256):
        e = fault_overhead(S)
        print(f"  {S} 卡：故障开销 {e:.1%}，有效吞吐 {step_1024(S)[1] / (1 + e) / 1e4:.1f} 万 token/s")


if __name__ == "__main__":
    main()
