"""第 40 章 · 远程访问的路径与并发。

讲义：ai-infra/notes/lessons/40-remote-access.md
运行：uv run python ch40_remote_access.py
一次 64 B 远程读取的阶段分解、快照的复用交点、并发与吞吐模型、KV-Direct 的 PCIe 上限取自 AII 第 7 章 §7.3。
"""

import math

# 阶段（ns）：PCIe 外设网卡、UB 异步读、UB Load；网卡流水线按 322 MHz 周期数未取整
CYC = 1e3 / 322
STAGES = [
    ("接口库提交", 50, 50, 0), ("构造描述符", 30, 30, 0), ("门铃写", 150, 0, 0), ("DMA 读描述符", 500, 0, 0),
    ("片上总线提交", 0, 30, 30), ("发送流水线", 9 * CYC, 25 * CYC, 8 * CYC), ("线路去程", 100, 100, 100),
    ("接收流水线", 9 * CYC, 25 * CYC, 8 * CYC), ("目标 DMA 读", 500, 0, 0), ("目标片上总线", 0, 30, 30),
    ("内存行命中", 30, 30, 30), ("目标发送响应", 9 * CYC, 25 * CYC, 8 * CYC), ("线路回程", 100, 100, 100),
    ("接收响应", 9 * CYC, 25 * CYC, 8 * CYC), ("载荷 DMA 写", 250, 0, 0), ("完成记录 DMA 写", 250, 0, 0),
    ("片上总线完成", 0, 30, 30), ("完成队列轮询", 70, 5, 0), ("接口库轮询", 30, 30, 0), ("序号分配", 50, 0, 0),
]


def read_latency_ns(wire_ns: float = 100) -> tuple[float, float, float]:
    tot = [0.0, 0.0, 0.0]
    for name, *v in STAGES:
        for i in range(3):
            tot[i] += v[i] if "线路" not in name else wire_ns
    return tuple(tot)


def snapshot_crossover(remote: float, stage: float, local: float) -> float:
    return stage / (remote - local)


def eff_bw(N: int, m: float, T: float, delta: float | None = None, path: float = 50e9) -> float:
    """min(路径, Nm/T, m/δ)。"""
    vals = [path, N * m / T]
    if delta:
        vals.append(m / delta)
    return min(vals)


def need_inflight(B: float, T: float, m: float) -> int:
    return math.ceil(B * T / m)


def main() -> None:
    print("\n第一步：64 B 远程读取（线路单程 100 ns）")
    p, u, l = read_latency_ns()
    pcie = sum(v[0] for n, *v in STAGES if n in ("门铃写", "DMA 读描述符", "目标 DMA 读", "载荷 DMA 写", "完成记录 DMA 写"))
    print(f"  PCIe 外设 {p:.0f} ns（PCIe 穿越 {pcie:.0f}），UB 异步 {u:.0f} ns，UB Load {l:.0f} ns")
    for L in (100, 500):
        a, _, c = read_latency_ns(L)
        print(f"  线路 {L} ns：外设 / Load = {a / c:.1f} 倍")
    print(f"  网卡处理器发起：{p - 650:.0f} ns；8 个独立读串行 {8 * p / 1e3:.1f} μs，并发 {p / 1e3:.1f} μs")

    print("\n第二步：144 MiB 快照，远端 3.02 ms / 次，搬回 3.07 ms + 本地 0.046 ms / 次")
    r = snapshot_crossover(3.02, 3.07, 0.046)
    print(f"  交点 r = {r:.2f}；读 4 次：远端 {4 * 3.02:.1f} ms，搬回 {3.07 + 4 * 0.046:.1f} ms")
    print(f"  每次只读 10%：交点 r = {snapshot_crossover(0.302, 3.07, 0.0046):.1f}")
    print(f"  base64：1 MiB → {4 / 3:.2f} MiB；启动 5 → 2 μs，8 KiB {5 + 8192 / 50e9 * 1e6:.2f} → {2 + 8192 / 50e9 * 1e6:.2f} μs")

    print("\n第三步：并发，256 B、槽位 2 μs、50 GB/s")
    print(f"  需要 {need_inflight(50e9, 2e-6, 256)} 个在途；128 个 {eff_bw(128, 256, 2e-6) / 1e9:.1f} GB/s；1 个 {eff_bw(1, 256, 2e-6) / 1e9:.3f} GB/s")
    print(f"  再加 δ = 18.6 ns：{eff_bw(391, 256, 2e-6, 18.6e-9) / 1e9:.1f} GB/s；用满要 δ ≤ {256 / 50e9 * 1e9:.2f} ns")
    print(f"  4 KiB：δ ≤ {4096 / 50e9 * 1e9:.0f} ns，槽位 {need_inflight(50e9, 2e-6, 4096)} 个")
    for name, d in (("UB", 2 / 322e6), ("RoCE RC", 6 / 322e6)):
        print(f"  {name}：δ = {d * 1e9:.1f} ns，{1 / d / 1e6:.0f} M 项/s，64 B {64 / d / 1e9:.1f} GB/s，4 KiB {4096 / d / 1e9:.0f} GB/s")

    print("\n第四步：KV-Direct 的 PCIe Gen3 x8")
    ops = 7.87e9 / (64 + 26)
    print(f"  报文开销上限 {ops / 1e6:.0f} M 次/s；用满要 {ops * 1050e-9:.0f} 个在途；64 个标签 {64 / 1050e-9 / 1e6:.0f} M 次/s（实测 60 M）")


if __name__ == "__main__":
    main()
