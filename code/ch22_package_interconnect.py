"""第 22 章 · 多芯片封装与互联：计算放在哪，数据走哪条路。

讲义：ai-infra/notes/lessons/22-package-interconnect.md
运行：uv run python ch22_package_interconnect.py
B200 与昇腾 910C 的双 die 参数、RTX 4090 的 PCIe 与显存、H100 / A100 NVLink 与 2 μs 固定开销取自 AII 第 4 章 §4.5。
"""


def dual_die_ms(hbm: float, link: float, gib: float = 32) -> tuple[float, float, float]:
    """计算都在 die 0：(本地读 ms, 跨 die 读 ms, 阶段时间 ms)。两路并行，跨 die 受远端 HBM 与链路较慢者限制。"""
    v = gib * 2**30
    local = v / hbm  # die 0 读自己的 32 GiB
    remote = v / min(hbm, link)  # 读 die 1 的：远端 HBM 和链路取较慢者
    return local * 1e3, remote * 1e3, max(local, remote) * 1e3


def h2d_vs_read(nbytes: float, pcie: float, hbm: float) -> tuple[float, float, float]:
    """(上传 ms, 显存读一次 μs, 两者之比)。"""
    up = nbytes / pcie
    rd = nbytes / hbm
    return up * 1e3, rd * 1e6, up / rd


def edge_us(nbytes: float, rate: float, alpha_s: float = 2e-6) -> float:
    """点对点传输时间（μs）= 固定开销 + 数据量 / 带宽。"""
    return (alpha_s + nbytes / rate) * 1e6


def crossover_bytes(alpha_s: float, rate: float) -> float:
    """固定开销与传输时间相等的数据量 V* = α·R。"""
    return alpha_s * rate


def main() -> None:
    print("\n第一步：两颗 die 各 32 GiB 权重，计算都放在 die 0")
    for name, hbm, link in (("B200", 4e12, 5e12), ("910C", 1.6e12, 270e9)):
        lo, re, st = dual_die_ms(hbm, link)
        print(f"  {name}：本地 {lo:.1f} ms，跨 die {re:.1f} ms，阶段 {st:.1f} ms（链路/本地 = {link / hbm:.2f}）")
    print(f"  910C 就地计算：两侧各 {32 * 2**30 / 1.6e12 * 1e3:.1f} ms，链路只传 64 MiB {64 * 2**20 / 270e9 * 1e3:.2f} ms")

    print("\n第二步：RTX 4090 上传 64 MiB")
    up, rd, ratio = h2d_vs_read(64 * 2**20, 32e9, 1008e9)
    print(f"  H2D {up:.1f} ms，显存读 {rd:.0f} μs，相差 {ratio:.1f} 倍；复用 100 次每次分摊 {up * 1e3 / 100:.0f} μs")

    print("\n第三步：卡间传 [M, 4096] BF16 激活")
    for M in (1, 256):
        v = 8192 * M
        print(f"  M = {M:3d}（{v // 1024} KiB）：A100 {edge_us(v, 300e9):.2f} μs，H100 {edge_us(v, 450e9):.2f} μs，H100 α 减半 {edge_us(v, 450e9, 1e-6):.2f} μs")
    vs = crossover_bytes(2e-6, 450e9)
    print(f"  转折点 V* = {vs:,.0f} B ≈ {vs / 1024:.0f} KiB，约 M = {vs / 8192:.1f}")


if __name__ == "__main__":
    main()
