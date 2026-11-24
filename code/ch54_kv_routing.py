"""第 54 章 · KV 的分布、共享与请求路由。

讲义：ai-infra/notes/lessons/54-kv-routing.md
运行：uv run python ch54_kv_routing.py
DGX A100 的四级存储、C ≥ rT、保存与重算的不等式、逐层预加载、SSD 写入寿命与例 9.7 的路由，取自 AII 第 9 章 §9.5。
"""

import math

MiB, GiB = 2**20, 2**30
V = 1.125 * GiB                 # Qwen3-8B 8K 前缀
T_PREFILL = 0.8564              # A100 重算 8K（s）
R = V / T_PREFILL               # KV 产生速率 ≈ 1.41 GB/s


def fetch_ms(bw: float, v: float = V) -> float:
    return v / bw * 1e3


def retention_s(capacity: float, r: float = R) -> float:
    """容量 C 能保存最近 C / r 秒产生的 KV。"""
    return capacity / r


def worth_saving(bw_write: float, bw_read: float, k: int, v: float = V, t_re: float = T_PREFILL) -> bool:
    """T_write + k T_read < k T_recompute。"""
    return v / bw_write + k * v / bw_read < k * t_re


def layerwise_ms(read_ms: float, compute_ms: float, layers: int = 36, preload: int = 0) -> float:
    """逐层预加载：第 i 层计算要等它读完；preload 层已在 HBM。"""
    r, c = read_ms / layers, compute_ms / layers
    t_read = t = 0.0
    for i in range(layers):
        if i >= preload:
            t_read += r
            t = max(t, t_read)
        t += c
    return t


def min_preload(read_ms: float, compute_ms: float, layers: int = 36) -> int:
    """最少预读几层，读取就完全藏在计算后面。"""
    return next(n for n in range(layers + 1) if layerwise_ms(read_ms, compute_ms, layers, n) <= compute_ms + 1e-9)


def buffer_bytes(bw: float, t_read: float, t_new: float) -> float:
    """S_buf = B (T_read − T_new)。"""
    return max(0.0, bw * (t_read - t_new))


def ttft(Q: float, R_: float, C: float) -> float:
    """T_first = max(Q, R) + C。"""
    return max(Q, R_) + C


def expected_a(p: float, hit: float = 110.9, miss: float = 967.2) -> float:
    return hit * p + miss * (1 - p)


def main() -> None:
    print(f"\nr = {R / 1e9:.2f} GB/s（A100 每 {T_PREFILL:.3f} s 产生一份 {V / GiB} GiB）")
    print("\n第一步：四级存储取回 8K 前缀")
    for name, ms in (("主机内存 PCIe 25 GB/s", fetch_ms(25e9)), ("本地 SSD 7.1 GB/s", fetch_ms(7.1e9)),
                     ("远端（网卡 + PCIe）", 2 * fetch_ms(25e9)), ("在 A100 上重算", T_PREFILL * 1e3)):
        print(f"  {name}：{ms:.1f} ms")

    print("\n第二步：C ≥ rT")
    for name, c in (("HBM 63.6 GB", 63.6e9), ("主机内存 256 GiB", 256 * GiB), ("SSD 3.84 TB", 3.84e12)):
        print(f"  {name}：保存最近 {retention_s(c):.0f} s")
    print(f"  复用间隔 10 s 要 {R * 10 / 1e9:.1f} GB，10 分钟要 {R * 600 / 1e9:.0f} GB")
    print(f"  Mooncake 50,000 块 × 512 token × 144 KiB = {50000 * 512 * 147456 / 1e12:.2f} TB")

    print("\n第三步：每步远程读 vs 取回一次 vs 重算")
    print(f"  100 次 decode 每次都远程读：一秒内要 {100 * V / 1e9:.0f} GB/s；MLA {100 * 8192 * 70272 / 1e9:.1f} GB/s")
    for bw in (25e9, 2.5e9, 1.25e9):
        ks = [k for k in (1, 2, 3, 10) if worth_saving(bw, bw, k)]
        print(f"  {bw / 1e9:g} GB/s：值得保存的复用次数 {ks if ks else '无（取回比重算还慢）'}")
    print(f"  临界带宽：复用一次 {2 * R / 1e9:.2f} GB/s，复用无穷次 {R / 1e9:.2f} GB/s；SSD 写 + 读 {(V / 4.2e9 + V / 7.1e9) * 1e3:.0f} ms")

    print("\n第四步：读取与计算重叠（新 256 token 计算 30.9 ms）")
    rd = fetch_ms(25e9)
    print(f"  先读后算 {rd + 30.9:.1f} ms；逐层 {layerwise_ms(rd, 30.9):.1f} ms")
    buf = buffer_bytes(25e9, rd / 1e3, 30.9e-3)
    n = min_preload(rd, 30.9)
    print(f"  缓冲 {buf / 1e6:.0f} MB ≈ {buf / (32 * MiB):.1f} 层；逐层模拟要预读 {n} 层（{n * 32} MiB）才完全藏住：{layerwise_ms(rd, 30.9, preload=n):.1f} ms")
    print(f"  SSD 一层 {32 * MiB / 7.1e9 * 1e3:.2f} ms；主机内存可放 {int(256 * GiB // V)} 份前缀")
    quota = 3.84e12 / 86400
    print(f"  SSD 1 DWPD = {quota / 1e6:.1f} MB/s，全写是额度的 {R / quota:.1f} 倍，只能写 {quota / R * 100:.1f}%")
    print(f"  重启：读 64 页 {64 * 2.25:.0f} MiB，连续可用 63 页 {63 * 2.25:.2f} MiB")

    print("\n第五步：例 9.7 路由（ms）")
    print(f"  A 本地命中 {ttft(250, 0, 30.9):.1f}；B 重算 {20 + 887.2:.1f}")
    for name, bw in (("50 GbE", 6.25e9), ("200 GbE", 25e9)):
        print(f"  B 经 {name} 取回 {ttft(20, 10 + fetch_ms(bw) + rd, 30.9):.1f}")
    print(f"  B 与 A 打平带宽 {V / ((250 - 10 - rd) / 1e3) / 1e9:.2f} GB/s；与重算打平 {V / ((907.2 - 10 - rd - 30.9) / 1e3) / 1e9:.2f} GB/s")
    p = (967.2 - 907.2) / 856.3
    print(f"  缓存只以概率 p 可用：E[T_A] = 967.2 − 856.3p，p > {p * 100:.1f}% 均值才赢；p = 0.9 → {expected_a(0.9):.1f} ms，但 p99 仍 967 ms")


if __name__ == "__main__":
    main()
