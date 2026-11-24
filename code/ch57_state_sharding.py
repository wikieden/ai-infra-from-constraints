"""第 57 章 · 训练状态的分片、重计算与卸载。

讲义：ai-infra/notes/lessons/57-state-sharding.md
运行：uv run python ch57_state_sharding.py
ZeRO 各阶段每卡容量、门控乘积重算、CPU / GPU 转换梯度的带宽分界与 RTX 4090 的容量方案，取自 AII 第 10 章 §10.2。
"""

MiB, GiB = 2**20, 2**30
N = 8_190_735_360


def zero_bytes(stage: int, d: int, n: int = N) -> float:
    """每卡常驻状态：M0 = 16N，M1 = 4N + 12N/d，M2 = 2N + 14N/d，M3 = 16N/d。"""
    return {0: 16 * n, 1: 4 * n + 12 * n / d, 2: 2 * n + 14 * n / d, 3: 16 * n / d}[stage]


def recompute_saving_mib(tokens: int = 128, layers: int = 9, mbs: int = 1) -> float:
    """每层省下 h（[t, 12288] FP32）和两个 [t, 4096] 归一化输出。"""
    per_layer = tokens * (12288 + 2 * 4096) * 4 / MiB
    return per_layer * layers * mbs


def wait_s(V: float, B: float, W: float) -> float:
    """T_wait = max(0, V/B − W)。"""
    return max(0.0, V / B - W)


def cast_paths_ms(G: float = 96 * MiB, B: float = 32e9, Cc: float = 307.2e9, Cg: float = 1008e9):
    """(CPU 转换, GPU 转换) 的串行时间：T_c = G/B + 3G/C_c，T_g = 3G/C_g + 2G/B。"""
    return (G / B + 3 * G / Cc) * 1e3, (3 * G / Cg + 2 * G / B) * 1e3


def cast_breakeven(Cc: float = 307.2e9, Cg: float = 1008e9) -> float:
    """B* = 1 / (3 (1/C_c − 1/C_g))。"""
    return 1 / (3 * (1 / Cc - 1 / Cg))


def main() -> None:
    print("\n第一步：ZeRO 各阶段（d = 8）")
    for st, name in enumerate(("普通 DP", "ZeRO-1", "ZeRO-2", "ZeRO-3")):
        print(f"  {name}：{zero_bytes(st, 8) / GiB:.1f} GiB/卡")
    print("  FSDP2 小实验：常驻 18 → 9 MiB，峰值 39 → 30 MiB（另有约 21 MiB 执行期张量）")

    print("\n第二步：选择性重计算")
    print(f"  h = [128, 12288] FP32 = {128 * 12288 * 4 / MiB:.0f} MiB；每层省 {recompute_saving_mib(layers=1):.0f} MiB，九层 {recompute_saving_mib():.0f} MiB")
    print(f"  九层 GEMM 输入 135 → {135 - recompute_saving_mib():.0f} MiB；四个 micro-batch 等反向省 {recompute_saving_mib(mbs=4):.0f} MiB")
    print(f"  重建 h 要 {128 * 12288 / 1e6:.2f} M 次乘法")

    print("\n第三步：卸载与转换位置（96 MiB BF16 梯度）")
    for name, B in (("PCIe 4.0 x16 32 GB/s", 32e9), ("NVLink-C2C 450 GB/s", 450e9)):
        c, g = cast_paths_ms(B=B)
        print(f"  {name}：CPU 转换 {c:.2f} ms，GPU 转换 {g:.2f} ms")
    print(f"  分界带宽 B* = {cast_breakeven() / 1e9:.0f} GB/s；GPU 转换峰值 {96 + 192} MiB")
    print(f"  例：1 GiB 状态、32 GB/s、提前 20 ms 开始 → 等 {wait_s(GiB, 32e9, 0.02) * 1e3:.1f} ms")

    print("\n第四步：RTX 4090 每卡 22 GiB")
    for d in (8, 16, 32, 48):
        s = zero_bytes(3, d) / GiB
        print(f"  ZeRO-3 {d:>2} 卡：常驻 {s:.1f} GiB + 10 GiB = {s + 10:.1f} GiB，余量 {22 - s - 10:+.1f}")


if __name__ == "__main__":
    main()
