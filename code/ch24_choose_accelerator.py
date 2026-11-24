"""第 24 章 · 根据架构分析性能、选择加速器：先估下界，再找差距，最后算账。

讲义：ai-infra/notes/lessons/24-choose-accelerator.md
运行：uv run python ch24_choose_accelerator.py
资源时间模型、Roofline 转折、Qwen3-8B 设备比较、vLLM 实测、能耗与例 4-3 取自 AII 第 4 章 §4.8。
"""

import math

# 型号: (容量 GB, 带宽 GB/s, BF16/FP32 稠密 TFLOP/s)
DEVICES = {
    "RTX 3090": (24, 936, 71.0),
    "RTX 4090": (24, 1008, 165.2),
    "RTX 5090": (32, 1792, 209.5),
    "RTX PRO 6000": (96, 1792, 503.8),
    "A100": (80, 2039, 312.0),
    "H100": (80, 3350, 989.4),
    "H200": (141, 4800, 989.5),
}
DECODE_BYTES = 16.345e9  # Qwen3-8B BF16 单请求 8K 每步权重 + KV
PREFILL_FLOPS = 61.85e12  # 4K prefill


def staged_time(stages: list[dict[str, float]]) -> tuple[float, dict[str, float]]:
    """阶段内各资源并行（取最大），阶段之间串行。返回 (总时间, 各资源合计)。"""
    total = sum(max(s.values()) for s in stages)
    per_res: dict[str, float] = {}
    for s in stages:
        for r, t in s.items():
            per_res[r] = per_res.get(r, 0) + t
    return total, per_res


def projection_us(m: int, p: float, r: float, d: int = 4096) -> tuple[float, float]:
    """(矩阵时间 μs, 片外读写 μs)：BF16 输入 + 权重读一次、输出写一次。"""
    return 2 * m * d * d / p * 1e6, (2 * m * d + d * d) * 2 / r * 1e6


def turning_rows(p: float, r: float, d: int = 4096) -> float:
    """I(M) = Md/(d+2M) 等于 P/R 时的 M*。"""
    i = p / r
    return i * d / (d - 2 * i)


def decode_flops() -> float:
    w_layer = 2 * 4096**2 + 2 * 4096 * 1024 + 3 * 4096 * 12288
    return 2 * 36 * w_layer + 4 * 36 * 8192 * 4096 + 1.245e9


def device_table() -> dict[str, tuple[float, float]]:
    """每张卡 (decode 读取下界 ms, 4K prefill 矩阵时间 ms)。"""
    return {k: (DECODE_BYTES / (bw * 1e9) * 1e3, PREFILL_FLOPS / (p * 1e12) * 1e3) for k, (_, bw, p) in DEVICES.items()}


def cheaper(c_a: float, t_a: float, c_b: float, t_b: float) -> bool:
    """按运行时间计费，A 每任务更便宜 ⇔ c_A/c_B < t_B/t_A。"""
    return c_a / c_b < t_b / t_a


def breakeven(f0: float, saving_per_m: float, machines: int, tok_s: float, util: float, years: float = 1) -> tuple[float, float, int]:
    """(回本 token 数, 寿命内产出, 寿命内回本所需最少机器数)。"""
    need = f0 / (saving_per_m / 1e6)
    per_machine = tok_s * util * 365 * 86400 * years
    return need, machines * per_machine, math.ceil(need / per_machine)


def main() -> None:
    print("\n第一步：资源相加、阶段取最大")
    total, per = staged_time([{"matrix": 10, "memory": 1}, {"matrix": 1, "memory": 10}])
    print(f"  两阶段 {total:.0f} μs，各资源合计 {per}（只看合计会估成 {max(per.values()):.0f} μs）")
    print(f"  4090 算 8.59 GFLOPs：FP16 累加 {8.59e9 / 330e12 * 1e6:.0f} μs，FP32 累加 {8.59e9 / 165.2e12 * 1e6:.0f} μs")
    for m in (1, 256):
        c, v = projection_us(m, 165.2e12, 1.008e12)
        print(f"  M = {m:3d}：矩阵 {c:.2f} μs，读写 {v:.1f} μs")
    print(f"  I* = {165.2e12 / 1.008e12:.0f} FLOPs/B，M* = {turning_rows(165.2e12, 1.008e12):.1f}")

    print("\n第二步：同一个 Qwen3-8B 比较设备")
    print(f"  decode 一步 {decode_flops() / 1e9:.2f} GFLOPs，读 {DECODE_BYTES / 1e9} GB")
    tab = device_table()
    for k, (dec, pre) in tab.items():
        print(f"  {k:13s} decode {dec:6.2f} ms（{1000 / dec:5.1f} token/s），prefill {pre:6.1f} ms")
    for a, b in (("RTX 3090", "RTX 4090"), ("A100", "H100")):
        print(f"  {a}→{b}：读取 −{(1 - tab[b][0] / tab[a][0]) * 100:.1f}%，矩阵 −{(1 - tab[b][1] / tab[a][1]) * 100:.1f}%")
    print(f"  235B-A22B 每步 13.697 GB：H200 {13.697e9 / 4.8e12 * 1e3:.2f} ms（{4.8e12 / 13.697e9:.1f} token/s）")

    print("\n第三步：预测和实测")
    lb = DECODE_BYTES / 1.792e12 * 1e3
    print(f"  PRO 6000 下界 {lb:.2f} ms，实测 25.83 ms → MBU {lb / 25.83 * 100:.1f}%，多出 {25.83 - lb:.1f} ms")
    print(f"  batch 64、2K：读 34.46 GB 下界 {34.46e9 / 1.792e12 * 1e3:.2f} ms，实测 29.63 ms → {34.46e9 / 29.63e-3 / 1e12:.2f} TB/s")

    print("\n第四步：成本、能耗与回本")
    for t_us in (42.9, 32.6):
        print(f"  600 W × {t_us} μs = {600 * t_us * 1e-3:.1f} mJ")
    print(f"  Mac 持平功率：单行 {600 * 42.9 / 166:.0f} W，256 行 {600 * 32.6 / 1835:.1f} W")
    print(f"  每步能耗：4090 {450 * tab['RTX 4090'][0] / 1e3:.2f} J，5090 {575 * tab['RTX 5090'][0] / 1e3:.2f} J")
    need, out, n = breakeven(20e6, 0.5, 100, 10_000, 0.5)
    print(f"  例 4-3：回本 {need:.1e} token，一年产出 {out:.2e}，至少 {n} 台；半年 {breakeven(20e6, 0.5, 100, 10_000, 0.5, 0.5)[2]} 台")


if __name__ == "__main__":
    main()
