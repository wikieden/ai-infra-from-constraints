"""第 52 章 · Attention–FFN 分离与异构执行。

讲义：ai-infra/notes/lessons/52-af-split.md
运行：uv run python ch52_af_split.py
KTransformers 实验机（Xeon 8452Y + A100 40GB PCIe）、Qwen3-235B-A22B 专家、micro-batch 流水与例 9.4，取自 AII 第 9 章 §9.3。
"""

KiB, MiB, GiB = 2**10, 2**20, 2**30
P_E = 3 * 4096 * 1536          # 一个专家的参数数 18,874,368
W_E = P_E * 2                  # BF16 36 MiB
H, B = 4096, 2                 # 激活宽度、每元素字节
CPU_DRAM = 220e9
CPU_FLOPS = {"AVX-512": 1.8e12, "AMX": 21.3e12}
GPU_FLOPS, GPU_HBM = 156e12, 778e9
LINK, ALPHA = 25e9, 5e-6


def t_cpu(m: int, kernel: str = "AVX-512", dram: float = CPU_DRAM) -> float:
    """T_C(m) = 2α + 2mhb/B_L + max(2mP_e/C_C, W_e/B_D)，秒。"""
    return 2 * ALPHA + 2 * m * H * B / LINK + max(2 * m * P_E / CPU_FLOPS[kernel], W_E / dram)


def t_gpu(m: int) -> float:
    """T_G(m) = α + W_e/B_L + max(2mP_e/C_G, W_e/B_H)，秒。"""
    return ALPHA + W_E / LINK + max(2 * m * P_E / GPU_FLOPS, W_E / GPU_HBM)


def crossover(kernel: str) -> int:
    """GPU 开始比 CPU 快的最小每专家 token 数。"""
    m = 1
    while t_gpu(m) >= t_cpu(m, kernel):
        m += 1
    return m


def active_experts(E: int = 128, K: int = 8, M: int = 64) -> float:
    return E * (1 - (1 - K / E) ** M)


def pipeline_ms(tA: float, tF: float, q: int) -> float:
    """两阶段交错流水：t_A + t_F + (q − 1) max(t_A, t_F)。"""
    return tA + tF + (q - 1) * max(tA, tF)


def af_step_ms(layers: int = 36, bw: float = 25e9, alpha: float = ALPHA) -> float:
    """一步 decode 的逐层 AF 往返：2 × layers 次、每次 h·b 字节。"""
    n = 2 * layers
    return (n * H * B / bw + n * alpha) * 1e3


def alpha_star_us(v_kv: float, bw: float = 25e9, layers: int = 36) -> float:
    """α* = (V_KV − V_AF) / ((2·layers − 1) B)。"""
    v_af = 2 * layers * H * B
    return (v_kv - v_af) / ((2 * layers - 1) * bw) * 1e6


def main() -> None:
    print(f"\n一个专家 {P_E:,} 参数，BF16 {W_E / MiB:.0f} MiB")
    print("\n第一步：八个专家的两条路径（ms）")
    for m in (1, 128):
        print(f"  每专家 {m:>3} 个 token：CPU AVX-512 {8 * t_cpu(m) * 1e3:.2f}、AMX {8 * t_cpu(m, 'AMX') * 1e3:.2f}、搬到 GPU {8 * t_gpu(m) * 1e3:.2f}")
    print(f"  交点：AVX-512 {crossover('AVX-512')} 个、AMX {crossover('AMX')} 个 token/专家")
    for k in CPU_FLOPS:
        comp = 8 * 2 * 128 * P_E / CPU_FLOPS[k] * 1e3
        print(f"  {k} 128 token：计算 {comp:.2f} ms vs 读取 同插槽 {8 * W_E / 220e9 * 1e3:.2f} / 跨插槽 {8 * W_E / 125e9 * 1e3:.2f} ms")

    print("\n第二步：容量与复用")
    print(f"  每层 128 专家 {128 * W_E / GiB:.1f} GiB；每层驻留 32 个 × 94 层 = {32 * 94 * W_E / GiB:.0f} GiB，16 个 = {16 * 94 * W_E / GiB:.0f} GiB")
    u = active_experts()
    print(f"  64 token × 8：期望活跃 {u:.1f} 个专家，读 {u * W_E / GiB:.2f} GiB；均匀 4.5 GiB {4.5 * GiB / 220e9 * 1e3:.1f} ms vs 集中 288 MiB {288 * MiB / 220e9 * 1e3:.2f} ms")

    print("\n第三步：micro-batch 交错流水")
    print(f"  注意力 2 ms、专家 3 ms、4 个 micro-batch：串行 {4 * 5} ms，流水 {pipeline_ms(2, 3, 4):.0f} ms")

    print("\n第四步：例 9.4 逐层交接")
    print(f"  一步 decode 72 次 × 8 KiB = {72 * 8} KiB：{af_step_ms():.3f} ms；1024 步 {1024 * af_step_ms():.0f} ms")
    for name, v in (("GQA 1.125 GiB", 1.125 * GiB), ("MLA 549 MiB", 8192 * 70272)):
        print(f"  {name}：α* = {alpha_star_us(v):.0f} μs（25 GB/s）、{alpha_star_us(v, 50e9):.0f} μs（50 GB/s）")


if __name__ == "__main__":
    main()
