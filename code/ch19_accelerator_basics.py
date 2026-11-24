"""第 19 章 · 从矩阵乘法认识加速器：算多少、搬多少、耗多少电。

讲义：ai-infra/notes/lessons/19-accelerator-basics.md
运行：uv run python ch19_accelerator_basics.py
Q 投影形状、60/40 μs 算例、LogicFolding 的电压与面积、能耗表、decode 能量账、HBM 与 TDP 取自 AII 第 4 章 §4.1。
"""

PJ_PER_BYTE = {"KB 级 SRAM": 1.25, "MB 级 SRAM": 12.5, "NVLink-C2C": 10.4, "HBM2": 31.76, "LPDDR": 160.0}
PJ_PER_FLOP = 0.75  # 16 位浮点乘加 1.5 pJ / 2 FLOPs


def proj_cost(M: int, K: int = 4096, N: int = 4096, b: int = 2) -> tuple[int, int, float]:
    """矩阵乘法 [M,K]×[K,N]：(运算 FLOPs, 片外读写字节, 算术强度)。"""
    F = 2 * M * K * N  # 运算
    V = b * (M * K + K * N + M * N)  # 输入 + 权重 + 输出，各读写一次
    return F, V, F / V


def intensity_limit(K: int = 4096, N: int = 4096) -> float:
    """M → ∞ 时算术强度的上限 KN/(K+N)。"""
    return K * N / (K + N)


def amdahl_us(matrix_us: float, other_us: float, s: float) -> float:
    """矩阵部分加速 s 倍后的总时间（μs）。"""
    return matrix_us / s + other_us


def decode_energy(w_gb: float = 15.137, kv_gb: float = 1.208, gflops: float = 19.97) -> tuple[float, float, float]:
    """一步 decode 的 (权重, KV, 计算) 能量（J），HBM2 31.76 pJ/byte，乘加 0.75 pJ/FLOP。"""
    hbm = PJ_PER_BYTE["HBM2"] * 1e-12
    return w_gb * 1e9 * hbm, kv_gb * 1e9 * hbm, gflops * 1e9 * PJ_PER_FLOP * 1e-12


def energy_per_token(B: int) -> float:
    """batch 为 B 时每 token 能量：权重读一次服务 B 个 token。"""
    w, kv, c = decode_energy()
    return w / B + kv + c


def hbm_stack_gbs(pin_gbit: float) -> float:
    """每堆 HBM 带宽（GB/s）= 1024 根引脚 × 速率 / 8。"""
    return 1024 * pin_gbit / 8


def power_budget(tdp_w: float = 700, bw_tbs: float = 3.35, peak_tflops: float = 989.4) -> float:
    """峰值速率下每 FLOP 的能量预算（pJ）=（TDP − HBM 功率）/ 峰值算力。"""
    mem_w = bw_tbs * 1e12 * PJ_PER_BYTE["HBM2"] * 1e-12
    return (tdp_w - mem_w) / (peak_tflops * 1e12) * 1e12


def sustained(e_over_b: float) -> tuple[float, float]:
    """每 FLOP 能耗超出预算时的持续频率比：(电压不变 b/e, 电压同步下降 (b/e)^(1/3))。"""
    r = min(1.0, 1 / e_over_b)
    return r, r ** (1 / 3)


def main() -> None:
    print("\n第一步：Qwen3-8B 的 Q 投影")
    for M in (1, 256):
        F, V, I = proj_cost(M)
        print(f"  M = {M:3d}：运算 {F / 1e9:.4g} GFLOPs，访问 {V / 2**20:.2f} MiB，算术强度 {I:.1f}")
    print(f"  上限 KN/(K+N) = {intensity_limit():.0f}")

    print("\n第三步：60 μs 矩阵 + 40 μs 其余")
    for s in (1, 2, 4, float("inf")):
        t = amdahl_us(60, 40, s)
        print(f"  矩阵加速 {s} 倍：{t:.0f} μs，加速比 {100 / t:.2f}")

    print("\n第四步：功耗与能量")
    print(f"  电压 0.85 → 0.55 V：功耗 ×{(0.55 / 0.85) ** 2:.2f}；功率密度 0.75 / 0.60 = {0.75 / 0.60:.2f}")
    w, kv, c = decode_energy()
    print(f"  decode 一步：权重 {w:.3f} J + KV {kv:.3f} J + 计算 {c:.3f} J = {w + kv + c:.3f} J，权重占 {w / (w + kv + c):.0%}")
    print(f"  batch 32 时每 token {energy_per_token(32):.3f} J")

    print("\n第五步：HBM 与功率封顶")
    per = hbm_stack_gbs(5.24)
    print(f"  H100：每堆 {per:.0f} GB/s，5 堆 {5 * per:,.0f} GB/s")
    print(f"  每 FLOP 能量预算 {power_budget():.2f} pJ；e/b = 1.5 时持续频率 {sustained(1.5)[0]:.2f} / {sustained(1.5)[1]:.2f}")


if __name__ == "__main__":
    main()
