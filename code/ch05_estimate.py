"""第 5 章 · 用几个数字估算一次模型执行：算和读各要多久，实测为什么达不到。

讲义：ai-infra/notes/lessons/05-estimate.md
运行：uv run python ch05_estimate.py
数字取自 AII 第 1 章 §1.3：例题 1-1（70B、每参数 1 字节、H100 SXM 989.4 TFLOP/s 与 3.35 TB/s）、
Qwen3-8B 在 RTX PRO 6000（1.792 TB/s）上的读取下界与实测输出间隔。
"""

N = 70e9  # 参数数（手算近似）
PI = 989.4e12  # H100 SXM BF16 稠密矩阵峰值，FLOP/s
BETA = 3.35e12  # H100 SXM HBM 带宽，bytes/s


def step_bounds(B: int, n: float = N, bw: float = 1, pi: float = PI, beta: float = BETA) -> tuple[float, float, float]:
    """一步生成的下界（ms）：(读权重, 矩阵计算, 两者取大)。"""
    read_ms = bw * n / beta * 1e3  # 读一遍权重，和 B 无关
    compute_ms = 2 * B * n / pi * 1e3  # 每个请求 2N 次运算
    return read_ms, compute_ms, max(read_ms, compute_ms)  # 充分重叠：取较大者


def batch_view(B: int) -> tuple[float, float, float]:
    """batch 为 B 时：(一步下界 ms, 每个输出分摊 ms, 吞吐上界 token/s)。"""
    lower = step_bounds(B)[2]
    return lower, lower / B, B / lower * 1e3


def b_star(bw: float = 1, pi: float = PI, beta: float = BETA) -> float:
    """计算追上读取的 batch：B* = b_W × Π / (2β)。"""
    return bw * pi / (2 * beta)


def qwen_lower_ms(weights_gb: float = 15.14, kv_gb: float = 0.302, beta_tb: float = 1.792) -> float:
    """Qwen3-8B 一步 decode 读主要权重 + 旧 KV 的下界（ms）。"""
    return (weights_gb + kv_gb) / (beta_tb * 1e3) * 1e3


def mbu(measured_ms: float, lower_ms: float | None = None) -> float:
    """带宽利用率 = 读取下界 ÷ 实测时间。"""
    return (lower_ms if lower_ms is not None else qwen_lower_ms()) / measured_ms


def main() -> None:
    print("\n第一步：70B、每参数 1 字节，batch = 1")
    print(f"  存 = 读 = {N / 1e9:.0f} GB；算 = 2N = {2 * N / 1e9:.0f} GFLOPs")

    print("\n第二步：两个下界（H100 SXM）")
    r, c, low = step_bounds(1)
    print(f"  读权重 {r:.2f} ms，矩阵计算 {c:.4f} ms，读是算的 {r / c:.0f} 倍 → 下界 {low:.2f} ms > 10 ms")
    print(f"  算力翻倍：下界 {step_bounds(1, pi=2 * PI)[2]:.2f} ms（计算 {step_bounds(1, pi=2 * PI)[1]:.4f} ms）")
    print(f"  带宽翻倍：下界 {step_bounds(1, beta=2 * BETA)[2]:.2f} ms")
    print(f"  权重减半：下界 {step_bounds(1, bw=0.5)[2]:.2f} ms")

    print("\n第三步：批内复用")
    for B in (1, 8):
        lower, per, tput = batch_view(B)
        print(f"  B = {B}：计算 {step_bounds(B)[1]:.2f} ms，一步 {lower:.2f} ms，每输出分摊 {per:.2f} ms，吞吐 {tput:.1f} token/s")
    print(f"  转折点 B* = b_W × Π / (2β) = {b_star():.1f}；Π/β = {PI / BETA:.0f} FLOP/字节")

    print("\n第五步：Qwen3-8B 在 RTX PRO 6000 上")
    low = qwen_lower_ms()
    print(f"  下界 (15.14 + 0.302) GB ÷ 1.792 TB/s = {low:.2f} ms")
    for t in (26.5, 12.1, 10.7):
        print(f"  实测 {t} ms → MBU {mbu(t):.0%}")


if __name__ == "__main__":
    main()
