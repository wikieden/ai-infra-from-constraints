"""第 49 章 · 请求负载下的性能与配置选择。

讲义：ai-infra/notes/lessons/49-serving-choice.md
运行：uv run python ch49_serving_choice.py
Little 定律、有效吞吐、实验 8-1 的 MBU / MFU、16 条长请求四种方案、Amdahl 取自 AII 第 8 章 §8.6。
"""

MiB = 2**20

READS = {1: 15.44, 4: 16.34, 16: 19.97, 64: 34.46}
STEPS = {1: 26.26, 4: 26.62, 16: 27.44, 64: 29.63}


def mbu(b: int, bw: float = 1792) -> float:
    return READS[b] / bw * 1e3 / STEPS[b]


def gpu_seconds(L: int, G: int, rP: float, rD: float) -> float:
    return L / rP + (G - 1) / rD


def interp_step(read_gb: float) -> float:
    return STEPS[16] + (read_gb - READS[16]) / (READS[64] - READS[16]) * (STEPS[64] - STEPS[16])


PLANS = {
    "A": dict(mem=16 * 1188, prefill=7.34, rounds=255, step=29.63),
    "B": dict(mem=864 + 16 * 324, prefill=2.04, rounds=255, step=27.35),
    "C": dict(mem=16 * 1188 * 34 / 64, prefill=7.34, rounds=255, step=None),
    "D": dict(mem=864 + 16 * 324 + 3188, prefill=2.23, rounds=85, step=27.35),
}


def plan_time(name: str) -> float:
    """整批耗时（s）= prefill + 轮数 × 每轮；C 每轮读 25.40 GB，按表插值。"""
    p = PLANS[name]
    step = p["step"] if p["step"] else interp_step(25.40)
    return p["prefill"] + p["rounds"] * step / 1e3


def amdahl(f: float, s: float) -> float:
    return 1 / ((1 - f) + f / s)


def main() -> None:
    print("\n第一步：Little 定律")
    print(f"  4 req/s × 0.5 s = {4 * 0.5:.0f}；× 2 s = {4 * 2:.0f} 条在系统内")
    print(f"  336 次回答：294 正确，155 正确且按时")

    print("\n第二步：实验 8-1 的效率")
    for b in READS:
        print(f"  batch {b:2d}：下界 {READS[b] / 1.792:.2f} ms，实测 {STEPS[b]} ms，MBU {mbu(b):.0%}")

    print("\n第三步：16 条 8K 长请求（前 6K 共享），12 GiB、7 s")
    rP, rD = 32768 / 2.04, 16 / 0.02735
    print(f"  B：r_P = {rP:,.0f} token/s，r_D = {rD:.0f} 调用/s，每条 {gpu_seconds(2048, 256, rP, rD):.3f} GPU s")
    print(f"  C 的一轮（读 25.40 GB）插值 {interp_step(25.40):.2f} ms")
    for k, p in PLANS.items():
        t = plan_time(k)
        ok = p["mem"] <= 12 * 1024 and t <= 7
        print(f"  {k}：内存 {p['mem'] / 1024:.2f} GiB，总耗时 {t:.2f} s {'✓' if ok else '✗'}")
    print(f"  D 每个合格结果 {plan_time('D') / 16:.3f} GPU s，≤ {600 * plan_time('D') / 16:.0f} J")
    print(f"  输出只要 16 个：B {2.04 + 15 * 0.02735:.2f} s，D {2.23 + 5 * 0.02735:.2f} s")

    print("\n第四步：完整任务")
    for f in (0.5, 0.2):
        print(f"  decode 占 {f:.0%}、加速 4 倍：整体 {amdahl(f, 4):.2f} 倍")


if __name__ == "__main__":
    main()
