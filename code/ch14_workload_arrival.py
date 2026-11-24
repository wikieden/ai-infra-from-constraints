"""第 14 章 · 从单次请求到持续到达的负载：什么时候来，比来多少更要紧。

讲义：ai-infra/notes/lessons/14-workload-arrival.md
运行：uv run python ch14_workload_arrival.py
例题 3-1（两类请求、RTX PRO 6000 1.792 TB/s、64 条序列、Qwen3-8B）取自 AII 第 3 章 §3.1；t_a = 0、t_1 = 0.8、t_G = 5.8 的请求是编的。
"""

import math

KV = 147_456  # Qwen3-8B 每 token 的 KV（字节）
R_W = 15_136_811_008  # 一步 decode 的共享权重读取（字节）


def request_metrics(t_a: float, t_1: float, t_g: float, G: int) -> tuple[float, float, float]:
    """(TTFT, 完整请求时间, 平均 token 间隔)，单位秒。"""
    return t_1 - t_a, t_g - t_a, (t_g - t_1) / (G - 1)


def p95_rank(n: int) -> int:
    """最近秩法：n 个观测里 p95 是排序后的第几个。"""
    return math.ceil(0.95 * n)


def state_occupancy(lam: float, m0: float, tau: float) -> float:
    """稳定时等待任务的平均状态占用 ≈ λ · M0 · τ。"""
    return lam * m0 * tau


def window_demand(rate_long_in: float, rate_long_out: float) -> tuple[float, float]:
    """(新输入 token/s, 后续 decode 步/s)。长输入类 8192 入 256 出；长输出类 1024 入 2048 出。"""
    return rate_long_in * 8192 + rate_long_out * 1024, rate_long_in * 255 + rate_long_out * 2047


def decode_rate(seqs: int = 64, avg_ctx: int = 2742, beta: float = 1.792e12) -> tuple[float, float]:
    """单卡 decode 上限：(一步毫秒, 每秒步数)。一步读权重 + 每条序列旧 KV，再写一份新 KV。"""
    step_bytes = R_W + seqs * avg_ctx * KV + seqs * KV
    t = step_bytes / beta
    return t * 1e3, seqs / t


def simulate_backlog(arrivals: list[float], mu: float, dt: float = 0.01) -> tuple[float, float | None]:
    """每个元素是一秒内的到达速率；返回 (积压峰值, 排空时刻)。之后补 60 秒不再到达。"""
    q, peak, t, t_empty = 0.0, 0.0, 0.0, None
    for a in arrivals + [0.0] * 60:
        for _ in range(int(round(1 / dt))):
            q = max(0.0, q + (a - mu) * dt)  # 到达 − 处理，不能为负
            t += dt
            peak = max(peak, q)
            if q == 0 and peak > 0 and t_empty is None:
                t_empty = t
    return peak, t_empty


def main() -> None:
    print("\n第二步：一个请求的三个指标（编的时间点）")
    ttft, req, tok = request_metrics(0, 0.8, 5.8, 101)
    print(f"  TTFT {ttft:.1f} s，完整 {req:.1f} s，平均 token 间隔 {tok * 1e3:.0f} ms")
    print(f"  p95：4 条取第 {p95_rank(4)} 个，480 条取第 {p95_rank(480)} 个")
    print(f"  1 GiB 占 10 秒 = {1 * 10} GiB·s")

    print("\n第三步：例题 3-1 的阶段需求（每秒 4 条请求）")
    for name, (a, b) in (("均匀混合", (2, 2)), ("前一分钟 9:1", (3.6, 0.4)), ("后一分钟 1:9", (0.4, 3.6))):
        inp, dec = window_demand(a, b)
        print(f"  {name}：输入 {inp:,.1f} token/s，decode {dec:,.1f} 步/s")
    ms, rate = decode_rate()
    print(f"  单卡一步 {ms:.2f} ms → {rate:,.2f} 步/s；两张 {2 * rate:,.2f} 步/s")
    mu = 2 * rate
    arrivals = [4604.0] * 60 + [1736.8] * 60 + [7471.2] * 60
    peak, t_empty = simulate_backlog(arrivals[60:], mu)
    print(f"  时段变化：积压峰值 {peak:,.0f} 步，到达停止后 {t_empty - 120:.2f} s 排空")


if __name__ == "__main__":
    main()
