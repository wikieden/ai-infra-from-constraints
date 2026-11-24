"""第 59 章 · 数据输入、checkpoint 与故障恢复。

讲义：ai-infra/notes/lessons/59-data-checkpoint.md
运行：uv run python ch59_data_checkpoint.py
数据准备进程数、checkpoint 内容与重分片、异步保存的可用恢复点、最优保存间隔、掉队者与损失尖峰，取自 AII 第 10 章 §10.4。
"""

import math

N = 8_190_735_360
CKPT = 14 * N                    # 权重 2 + 主权重 4 + 两份矩 8 字节/参数
MTBF_CARD = 1024 * 7.9 * 3600    # 1024 卡 7.9 h 中断一次 → 每卡约 337 天


def workers_needed(tok_s: float, per_worker_seq: float = 2, seq: int = 8192) -> int:
    return math.ceil(tok_s / seq / per_worker_seq)


def commit_times(capture: list[float], size: float, bw: float, copy: float = 0.5) -> list[float]:
    """异步保存：先复制 copy 秒到缓冲，再按 bw 上传；上一份传完才传下一份。"""
    out, free = [], 0.0
    for t in capture:
        start = max(t + copy, free)
        free = start + size / bw
        out.append(free)
    return out


def overhead(tau: float, c: float, lam: float, r: float = 120) -> tuple[float, float, float]:
    """L(τ) = c/τ + λτ/2 + λr 的三项。"""
    return c / tau, lam * tau / 2, lam * r


def tau_opt(c: float, lam: float) -> float:
    return math.sqrt(2 * c / lam)


def job_rate(cards: int) -> float:
    return cards / MTBF_CARD


def expected_max_std(n: int) -> float:
    """n 个独立标准正态的最大值的期望（数值积分）。"""
    if n == 1:
        return 0.0
    phi = lambda x: math.exp(-x * x / 2) / math.sqrt(2 * math.pi)
    Phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    h, s = 0.001, 0.0
    x = -8.0
    while x < 8:
        s += x * n * phi(x) * Phi(x) ** (n - 1) * h
        x += h
    return s


def main() -> None:
    print("\n第一步：数据输入")
    need = 1e11 / (25 * 86400)
    print(f"  {need:,.0f} token/s = {need / 8192:.1f} 条/s，int64 只有 {need * 8 / 1e6:.2f} MB/s；每进程 2 条/s → 至少 {workers_needed(need)} 个")
    print(f"  48 卡每步 52.3 s 处理 384 条 → {384 / 52.3:.1f} 条/s；四个进程 8 条/s 够用")
    print("  训练完成 batch 100、预取到 108：恢复从 101 开始，不是 109")

    print("\n第二步：checkpoint 内容")
    print(f"  每参数 14 B → {CKPT / 1e9:.1f} GB；门控投影 96 MiB × (1 + 2 + 2 + 2) = {96 * 7} MiB；TP4 → TP8 每片 24 → 12 MiB")

    print("\n第三步：异步保存与可用恢复点（112 GB）")
    for bw in (7e9, 20e9):
        ts = commit_times([20, 40], 112e9, bw)
        ok = [t for t, cm in zip((20, 40), ts) if cm <= 50]
        print(f"  {bw / 1e9:.0f} GB/s：提交于 {ts[0]:.1f} / {ts[1]:.1f} s；50 s 故障恢复到 {max(ok)} s，重做 {50 - max(ok)} s")
    print(f"  每 10 s 一份要 {112 / 10:.1f} GB/s > 7 → 队列增长；每 20 s 一份 {112 / 20:.1f} GB/s")

    print("\n第四步：保存间隔")
    c = CKPT / 7e9
    lam = job_rate(1024)
    print(f"  1024 卡 λ = 1/{1 / lam / 3600:.1f} h，c = {c:.1f} s，τ* = {tau_opt(c, lam):.0f} s")
    for tau in (300, 900, 1800):
        a, b, r = overhead(tau, c, lam)
        print(f"  τ = {tau:>4} s：保存 {a * 100:.1f}% + 重做 {b * 100:.1f}% + 恢复 {r * 100:.1f}% = {(a + b + r) * 100:.1f}%")
    for cards in (32, 48):
        print(f"  {cards} 卡 τ = 1800 s：{sum(overhead(1800, c, job_rate(cards))) * 100:.2f}%")

    print("\n第五步：掉队者（μ = 52.2 s，另加 0.58 s）")
    for n in (8, 48, 1024):
        k = expected_max_std(n)
        print(f"  {n:>4} 卡：E[max] − μ = {k:.2f}σ；σ = 2% → {52.2 + k * 1.044 + 0.58:.1f} s，5% → {52.2 + k * 2.61 + 0.58:.1f} s")
    for n in (8, 48):
        wait = max(0, 3 - expected_max_std(n - 1))
        re = 52.2 * n / (n - 1) + expected_max_std(n - 1) * 1.044 + 0.58
        print(f"  {n} 卡一张慢到 μ + 3σ：等待多 {wait:.2f}σ；重分配每步 {re:.1f} s")
    print(f"  驱逐：τ/2 + r = {600 / 2 + 120} s ≈ {420 / 52.8:.0f} 步")
    l48, ls = job_rate(48), 1 / (7 * 86400)
    print(f"  损失尖峰每 7 天一次：τ* {tau_opt(c, l48):.0f} → {tau_opt(c, l48 + ls):.0f} s；τ = 600 s {sum(overhead(600, c, l48)) * 100:.2f}% → {(c / 600 + (l48 + ls) * 420) * 100:.2f}%；1800 s {sum(overhead(1800, c, l48)) * 100:.2f}% → {(c / 1800 + (l48 + ls) * 1020) * 100:.2f}%")


if __name__ == "__main__":
    main()
