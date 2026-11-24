"""第 63 章 · 共享资源池与 RL 资源配比。

讲义：ai-infra/notes/lessons/63-resource-pool.md
运行：uv run python ch63_resource_pool.py
资源碎片与迁移、突发排队、RL 迭代各段、权重分发的共享出口、临时实例窗口与成本、验证批次的长尾与最迟开始，取自 AII 第 11 章 §11.3。
"""


def place_options(wait: float = 12, move: float = 4, run_local: float = 20, ready_spread: float = 1, run_spread: float = 32) -> dict[str, float]:
    return {"等待": wait + run_local, "迁移": move + run_local, "分散": ready_spread + run_spread}


def burst_waits(n: int, cores: int, service: float = 1.0) -> list[float]:
    """n 个同时到达、每个 service 秒的工作，cores 个执行者按 FIFO。"""
    return [(i // cores) * service for i in range(n)]


def iteration(gen: float = 40, verify: float = 10, update: float = 30, publish: float = 5, gen_speed: float = 1) -> float:
    return gen / gen_speed + verify + update + publish


def broadcast_s(bytes_each: float, n: int, recv_bw: float, send_bw: float) -> float:
    """T_all ≥ max(单份 / 接收带宽, n 份 / 发送出口)。"""
    return max(bytes_each / recv_bw, n * bytes_each / send_bw)


def window(L: float, T0: float) -> float:
    return max(L - T0, 0.0)


def cost_ratio(base: float, extra: float, speedup: float) -> float:
    """单位工作成本相对原来的倍数。"""
    return (base + extra) / base / speedup


def residual(samples: list[float], e: float) -> float:
    """E[S − e | S > e]。"""
    alive = [s for s in samples if s > e]
    return sum(s - e for s in alive) / len(alive)


def batch_done(arrivals: list[float], work: float, eager: bool = True) -> float:
    """单个验证进程：逐条提交 vs 等全部就绪。"""
    t = 0.0 if eager else max(arrivals)
    for a in arrivals:
        t = max(t, a) + work
    return t


def main() -> None:
    print("\n第一步：碎片与迁移")
    print(f"  {place_options()}；迁移要 12 s 时与等待打平")
    print("  ASI 加可抢占低优先级作业：GPU 分配 68% → 93%")

    print("\n第二步：突发与优先级")
    w = burst_waits(10, 1)
    print(f"  十个同时到达、一个执行者：平均等 {sum(w) / 10:.1f} s")
    w = burst_waits(80, 48)
    print(f"  80 个工具调用、48 核：{sum(1 for x in w if x > 0)} 个等 1 s，平均 {sum(w) / 80:.1f} s；96 核全在 1 s 内；80 个环境 {80 * 2} GiB")
    print("  在线最多等 2 s、离线保存 30 s → 必须另留能 2 s 内交出的资源；SCHED_IDLE 下 LS 慢 45%，加 core scheduling 17%")

    print("\n第三步：RL 迭代")
    print(f"  40 + 10 + 30 + 5 = {iteration():.0f} s；生成快一倍 {iteration(gen_speed=2):.0f} s（{iteration() / iteration(gen_speed=2):.2f} 倍）；生成为零上限 {iteration() / iteration(gen_speed=1e9):.2f} 倍")
    W = 16_381_470_720
    print(f"  Qwen3-8B 权重 {W / 1e9:.2f} GB 发 6 份：单份 {W / 6.25e9:.2f} s，全部 ≥ {broadcast_s(W, 6, 6.25e9, 25e9):.2f} s")
    print(f"  临时实例准备 8 s：可用 20 s → {window(20, 8):.0f} s 生成；可用 5 s → {window(5, 8):.0f}")
    print(f"  训练 84 $/h + 6 × 5.3 $/h = {84 + 6 * 5.3:.1f}：吞吐要 {(84 + 31.8) / 84:.2f} 倍；1.2 倍 → 成本 {cost_ratio(84, 31.8, 1.2):.2f}，1.6 倍 → {cost_ratio(84, 31.8, 1.6):.2f}")
    print("  PolyRL 截断：4000 与 1000 token → 各取 1000，复用 2000、重生成 3000")

    print("\n第四步：验证")
    print(f"  第 0/10/20 s 到达、每条 10 s：逐条提交 {batch_done([0, 10, 20], 10):.0f} s，等齐再做 {batch_done([0, 10, 20], 10, False):.0f} s")
    s = [1] * 9 + [100]
    print(f"  九条 1 s、一条 100 s：均值 {sum(s) / 10} s；已跑 10 s 的剩余 {residual(s, 10):.0f} s（不是 0.9）")
    print("  整批最早 120 s 完成：100 s 的活最迟第 20 s 开始，准备 5 s 则第 15 s")
    for g in (64, 16):
        print(f"  验证省 60 H100·秒、训练组 {g} 卡多等 2 s：净 {g * 2 - 60:+d} H100·秒")


if __name__ == "__main__":
    main()
