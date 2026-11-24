"""第 53 章 · 专家放置、复制与负载均衡。

讲义：ai-infra/notes/lessons/53-expert-balance.md
运行：uv run python ch53_expert_balance.py
大 EP 的 skew、例 9.5（两台 HGX H100 的 dispatch / combine）、例 9.6（专家副本回本）与 Grouped GEMM 补齐，取自 AII 第 9 章 §9.4。
"""

import math
import random

MiB = 2**20
H100 = 494.7e12                       # BF16 峰值的 50%
FLOP_PER = 6 * 4096 * 1536            # 一份 token—专家任务
ROW = 8 * 1024                        # 每行 8 KiB


def hot_skew(ep: int, n: int = 1024, k: int = 8, E: int = 256, hot: int = 128) -> tuple[float, float]:
    """一个热点专家（hot 行）、其余均分：返回 (热点卡行数, 热点卡 / 平均)。"""
    rest = (n * k - hot) / (E - 1)
    per_card = E // ep
    rows = hot + (per_card - 1) * rest
    return rows, rows / (n * k / ep)


def random_skew(ep: int, batches: int = 1000, n: int = 1024, k: int = 8, E: int = 256, seed: int = 0) -> float:
    """均匀随机路由下 最忙卡 / 平均 的批平均。"""
    rng = random.Random(seed)
    per_card = E // ep
    tot = 0.0
    for _ in range(batches):
        cards = [0] * ep
        for _ in range(n):
            for e in rng.sample(range(E), k):
                cards[e // per_card] += 1
        tot += max(cards) / (n * k / ep)
    return tot / batches


def exchange(groups: list[int], bw: float = 50e9, dispatch_bytes: int = ROW, combine_bytes: int = ROW, shared: float | None = None):
    """例 9.5：(dispatch, 最忙卡计算, combine, 屏障总和) ms。shared 是两池间总带宽上限。"""
    total = sum(groups)
    d = max(max(groups) * dispatch_bytes / bw, total * dispatch_bytes / shared if shared else 0)
    c = max(max(groups) * combine_bytes / bw, total * combine_bytes / shared if shared else 0)
    comp = max(groups) * FLOP_PER / H100
    return tuple(round(x * 1e3, 3) for x in (d, comp, c, d + comp + c))


def replica_breakeven(dt: float, n_copies: int = 7, bw: float = 450e9, alpha: float = 5e-6, w: float = 36 * MiB):
    """(复制时间, 回本批数)：N_min = ⌊T_copy / Δt⌋ + 1。"""
    t = n_copies * (w / bw + alpha)
    return t, math.floor(t / dt) + 1


def padded_rows(rows: list[int], tile: int = 32) -> tuple[int, int]:
    """(非空专家各补到 tile 的倍数, 全部补到最大值)。"""
    a = sum(math.ceil(r / tile) * tile for r in rows if r)
    return a, len(rows) * max(rows)


def three_stage(d: float, c: float, m: float, q: int = 2, slow: float = 1.0) -> float:
    """三段流水：一个 micro-batch 三段之和 + (q − 1) × 最长段；slow 是通信被拖慢的倍数。"""
    s = [d / q * slow, c / q, m / q * slow]
    return sum(s) + (q - 1) * max(s)


def main() -> None:
    print("\n第一步：总量一样，最忙的卡定完成时刻")
    gf = 512 * 2 * 18874368
    print(f"  512 次分派 {gf / 1e9:.1f} GFLOPs：均分每卡 {gf / 8 / H100 * 1e6:.1f} μs，全在一张卡 {gf / H100 * 1e6:.1f} μs")
    for ep in (8, 32, 256):
        r, s = hot_skew(ep)
        print(f"  EP{ep}：热点卡 {r:.0f} 行，是平均的 {s:.2f} 倍")
    print(f"  均匀随机（1000 批）：EP8 {random_skew(8):.2f}，EP256 {random_skew(256):.2f}")

    print("\n第二步：例 9.5（ms：dispatch / 计算 / combine / 合计）")
    bal, hot = [2048] * 4, [5120, 1024, 1024, 1024]
    print(f"  均衡跨服务器     {exchange(bal)}")
    print(f"  热点跨服务器     {exchange(hot)}")
    print(f"  均衡、只剩一条链路 {exchange(bal, shared=50e9)}")
    print(f"  热点、dispatch FP8 {exchange(hot, dispatch_bytes=ROW // 2)}")
    print(f"  热点、同一台 NVLink {exchange(hot, bw=450e9)}")

    print("\n第三步：例 9.6 专家副本")
    before = 921e6 / 1675e9
    dt = before - before * 576 / 1024
    for name, bw in (("NVLink 450 GB/s", 450e9), ("网卡 50 GB/s", 50e9)):
        t, n = replica_breakeven(dt, bw=bw)
        print(f"  {name}：复制 {t * 1e3:.3f} ms，每批省 {dt * 1e3:.3f} ms，第 {n} 批回本；16 批净 {(16 * dt - t) * 1e3:+.1f} ms，64 批 {(64 * dt - t) * 1e3:+.1f} ms")
    print(f"  每卡每层一个副本：94 × 36 MiB = {94 * 36 / 1024:.2f} GiB；只给 8 层：{8 * 36} MiB")

    print("\n第四步：补齐与重叠")
    print(f"  [32,16,8,4,2,1,1,0] 共 64 行：补到 32 的倍数 {padded_rows([32, 16, 8, 4, 2, 1, 1, 0])[0]}，全补到最大 {padded_rows([32, 16, 8, 4, 2, 1, 1, 0])[1]}")
    print(f"  0.336 / 0.156 / 0.336：串行 0.827，两个 micro-batch {three_stage(0.336, 0.156, 0.336):.3f}，通信慢 49% {three_stage(0.336, 0.156, 0.336, slow=1.49):.3f} ms")


if __name__ == "__main__":
    main()
