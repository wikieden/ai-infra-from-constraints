"""第 58 章 · 流水调度与通信重叠。

讲义：ai-infra/notes/lessons/58-pipeline-overlap.md
运行：uv run python ch58_pipeline_overlap.py
按 token 平均的损失、填满排空与 1F1B 的事件模型、各调度的气泡公式、ZeRO-3 通信量与暴露部分、
MoE 容量因子、梯度切桶与不等长序列的注意力配对，取自 AII 第 10 章 §10.3。
"""

import math

N = 8_190_735_360


def token_mean_grad() -> tuple[float, float, float]:
    """八条回答：4 条 2 token（每 token 损失 θ/5）、4 条 3 token（2θ/5）。返回 (按 token, 按回答, 错除以 8)。"""
    by_token = (8 * (1 / 5) + 12 * (2 / 5)) / 20
    by_answer = (1 / 5 + 2 / 5) / 2
    return by_token, by_answer, by_token * 20 / 8 / 20


def simulate(order: dict[int, list[tuple[str, int]]], p: int, tf: float = 10, tb: float = 20, link: float = 1, update: float = 1) -> float:
    """按每阶段给定的 (F/B, micro-batch) 顺序排事件，返回包括参数更新的完成时间（ms）。"""
    done: dict[tuple[str, int, int], float] = {}
    free = [0.0] * p
    idx = [0] * p
    total = sum(len(v) for v in order.values())
    while len(done) < total:
        progressed = False
        for s in range(p):
            if idx[s] >= len(order[s]):
                continue
            kind, j = order[s][idx[s]]
            if kind == "F":
                dep = 0.0 if s == 0 else done.get(("F", s - 1, j))
                dep = dep if s == 0 or dep is None else dep + link
            else:
                dep = done.get(("F", s, j)) if s == p - 1 else done.get(("B", s + 1, j))
                dep = dep if s == p - 1 or dep is None else dep + link
            if dep is None:
                continue
            start = max(free[s], dep)
            free[s] = start + (tf if kind == "F" else tb)
            done[(kind, s, j)] = free[s]
            idx[s] += 1
            progressed = True
        if not progressed:
            raise RuntimeError("调度死锁")
    return max(free) + update


def fill_drain(p: int = 4, m: int = 8) -> dict:
    return {s: [("F", j) for j in range(m)] + [("B", j) for j in range(m)] for s in range(p)}


def one_f_one_b(p: int = 4, m: int = 8) -> dict:
    order = {}
    for s in range(p):
        w = min(p - s - 1, m)
        seq = [("F", j) for j in range(w)]
        for i in range(m - w):
            seq += [("F", w + i), ("B", i)]
        seq += [("B", j) for j in range(m - w, m)]
        order[s] = seq
    return order


def bubble(p: int, m: int) -> float:
    """填满排空 / 1F1B 的时间利用率 u = m / (m + p − 1)。"""
    return m / (m + p - 1)


def zero3_bytes(d: int, m: int, n: int = N) -> float:
    """V = 3m · (d − 1)/d · 2N：每个 micro-batch 两次 AllGather + 一次 ReduceScatter。"""
    return 3 * m * (d - 1) / d * 2 * n


def exposed_ms(comm: float, window: float, busy: float = 0.0) -> float:
    """链路先被占 busy，通信在独立计算 window 内能藏多少；返回延伸到计算之后的部分。"""
    return max(0.0, busy + comm - window)


def capacity(c: float, tokens: int = 8192, k: int = 8, E: int = 128, hot_frac: float = 0.5, hot: float = 1.5):
    """容量因子：返回 (每专家容量, 丢弃占比, 补零行占执行量)。"""
    avg = tokens * k / E
    cap = math.floor(c * avg)
    nh = int(E * hot_frac)
    loads = [hot * avg] * nh + [(2 - hot) * avg] * (E - nh)
    dropped = sum(max(0, x - cap) for x in loads)
    padded = sum(max(0, cap - x) for x in loads)
    return cap, dropped / (tokens * k), padded / (cap * E)


def bucket_ms(mib: float, members: int = 4, bw: float = 25e9, alpha: float = 0.02e-3) -> float:
    """环形 AllReduce：每卡发 2(n−1)/n × 桶大小。"""
    return (2 * (members - 1) / members * mib * 2**20 / bw + alpha) * 1e3


def causal_pairs(lengths: list[int]) -> int:
    return sum(s * (s + 1) // 2 for s in lengths)


def main() -> None:
    print("\n第一步：损失分母")
    a, b, c = token_mean_grad()
    print(f"  按 token 平均 {a:.2f}，先按回答平均 {b:.2f}，用 20 再除以 8 → {c:.2f}")

    print("\n第二步：四阶段、八个 micro-batch（前向 10、反向 20、边界 1、更新 1 ms）")
    print(f"  公式 (m + p − 1)(t_f + t_b) = {(8 + 3) * 30} ms，u = {bubble(4, 8) * 100:.1f}%")
    print(f"  事件模型：填满排空 {simulate(fill_drain(), 4):.0f} ms，1F1B {simulate(one_f_one_b(), 4):.0f} ms")
    print(f"  十六个 micro-batch：填满排空 {simulate(fill_drain(m=16), 4):.0f} ms，1F1B {simulate(one_f_one_b(m=16), 4):.0f} ms；气泡占比 {3 / 8 * 100:.1f}% → {3 / 16 * 100:.1f}%")
    print("  原文事件模型：交错式 298、ZB-H1 283、DualPipe 306 ms（八个）；599 / 538 / 523 / 558 ms（十六个，含 1F1B）")
    print(f"  气泡下界：1F1B {3 * 30} ms、ZB-H1 {3 * (10 + 10 - 10)} ms、ZB-H2 {3 * (10 + 10 - 20)} ms、DualPipe {(2 - 1) * (30 + 20 - 30)} ms")

    print("\n第三步：通信重叠")
    print(f"  3 ms 归约、5 ms 计算：链路空闲暴露 {exposed_ms(3, 5):.0f} ms；先被占 4 ms 暴露 {exposed_ms(3, 5, 4):.0f} ms")
    for d, m in ((48, 8), (32, 12)):
        v = zero3_bytes(d, m)
        print(f"  {d} 卡 m = {m}：每步 {v / 1e9:.0f} GB，PCIe 32 GB/s 链路时间 {v / 32e9:.1f} s")
    emb = 151936 * 4096 * 2
    print(f"  每 micro-batch 链路 {zero3_bytes(48, 1) / 32e9:.2f} s vs 计算 {52.2 / 8:.2f} s；只暴露首尾两次 {emb / 1e9:.2f} GB 词嵌入 ≈ {2 * 47 / 48 * emb / 32e9:.2f} s")

    print("\n第四步：MoE 与长上下文")
    for cf in (1.0, 1.25, 1.5, 2.0):
        cap, dr, pd = capacity(cf)
        print(f"  c = {cf}：每专家 {cap} 行，丢弃 {dr * 100:.1f}%，补零 {pd * 100:.0f}%")
    print(f"  72 MiB 梯度整桶 {bucket_ms(72):.2f} ms；切三桶每桶 {bucket_ms(24):.2f} ms，合计 {3 * bucket_ms(24):.2f} ms；400 Gbit/s 整桶 {bucket_ms(72, bw=50e9):.2f} ms")
    e, u = causal_pairs([4096, 4096]), causal_pairs([7168, 1024])
    print(f"  因果配对：4096 + 4096 = {e:,}，7168 + 1024 = {u:,}（+{(u / e - 1) * 100:.0f}%）")


if __name__ == "__main__":
    main()
