"""第 48 章 · 推测解码。

讲义：ai-infra/notes/lessons/48-speculative.md
运行：uv run python ch48_speculative.py
接受概率 min(1, p/q)、每轮期望输出、每 token 耗时、查询时间的交点、草稿来源与草稿长度取自 AII 第 8 章 §8.5。
"""

import math
import random


def accept_prob(p: float, q: float) -> float:
    return min(1.0, p / q)


def speculative_sample(p: dict[str, float], q: dict[str, float], rng: random.Random) -> str:
    """从 q 采草稿，按 min(1, p/q) 接受，否则从 [p − q]+ 归一化后重采。"""
    x = rng.choices(list(q), weights=list(q.values()))[0]
    if rng.random() < accept_prob(p[x], q[x]):
        return x
    resid = {k: max(p[k] - q.get(k, 0), 0) for k in p}
    return rng.choices(list(resid), weights=list(resid.values()))[0]


def expected_tokens(a: float | list[float], m: int = 4) -> float:
    """E[N] = 1 + Σ_j Π_{i≤j} a_i。"""
    acc = [a] * m if isinstance(a, float) else a
    total, prod = 1.0, 1.0
    for ai in acc[:m]:
        prod *= ai
        total += prod
    return total


def per_token_ms(round_ms: float, n: float) -> float:
    return round_ms / n


def main() -> None:
    print("\n第一步：保持目标分布（p(A) = 1/4，p(B) = 3/4）")
    rng = random.Random(0)
    p = {"A": 0.25, "B": 0.75}
    for name, q in (("草稿恒为 A", {"A": 1.0, "B": 0.0}), ("草稿恒为 B", {"A": 0.0, "B": 1.0})):
        s = [speculative_sample(p, {k: v for k, v in q.items() if v > 0}, rng) for _ in range(20000)]
        print(f"  {name}：接受概率 {accept_prob(p[max(q, key=q.get)], 1.0):.2f}，采样得 P(A) ≈ {s.count('A') / len(s):.3f}")

    print("\n第二步：每轮输出与每 token 耗时（验证 26.26 ms + 查询 0.1 ms）")
    for name, a in (("AAAA", 0.25), ("BBBB", 0.75)):
        n = expected_tokens(a)
        print(f"  {name}：E[N] = {n:.2f}，每 token {per_token_ms(26.36, n):.2f} ms；查询上限 {n * 26.26 - 26.26:.1f} ms")
    print(f"  两轮 1.5 ms 各出 1、5 个：{3 / 6:.1f} ms/token（不是 {(1.5 + 0.3) / 2:.1f}）")
    save = 26.26 - per_token_ms(26.36, expected_tokens(0.75))
    print(f"  建索引 2 s：{math.ceil(2000 / save)} 个 token 回本；256 token {256 * 26.26 / 1e3:.2f} → {2 + 256 * per_token_ms(26.36, expected_tokens(0.75)) / 1e3:.2f} s")

    print("\n第三步：草稿来源（DFlash 2.10 GB，588 GB/s）")
    fwd = 2.10e9 / 588e9 * 1e3
    print(f"  一次前向 {fwd:.2f} ms；串行 4 次 {4 * fwd:.2f} ms")
    print(f"  每轮 3 个 token：串行 {(4 * fwd + 26.26) / 3:.1f}，并行 {(fwd + 26.26) / 3:.1f} ms；并行只出 2 个 {(fwd + 26.26) / 2:.1f} ms")

    print("\n第四步：草稿长度（a = 3/4，T = 26.3 + 3.6m）")
    for m in (1, 2, 4, 8):
        n = expected_tokens(0.75, m)
        print(f"  m = {m}：E[N] = {n:.2f}，{(26.3 + 3.6 * m) / n:.1f} ms/token")


if __name__ == "__main__":
    main()
