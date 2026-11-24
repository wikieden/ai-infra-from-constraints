"""第 60 章 · 强化学习训练系统。

讲义：ai-infra/notes/lessons/60-rl-system.md
运行：uv run python ch60_rl_system.py
组内优势、三阶段有效吞吐、共享加速器切换次序、按分片同步权重、概率比、异步周期与路由重放记录，取自 AII 第 10 章 §10.5。
"""

import math


def advantages(rewards: list[float]) -> list[float]:
    m = sum(rewards) / len(rewards)
    return [r - m for r in rewards]


def effective_rate(rg: float, rv: float, rl: float, q: float) -> float:
    """r_eff = min(r_l, q · min(r_g, r_v))。"""
    return min(rl, q * min(rg, rv))


def switch_peak(order: str, train: float = 40, weights: float = 15.3, kv: float = 24, common: float = 4) -> float:
    """共享加速器从训练切到生成的显存峰值（GiB）。"""
    if order == "加载权重和 KV 再释放训练":
        return train + weights + kv + common
    return train + weights + common          # 先加载权重，释放训练后再分配 KV


def ratio(logp_theta: float, logp_mu: float) -> float:
    return math.exp(logp_theta - logp_mu)


def async_gain(q: float, gen: float = 40, learn: float = 16, sync: float = 4) -> float:
    """异步有效吞吐 / 同步有效吞吐 = (q / (max + sync)) / (1 / (gen + learn + sync))。"""
    return q / (max(gen, learn) + sync) * (gen + learn + sync)


def survive(lam: float, t: float) -> float:
    return math.exp(-lam * t)


def replay_bytes(tokens: int = 8192, layers: int = 48, k: int = 8, width: int = 2) -> int:
    return tokens * layers * k * width


def top2(scores: list[float]) -> list[int]:
    return sorted(range(len(scores)), key=lambda i: -scores[i])[:2]


def main() -> None:
    print("\n第一步：组内优势")
    print(f"  奖励 [1, 1] → {advantages([1, 1])}；[1, 0] → {advantages([1, 0])}")

    print("\n第二步：三阶段有效吞吐（条/s）")
    print(f"  生成 12、验证 6、学习 8、保留 75% → {effective_rate(12, 6, 8, 0.75)}")
    print(f"  生成翻倍 → {effective_rate(24, 6, 8, 0.75)}；验证翻倍 → {effective_rate(12, 12, 8, 0.75)}（学习成瓶颈）")

    print("\n第三步：切换次序与权重同步")
    for o in ("加载权重和 KV 再释放训练", "先加载权重，释放后再分配 KV"):
        print(f"  {o}：峰值 {switch_peak(o):.1f} GiB（H100 74.5 GiB）")
    print(f"  Qwen3-235B 专家 423 GiB、EP16：整份发给每个接收方 {16 * 423} GiB，按分片 423 GiB（每方 {423 / 16:.1f} GiB）")

    print("\n第四步：策略版本与异步")
    print(f"  logprob −3 vs 误记 −3.25：比值 {ratio(-3, -3.25):.2f}")
    print(f"  同步周期 60 s、异步 44 s：q 门槛 {44 / 60 * 100:.1f}%；q = 0.8 → {async_gain(0.8):.2f} 倍，q = 0.7 → {async_gain(0.7):.2f} 倍")
    print(f"  中断率每小时 0.1 次：10 分钟轨迹完整概率 {survive(0.1 / 3600, 600):.3f}，2 小时 {survive(0.1 / 3600, 7200):.3f}")

    print("\n第五步：路由重放")
    print(f"  top-2：[0.500, 0.301, 0.300] → {top2([0.5, 0.301, 0.3])}；第三个变 0.302 → {top2([0.5, 0.301, 0.302])}")
    print(f"  Qwen3-30B-A3B 8192 token × 48 层 × top-8：uint16 {replay_bytes() / 2**20:.0f} MiB，int32 {replay_bytes(width=4) / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
