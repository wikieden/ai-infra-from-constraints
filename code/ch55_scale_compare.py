"""第 55 章 · 扩缩容、故障恢复与部署方案比较。

讲义：ai-infra/notes/lessons/55-scale-compare.md
运行：uv run python ch55_scale_compare.py
启动积压的排空、后台迁移追赶、重配置、输出记录恢复与例 9.8 的方案比较，取自 AII 第 9 章 §9.6–§9.7。
"""

GiB = 2**30
K_TOKEN = 147456
MU = {"八个完整副本": 3.018, "理想分块共置": 4.17, "直接 PD": 4 / 0.8782}


def drain_s(lam: float, Ts: float, mu: float) -> float:
    """T_drain = λ T_s / (μ − λ)；μ ≤ λ 时返回无穷。"""
    return lam * Ts / (mu - lam) if mu > lam else float("inf")


def min_rate(lam: float, Ts: float, deadline: float) -> float:
    """在 deadline 前清空启动积压所需的最低服务率。"""
    return lam + lam * Ts / (deadline - Ts)


def catch_up_s(state: float, copy_bw: float, growth: float) -> float:
    """后台复制追上仍在增长的状态：state / (copy − growth)。"""
    return state / (copy_bw - growth)


def breakeven_steps(Ts: float, delta: float) -> int:
    return round(Ts / delta)


def cost_per_k(yuan_per_h: float, rate: float, qualified: float = 1.0) -> float:
    """每千个合格请求的成本。"""
    return yuan_per_h / (rate * qualified * 3600) * 1000


def main() -> None:
    print("\n第一步：启动与预热")
    print(f"  启动多 10 s、每步省 0.2 ms：{breakeven_steps(10, 0.2e-3):,} 步持平（batch 32 约 {breakeven_steps(10, 0.2e-3) * 32 / 1e6:.1f} M token）")
    for name, mu in MU.items():
        d = drain_s(3.5, 10, mu)
        print(f"  {name} {mu:.2f} 请求/s：" + (f"就绪后 {d:.1f} s 排空，第 {10 + d:.1f} 秒" if d != float("inf") else "积压一直增长"))

    print("\n第二步：迁移与重配置")
    growth = 1166 * K_TOKEN / 1e9
    state = 32 * 8704 * K_TOKEN / 1e9
    for name, bw in (("200 Gbit/s 25 GB/s", 25.0), ("50 GbE 6.25 GB/s", 6.25)):
        t = catch_up_s(state, bw, growth)
        print(f"  {state:.1f} GB、每秒长 {growth:.3f} GB，{name}：{t:.2f} s 追上，期间新增 {t * growth:.2f} GB")
    for name, bw in (("50 GbE", 6.25e9), ("200 Gbit/s", 25e9)):
        print(f"  TP4 → TP8 16.4 GB 走 {name}：{16.4e9 / bw:.2f} s，加 9 s 准备 {16.4e9 / bw + 9:.1f} s")
    print(f"  同机 NVLink 最忙卡 4.7 GB / 300 GB/s = {4.7e9 / 300e9 * 1e3:.1f} ms，总时间仍约 9 s")

    print("\n第三步：恢复")
    print("  已返回 1025 个输出、KV 只存了输入：把前 1024 个输出当 prefill 补算，重建 9216 个 token 的 KV")

    print("\n第四步：例 9.8")
    v = 1.125 * GiB
    for name, mult in (("直接 PD", 1), ("PD + 共享池", 2)):
        print(f"  {name}：每请求 {mult * 1.125} GiB，通道 {25e9 / (mult * v):.1f} 请求/s；3.5 请求/s 时 {3.5 * mult * v / 1e9:.1f} GB/s")
    need = min_rate(3.5, 10, 60)
    print(f"  60 s 内清空：μ ≥ {need:.1f}；直接 PD 余量 {(MU['直接 PD'] / need - 1) * 100:.0f}%")
    mu9 = MU["直接 PD"] * 0.9
    print(f"  效率只有九成：{mu9:.2f} 请求/s → 第 {10 + drain_s(3.5, 10, mu9):.1f} 秒才清空")
    print(f"  完整副本到第 60 秒积压 {35 + 50 * (3.5 - MU['八个完整副本']):.0f} 个")
    print(f"  成本：每小时 8 元、3.5 请求/s → {cost_per_k(8, 3.5):.2f} 元/千请求；满载 3.02 / 4.55 → {cost_per_k(8, 3.02):.2f} / {cost_per_k(8, 4.55):.2f}；只有 1/4 合格 → {cost_per_k(8, 4.55, 0.25):.2f}")

    print("\n第五步：负载变了")
    print(f"  后续复用一次：中转 + 取回 96.6 ms vs 重算 856 ms → 省 {856 - 96.6:.0f} ms；每对搬 3.375 GiB，3.5 对/s 要 {3.5 * 3.375 * GiB / 1e9:.1f} GB/s，上限 {25e9 / (3.375 * GiB):.1f} 对/s")
    for name, mu in (("完整副本", 4.897), ("原 PD 4P/4D", MU["直接 PD"]), ("2P/6D", 5.69)):
        print(f"  命中 6144 前缀 · {name} {mu:.2f}：第 {10 + drain_s(3.5, 10, mu):.1f} 秒清空")
    print("  输出 4097：最佳 2P/6D 只有 1.26 请求/s → 持续要 3 组（3.78），按期清空要 4 组（5.04）")


if __name__ == "__main__":
    main()
