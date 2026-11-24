"""第 64 章 · 模型服务的选择、成本与扩容。

讲义：ai-infra/notes/lessons/64-service-cost.md
运行：uv run python ch64_service_cost.py
思考预算、前缀缓存计费、命中率与成功率的路由、自建 / 按量 / 订阅、提交粒度、恢复树与例 11-9 的方案选择，取自 AII 第 11 章 §11.4–§11.5。
"""


def call_cost(inp: float, out: float, p_in: float, p_out: float, cached: float = 0, p_cache: float = 0, write: float = 0, p_write: float = 0) -> float:
    """C = (I_u p_u + I_w p_w + I_h p_h + O p_o) / 1e6，价格按美元/百万 token。"""
    return (inp * p_in + write * p_write + cached * p_cache + out * p_out) / 1e6


def cache_worth(n: int, pu: float, pw: float, ph: float) -> bool:
    """n p_u > p_w + (n − 1) p_h。"""
    return n * pu > pw + (n - 1) * ph


def breakeven_hit(cA: float, pA: float, hit: float, miss: float, pB: float) -> float:
    """令 B 的每成功成本等于 A，求 B 的命中率。"""
    target = cA / pA * pB
    return (miss - target) / (miss - hit)


def selfhost_breakeven(F: float, c: float, v: float = 0) -> float:
    return F / (c - v)


def commit_plan(n: int = 100, work_ms: float = 200, fixed: float = 8, per: float = 2, batch: int = 1):
    """(保存总 ms, 无故障总 s, 最多重做 s)。"""
    k = n // batch
    save = k * (fixed + batch * per)
    return save, (n * work_ms + save) / 1000, batch * work_ms / 1000


def recovery_tree():
    """例 11-8：路径 (概率, 成本, 时间)。"""
    return [(0.80, 0.010, 10), (0.12 * 0.6, 0.016, 14), (0.12 * 0.4 * 0.98, 0.046, 22), (0.12 * 0.4 * 0.02, 0.046, 22),
            (0.08 * 0.98, 0.040, 18), (0.08 * 0.02, 0.040, 18)]


def plan_cost(call_b200_s: float, cpu_s: float, mem_gib_s: float, rounds: int = 3) -> float:
    """每项任务：模型（6.79 $/卡时）+ CPU（0.000014 $/CPU·秒）+ 内存（0.0000045 $/GiB·秒）。"""
    return rounds * call_b200_s * 6.79 / 3600 + cpu_s * 0.000014 + mem_gib_s * 0.0000045


def amdahl(f: float, s: float) -> float:
    return 1 / ((1 - f) + f / s)


def main() -> None:
    print("\n第一步：思考预算")
    a, b = call_cost(10000, 1200, 2, 10), call_cost(10000, 300, 2, 10)
    print(f"  思考 1000 vs 100：{a:.3f} vs {b:.3f} 美元（−{(1 - b / a) * 100:.0f}%）；100 次省 {100 * (a - b):.2f}、30 次重试花 {30 * a:.2f}")

    print("\n第二步：前缀缓存")
    plain = 10 * call_cost(10000, 0, 2, 0)
    cached = call_cost(2000, 0, 2, 0, write=8000, p_write=2.5) + 9 * call_cost(2000, 0, 2, 0, cached=8000, p_cache=0.2)
    print(f"  十轮输入：无缓存 {plain:.2f}，有缓存 {cached:.4f} 美元；第 2 次就回本：{cache_worth(2, 2, 2.5, 0.2)}")
    print("  请求命中率都 50%：只命中 1K 前缀读 10%，只命中 9K 前缀读 90%")

    print("\n第三步：路由")
    print(f"  A 排 200 + 命中 50 + 网络 50 = 300 ms；B 20 + 500 + 50 = 570 ms")
    cA = call_cost(1000, 2000, 1, 5, cached=19000, p_cache=0.1)
    hit, miss = call_cost(1000, 300, 2, 10, cached=19000, p_cache=0.2), call_cost(20000, 300, 2, 10)
    h = breakeven_hit(cA, 0.8, hit, miss, 0.98)
    print(f"  A 每次 {cA:.4f}、成功 {cA / 0.8:.4f}；B 命中 {hit:.4f}、未命中 {miss:.4f} → 成本交点 h = {h:.1%}；6 s 期限要 h ≥ {0.90 / 0.98:.1%}")
    print(f"  快速服务：16 会话 6 s/调用，180 并发 > 160 槽 → 12 个副本；每次 {4 * 9 / 32} → {4 * 6 / 16} B200·秒")

    print("\n第四步：自建 vs 按量")
    F = 4 * 720 * 6.79
    c = 27 * 4 / 32 * 8.64 / 3600
    n = selfhost_breakeven(F, c)
    cap = 32 * 2_592_000 / 27
    print(f"  预留 {F:,.1f} 美元/月；按量每项 {c:.4f}；交点 {n / 1e4:.0f} 万项，上限 {cap / 1e4:.1f} 万项，利用率 {n / cap:.1%} = {6.79 / 8.64:.1%}")
    print(f"  decode 快一倍：{15.9 / 2 + 9.4:.1f} ms/token，总加速 {25.3 / (15.9 / 2 + 9.4):.2f} 倍")

    print("\n第五步：恢复与提交")
    for bt in (1, 10):
        s, t, r = commit_plan(batch=bt)
        print(f"  每 {bt} 个提交：保存 {s:.0f} ms，总 {t:.2f} s，最多重做 {r:.1f} s")
    tree = recovery_tree()
    cost = sum(p * c for p, c, _ in tree)
    ok = sum(p for (p, _, _), good in zip(tree, (1, 1, 1, 0, 1, 0)) if good)
    ontime = sum(p for (p, _, t), good in zip(tree, (1, 1, 1, 0, 1, 0)) if good and t <= 20)
    print(f"  恢复树：每项 {cost:.5f} 美元，成功 {ok:.1%}，按时 {ontime:.1%}；每成功 {cost / ok:.4f}，每按时成功 {cost / ontime:.4f}（只做首次 0.0125）")

    print("\n第六步：例 11-9（每千项提交）")
    for name, call, cpu, mem in (("普通 + 驻留", 1.125, 3, 60), ("普通 + 重建", 1.125, 3.3, 18), ("快速 + 驻留", 1.5, 3, 42), ("快速 + 重建", 1.5, 3.3, 18)):
        print(f"  {name}：{plan_cost(call, cpu, mem) * 1000:.2f} 美元")
    print(f"  每千项按时成功：{plan_cost(1.5, 3.3, 18) * 1000 / 0.95:.2f} vs {plan_cost(1.5, 3, 42) * 1000 / 0.95:.2f}")
    print(f"  保留与重建打平的等待 G = {(2 * 2 * 0.0000045 + 0.1 * 0.000014) / (2 * 0.0000045):.2f} s")
    print(f"  Amdahl：工具加速 2 倍 {amdahl(0.1, 2):.2f}，模型加速 2 倍 {amdahl(0.9, 2):.2f}")


if __name__ == "__main__":
    main()
