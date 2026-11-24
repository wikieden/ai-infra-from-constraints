"""第 41 章 · 数据交接、顺序与状态管理。

讲义：ai-infra/notes/lessons/41-handoff-state.md
运行：uv run python ch41_handoff_state.py
共享传输状态、OpenURMA 的状态尺寸与缓存溢出点、例题 7.4、槽位释放与背压取自 AII 第 7 章 §7.4。
"""

KiB, MiB = 2**10, 2**20


def state_bytes(L: int, P: int, e: int = 256, r: int = 64, t: int = 1024, groups: int | None = None) -> float:
    """groups=None：每关系独占；groups=K：K 组共享传输状态。"""
    if groups is None:
        return L * e + L * P * (r + t)
    return L * e + L * P * r + groups * P * t


def impl_state(N: int, M: int) -> tuple[int, int]:
    return 512 * N * M + 32 * N, 52 * N + 56 * M


def overflow_n(cache: int = 256 * KiB) -> tuple[int, int]:
    pair = next(n for n in range(1, 10**5) if 512 * n * n + 32 * n > cache)
    ub = next(n for n in range(1, 10**5) if 108 * n > cache)
    return pair, ub


def ordering(write: float = 20, recover: float = 80, notify: float = 2, c: float = 10) -> tuple[float, float, float]:
    """(串行时 C 完成, 只保留 A→B 时 C 完成, 发布链完成)。"""
    chain = write + recover + notify
    return chain + c, c, chain


def slots_sim(n_ops: int = 16, slots: int = 8, lat: float = 5, batch: int = 4, period: float = 20) -> tuple[float, float]:
    """每 1 μs 最多提交一次，提交后 lat 完成；软件每 period 处理 batch 条完成通知并释放槽位。返回 (全部传完, 全部槽位释放)。"""
    t, free, submit, done = 0.0, slots, [], []
    released = 0
    release_times = []
    k = 1
    while len(submit) < n_ops:
        while free == 0:
            # 等下一次批处理
            ready = sorted(s + lat for s in submit)[released:]
            tick = period * k
            k += 1
            n = len([x for x in ready if x <= tick][:batch])
            released += n
            free += n
            release_times += [tick] * n
            t = max(t, tick)
        submit.append(t)
        free -= 1
        t += 1
    done = [s + lat for s in submit]
    while released < n_ops:
        ready = sorted(done)[released:]
        tick = period * k
        k += 1
        n = len([x for x in ready if x <= tick][:batch])
        released += n
        release_times += [tick] * n
    return max(done), max(release_times)


def main() -> None:
    print("\n第一步：64 个端点 × 128 个目标")
    p, s = state_bytes(64, 128), state_bytes(64, 128, groups=1)
    print(f"  独占 {p / MiB:.2f} MiB，共享 {s / MiB:.2f} MiB（约 1/{p / s:.0f}）")
    for K in (3, 4, 8):
        print(f"  {K} 组隔离：{state_bytes(64, 128, groups=K) / KiB:.0f} KiB")
    a, b = impl_state(1024, 1024)
    print(f"  实现尺寸 N = M = 1024：逐对 {a / MiB:.0f} MiB，UB {b / KiB:.0f} KiB，{a / b:.0f} 倍；256 KiB 缓存溢出点 {overflow_n()}")

    print("\n第二步：例题 7.4")
    for rec in (80, 200):
        cs, ci, ch = ordering(recover=rec)
        print(f"  恢复 {rec} μs：串行 C 在 {cs:.0f} μs，独立 C 在 {ci:.0f} μs，发布链 {ch:.0f} μs")

    print("\n第三步：槽位与完成通知")
    for n in (8, 16):
        d, r = slots_sim(slots=n)
        print(f"  {n} 个槽位：全部传完 {d:.0f} μs，槽位全部释放 {r:.0f} μs")
    rate = 4 / 20e-6
    print(f"  完成通知 {rate:,.0f} 项/s × 8 KiB = {rate * 8192 / 1e9:.2f} GB/s；50 GB/s 要 {50e9 / 8192 / 1e6:.1f} M 项/s")


if __name__ == "__main__":
    main()
