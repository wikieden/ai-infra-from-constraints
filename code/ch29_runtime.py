"""第 29 章 · 运行时：重叠、CUDA Graph、动态形状与 persistent kernel。

讲义：ai-infra/notes/lessons/29-runtime.md
运行：uv run python ch29_runtime.py
例 5-10 主机流水、例 5-11 图输入复制、例 5-12 分桶与特化、自动调优回本、persistent kernel 取自 AII 第 5 章 §5.5。
"""

MiB = 2**20


def host_pipeline(n: int, host: float, dev: float) -> tuple[float, float]:
    """(串行, 流水) 的总时间（μs）：流水时先准备第一段，之后每 max 完成一段。"""
    return n * (host + dev), host + (n - 1) * max(host, dev) + dev


def graph_total_us(nbytes: float, compute: float = 20, prep: float = 5, bw: float = 1.792e12) -> float:
    """图重放总时间 = 准备 + 输入复制（读 + 写）+ 计算。"""
    return prep + 2 * nbytes / bw * 1e6 + compute


def graph_breakeven_bytes(saving_us: float = 15, bw: float = 1.792e12) -> float:
    return saving_us * 1e-6 * bw / 2


def group_ms(rows: int, tflops: float, h: int = 4096, f: int = 12288) -> float:
    """一组 FFN（每行 6HF FLOPs）的执行时间。"""
    return rows * 6 * h * f / (tflops * 1e12) * 1e3


STRATS = {"通用": (100, 5632, 100), "分桶": (400, 8192, 200), "特化": (900, 5632, 250)}


def total_ms(name: str, r: int) -> float:
    prep, rows, tf = STRATS[name]
    return prep + r * group_ms(rows, tf)


def best(r: int) -> str:
    return min(STRATS, key=lambda s: total_ms(s, r))


def persistent_us(blocks: int = 8, launch: float = 5, task: float = 0.52,
                  rows: int = 64, h: int = 4096, f: int = 12288, peak: float = 503.8e12, bw: float = 1.792e12) -> tuple[float, float]:
    """(粗粒度两次 launch, persistent 按块) 的总时间。"""
    proj = 2 * rows * h * f / peak * 1e6
    act = rows * f * 4 / bw * 1e6
    coarse = 2 * launch + blocks * (proj + act)
    fine = launch + blocks * (proj + task) + act + task
    return coarse, fine


def main() -> None:
    print("\n第一步：例 5-10，100 段")
    for h, d in ((20, 20), (20, 5), (5, 5)):
        s, p = host_pipeline(100, h, d)
        print(f"  主机 {h} μs、加速器 {d} μs：串行 {s:.0f}，流水 {p:.0f} μs")

    print("\n第二步：例 5-11，图重放省 15 μs，但要先把输入复制到固定地址")
    for mib in (2, 16):
        print(f"  {mib:2d} MiB：图执行 {graph_total_us(mib * MiB):.1f} μs（普通 40 μs）")
    print(f"  临界输入 {graph_breakeven_bytes() / MiB:.1f} MiB")

    print("\n第三步：例 5-12，10 次 FFN 一组")
    print(f"  实际 5632 行，分桶后 8192 行（多 {(8192 / 5632 - 1) * 100:.0f}%）")
    for k, (prep, rows, tf) in STRATS.items():
        print(f"  {k}：准备 {prep} ms，每组 {group_ms(rows, tf):.1f} ms，70 组 {total_ms(k, 70) / 1e3:.2f} s")
    ranges, cur, start = [], best(1), 1
    for r in range(2, 200):
        if best(r) != cur:
            ranges.append((cur, start, r - 1))
            cur, start = best(r), r
    ranges.append((cur, start, "…"))
    print(f"  最省时：{ranges}")
    calls = 600 / 5e-6
    print(f"  调优 10 分钟、每次省 5 μs：{calls:.1e} 次；每秒 1 万次 {calls / 1e4 / 3600:.1f} 小时，每秒 100 次 {calls / 100 / 86400:.0f} 天")

    print("\n第四步：persistent kernel，8 块 up 投影 + SiLU")
    c, f = persistent_us()
    print(f"  两次 launch {c:.0f} μs → 按块 {f:.0f} μs（{c / f:.2f} 倍）")


if __name__ == "__main__":
    main()
