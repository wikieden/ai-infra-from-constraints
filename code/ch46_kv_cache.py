"""第 46 章 · KV 缓存的分配、复用与释放。

讲义：ai-infra/notes/lessons/46-kv-cache.md
运行：uv run python ch46_kv_cache.py
分页与前缀共享的块数、写时复制、16 条共享前缀长请求的容量、前缀保留的期望收益取自 AII 第 8 章 §8.3。
"""

import math


def paged(lengths: list[int], p: int) -> tuple[int, int]:
    """(分配的 token 容量, 未用槽位)。"""
    alloc = sum(math.ceil(L / p) * p for L in lengths)
    return alloc, alloc - sum(lengths)


def shared_prefix_mib(b: int, prefix: float = 864, private: float = 324) -> float:
    return prefix + b * private


def max_shared(budget: float = 12 * 1024, prefix: float = 864, private: float = 324) -> int:
    return int((budget - prefix) // private)


def keep_value(p: float, Tr: float, Tf: float, Tm: float) -> float:
    """V_h = p(Tr − Tf) − Tm（ms）。"""
    return p * (Tr - Tf) - Tm


def main() -> None:
    print("\n第一步：分页")
    L = [9, 13, 5, 15]
    print(f"  整段预留 {4 * 16}（未用 {64 - sum(L)}）")
    for p in (2, 4, 8):
        a, w = paged(L, p)
        print(f"  每块 {p}：分配 {a}，未用 {w}")
    print(f"  A、B 共享前 8 个 token：{paged(L, 4)[0] - 8} 个")

    print("\n第二步：共享与写时复制")
    refs, phys = 4 * (95 + 9), 95 + 4 * 9
    print(f"  四分支：引用 {refs}，物理块 {phys}，少 {1 - phys / refs:.0%}")
    print(f"  16 条 8K 长请求：独立 {16 * 1188:,} MiB，共享前缀 {shared_prefix_mib(16):,.0f} MiB（{shared_prefix_mib(16) / 1024:.2f} GiB）")
    print(f"  12 GiB 最多：独立 {12 * 1024 // 1188} 条，共享 {max_shared()} 条")

    print("\n第三步：前缀值不值得留（RTX PRO 6000）")
    gen5, gen4 = 0.906e9 / 64e9 * 1e3, 0.906e9 / 32e9 * 1e3
    va = keep_value(0.5, 307.9, gen5, gen5)
    print(f"  A（6144 token，864 MiB）：取回 {gen5:.1f} ms，命中 50% 净省 {va:.1f} ms，门槛 {gen5 / (307.9 - gen5):.1%}")
    print(f"  PCIe Gen4：取回 {gen4:.1f} ms，门槛 {gen4 / (307.9 - gen4):.1%}")
    vb = keep_value(0.5, 94.7, 4.7, 4.7)
    print(f"  B（2048 token，288 MiB）：净省 {vb:.1f} ms，三个 {3 * vb:.1f} ms；每 MiB A {va / 864:.3f}、B {vb / 288:.3f} ms")


if __name__ == "__main__":
    main()
