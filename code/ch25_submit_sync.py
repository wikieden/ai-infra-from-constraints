"""第 25 章 · 提交、搬移与同步：从调用返回到结果可用。

讲义：ai-infra/notes/lessons/25-submit-sync.md
运行：uv run python ch25_submit_sync.py
例 5-1、例 5-2、首次编译摊销、H100 单 SM 的驻留与在途字节、MMA 指令数、生产者 / 消费者 warp 时序取自 AII 第 5 章 §5.1。
"""

import math

MiB = 2**20


def transfer_ms(nbytes: float, rate: float) -> float:
    return nbytes / rate * 1e3


def ready_time(submit: float, h2d: float, kernel: float, d2h: float) -> float:
    """结果可用时刻（μs）：提交后加速器按依赖依次做 H2D、kernel、D2H。"""
    return submit + h2d + kernel + d2h


def amortized_ms(prep: float, run: float, calls: int) -> float:
    """首次准备 prep、每次执行 run，调用 calls 次的平均耗时。"""
    return (prep + run * calls) / calls


def resident_blocks(threads: int, regs_per_thread: int, smem_bytes: int,
                    sm_threads: int = 2048, sm_regs: int = 65536, sm_smem: int = 233_472,
                    sm_blocks: int = 32, reserve: int = 1024) -> dict[str, int]:
    """各类资源各自允许的驻留线程块数（H100 一个 SM 的默认值）。"""
    return {
        "threads": sm_threads // threads,
        "regs": sm_regs // (threads * regs_per_thread),
        "smem": sm_smem // (smem_bytes + reserve),
        "blocks": sm_blocks,
    }


def occupancy(blocks: int, threads: int, max_warps: int = 64) -> float:
    return blocks * threads // 32 / max_warps


def inflight_bytes(bw: float, latency_s: float, sms: int = 132) -> float:
    """每个 SM 需要保持的在途字节 = 带宽 × 延迟 / SM 数。"""
    return bw * latency_s / sms


def inflight_have(warps: int, loads_per_warp: int, bytes_per_load: int = 512) -> int:
    """驻留 warp 能提供的在途字节：每条加载 32 线程 × 16 B = 512 B。"""
    return warps * loads_per_warp * bytes_per_load


def mma_count(m: int, n: int, k: int, shape: tuple[int, int, int] = (16, 8, 16)) -> int:
    sm, sn, sk = shape
    return (m // sm) * (n // sn) * (k // sk)


def main() -> None:
    print("\n第一步：例 5-1，64 MiB 经 PCIe Gen5 x16（每方向 64 GB/s）")
    print(f"  一次 {transfer_ms(64 * MiB, 64e9):.2f} ms；分两份各 {transfer_ms(32 * MiB, 64e9):.2f} ms")

    print("\n第二步：例 5-2，提交 3 + H2D 8 + kernel + D2H 4（μs）")
    a, b = ready_time(3, 8, 20, 4), ready_time(3, 8, 5, 4)
    print(f"  kernel 20 μs → {a:.0f} μs；kernel 5 μs → {b:.0f} μs；kernel 快 4 倍，整体快 {a / b:.2f} 倍")
    print(f"  首次编译 100 ms + 每次 1 ms：1 次 {amortized_ms(100, 1, 1):.0f} ms，100 次平均 {amortized_ms(100, 1, 100):.1f} ms")

    print("\n第三步：H100 一个 SM，256 线程、128 寄存器、96 KiB 工作集")
    r = resident_blocks(256, 128, 96 * 1024)
    n = min(r.values())
    print(f"  各资源允许 {r} → 驻留 {n} 块，占用率 {occupancy(n, 256):.0%}")
    r2 = resident_blocks(256, 64, 32 * 1024)
    n2 = min(r2.values())
    print(f"  寄存器 64、共享内存 32 KiB：{r2} → {n2} 块，{occupancy(n2, 256):.0%}")
    need, have = inflight_bytes(3.35e12, 600e-9), inflight_have(16, 4)
    print(f"  在途字节：需要 {need / 1e3:.1f} KB，16 个 warp 提供 {have // 1024} KiB，覆盖 {have / need:.2f} 倍")
    need2, have2 = inflight_bytes(3.35e12, 1000e-9), inflight_have(16, 2)
    print(f"  延迟 1000 ns、每 warp 2 条：需要 {need2 / 1e3:.1f} KB，提供 {have2 // 1024} KiB，覆盖 {have2 / need2:.2f}，至少 {math.ceil(need2 / 1024)} 个 warp")

    print("\n第四步：矩阵指令与生产者 / 消费者 warp")
    print(f"  128×128×64 块要 {mma_count(128, 128, 64)} 条 16×8×16 指令，每条 {2 * 16 * 8 * 16} FLOPs")
    bw, pk = 3.35e12 / 132, 989.4e12 / 132
    print(f"  每 SM {bw / 1e9:.1f} GB/s、{pk / 1e12:.2f} TFLOP/s：一块复制 {32 * 1024 / bw * 1e6:.2f} μs，计算 {2 * 128 * 128 * 64 / pk * 1e6:.2f} μs")


if __name__ == "__main__":
    main()
