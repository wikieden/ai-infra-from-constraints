"""第 31 章 · 从单卡到多卡：容量、单步时间与六种切分。

讲义：ai-infra/notes/lessons/31-multi-gpu.md
运行：uv run python ch31_multi_gpu.py
HGX H100 八卡与 Qwen3-235B-A22B 的容量、专家读取时间取自 AII 第 6 章 §6.1。
"""

KiB, MiB, GiB = 2**10, 2**20, 2**30
HBM, PEAK = 3350e9, 989.4e12  # H100 SXM


def kv_per_token(layers: int, kv_heads: int, head_dim: int, b: int = 2) -> int:
    """每 token 的 KV 字节：K、V 各一份 × 层 × KV 头 × 头维 × 字节。"""
    return 2 * layers * kv_heads * head_dim * b


def per_card_gb(weights_gb: float, kv_gb: float, cards: int, kv_share: float, work_gb: float = 2 * GiB / 1e9,
                weights_per_card_gb: float | None = None) -> float:
    """每卡内存：权重分片 + KV 份额 + 每卡一份工作区。"""
    w = weights_per_card_gb if weights_per_card_gb is not None else weights_gb / cards
    return w + kv_gb * kv_share + work_gb


def op_time(flops: float, nbytes: float, peak: float = PEAK, bw: float = HBM) -> tuple[float, float, float]:
    """式 (6-2)：(计算 s, 读取 s, max)。"""
    tc, tm = flops / peak, nbytes / bw
    return tc, tm, max(tc, tm)


def main() -> None:
    print("\n第一步：Qwen3-235B-A22B，8192 个上下文 token")
    kv = kv_per_token(94, 4, 128)
    kv_gb = kv * 8192 / 1e9
    print(f"  每 token KV {kv // KiB} KiB，8192 个 {kv_gb:.2f} GB；单卡共需 {470.2 + kv_gb + 2 * GiB / 1e9:.1f} GB")
    pc = per_card_gb(470.2, kv_gb, 8, 1 / 4, weights_per_card_gb=58.96)
    print(f"  八卡每卡 {pc:.2f} GB（KV 每卡存 1/4：{kv_gb / 4:.2f} GB）；空余可再放 {int((80 - 58.96 - 2 * GiB / 1e9) / (kv_gb / 4))} 个会话")

    print("\n第二步：一层专家，单 token 选 8 个")
    w = 8 * 3 * 4096 * 1536 * 2
    tc, tm, _ = op_time(2 * 8 * 3 * 4096 * 1536, w)
    print(f"  读 {w // MiB} MiB {tm * 1e6:.1f} μs，算 {tc * 1e6:.2f} μs，MFU 上限 {tc / tm:.1%}")
    print(f"  全模型一步：单卡 {(43.1e9 + kv_gb * 1e9) / HBM * 1e3:.1f} ms；八卡每卡 {43.1 / 8 + kv_gb / 4:.2f} GB，{(43.1e9 / 8 + kv_gb * 1e9 / 4) / HBM * 1e3:.2f} ms")
    tc, tm, _ = op_time(2 * 8192 * 8 * 3 * 4096 * 1536, 128 * 3 * 4096 * 1536 * 2)
    print(f"  prefill 8192 token：读 4.5 GiB {tm * 1e3:.2f} ms，算 {tc * 1e3:.2f} ms → 计算主导")
    print(f"  隐藏张量：单 token {4096 * 2 // KiB} KiB，8192 token {8192 * 4096 * 2 // MiB} MiB")


if __name__ == "__main__":
    main()
