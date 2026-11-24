"""第 65 章 · 从模型运行时间到完整交互时间。

讲义：ai-infra/notes/lessons/65-interaction-time.md
运行：uv run python ch65_interaction_time.py
图片精修的串行时间与压缩、分块重叠，实时语音的逐块递推与播放缓冲，Computer Use 的轮次，手机端侧的带宽、容量与能耗，取自 AII 第 12 章 §12.1。
"""

MB = 1e6


def serial(Su: float, Bu: float, R: float, Tc: float, Sd: float, Bd: float) -> float:
    """T = S_u/B_u + R + T_c + S_d/B_d（字节、字节/秒、秒）。"""
    return Su / Bu + R + Tc + Sd / Bd


def compress_gain(saved_bytes: float, Bu: float, Te: float) -> float:
    """ΔT = (S_u − S'_u)/B_u − T_e。"""
    return saved_bytes / Bu - Te


def chunked(up: list[float], proc: list[float], down: list[float], R: float) -> float:
    """逐块上传、处理、回传，上传 / 处理 / 回传各自是一个串行资源。"""
    u = c = d = 0.0
    for a, b, e in zip(up, proc, down):
        u += a
        c = max(c, u + R / 2) + b
        d = max(d, c) + e
    return d + R / 2


def audio(n: int = 6, chunk: float = 20, proc: float = 12, send: float = 1, prop=None, J: float = 40):
    """c_i = max(a_i, c_{i−1}) + m_i；u_i = max(c_i, u_{i−1}) + s_i；r_i = u_i + d_i；p_i = max(r_i, p_{i−1} + τ)。返回 (到达, 播放, 停顿 ms)。"""
    prop = prop or [5] * n
    c = u = 0.0
    arr, play = [], []
    for i in range(n):
        a = (i + 1) * chunk
        c = max(a, c) + proc
        u = max(c, u) + send
        arr.append(u + prop[i])
    p = arr[0] + J
    play.append(p)
    for r in arr[1:]:
        p = max(r, p + chunk)
        play.append(p)
    stall = play[-1] - (play[0] + (n - 1) * chunk)
    return arr, play, stall


def drain_s(buffer_ms: float, need_bps: float, have_bps: float) -> float:
    return buffer_ms / 1000 * need_bps / (need_bps - have_bps)


def rounds_time(n: int, upload_mb: float, uplink_mbps: float, other: float, extra: float = 0) -> float:
    return n * (upload_mb * 8 / uplink_mbps + extra + other)


def phone_decode(read_gb: float, bw: float = 84.8) -> float:
    """每步读取 / 带宽 → token/s。"""
    return bw / read_gb


def phone_context(mem_gb: float, weights_gb: float = 16.38 * 18 / 64, reserve: float = 4) -> tuple[float, int, int]:
    left = mem_gb - reserve - weights_gb
    return left, max(0, int(left // 1.208)), max(0, int(left * 1e9 // 147456))


def main() -> None:
    print("\n第一步：图片精修（30 MB 上传、5 MB 返回）")
    t = serial(30 * MB, 20e6 / 8, 0.1, 0.3, 5 * MB, 100e6 / 8)
    print(f"  串行 {t:.1f} s；模型快 10 倍 {serial(30 * MB, 2.5e6, 0.1, 0.03, 5 * MB, 12.5e6):.2f} s；上行 100 Mbit/s {serial(30 * MB, 12.5e6, 0.1, 0.3, 5 * MB, 12.5e6):.1f} s")
    print(f"  输入压一半、编解码 0.15 s：{serial(15 * MB, 2.5e6, 0.1, 0.3, 5 * MB, 12.5e6) + 0.15:.2f} s；收益在 20 / 100 / 800 Mbit/s 时 {compress_gain(15 * MB, 2.5e6, 0.15):.2f} / {compress_gain(15 * MB, 12.5e6, 0.15):.2f} / {compress_gain(15 * MB, 100e6, 0.15):.2f} s")
    print(f"  分三块：{chunked([4, 4, 4], [0.1] * 3, [0.08, 0.16, 0.16], 0.1):.2f} s")

    print("\n第二步：实时语音（24 kHz 16-bit 单声道，每块 20 ms = 960 B）")
    arr, play, st = audio()
    print(f"  到达 {[round(x) for x in arr[:3]]} ms，播放 {[round(x) for x in play[:3]]} ms")
    arr, play, st = audio(prop=[5, 5, 50, 5, 5, 5])
    print(f"  第三块传播 50 ms：到达 {arr[2]:.0f} ms，计划 118 ms → 停顿 {st:.0f} ms；缓冲 45 ms → 停顿 {audio(prop=[5, 5, 50, 5, 5, 5], J=45)[2]:.0f}")
    print(f"  256 kbit/s 只来 130 kbit/s：60 ms 缓冲 {drain_s(60, 256e3, 130e3):.2f} s 耗尽")

    print("\n第三步：Computer Use 30 轮")
    print(f"  每轮 1 + 0.3 + 2.0 + 0.2 = 3.5 s，共 {rounds_time(30, 0.8, 6.4, 2.5):.0f} s；压到 0.2 MB +30 ms：{rounds_time(1, 0.2, 6.4, 2.5, 0.03):.2f} s/轮、{rounds_time(30, 0.2, 6.4, 2.5, 0.03):.1f} s；轮数不超过 {int(105 / 2.78 - 1e-9)} 才划算")
    print(f"  光纤 10,000 km 往返至少 {2 * 10_000e3 / 2e8 * 1e3:.0f} ms")

    print("\n第四步：手机端侧（LPDDR5X 4 × x16 × 10.6 Gbit/s = 84.8 GB/s）")
    print(f"  BF16 只读权重 {15.14 / 84.8:.3f} s/步（{phone_decode(15.14):.1f} token/s），含 KV {phone_decode(16.345):.1f}；q4_0 {15.14 * 18 / 64 + 1.21:.2f} GB → {phone_decode(15.14 * 18 / 64 + 1.21):.1f} token/s")
    for m in (6, 12, 16, 24):
        left, n8k, single = phone_context(m)
        print(f"  {m} GB：剩 {left:.2f} GB → {n8k} 条 8K，单条 {single:,} token")
    print(f"  14.8 token/s × 0.576 J = {14.8 * 0.576:.1f} W；2.9 s 每轮 ≈ {2.9 * 15.5:.0f} 个 token")
    print(f"  24 GB 全参数 16 B/参数 → {24 / 16:.1f}B 参数；LoRA 只存 BF16 基础权重 → 约 {24 / 2:.0f}B")


if __name__ == "__main__":
    main()
