"""第 66 章 · 跨设备分工与会话迁移。

讲义：ai-infra/notes/lessons/66-cross-device.md
运行：uv run python ch66_cross_device.py
端侧编码后上传特征、会话迁移的回本轮数、跨设备张量并行的启动开销，取自 AII 第 12 章 §12.2。
"""

MiB = 2**20


def feature_bytes(tokens: int = 400, width: int = 2560, groups: int = 4, b: int = 2) -> int:
    """最终投影 + 三组 DeepStack，每组同宽。"""
    return tokens * width * groups * b


def encode_delta(T_el: float, T_er: float, Tx: float, S_E: float, S_I: float, Bu: float) -> float:
    """ΔT = T_E,l − T_E,r + T_x + (S_E − S_I)/B_u；< 0 时端侧编码更快。"""
    return T_el - T_er + Tx + (S_E - S_I) / Bu


def migrate_cost(state_bytes: float, bw: float, restore: float) -> float:
    return state_bytes * 8 / bw + restore


def gain(N: int, dt: float, M: float) -> float:
    """G(N) = N Δt − M。"""
    return N * dt - M


def breakeven_rounds(dt: float, M: float) -> int:
    n = 1
    while gain(n, dt, M) <= 0:
        n += 1
    return n


def tp_startup_ms(layers: int = 36, alpha_us: float = 2) -> float:
    """每层两次归约 × 每次两阶段启动 = 144α。"""
    return layers * 2 * 2 * alpha_us / 1000


def main() -> None:
    print("\n第一步：上传图片还是上传特征")
    S_E, S_I = feature_bytes(), 0.8e6
    print(f"  特征 400 × 10240 × 2 = {S_E:,} B（{S_E / MiB:.1f} MiB），截图 0.8 MB；6.4 Mbit/s 上传 {S_E * 8 / 6.4e6:.1f} s vs {S_I * 8 / 6.4e6:.1f} s")
    le, re = 1.31e12 / 165.2e12, 1.31e12 / 989.4e12
    print(f"  编码 1.31 TFLOPs：4090 {le * 1e3:.1f} ms、H100 {re * 1e3:.1f} ms；ΔT = {encode_delta(le, re, 0, S_E, S_I, 0.8e6):.2f} s（端侧更慢）")
    print(f"  同机房 50 GB/s：传输差 {(S_E - S_I) / 50e9 * 1e3:.2f} ms")
    print(f"  400 个视觉 token 的语言 KV：{400 * 147456 / MiB:.1f} MiB")

    print("\n第二步：会话迁移")
    M = migrate_cost(64 * MiB, 80e6, 1)
    print(f"  64 MiB 经 80 Mbit/s + 恢复 1 s：M = {M:.1f} s")
    for n in (10, 20, 40):
        print(f"  剩 {n} 轮、每轮快 0.4 s：{gain(n, 0.4, M):+.1f} s")
    print(f"  回本 {breakeven_rounds(0.4, M)} 轮；恢复再多 1 s → {breakeven_rounds(0.4, M + 1)} 轮")

    print("\n第三步：模型内部切分")
    print(f"  TP = 2、36 层：144α；α = 2 μs → {tp_startup_ms():.2f} ms；Wi-Fi 短帧交换 134 μs → {tp_startup_ms(alpha_us=134):.1f} ms（RTX PRO 6000 一步读取下界 9.12 ms）")


if __name__ == "__main__":
    main()
