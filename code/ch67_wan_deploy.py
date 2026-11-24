"""第 67 章 · 广域网、无线与部署选择。

讲义：ai-infra/notes/lessons/67-wan-deploy.md
运行：uv run python ch67_wan_deploy.py
连接复用、窗口与有效吞吐、丢包下的 TCP 速率、双路径分流，以及剩余二十轮截图任务的端 / 边 / 云选择，取自 AII 第 12 章 §12.3–§12.5。
"""

import math


def useful_bw(path_bps: float, window_bytes: float, rtt: float) -> float:
    """B_useful = min(B_path, W / RTT)，返回 bit/s。"""
    return min(path_bps, window_bytes * 8 / rtt)


def mathis_bps(p: float, rtt: float = 0.2, mss: int = 1460, C: float = 1.0) -> float:
    """丢包率 p 下 TCP 的稳态速率 ≈ MSS·C / (RTT √p)。"""
    return mss * 8 * C / (rtt * math.sqrt(p))


def split(B1: float, B2: float, total: float, egress: float | None = None) -> float:
    """两条路径按 x = B1/(B1+B2) 分，同时完成；有共同出口时再取出口下界。"""
    t = total / (B1 + B2)
    return max(t, total / egress) if egress else t


SITES = {   # 准备 s、每轮模型 s、上行 Mbit/s、RTT s、功率 W（端侧按每 token 能耗算）
    "端侧手机": (0.0, 2.90, None, 0.0, None),
    "附近工作站": (0.27, 0.137, 80, 0.02, 600),
    "云地域 H100": (0.14, 0.073, 6.4, 0.20, 700),
}


def remaining(site: str, rounds: int = 20, shot_mb: float = 0.8, terminal: float = 0.3, uplink: float | None = None) -> float:
    prep, model, up, rtt, _ = SITES[site]
    up = uplink if uplink is not None else up
    upload = shot_mb * 8 / up if up else 0.0
    return prep + rounds * (upload + terminal + model + rtt)


def energy_j(site: str, rounds: int = 20) -> tuple[float, float]:
    prep, model, _, _, watt = SITES[site]
    if watt is None:
        return rounds * 45 * 0.576, rounds * 45 * 0.756
    e = watt * (prep + rounds * model)
    return e, e


def main() -> None:
    print("\n第一步：连接与窗口")
    print(f"  30 轮每轮重新建链（2 RTT × 0.2 s）{30 * 0.4:.0f} s vs 只建一次 0.4 s")
    print(f"  20 Mbit/s × 100 ms 要 {20e6 * 0.1 / 8 / 1e3:.0f} KB 在途；窗口 64 KB 只能 {useful_bw(20e6, 64000, 0.1) / 1e6:.2f} Mbit/s（发 25.6 ms 等 100 ms）")
    print(f"  慢启动 10 段起步翻倍：前四轮共 {14600 * 15:,} B")
    print(f"  每包 1168 B 载荷 + 60 B 头 + 92 B ACK：三万包多 {30000 * 152 / 1e6:.1f} MB")

    print("\n第二步：丢包与多流")
    print(f"  丢包 14%、RTT 200 ms：C = 1 → {mathis_bps(0.14) / 1e6:.3f} Mbit/s，C = 1.22 → {mathis_bps(0.14, C=1.22) / 1e6:.3f}；355 KB 需 {354640 * 8 / mathis_bps(0.14):.1f} s")
    print("  FIFO：图片 5 s、音频 6 s；音频优先：音频 3 s、图片 6 s")
    print("  跨地域语音：新建连接 1.19 vs 0.30 s，复用并调优都约 0.24 s；模型 30 → 3 ms 省 27 ms，RTT 200 → 20 ms 省 180 ms")

    print("\n第三步：无线与双路径")
    print(f"  数据帧 208 → 40 μs，固定 94 μs：整次交换 {(208 + 94):.0f} → {(40 + 94):.0f} μs（−{(1 - 134 / 302) * 100:.0f}%）")
    print(f"  30 MB 分 20 + 10 Mbit/s：{split(20e6, 10e6, 240e6):.0f} s（只用快路 12 s）；共同出口 24 Mbit/s → {split(20e6, 10e6, 240e6, 24e6):.0f} s")

    print("\n第四步：剩余 20 轮、45 s 期限")
    for s in SITES:
        lo, hi = energy_j(s)
        print(f"  {s}：{remaining(s):.1f} s，能耗 {lo:.0f}{'–%.0f' % hi if hi != lo else ''} J")
    base = remaining("云地域 H100", uplink=1e12)
    print(f"  云端不含上传 {base:.1f} s；期限要上行 ≥ {128 / (45 - base):.1f} Mbit/s；4 / 3.5 Mbit/s → {remaining('云地域 H100', uplink=4):.1f} / {remaining('云地域 H100', uplink=3.5):.1f} s")
    mix = base + 10 * 0.8 * 8 / 6 + 10 * 0.8 * 8 / 10
    print(f"  十轮 6 + 十轮 10 Mbit/s：{mix:.1f} s；全程 8 Mbit/s：{remaining('云地域 H100', uplink=8):.1f} s")
    per = remaining("云地域 H100", rounds=1) - 0.14
    print(f"  断连恢复 1 s：保留九轮重做一轮 {remaining('云地域 H100') + 1 + per:.1f} s，丢十轮 {remaining('云地域 H100') + 1 + 10 * per:.1f} s")
    print(f"  模型 8 s + 其他 2 s，模型快 10 倍 → {8 / 10 + 2:.1f} s")


if __name__ == "__main__":
    main()
