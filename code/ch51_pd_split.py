"""第 51 章 · Prefill–Decode 分离与资源配比。

讲义：ai-infra/notes/lessons/51-pd-split.md
运行：uv run python ch51_pd_split.py
A100 / H20 的阶段能力按峰值 50% 从 Roofline 推出；例 9.1、9.2 与 §9.2.4 的配比，取自 AII 第 9 章 §9.2。
"""

KiB, MiB, GiB = 2**10, 2**20, 2**30
K_GQA = 2 * 36 * 8 * 128 * 2            # Qwen3-8B 每 token 144 KiB
K_MLA = 61 * (512 + 64) * 2             # DeepSeek-V3 紧凑 MLA 每 token 70,272 B
D_W = 15_136_811_008                    # Qwen3-8B 一次 decode 读的权重字节
PREFILL_FLOP = 133.6e12                 # 8192 token prefill 的矩阵运算
DECODE_FLOP = 0.65e12                   # batch 32 一步 decode
CARDS = {"A100": (156e12, 1020e9), "H20": (74e12, 2048e9)}  # 有效算力、带宽（峰值 50%）


def v_kv(L: int, per_token: int = K_GQA) -> int:
    """V_KV = 2 n L h_kv d_h b（MLA 时是 n L (d_c + d_r) b）。"""
    return L * per_token


def handoff_ms(nbytes: float, bw: float = 25e9, alpha: float = 5e-6) -> float:
    return (nbytes / bw + alpha) * 1e3


def decode_step(card: str, batch: int = 32, ctx: int = 8704) -> float:
    """一步 decode（秒）：max(计算, 读权重 + 读 KV)。"""
    C, B = CARDS[card]
    return max(DECODE_FLOP / C, (D_W + batch * ctx * K_GQA) / B)


def stage_gpu_s(card: str, new_tokens: int = 8192, G: int = 1025, batch: int = 32, ctx: int | None = None):
    """(d_P, d_D)：每请求在该卡上的 P、D GPU 秒。"""
    C, _ = CARDS[card]
    d_p = PREFILL_FLOP * new_tokens / 8192 / C
    ctx = ctx if ctx is not None else 8192 + (G - 1) // 2
    d_d = (G - 1) * decode_step(card, batch, ctx) / batch
    return d_p, d_d


def pd_rate(p_cards: dict, d_cards: dict, d: dict, link: float = 25e9, state: float = 8192 * K_GQA):
    """μ_PD = min(μ_P, μ_D, B_net / V_KV)。d[card] = (d_P, d_D)。"""
    mp = sum(n / d[c][0] for c, n in p_cards.items())
    md = sum(n / d[c][1] for c, n in d_cards.items())
    return min(mp, md, link / state), mp, md


def colocated(cards: dict, d: dict) -> float:
    return sum(n / sum(d[c]) for c, n in cards.items())


def piggyback_tokens(card: str, batch: int = 32, ctx: int = 8704) -> float:
    """一步 decode 里空闲算力能捎带多少个 prefill token。"""
    C, _ = CARDS[card]
    idle = decode_step(card, batch, ctx) - DECODE_FLOP / C
    return idle * C / (PREFILL_FLOP / 8192)


def main() -> None:
    print("\n第一步：交接多少字节、多久")
    for name, k in (("Qwen3-8B GQA", K_GQA), ("DeepSeek-V3 MLA", K_MLA)):
        v = v_kv(8192, k)
        print(f"  {name}：每 token {k / KiB:.1f} KiB，8K {v / MiB:.0f} MiB；25 GB/s {handoff_ms(v):.1f} ms，50 GB/s {handoff_ms(v, 50e9):.1f} ms")
    v = v_kv(8192)
    print(f"  双缓冲两端共 {2 * v / GiB:.2f} GiB；每秒 8 / 16 次交接要 {8 * v / 1e9:.1f} / {16 * v / 1e9:.1f} GB/s")

    print("\n第二步：两种卡的阶段能力")
    d = {c: stage_gpu_s(c) for c in CARDS}
    for c, (dp, dd) in d.items():
        print(f"  {c}：prefill {dp:.3f} s（{8192 / dp:.0f} token/s），decode 一步 {decode_step(c) * 1e3:.1f} ms（{32 / decode_step(c):.0f} 次/s），每请求 D {dd:.3f} GPU 秒")
    print(f"  batch 32 生成到最后 KV {32 * 9216 * K_GQA / 1e9:.1f} GB")

    print("\n第三步：同构八张 A100")
    print(f"  共置 {colocated({'A100': 8}, d):.2f} 请求/s；3P/5D → {pd_rate({'A100': 3}, {'A100': 5}, d)[0]:.2f}")
    pb = piggyback_tokens("A100")
    print(f"  分块捎带：每步空闲可做 {pb:.0f} 个 prefill token，8192 分 {-(-8192 // int(pb))} 块，首 token 约 {-(-8192 // int(pb)) * decode_step('A100'):.2f} s；理想上限 {8 / d['A100'][1]:.2f} 请求/s")

    print("\n第四步：异构 4×A100 + 4×H20")
    print(f"  共置 {colocated({'A100': 4, 'H20': 4}, d):.2f} 请求/s")
    mu, mp, md = pd_rate({"A100": 4}, {"H20": 4}, d)
    print(f"  A100 做 P、H20 做 D：P {mp:.2f}、D {md:.2f}、链路 {25e9 / v:.1f} → {mu:.2f} 请求/s")
    print(f"  角色对调：P 只剩 {pd_rate({'H20': 4}, {'A100': 4}, d)[1]:.2f}")
    print(f"  MLA 状态链路项 {25e9 / v_kv(8192, K_MLA):.1f}；链路成为瓶颈的门槛 GQA {md * v / 1e9:.1f}、MLA {md * v_kv(8192, K_MLA) / 1e9:.1f} GB/s")
    h = piggyback_tokens("H20")
    left = 8192 - 32 * h
    rate = 4 / d["A100"][1] + 4 / (d["H20"][1] + left / 8192 * d["H20"][0])
    print(f"  H20 每步只能捎带 {h:.0f} 个；八卡理想捎带共置约 {rate:.2f} 请求/s（原文 4.17）")
    grid = [[pd_rate({"A100": a, "H20": b}, {"A100": 4 - a, "H20": 4 - b}, d)[0] for b in range(5)] for a in range(5)]
    best = max((grid[a][b], a, b) for a in range(5) for b in range(5))
    print(f"  25 种配比里最好：{best[1]} 张 A100 + {best[2]} 张 H20 做 P → {best[0]:.2f}")

    print("\n第五步：请求变了，配比跟着变")
    dh = {"A100": (0.2378, d["A100"][1]), "H20": (0.2378 * 156 / 74, d["H20"][1])}
    print(f"  命中 6144 前缀（A100 prefill 0.238 s）：2 张 A100 做 P → {pd_rate({'A100': 2}, {'A100': 2, 'H20': 4}, dh)[0]:.2f}")
    ds = {c: stage_gpu_s(c, G=129) for c in CARDS}
    print(f"  输出 129：H20 D 能力 {1 / ds['H20'][1]:.2f}/卡；4A100+3H20 做 P → {pd_rate({'A100': 4, 'H20': 3}, {'H20': 1}, ds)[0]:.2f}，共置 {colocated({'A100': 4, 'H20': 4}, ds):.2f}")


if __name__ == "__main__":
    main()
