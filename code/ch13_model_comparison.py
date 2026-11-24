"""第 13 章 · 整个模型的资源需求与比较：存多少、算多少、能放几条请求。

讲义：ai-infra/notes/lessons/13-model-comparison.md
运行：uv run python ch13_model_comparison.py
表 2-B 参数分解、表 2-E 长上下文交互、RTX 4090 与 H100 的容量算例取自 AII 第 2 章 §2.6。
"""

GIB = 2**30

# 表 2-B（十亿）：嵌入与输出头、注意力投影、稠密/共享 FFN、路由专家、Engram、其他
PARAMS = {
    "V4.1 Flash": (1.324, 5.126, 1.416, 543.582, 196.929, 0.118),
    "Qwen3-8B": (1.245, 1.510, 5.436, 0.0, 0.0, 0.0),
    "Qwen3.6": (1.017, 1.283, 0.126, 32.212, 0.0, 0.022),
    "V4-Flash": (1.059, 5.085, 1.082, 277.025, 0.0, 0.081),
    "Kimi K3": (2.349, 36.180, 12.882, 2722.741, 0.0, 5.333),
}

# 表 2-E：(8K 交互 GFLOPs, 1M 交互 GFLOPs, 8K 状态 GiB, 1M 状态 GiB)
LONG = {
    "V4.1 Flash": (3.66, 25.23, 0.029, 3.138),
    "Qwen3-8B": (4.83, 618.48, 1.125, 144.000),
    "Qwen3.6": (1.34, 171.80, 0.217, 20.060),
    "V4-Flash": (3.00, 113.80, 0.069, 6.735),
    "Kimi K3": (41.08, 5257.04, 0.634, 27.423),
}

QWEN8B_W = 16_381_470_720  # Qwen3-8B BF16 权重字节


def expert_share(name: str) -> float:
    """路由专家占全部参数的比例。"""
    p = PARAMS[name]
    return p[3] / sum(p)


def bf16_gb(name: str) -> float:
    """统一 BF16 时的完整权重（GB）= 总参数（十亿）× 2。"""
    return sum(PARAMS[name]) * 2


def b_max(device: float, weight: float, workspace: float, per_request: float) -> int:
    """一张卡能同时放几条请求：(容量 - 权重 - 工作区) ÷ 每请求状态，向下取整。"""
    return int((device - weight - workspace) // per_request)


def n_max(cap: float, workspace: float, B: int, kv: float, bytes_per_param: float) -> float:
    """反推权重上限（参数个数）：(C - M0 - B·K) / b_W。"""
    return (cap - workspace - B * kv) / bytes_per_param


def growth() -> dict[str, float]:
    """8K → 1M 时交互运算涨了几倍。"""
    return {k: v[1] / v[0] for k, v in LONG.items()}


def main() -> None:
    print("\n第一步：路由专家占参数的比例")
    for name in ("Qwen3.6", "V4-Flash", "Kimi K3"):
        print(f"  {name:10s} {expert_share(name):.1%}")

    print("\n第二步：统一 BF16 的完整权重")
    for name in PARAMS:
        print(f"  {name:10s} {bf16_gb(name):8.1f} GB")

    print("\n第三步：RTX 4090（24 GB），Qwen3-8B BF16，工作区 2 GiB")
    for ctx, kv in ((4096, 0.5625), (8192, 1.125), (16384, 2.25)):
        print(f"  {ctx // 1024}K 上下文（KV {kv} GiB）→ {b_max(24e9, QWEN8B_W, 2 * GIB, kv * GIB)} 条")
    print("  70B 在 H100（80 GB），工作区 2 GiB，8K 请求 KV 2.5 GiB：")
    print(f"    8 比特 73.726 GB → {b_max(80e9, 73.726e9, 2 * GIB, 2.5 * GIB)} 条")
    print(f"    4 比特 39.500 GB → {b_max(80e9, 39.5e9, 2 * GIB, 2.5 * GIB)} 条；32K（KV 10 GiB）→ {b_max(80e9, 39.5e9, 2 * GIB, 10 * GIB)} 条")
    print(f"  反推：4090 放 4 条 8K 请求后，BF16 最多 {n_max(24e9, 2 * GIB, 4, 1.125 * GIB, 2) / 1e9:.2f} 十亿参数")

    print("\n第四步：8K → 1M，交互运算涨了几倍")
    for name, g in growth().items():
        print(f"  {name:10s} {g:6.1f} 倍")


if __name__ == "__main__":
    main()
