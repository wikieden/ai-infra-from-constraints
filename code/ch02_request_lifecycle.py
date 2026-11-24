"""第 2 章 · 一次生成请求的处理过程：权重加载一次，每个 token 都要再读一遍。

讲义：ai-infra/notes/lessons/02-request-lifecycle.md
运行：uv run python ch02_request_lifecycle.py
Qwen3-8B 的 16.38 GB、15.14 GB、144 KiB 取自 AII 第 1 章"本章数字速查"；实例负载、4 个输入 token、请求文字 8 KB 是编的。
"""

WEIGHT_LOAD_GB = 16.38  # 完整 BF16 权重：加载进显存的量
WEIGHT_STEP_GB = 15.14  # 一次前向的主要权重读取（嵌入层只读当前 token 那一行）
KV_KIB_PER_TOKEN = 144  # 每个上下文 token 的全模型 KV


def route(loads: dict[str, int]) -> str:
    """路由器：在各实例里挑排队请求最少的那个（并列时取第一个）。"""
    return min(loads, key=loads.get)


def simulate(prompt_len: int, out_len: int) -> list[tuple[str, int, int, int]]:
    """逐次前向：返回 (阶段, 送进几个 token, 读几份旧 KV, 结束时 KV 份数)。"""
    kv, steps = 0, []
    for i in range(out_len):  # 输出 N 个 token = N 次前向
        fed = prompt_len if i == 0 else 1  # prefill 送进全部输入，decode 只送 1 个
        read_old = kv  # 先读已有的整份 KV
        kv += fed  # 再追加这次送进来的 token 的 k、v
        steps.append(("prefill" if i == 0 else "decode", fed, read_old, kv))
    return steps


def weight_read_gb(out_len: int, step_gb: float = WEIGHT_STEP_GB) -> float:
    """权重读取总量 ≈ N × W_step（每次前向读一遍）。"""
    return out_len * step_gb


def kv_mib(tokens: int, kib_per_token: float = KV_KIB_PER_TOKEN) -> float:
    """KV 缓存大小（MiB）= 份数 × 每份 KiB ÷ 1024。"""
    return tokens * kib_per_token / 1024


def ratio(read_gb: float = 757.0, request_kb: float = 8.0) -> float:
    """内部读取量是请求文字的多少倍：先都换成字节（GB = 1e9，KB = 1e3）。"""
    return read_gb * 1e9 / (request_kb * 1e3)


def main() -> None:
    print("\n第一步：路由器挑更闲的实例（编的负载）")
    loads = {"A": 6, "B": 2}
    print(f"  各实例排队数 {loads} → 派给 {route(loads)}")

    print("\n第三步：输入 4 个 token，输出 3 个 token（编的）")
    for n, (phase, fed, read_old, kv) in enumerate(simulate(4, 3), 1):
        print(f"  第 {n} 次前向 {phase:7s} 送进 {fed} 个 token，读 {read_old} 份旧 KV，结束时 KV {kv} 份")
    print("  输出 N 个 token = N 次前向；结束时 KV = 输入长度 + N − 1 = 4 + 2 = 6")

    print("\n第四步：Qwen3-8B、BF16，2048 个 token 的上下文，输出 50 个 token")
    print(f"  权重从存储加载 1 次：{WEIGHT_LOAD_GB} GB（实例启动时）")
    print(f"  权重从显存读 50 次：50 × {WEIGHT_STEP_GB} ≈ {weight_read_gb(50):.0f} GB")
    print(f"  prefill 结束时 KV：2048 × {KV_KIB_PER_TOKEN} KiB = {kv_mib(2048):.0f} MiB；之后每步追加 {KV_KIB_PER_TOKEN} KiB")
    print(f"  请求文字约 8 KB（编的）→ 读权重总量是它的 {ratio():.2e} 倍")


if __name__ == "__main__":
    main()
