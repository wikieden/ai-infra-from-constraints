"""第 44 章 · 推理请求的执行过程与资源需求。

讲义：ai-infra/notes/lessons/44-inference-request.md
运行：uv run python ch44_inference_request.py
RTX PRO 6000 上 Qwen3-8B 的请求时间线、KV 容量、TTFT / ITL 与吞吐取自 AII 第 8 章 §8.1。
"""

import math

KiB, MiB, GiB = 2**10, 2**20, 2**30
K_TOKEN = 2 * 36 * 8 * 128 * 2  # 147,456 B


def request_time(queue: float, prefill: float, itl: float, G: int = 256) -> tuple[float, float]:
    """(TTFT, 完整请求时间) 秒。"""
    ttft = queue + prefill
    return ttft, ttft + (G - 1) * itl


def kv_alloc_mib(input_len: int, G: int = 256, block: int = 16) -> float:
    """生成完毕时按块分配的 KV（MiB）：输入 + G − 1 个回送的输出 token，按块上取整。"""
    tokens = math.ceil((input_len + G - 1) / block) * block
    return tokens * K_TOKEN / MiB


def fit(budget_mib: float, per_mib: float) -> int:
    return int(budget_mib // per_mib)


def main() -> None:
    print("\n第一步：一条 2K 短请求（排队 0.1 s、prefill 0.096 s、ITL 26.5 ms）")
    ttft, total = request_time(0.1, 0.096, 0.0265)
    print(f"  TTFT {ttft:.3f} s，完整 {total:.2f} s；插入一次 prefill 后 {total + 0.096:.2f} s（超 7 s）")

    print("\n第二步：KV 容量")
    print(f"  每 token {K_TOKEN:,} B = {K_TOKEN // KiB} KiB；2K {2048 * K_TOKEN / MiB:.0f} MiB，8K {8192 * K_TOKEN / MiB:.0f} MiB")
    for name, L in (("短对话 2K", 2048), ("长上下文 8K", 8192)):
        a = kv_alloc_mib(L)
        print(f"  {name}：生成完 {a:.0f} MiB，12 GiB 容纳 {fit(12 * 1024, a)} 条")
    print(f"  只按 prefill 算会放 {fit(12 * 1024, 288)} 条短请求，生成完要 {42 * 324:,} MiB")
    print(f"  16 条长请求只算输入就要 {16 * 1152 / 1024:.0f} GiB")

    print("\n第三步：吞吐")
    print(f"  batch 16：每轮 27.44 ms 出 16 个 → {16 / 0.02744:.0f} token/s；batch 1：{1 / 0.02626:.0f} token/s")


if __name__ == "__main__":
    main()
