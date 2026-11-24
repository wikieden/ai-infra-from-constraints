"""第 16 章 · 多模态与实时交互：数据换形状，时间有死线。

讲义：ai-infra/notes/lessons/16-multimodal-realtime.md
运行：uv run python ch16_multimodal_realtime.py
Qwen3-VL-4B 的视觉 token 与特征宽度、Qwen-Image 的 50 步 × 2 路 CFG、20 ms 音频块的教学时间线取自 AII 第 3 章 §3.3。
"""


def vision_tokens(h: int, w: int, patch: int = 16, merge: int = 2) -> int:
    """视觉 token 数 n_v = H·W / (p² r²)。"""
    return h * w // (patch**2 * merge**2)


def vision_sizes(n_v: int, width: int = 2560, groups: int = 4, b: int = 2, kv_kib: int = 144) -> tuple[float, float]:
    """(完整 EC MiB, 视觉 token 的语言 KV MiB)。EC = n_v × (groups × width) × b。"""
    return n_v * groups * width * b / 2**20, n_v * kv_kib / 1024


def dit_forwards(steps: int, branches: int) -> int:
    """图像生成的 DiT 前向次数 = 去噪步数 × 引导分支数。"""
    return steps * branches


def chunk_bytes(rate: int = 24_000, ms: int = 20, channels: int = 1, bytes_per_sample: int = 2) -> int:
    """一块音频的字节数 = 采样率 × 时长 × 声道 × 每样本字节。"""
    return rate * ms // 1000 * channels * bytes_per_sample


def playback(buf_ms: int, prop_override: dict[int, int], n: int = 8, chunk: int = 20, proc: int = 13, prop: int = 5) -> tuple[int, int]:
    """逐块模拟：(首播时刻 ms, 总停顿 ms)。第 i 块在 20(i+1) ms 采集完，处理 + 发送 13 ms，传播默认 5 ms。"""
    arrive = [chunk * (i + 1) + proc + prop_override.get(i, prop) for i in range(n)]
    t = arrive[0] + buf_ms  # 第一块到达后再等 buf_ms 开播
    start, stall = t, 0
    for i in range(1, n):
        t += chunk  # 上一块播完，下一块的计划时刻
        if arrive[i] > t:  # 没到：停顿、顺延
            stall += arrive[i] - t
            t = arrive[i]
    return start, stall


def main() -> None:
    print("\n第一步：Qwen3-VL-4B，640×640")
    n_v = vision_tokens(640, 640)
    ec, kv = vision_sizes(n_v)
    print(f"  {n_v} 个视觉 token；EC {ec:.4f} MiB；语言 KV {kv:.2f} MiB")
    print(f"  四张图 + 400 文本 token：EC {4 * ec:.2f} MiB，输入 KV {4 * kv + 400 * 144 / 1024:.2f} MiB")

    print("\n第二步：Qwen-Image，50 步 × 2 路 CFG")
    print(f"  {dit_forwards(50, 2)} 次 DiT 前向")

    print("\n第三步：20 ms 一块音频")
    print(f"  每块 {chunk_bytes()} 字节")
    for buf in (40, 60):
        start, stall = playback(buf, {2: 50})
        print(f"  缓冲 {buf} ms：首播 {start} ms，停顿 {stall} ms")


if __name__ == "__main__":
    main()
