"""第 4 章 · 系统设计的关键数字：容量、算力、带宽、延迟。

讲义：ai-infra/notes/lessons/04-key-numbers.md
运行：uv run python ch04_key_numbers.py
数字取自 AII 第 1 章 §1.2：Jeff Dean 2009 年的 12 个操作耗时、三张 GPU 的速查表、DeepSeek-R1-Distill-Llama-70B 的参数数与量化结果。
"""

DEAN_NS = {
    "访问一级缓存": 0.5,
    "分支预测错误后的恢复": 5,
    "访问二级缓存": 7,
    "对互斥锁加锁或解锁": 25,
    "访问主存": 100,
    "用 Zippy 压缩 1K 字节": 3_000,
    "经 1 Gbit/s 网络发送 2K 字节": 20_000,
    "从主存连续读取 1 MB": 250_000,
    "同一数据中心内往返": 500_000,
    "磁盘寻道": 10_000_000,
    "从磁盘连续读取 1 MB": 20_000_000,
    "数据包从加州到荷兰再返回": 150_000_000,
}

# (显存 GB, 带宽 TB/s, BF16 矩阵峰值 TFLOP/s)
GPUS = {
    "RTX 4090": (24, 1.008, 165.2),
    "A100 80GB SXM": (80, 2.039, 312),
    "H100 SXM": (80, 3.35, 989.4),
}

GIB = 2**30 / 1e9  # 1 GiB 合多少 GB


def seek_example(seeks: int = 20, seek_ms: float = 10, cpu_ms: float = 1) -> float:
    """串行寻道 + CPU 计算的总时间（ms）。"""
    return seeks * seek_ms + cpu_ms


def amdahl(f: float, s: float) -> float:
    """Amdahl 定律：比例 f 的部分加速 s 倍，总加速比 1 / ((1-f) + f/s)。"""
    return 1 / ((1 - f) + f / s)


def read_ms(nbytes: float, gpu: str) -> float:
    """按带宽峰值读 nbytes 字节要几毫秒：R / β。"""
    return nbytes / (GPUS[gpu][1] * 1e12) * 1e3


def weight_gb(n_params: int, bytes_per: float) -> float:
    """权重大小 M_W = N × b_W（十进制 GB）。"""
    return n_params * bytes_per / 1e9


def budget_gb(weights_gb: float, kv_gib: float, work_gib: float) -> float:
    """容量预算：权重（GB）+ KV（GiB）+ 工作区（GiB），统一换成 GB 再相加。"""
    return weights_gb + kv_gib * GIB + work_gib * GIB


def gbit_to_gbyte(gbit_per_s: float) -> float:
    """网速换算：Gbit/s ÷ 8 = GB/s。"""
    return gbit_per_s / 8


def main() -> None:
    print("\n第一步：数量级（Jeff Dean 2009）")
    print(f"  磁盘寻道 ÷ 访问主存 = {DEAN_NS['磁盘寻道'] / DEAN_NS['访问主存']:.0f} 倍")

    print("\n第二步：串行寻道的例子")
    print(f"  原来 {seek_example():.1f} ms；CPU 快一倍 {seek_example(cpu_ms=0.5):.1f} ms；寻道减半 {seek_example(seeks=10):.1f} ms")
    print("  AI 编程工具让编码快 5 倍：")
    for f in (0.2, 0.3, 0.7, 0.8):
        print(f"    编码占 {f:.0%} → 总加速比 {amdahl(f, 5):.2f}")

    print("\n第三步：按带宽峰值读 1 GB")
    for gpu in GPUS:
        print(f"  {gpu:14s} {read_ms(1e9, gpu):.2f} ms")

    print("\n第四步：DeepSeek-R1-Distill-Llama-70B")
    w = weight_gb(70_553_706_496, 2)
    print(f"  BF16 权重 {w:.2f} GB = {w / GIB:.2f} GiB；均分两张卡每张 {w / 2:.2f} GB，剩 {80 - w / 2:.2f} GB")
    print(f"  8 比特方案 73.73 GB，一张 H100 剩 {80 - 73.73:.2f} GB")
    print(f"  容量预算 73.73 + 2.5 GiB + 2 GiB = {budget_gb(73.73, 2.5, 2):.2f} GB ≤ 80 GB")

    print("\n第五步：网卡 400 Gbit/s")
    bw = gbit_to_gbyte(400)
    print(f"  = {bw:.0f} GB/s，按线速传 1 GB 要 {1 / bw * 1e3:.0f} ms")


if __name__ == "__main__":
    main()
