"""第 3 章 · 从存储程序到数据流执行：存储层次就像大楼里的厕所。

讲义：ai-infra/notes/lessons/03-dataflow-hierarchy.md
运行：uv run python ch03_dataflow_hierarchy.py
H100 SXM 的容量取自公开规格（与《CUDA 从线程到 Tensor Core》第 5 章一致）；n = 4 的矩阵和 MoE 的 64 个 token、8 个专家是编的。
"""

# (层, 容量字节, 范围) —— H100 SXM
LEVELS = [
    ("寄存器", 256 * 1024, "每 SM"),
    ("shared memory + L1", 256 * 1024, "每 SM"),
    ("L2 缓存", 50 * 10**6, "全卡"),
    ("显存 HBM", 80 * 10**9, "全卡"),
]


def far_reads(n: int, t: int = 1) -> int:
    """n×n 矩阵乘法从远处（显存）读多少次：不分块 2n³，分块 2n³/t。"""
    return 2 * n**3 // t


def simulate_tiled(n: int, t: int) -> int:
    """一块一块数：每个 C 块沿 K 方向滑 n/t 次，每次搬 A、B 各一个 t×t 块。"""
    reads = 0
    for _ in range(n // t):  # C 的块行
        for _ in range(n // t):  # C 的块列
            for _ in range(n // t):  # 沿 K 方向滑动
                reads += 2 * t * t  # A、B 各搬一个 t×t 块进近处
    return reads


def moe_avg(tokens: int, experts: int, k: int) -> float:
    """MoE 平均每个专家分到几个 token：tokens × k ÷ 专家数。"""
    return tokens * k / experts


def main() -> None:
    print("\n第一步：H100 SXM 的存储层次（越往下越大、越远）")
    for name, size, scope in LEVELS:
        print(f"  {name:20s} {scope} {size:>15,} 字节")

    print("\n第二步：n = 4 的矩阵乘法，从显存读多少次")
    print(f"  不分块 2n³ = {far_reads(4)}")
    for t in (2, 4):
        print(f"  分块 t = {t}：2n³/t = {far_reads(4, t)}，逐块数一遍 = {simulate_tiled(4, t)}")

    print("\n第三步：MoE 平均每个专家分到几个 token（编的：64 个 token、8 个专家、每个选 2 个）")
    print(f"  64 × 2 / 8 = {moe_avg(64, 8, 2):.0f}")


if __name__ == "__main__":
    main()
