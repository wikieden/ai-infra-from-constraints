"""第 47 章 · 压缩与卸载。

讲义：ai-infra/notes/lessons/47-compress-offload.md
运行：uv run python ch47_compress_offload.py
Q2_K 位宽、M2 Max 容量、KV 的 q8_0 / q4_0、压缩的净节省、权重卸载与预取、压缩搬移的解压门槛取自 AII 第 8 章 §8.4。
"""

MiB, GiB = 2**20, 2**30


def group_bits(n: int, code_bits: int, meta_bytes: int) -> float:
    return (n * code_bits / 8 + meta_bytes) * 8 / n


def kv_mib(base: float, fmt: str) -> float:
    return base * {"bf16": 64, "q8_0": 34, "q4_0": 18}[fmt] / 64


def net_saving_ms(saved_mib: float, conv_ms: float, bw: float = 1.16e12) -> float:
    return saved_mib * MiB / bw * 1e3 - conv_ms


def offload(n_layers: int = 9, layer_mib: float = 288, buffers: int = 1) -> tuple[float, float]:
    """(腾出 MiB, 净省 MiB)。"""
    freed = n_layers * layer_mib
    return freed, freed - buffers * layer_mib


def copy_ms(mib: float, bw: float) -> float:
    return mib * MiB / bw * 1e3


def min_decode_gbps(link: float, r: float) -> float:
    return link / (1 - r)


def main() -> None:
    print("\n第一步：权重量化")
    print(f"  Q2_K：{group_bits(256, 2, 20):.3f} bit/值，元数据占 {20 / 84:.0%}")
    free = 96e9 / GiB - 8
    print(f"  M2 Max 可用 {free:.2f} GiB：79.81 GiB 文件剩 {free - 79.81:.2f}，81.97 GiB 超 {81.97 - free:.2f}；放一条 8K（1.47 GiB）剩 {free - 79.81 - 1.47:.2f}")

    print("\n第二步：KV 压缩")
    for f in ("q8_0", "q4_0"):
        print(f"  8K {f}：{kv_mib(1152, f):.0f} MiB（省 {1152 - kv_mib(1152, f):.0f}）")
    print(f"  完整长请求 q8_0 {kv_mib(1188, 'q8_0'):.3f} MiB，16 条 {16 * kv_mib(1188, 'q8_0') / 1024:.2f} GiB")
    for c in (0.2, 0.8):
        print(f"  少读 540 MiB、转换 {c} ms：净 {net_saving_ms(540, c):+.3f} ms")
    print(f"  FP8 vs BF16 交点 {0.14 / 2e-5:.0f} token；2K {6.44 + 4.37e-5 * 2048:.2f} vs {6.58 + 2.37e-5 * 2048:.2f} ms，16K {6.44 + 4.37e-5 * 16384:.2f} vs {6.58 + 2.37e-5 * 16384:.2f} ms")

    print("\n第三步：卸载九份 FFN")
    for buf in (1, 2):
        freed, net = offload(buffers=buf)
        print(f"  {buf} 组预取缓冲：净省 {net:.0f} MiB → 多放 {int(net // 1152)} 条 8K 输入 / {int(net // 1188)} 条完整请求")
    for name, bw in (("PCIe Gen5", 64e9), ("NVLink-C2C", 450e9)):
        t = copy_ms(2592, bw)
        print(f"  {name}：每轮 {t:.1f} ms，255 轮 {255 * t / 1e3:.2f} s")
    print(f"  batch 4 上限 {4 / 0.0425:.0f} token/s")
    print(f"  压到 1/4 再传：96 MiB 直传 {copy_ms(96, 64e9):.2f} ms、压缩后 {copy_ms(24, 64e9):.2f} ms；解压要 > {min_decode_gbps(64, 0.25):.1f} GB/s")


if __name__ == "__main__":
    main()
