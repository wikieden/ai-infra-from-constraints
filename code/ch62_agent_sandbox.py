"""第 62 章 · Agent 任务的资源需求与运行环境。

讲义：ai-infra/notes/lessons/62-agent-sandbox.md
运行：uv run python ch62_agent_sandbox.py
代码修复平台（每秒 10 项任务、三轮、模型 9 s、工具 1 s、2 GiB microVM、一台 m5d.metal）的资源账、
模板加载方式、暂停与每轮重建、提前准备的期望等待，取自 AII 第 11 章 §11.1–§11.2。
"""

MiB, GiB = 2**20, 2**30
LAM = 10            # 项/s
ROUNDS, MODEL_S, TOOL_S, ENV_GIB = 3, 9.0, 1.0, 2
CORES, MEM_GIB, SLOTS = 48, 384e9 / 2**30, 320   # 384 GB ≈ 357.6 GiB


def task_time(model: float = MODEL_S, tool: float = TOOL_S, rounds: int = ROUNDS) -> float:
    return rounds * (model + tool)


def demands(model: float = MODEL_S, lam: float = LAM) -> dict[str, float]:
    """资源需求率 = 到达率 × 每任务平均需求。"""
    t = task_time(model)
    return {"cpu_cores": lam * ROUNDS * TOOL_S, "mem_gib": lam * ENV_GIB * t, "model_calls": lam * ROUNDS * model}


def max_task_rate(model: float = MODEL_S) -> float:
    """内存装得下的环境数 ÷ 每项占用时间。"""
    return (MEM_GIB // ENV_GIB) / task_time(model)


def template_local_mib(mode: str, template: int = 2048, read: int = 512, dirty: int = 128, admin: int = 4, hot: int = 256) -> int:
    return {"完整复制": template + admin, "按需加载": read + admin, "只读共享": dirty + admin, "共享 + 热点": dirty + admin + hot}[mode]


def fetch_s(gib: float, bw: float = 3.125e9) -> float:
    return gib * GiB / bw


def pause_saving(env_gib: float, gap: float, save_per_gib: float = 4, resume: float = 1) -> float:
    """暂停后真正空出内存的秒数。"""
    return max(0.0, gap - env_gib * save_per_gib - resume)


def rebuild_gib_s(prep: float = 2, tool: float = TOOL_S) -> float:
    return ROUNDS * (prep + tool) * ENV_GIB


def expected_wait(P: float, L: float, h: float) -> float:
    """E[W] = P − h·min(P, L)。"""
    return P - h * min(P, L)


def main() -> None:
    print("\n第一步：一项任务")
    print(f"  三轮 × (9 + 1) = {task_time():.0f} s；工具 CPU 3 CPU·秒；环境 2 GiB × 30 s = {ENV_GIB * task_time():.0f} GiB·秒")

    print("\n第二步：平台需求（每秒 10 项）")
    for m in (9, 12):
        d = demands(m)
        print(f"  模型 {m} s：CPU {d["cpu_cores"]:.0f}/{CORES} 核（{d["cpu_cores"] / CORES:.1%}）、内存 {d["mem_gib"]:.0f}/{MEM_GIB:.0f} GiB（{d['mem_gib'] / MEM_GIB:.0%}）、并发调用 {d['model_calls']:.0f}/{SLOTS}（{d['model_calls'] / SLOTS:.0%}）")
    print(f"  内存最多 {MEM_GIB // ENV_GIB:.0f} 个环境 → 每秒只能开始 {max_task_rate():.1f} 项；Little 定律 10 × 30 = 300 个环境")

    print("\n第三步：模板加载（100 个环境）")
    for mode in ("完整复制", "按需加载", "只读共享", "共享 + 热点"):
        v = template_local_mib(mode)
        print(f"  {mode}：每个 {v} MiB，100 个 {100 * v / 1024:.0f} GiB")
    print(f"  冷路径取回 2 GiB + 加载 0.5 GiB：{fetch_s(2.5):.2f} s；热路径 {fetch_s(0.5):.2f} s；再缺页 1 GiB 多 {fetch_s(1):.2f} s")

    print("\n第四步：暂停与重建")
    for g in (2, 0.5):
        print(f"  {g} GiB 环境、9 s 间隔：暂停能空出 {pause_saving(g, 9):.0f} s")
    print(f"  32K 个 2 GiB 沙箱 = {32768 * 2 * GiB / 1e12:.0f} TB；暂停 2 小时空出时间是保存恢复的 {7200 / 9:.0f} 倍")
    r = rebuild_gib_s()
    print(f"  每轮重建：{ENV_GIB * task_time():.0f} → {r:.0f} GiB·秒/项，平台 {LAM * ENV_GIB * task_time():.0f} → {LAM * r:.0f} GiB；CPU 30 → {30 + 30 * 0.1:.0f} 核")

    print("\n第五步：提前准备（P = 2 s，h = 0.75）")
    for L in (0, 1, 2, 3):
        print(f"  提前 {L} s：平均等 {expected_wait(2, L, 0.75):.2f} s")
    print(f"  提前 3 s 多出的空闲占用 {0.75 * 2 * 1:.1f} GiB·秒/次")


if __name__ == "__main__":
    main()
