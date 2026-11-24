"""第 15 章 · 多轮交互与 Agent 任务：前缀、成功成本、关键路径。

讲义：ai-infra/notes/lessons/15-agent-tasks.md
运行：uv run python ch15_agent_tasks.py
例题 3-2、工具 2/6/10/3 秒、CED 的 8192 个 token、决策请求费用取自 AII 第 3 章 §3.2；token 序列是编的。
"""


def reusable_prefix(old: list[int], new: list[int]) -> int:
    """最长可直接复用的前缀：第一个不同的 token 之前。"""
    n = 0
    for a, b in zip(old, new):
        if a != b:
            break  # 第一个不同的位置之后全部失效
        n += 1
    return n


def success_cost(c: float, p: float) -> float:
    """每个成功任务的期望成本 c / p。"""
    return c / p


def success_cost_ratio(price: float, usage: float, p_old: float, p_new: float) -> float:
    """新旧方法每个成功任务的成本比：单价比 × 用量比 × (旧成功率 / 新成功率)。"""
    return price * usage * p_old / p_new


def amdahl_task(total: float, part: float, s: float) -> tuple[float, float]:
    """把占 part 秒的一段加速 s 倍：(新总时间, 总加速比)。"""
    new = total - part + part / s
    return new, total / new


def task_time(pre: float, tools: tuple[float, ...], post: float, parallel: bool) -> float:
    """模型 → 一组工具 → 模型：工具串行相加，独立时取最长。"""
    return pre + (max(tools) if parallel else sum(tools)) + post


def ced_work(P: int) -> tuple[int, int]:
    """从空状态输入 P 个 token 的专家层工作：(普通 40 层, CED)。"""
    return 40 * P, 20 * P + 20 * min(P, 128)


def handoff_breakeven(c_dec: float = 0.000081, c_llm: float = 0.013880) -> float:
    """转交比例超过多少时，转交 LLM 的费用超过决策模型本身。"""
    return c_dec / c_llm


def main() -> None:
    print("\n第一步：改写位置决定可复用前缀（编的 2000 个 token）")
    old = list(range(2000))
    print(f"  末尾追加 100 个 → 复用 {reusable_prefix(old, old + list(range(100)))}")
    changed = old.copy()
    changed[500] = -1  # 改第 501 个
    print(f"  改第 501 个 → 复用 {reusable_prefix(old, changed)}")

    print("\n第二步：单价 1/10、用量 20 倍、成功率 50% → 80%")
    print(f"  成功任务成本比 = {success_cost_ratio(0.1, 20, 0.5, 0.8):.2f}")

    print("\n第三步：例题 3-2，首轮 36.375 s 减半")
    new, sp = amdahl_task(76.510, 36.375, 2)
    print(f"  76.510 → {new:.3f} s，加速 {sp:.2f} 倍")

    print("\n第四步：模型 2 s → 工具 6 s、10 s → 模型 3 s")
    print(f"  串行 {task_time(2, (6, 10), 3, False):.0f} s，并行 {task_time(2, (6, 10), 3, True):.0f} s")

    print("\n第五步：CED 与决策请求")
    full, ced = ced_work(8192)
    print(f"  8192 个输入：普通 {full:,}，CED {ced:,}（{ced / full:.1%}）")
    print(f"  转交比例超过 {handoff_breakeven():.2%} 时，转交费用超过决策模型本身")


if __name__ == "__main__":
    main()
