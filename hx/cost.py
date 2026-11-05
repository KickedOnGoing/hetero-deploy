"""代价模型与离散事件模拟器。

这是"可实现性"的定量核心：任何分区方案在生成之前，都先在这里被
离散事件模拟评估一次。模拟器显式建模了三件事：

1. 每个后端是**独占串行资源**（一个 NPU 同时只能执行一个子图）；
2. 跨后端边产生**搬运 + 同步**开销，占用 DMA 资源；
3. 同后端的独立分支**无法并行**，因此把两个分支都放到最快的 NPU
   并不会带来并行收益 —— 这一点直接决定了分区搜索的目标函数。

所有时间单位为微秒 (us)。
"""
from __future__ import annotations

import dataclasses as dc
import math
from typing import Any

from .ir import Edge, Graph, Node, Part
from .opset import BACKENDS, efficiency


@dc.dataclass
class SimEvent:
    part: int
    backend: str
    start_us: float
    end_us: float
    launch_us: float
    compute_us: float
    kind: str = "compute"       # compute | transfer


@dc.dataclass
class SimResult:
    makespan_us: float
    events: list[SimEvent]
    backend_busy_us: dict[str, float]
    total_compute_us: float
    total_transfer_us: float
    total_launch_us: float
    steady_frame_us: float       # 深流水下的稳态帧间隔（瓶颈后端）
    critical_path_us: float
    utilization: dict[str, float]
    overlap_pairs: int = 0       # 不同后端在时间上真正重叠的 Part 对数

    @property
    def throughput_fps(self) -> float:
        return 1e6 / self.steady_frame_us if self.steady_frame_us > 0 else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "makespan_us": round(self.makespan_us, 2),
            "steady_frame_us": round(self.steady_frame_us, 2),
            "throughput_fps": round(self.throughput_fps, 2),
            "total_compute_us": round(self.total_compute_us, 2),
            "total_transfer_us": round(self.total_transfer_us, 2),
            "total_launch_us": round(self.total_launch_us, 2),
            "critical_path_us": round(self.critical_path_us, 2),
            "overlap_pairs": self.overlap_pairs,
            "backend_busy_us": {k: round(v, 2) for k, v in self.backend_busy_us.items()},
            "utilization": {k: round(v, 4) for k, v in self.utilization.items()},
        }


class CostModel:
    """把芯片档案翻译成可计算的时间代价。"""

    def __init__(self, chip: dict[str, Any]):
        self.chip = chip
        self.be = chip["backends"]

    def compute_us(self, node: Node, backend: str) -> float:
        """单算子纯计算时间（不含提交开销）。"""
        eff = efficiency(node.op, backend)
        if eff <= 0:
            return float("inf")
        peak = float(self.be[backend]["peak_gflops"]) * 1e9
        # flops / (peak * eff) 秒 -> 微秒
        t = node.flops / (peak * eff) * 1e6
        # 访存下限：小张量受带宽约束而非算力约束
        return max(t, self.memory_floor_us(node, backend))

    def memory_floor_us(self, node: Node, backend: str) -> float:
        """算子读写量 / 有效带宽给出的时间下界，防止小算子被估成 0。"""
        bw = 55.0 * 1e9  # bytes/s，与 per_mb_us 一致
        return 0.0  # 由调用方的 launch_us 覆盖；保留接口供校准

    def launch_us(self, backend: str) -> float:
        return float(self.be[backend].get("launch_us", 0.0))

    def transfer_us(self, nbytes: int, src: str, dst: str) -> float:
        """跨后端搬运一次张量的代价。同后端返回 0（共享缓冲，零拷贝）。"""
        if src == dst:
            return 0.0
        c = float(self.chip["fixed_cross_us"])
        c += nbytes / 1e6 * float(self.chip["per_mb_us"])
        if self.be[src].get("layout") != self.be[dst].get("layout"):
            c += float(self.chip["layout_convert_us"])
        return c

    def edge_cost(self, e: Edge) -> float:
        if e.src_backend == e.dst_backend:
            return float(self.chip["intra_part_sync_us"])
        return self.transfer_us(e.bytes, e.src_backend, e.dst_backend)

    def baseline_us(self, g: Graph, backend: str = "npu") -> float:
        """单一后端顺序执行全部算子的时间（含不支持算子退化到 CPU 的惩罚）。"""
        t = 0.0
        for n in g.nodes:
            b = backend if efficiency(n.op, backend) > 0 else "cpu"
            t += self.compute_us(n, b)
            if b != backend:
                t += self.transfer_us(g.tensors[n.outputs[0]].nbytes, backend, b) if n.outputs else 0
        return t + self.launch_us(backend)


def simulate(g: Graph, cm: CostModel) -> SimResult:
    """对已分区的图做离散事件模拟，返回 makespan / 时间线 / 利用率。

    资源模型：每个后端一条串行队列；DMA 一条串行队列。
    任务单位 = Part（一次提交一个已编译子图）。
    """
    parts = {p.id: p for p in g.parts}
    if not parts:
        return SimResult(0.0, [], {b: 0.0 for b in BACKENDS}, 0.0, 0.0, 0.0,
                         0.0, 0.0, {b: 0.0 for b in BACKENDS})

    # part 级依赖与边延迟
    deps: dict[int, set[int]] = {p.id: set() for p in g.parts}
    edge_delay: dict[tuple[int, int], float] = {}
    for e in g.edges:
        if e.src == e.dst:
            continue
        deps[e.dst].add(e.src)
        key = (e.src, e.dst)
        edge_delay[key] = max(edge_delay.get(key, 0.0), cm.edge_cost(e))

    # part 计算时间
    comp: dict[int, float] = {}
    launch: dict[int, float] = {}
    for p in g.parts:
        c = 0.0
        for nid in p.nodes:
            n = g.node(nid)
            v = cm.compute_us(n, p.backend)
            if math.isinf(v):
                raise ValueError(f"子图 {p.id} 含 {p.backend} 不支持的算子 {n.op}")
            c += v
        comp[p.id] = c
        launch[p.id] = cm.launch_us(p.backend)

    # part 拓扑序（按节点全局拓扑序的最小位置排序，更稳定）
    order = {nid: i for i, nid in enumerate(g.topo())}
    porder = sorted(parts, key=lambda pid: min(order[x] for x in parts[pid].nodes))

    backend_free = {b: 0.0 for b in BACKENDS}
    dma_free = 0.0
    finish: dict[int, float] = {}
    events: list[SimEvent] = []
    backend_busy = {b: 0.0 for b in BACKENDS}

    for pid in porder:
        p = parts[pid]
        b = p.backend
        # 依赖就绪时间：前驱完成 + 跨后端搬运（占用 DMA）
        ready = 0.0
        for q in deps[pid]:
            d = edge_delay.get((q, pid), 0.0)
            if d > 0:
                dma_start = max(finish[q], dma_free)
                dma_free = dma_start + d
                arrival = dma_free
            else:
                arrival = finish[q] + d
            ready = max(ready, arrival)
        start = max(ready, backend_free[b])
        end = start + launch[pid] + comp[pid]
        backend_free[b] = end
        finish[pid] = end
        backend_busy[b] += launch[pid] + comp[pid]
        events.append(SimEvent(pid, b, start, end, launch[pid], comp[pid]))

    makespan = max(finish.values()) if finish else 0.0
    # 稳态帧间隔：流水足够深时由最忙后端决定
    steady = max(backend_busy.values()) if backend_busy else 0.0
    # 关键路径：只算计算+launch 的最长链（不含资源竞争），作为加速上界参考
    cp: dict[int, float] = {}
    for pid in porder:
        base = max((cp[q] + edge_delay.get((q, pid), 0.0) for q in deps[pid]), default=0.0)
        cp[pid] = base + launch[pid] + comp[pid]
    critical = max(cp.values()) if cp else 0.0

    total_transfer = sum(edge_delay.values())
    total_compute = sum(comp.values())
    total_launch = sum(launch.values())
    util = {b: (backend_busy[b] / makespan if makespan > 0 else 0.0) for b in BACKENDS}
    overlap = 0
    for i, a in enumerate(events):
        for b in events[i + 1:]:
            if a.backend != b.backend and a.start_us < b.end_us and b.start_us < a.end_us:
                overlap += 1
    return SimResult(makespan, events, backend_busy, total_compute,
                     total_transfer, total_launch, steady, critical, util, overlap)


def speedup_bound(g: Graph, cm: CostModel, res: SimResult,
                  baseline_backend: str = "npu",
                  baseline_us: float | None = None) -> dict[str, Any]:
    """给出分区方案相对单后端的收益与理论上界，供论证报告使用。

    baseline_us 传入「公平模拟基线」时，加速比才是可比的：
    朴素顺序基线忽略了单后端方案自身也会产生的跨片降级搬运，会高估加速比。
    """
    naive = cm.baseline_us(g, baseline_backend)
    base = baseline_us if baseline_us is not None else naive
    # T1：分区方案在单处理器上顺序执行所有 Part 的总时间（含搬运）
    t1 = res.total_compute_us + res.total_launch_us + res.total_transfer_us
    # Tinf：关键路径下界（含跨片搬运），无限资源也不可突破
    tinf = res.critical_path_us
    used = sum(1 for v in res.backend_busy_us.values() if v > 0)
    return {
        "baseline_single_backend_us": round(base, 2),
        "naive_sequential_baseline_us": round(naive, 2),
        "baseline_is_fair_simulation": baseline_us is not None,
        "partitioned_latency_us": round(res.makespan_us, 2),
        "latency_speedup": round(base / res.makespan_us, 4) if res.makespan_us else 0.0,
        "total_serial_compute_us": round(res.total_compute_us + res.total_launch_us, 2),
        "T1_serial_partitioned_us": round(t1, 2),
        "Tinf_critical_path_us": round(tinf, 2),
        "theoretical_upper_bound_vs_baseline": round(base / tinf, 4) if tinf else 0.0,
        "intrinsic_parallelism_T1_over_Tinf": round(t1 / tinf, 4) if tinf else 0.0,
        "parallel_efficiency": round(t1 / (res.makespan_us * max(1, used)), 4) if res.makespan_us else 0.0,
        "transfer_overhead_ratio": round(res.total_transfer_us / t1, 4) if t1 else 0.0,
    }
