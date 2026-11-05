"""代价模型与离散事件模拟器。

这是"可实现性"的定量核心：任何分区方案在生成之前，都先在这里被
离散事件模拟评估一次。模拟器显式建模了四件事：

1. 每个后端是**独占串行资源**（可由 chip 的 slots 配置多个执行槽）；
2. 跨后端边产生**搬运 + 同步**开销，占用一条串行 DMA 队列；
3. 同后端的独立分支**无法并行**（除非 slots>1），因此把两个分支都放到最快的
   NPU 并不会带来并行收益 —— 这直接决定分区搜索的目标函数；
4. 算子耗时取 **算力下界与访存下界的最大值**（roofline）：element-wise / view
   类算子 FLOPs≈0 但依然要读写内存，估成 0 会让分区器凭空制造 NPU 碎片。

确定性保证：所有遍历均排序，不依赖 set/dict 的迭代顺序；同一分区方案
重复 simulate 必须得到逐位相同的结果（见 tests/test_core.py）。

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
    slot: int = 0               # 占用的执行槽


@dc.dataclass
class SimResult:
    makespan_us: float
    events: list[SimEvent]
    backend_busy_us: dict[str, float]
    total_compute_us: float
    total_transfer_us: float
    total_launch_us: float
    steady_frame_us: float       # 深流水下的稳态帧间隔（瓶颈串行资源）
    critical_path_us: float
    utilization: dict[str, float]
    overlap_pairs: int = 0       # 不同后端在时间上真正重叠的 Part 对数
    dma_busy_us: float = 0.0     # DMA 队列被占用的总时长
    transfer_edges: int = 0      # 参与模拟的跨 part 数据边条数

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
            "dma_busy_us": round(self.dma_busy_us, 2),
            "transfer_edges": self.transfer_edges,
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

    def slots(self, backend: str) -> int:
        """该后端可并发的执行槽数（对应 chip.backends[b].slots / backend_slots）。"""
        if "slots" in self.be.get(backend, {}):
            return max(1, int(self.be[backend]["slots"]))
        return max(1, int(self.chip.get("backend_slots", {}).get(backend, 1)))

    def compute_us(self, node: Node, backend: str) -> float:
        """单算子纯计算时间（不含提交开销）= max(算力下界, 访存下界)。"""
        eff = efficiency(node.op, backend)
        if eff <= 0:
            return float("inf")
        peak = float(self.be[backend]["peak_gflops"]) * 1e9
        t_flops = node.flops / (peak * eff) * 1e6 if peak > 0 else 0.0
        return max(t_flops, self.memory_floor_us(node, backend))

    def memory_floor_us(self, node: Node, backend: str) -> float:
        """roofline 访存下界：读写字节 / 片内有效带宽。

        node.io_bytes 在 lower 阶段由 estimate_io_bytes 计算（输入+输出+权重）。
        带宽缺省 55 GB/s，与跨片 per_mb_us 的假设一致。
        """
        bw = float(self.be[backend].get("mem_bw_gbps", 55.0)) * 1e9
        if bw <= 0 or node.io_bytes <= 0:
            return 0.0
        return node.io_bytes / bw * 1e6

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


def _part_topo(parts: dict[int, Part], deps: dict[int, set[int]]) -> list[int]:
    """子图商图的**确定性**拓扑序（固定取最小可用 id），不依赖 set 迭代顺序。"""
    indeg = {pid: len(deps[pid]) for pid in parts}
    succ: dict[int, list[int]] = {pid: [] for pid in parts}
    for pid, ps in deps.items():
        for q in sorted(ps):
            if q in succ:
                succ[q].append(pid)
    ready = sorted(pid for pid, d in indeg.items() if d == 0)
    out: list[int] = []
    while ready:
        u = ready.pop(0)
        out.append(u)
        for v in sorted(succ[u]):
            indeg[v] -= 1
            if indeg[v] == 0:
                ready.append(v)
                ready.sort()
    if len(out) != len(parts):
        raise ValueError("子图商图存在环，无法模拟")
    return out


def simulate(g: Graph, cm: CostModel) -> SimResult:
    """对已分区的图做离散事件模拟，返回 makespan / 时间线 / 利用率。

    资源模型：每个后端 slots 条串行队列；DMA 一条串行队列。
    任务单位 = Part（一次提交一个已编译子图）。
    """
    parts = {p.id: p for p in g.parts}
    if not parts:
        return SimResult(0.0, [], {b: 0.0 for b in BACKENDS}, 0.0, 0.0, 0.0,
                         0.0, 0.0, {b: 0.0 for b in BACKENDS})

    # part 级依赖与**逐张量**入边（不再按 (src,dst) 取 max，避免低估搬运）
    deps: dict[int, set[int]] = {pid: set() for pid in parts}
    in_edges: dict[int, list[tuple[int, str, float, bool]]] = {pid: [] for pid in parts}
    for e in g.edges:
        if e.src == e.dst or e.src not in parts or e.dst not in parts:
            continue
        deps[e.dst].add(e.src)
        cross = e.src_backend != e.dst_backend
        in_edges[e.dst].append((e.src, e.tensor, cm.edge_cost(e), cross))

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

    porder = _part_topo(parts, deps)

    backend_free: dict[str, list[float]] = {
        b: [0.0] * cm.slots(b) for b in BACKENDS}
    dma_free = [0.0]
    finish: dict[int, float] = {}
    events: list[SimEvent] = []
    backend_busy = {b: 0.0 for b in BACKENDS}
    dma_busy = 0.0
    total_transfer = 0.0
    transfer_edges = 0

    for pid in porder:
        p = parts[pid]
        b = p.backend
        # 依赖就绪时间。跨片边按**前驱实际完成时刻**排序后依次占用 DMA，
        # 保证物理因果与结果确定性（旧实现按 set 顺序排队，结果会漂移）。
        pending = sorted(in_edges[pid], key=lambda x: (finish.get(x[0], 0.0), x[0], x[1]))
        ready = 0.0
        for q, _tensor, cost, cross in pending:
            q_done = finish.get(q, 0.0)
            if cross and cost > 0:
                di = min(range(len(dma_free)), key=lambda k: dma_free[k])
                dma_start = max(q_done, dma_free[di])
                dma_end = dma_start + cost
                dma_free[di] = dma_end
                dma_busy += cost
                total_transfer += cost
                transfer_edges += 1
                arrival = dma_end
            elif cost > 0:
                # 同后端 part 之间只有 fence 同步，不占 DMA
                total_transfer += cost
                arrival = q_done + cost
            else:
                arrival = q_done
            ready = max(ready, arrival)

        si = min(range(len(backend_free[b])), key=lambda k: backend_free[b][k])
        start = max(ready, backend_free[b][si])
        end = start + launch[pid] + comp[pid]
        backend_free[b][si] = end
        finish[pid] = end
        backend_busy[b] += launch[pid] + comp[pid]
        events.append(SimEvent(pid, b, start, end, launch[pid], comp[pid], slot=si))

    makespan = max(finish.values()) if finish else 0.0
    # 稳态帧间隔：深流水下由最忙的串行资源决定；多槽后端按槽数均摊，DMA 也参与竞争
    steady = max(
        [backend_busy[b] / cm.slots(b) for b in BACKENDS] + [dma_busy / len(dma_free)]
    )

    # 关键路径：无限资源下的最短完成时间（含跨片搬运），作为加速上界参考
    edge_max: dict[tuple[int, int], float] = {}
    for pid in porder:
        for q, _t, cost, _c in in_edges[pid]:
            key = (q, pid)
            edge_max[key] = max(edge_max.get(key, 0.0), cost)
    cp: dict[int, float] = {}
    for pid in porder:
        base = max((cp[q] + edge_max.get((q, pid), 0.0) for q in sorted(deps[pid])),
                   default=0.0)
        cp[pid] = base + launch[pid] + comp[pid]
    critical = max(cp.values()) if cp else 0.0

    total_compute = sum(comp.values())
    total_launch = sum(launch.values())
    util = {b: (backend_busy[b] / (makespan * cm.slots(b)) if makespan > 0 else 0.0)
            for b in BACKENDS}
    overlap = 0
    for i, a in enumerate(events):
        for bb in events[i + 1:]:
            if a.backend != bb.backend and a.start_us < bb.end_us and bb.start_us < a.end_us:
                overlap += 1
    return SimResult(makespan, events, backend_busy, total_compute,
                     total_transfer, total_launch, steady, critical, util,
                     overlap, dma_busy, transfer_edges)


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
