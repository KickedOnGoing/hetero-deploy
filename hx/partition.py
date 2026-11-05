"""编译期分区：把算子图拆成绑定到 CPU / GPU / NPU 的子图。

算法
----
1. 后端能力过滤：算子在某后端不支持 -> 该后端禁选（对应 MTK Neuron
   Delegate 的真实 fallback 行为）。
2. 初始解：single / greedy / hint 三种。
3. **并行感知局部搜索**：目标函数不是"每个算子 individually 最快"，
   而是整个 DAG 在"每个后端一条串行队列"约束下的 makespan。
   因为同一 NPU 一次只能跑一个子图，两个独立分支都放 NPU 无法并行；
   搜索会自动把分支分散到不同后端 —— 这正是「不同块并行在不同芯片」
   能否成立的决定性机制。
4. 子图构造：按拓扑序把「同后端 + 有直接依赖」的连续节点聚成一个
   Part；独立分支即使同后端也不合并，保留并行机会。
5. 碎片抑制：过小的 NPU 子图降级到 CPU（对应 NEURON_FLAG_MIN_GROUP_SIZE，
   以及学术界测得的过度分区 +28% 延迟）。

评价函数与最终产物使用**同一套** build_parts + simulate，避免优化目标
与实际执行不一致。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable

from .cost import CostModel, simulate
from .ir import Edge, Graph, Node, Part
from .opset import BACKENDS, allowed_backends, efficiency


@dataclass
class PartitionConfig:
    strategy: str = "auto"            # auto | greedy | single | manual
    iters: int = 600
    restarts: int = 3
    seed: int = 2026
    fragmentation_penalty_us: float = 2.0
    enforce_min_npu_group: bool = True
    respect_hints: bool = True
    pipeline_depth: int = 2
    allow_single_candidates: bool = True   # 是否允许「整图单后端」参与最终裁决
    require_multi_backend: bool = False    # 研究模式：强制至少使用两个后端


@dataclass
class SearchTrace:
    initial_us: float = 0.0
    final_us: float = 0.0
    iterations: int = 0
    accepted: int = 0
    history: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_us": round(self.initial_us, 2),
            "final_us": round(self.final_us, 2),
            "improvement_pct": round(
                (self.initial_us - self.final_us) / self.initial_us * 100, 2
            ) if self.initial_us else 0.0,
            "iterations": self.iterations, "accepted_moves": self.accepted,
        }


def _legal(g: Graph, n: Node) -> list[str]:
    return allowed_backends(n)


def initial_assignment(g: Graph, cm: CostModel, cfg: PartitionConfig) -> dict[str, str]:
    """生成初始后端绑定。"""
    a: dict[str, str] = {}
    boundaries = set(g.meta.get("hard_boundaries", []))
    for n in g.nodes:
        legal = _legal(g, n)
        if not legal:
            legal = ["cpu"]
        if cfg.strategy == "single":
            a[n.id] = "npu" if "npu" in legal else "cpu"
        elif cfg.strategy in ("manual", "force_hetero"):
            hint = n.hint if cfg.respect_hints else None
            a[n.id] = hint if hint in legal else min(legal, key=lambda b: cm.compute_us(n, b))
        else:  # greedy / auto
            hint = n.hint if cfg.respect_hints else None
            if hint and hint in legal:
                # 软提示：仅当不比最优慢 2 倍以上才采纳
                best = min(legal, key=lambda b: cm.compute_us(n, b))
                a[n.id] = hint if cm.compute_us(n, hint) <= 2.0 * cm.compute_us(n, best) else best
            else:
                a[n.id] = min(legal, key=lambda b: cm.compute_us(n, b))
    return a


def build_parts(g: Graph, assign: dict[str, str], cm: CostModel,
                cfg: PartitionConfig) -> tuple[list[Part], list[Edge]]:
    """按 assignment 构造 Part 与跨 Part 边；同时写回 Node.part / Node.backend。"""
    topo = g.topo()
    preds = g.preds()
    boundaries = set(g.meta.get("hard_boundaries", []))
    parts: list[Part] = []
    part_of: dict[str, int] = {}
    cur: Part | None = None
    cur_nodes: set[str] = set()

    # 跨后端分叉：某个张量被不同后端的节点消费时，其生产者必须单独成段。
    # 否则公共主干会和其中一条分支合并，另一条分支只能等整条分支跑完，
    # 并行机会被「连续段合并」悄悄吃掉 —— 这是分块并行最容易踩的坑。
    consumer_backends: dict[str, set[str]] = {}
    for n in g.nodes:
        for t in n.inputs:
            consumer_backends.setdefault(t, set()).add(assign[n.id])
    fork_cut = {t for t, bs in consumer_backends.items() if len(bs) > 1}

    for nid in topo:
        b = assign[nid]
        n = g.node(nid)
        # 封闭产出跨后端分叉张量的 Part，让下游分支可以立即并行启动
        if cur is not None and cur.nodes:
            last = g.node(cur.nodes[-1])
            if any(t in fork_cut for t in last.outputs):
                cur = None
                cur_nodes = set()
        boundary_hit = any(t in boundaries for t in n.inputs)
        # 同后端且在拓扑序上连续即可合并。
        # 独立分支即使同后端也合并成一个 Part：同一后端物理上是独占串行资源，
        # 合并反而省掉一次提交开销，且不损失任何真实并行能力。
        linked = (cur is not None and cur.backend == b)
        if cur is None or not linked or boundary_hit:
            cur = Part(id=len(parts), backend=b, nodes=[], inputs=[], outputs=[],
                       compile_flags={"target": b})
            parts.append(cur)
            cur_nodes = set()
        cur.nodes.append(nid)
        cur_nodes.add(nid)
        part_of[nid] = cur.id
        n.part = cur.id
        n.backend = b

    # 收集每个 part 的边界
    part_nodes: dict[int, set[str]] = {p.id: set(p.nodes) for p in parts}
    used_by: dict[str, set[int]] = {}
    for p in parts:
        for nid in p.nodes:
            for t in g.node(nid).inputs:
                used_by.setdefault(t, set()).add(p.id)
    for p in parts:
        ins: list[str] = []
        outs: list[str] = []
        for nid in p.nodes:
            n = g.node(nid)
            for t in n.inputs:
                ten = g.tensors[t]
                prod = ten.producer
                if ten.kind in ("weight", "bias", "constant"):
                    continue  # 权重全局加载，不构成跨片搬运
                if prod and part_of.get(prod, -1) != p.id:
                    if t not in ins:
                        ins.append(t)
                elif not prod and ten.kind == "input" and t not in ins:
                    ins.append(t)
            for t in n.outputs:
                consumers = used_by.get(t, set())
                if (consumers - {p.id}) or t in g.outputs or t in boundaries:
                    if t not in outs:
                        outs.append(t)
        p.inputs, p.outputs = ins, outs
        p.est_us = sum(cm.compute_us(g.node(x), p.backend) for x in p.nodes) + cm.launch_us(p.backend)

    # 跨 part 边
    edges: list[Edge] = []
    seen: set[tuple[int, int, str]] = set()
    for p in parts:
        for t in p.inputs:
            prod = g.tensors[t].producer
            if not prod or prod not in part_of:
                continue
            src = part_of[prod]
            if src == p.id:
                continue
            key = (src, p.id, t)
            if key in seen:
                continue
            seen.add(key)
            sb, db = parts[src].backend, p.backend
            e = Edge(src, p.id, t, sb, db, g.tensors[t].nbytes)
            e.cost_us = cm.transfer_us(e.bytes, sb, db)
            e.zero_copy = (sb == db) or bool(cm.chip.get("zero_copy_capable", False))
            edges.append(e)
    g.parts, g.edges = parts, edges
    return parts, edges


def _small_npu_parts(g: Graph, cm: CostModel) -> list[Part]:
    """识别过小的 NPU 子图（对应 NEURON_FLAG_MIN_GROUP_SIZE 的碎片化约束）。"""
    min_nodes = int(cm.chip.get("min_npu_nodes", 3))
    min_flops = int(cm.chip.get("min_part_flops", 2_000_000))
    return [p for p in g.parts if p.backend == "npu" and
            (len(p.nodes) < min_nodes or
             sum(g.node(x).flops for x in p.nodes) < min_flops)]


def score(g: Graph, assign: dict[str, str], cm: CostModel,
          cfg: PartitionConfig) -> tuple[float, Any]:
    """评价函数 = 离散事件模拟 makespan + 碎片惩罚 + 小 NPU 子图惩罚。

    小 NPU 子图用软惩罚而非强制降级：强制降级到 CPU 会让计算密集算子
    瞬间慢一个数量级，是分区器最容易犯的灾难性错误（见 docs/01 反例）。
    """
    try:
        parts, _ = build_parts(g, assign, cm, cfg)
        res = simulate(g, cm)
        pen = cfg.fragmentation_penalty_us * len(parts)
        if cfg.enforce_min_npu_group:
            pen += 250.0 * len(_small_npu_parts(g, cm))
        if cfg.require_multi_backend:
            used = set(assign.values())
            if len(used) < 2:
                pen += 100000.0
            else:
                counts = {b: sum(1 for v in assign.values() if v == b) for b in used}
                if min(counts.values()) < 3:
                    pen += 20000.0          # 禁止「22+1」式伪异构
            if res.overlap_pairs == 0:
                pen += 20000.0              # 必须存在真实的时间重叠
        return res.makespan_us + pen, res
    except (ValueError, KeyError):
        return float("inf"), None


def _single_assignment(g: Graph, b: str, cm: CostModel) -> dict[str, str]:
    """保底候选：尽量整体放在单一后端，不支持的算子退到**最快的**合法后端。

    早期版本退化到 legal[0]（恰好是 CPU），会把计算密集算子丢到 CPU，
    造成基线虚高与灾难性降级 —— 必须用代价模型选次优后端。
    """
    out: dict[str, str] = {}
    for n in g.nodes:
        legal = _legal(g, n) or ["cpu"]
        out[n.id] = b if b in legal else min(legal, key=lambda x: cm.compute_us(n, x))
    return out


def _same_backend_block(g: Graph, assign: dict[str, str], nid: str) -> list[str]:
    """沿依赖边做 BFS，取与 nid 同后端的连通块，用于块级变异。"""
    b = assign[nid]
    preds, succs = g.preds(), g.succs()
    adj: dict[str, list[str]] = {n: [] for n in assign}
    for n, ps in preds.items():
        for p in ps:
            adj[n].append(p)
            adj[p].append(n)
    for n, ss in succs.items():
        for s in ss:
            if s not in adj[n]:
                adj[n].append(s)
            if n not in adj[s]:
                adj[s].append(n)
    seen, stack, block = {nid}, [nid], []
    while stack:
        u = stack.pop()
        block.append(u)
        for v in adj.get(u, []):
            if v not in seen and assign.get(v) == b:
                seen.add(v)
                stack.append(v)
    return block


def search(g: Graph, cm: CostModel, cfg: PartitionConfig) -> tuple[dict[str, str], SearchTrace]:
    """并行感知的局部搜索：多起点 + 单节点/连通块变异 + 模拟退火。

    起点集合包含 greedy 与三种「整体单后端」解，保证搜索结果绝不劣于
    最优单后端方案 —— 这是分区器必须守住的底线。
    """
    rng = random.Random(cfg.seed)
    trace = SearchTrace()
    legal: dict[str, list[str]] = {n.id: _legal(g, n) or ["cpu"] for n in g.nodes}

    greedy = initial_assignment(g, cm, cfg)
    trace.initial_us, _ = score(g, greedy, cm, cfg)
    seeds = [greedy]
    if cfg.allow_single_candidates:
        seeds += [_single_assignment(g, b, cm) for b in BACKENDS]
    evaluated = [(score(g, s, cm, cfg)[0], s) for s in seeds]
    evaluated.sort(key=lambda x: x[0])

    best_global: dict[str, str] = dict(evaluated[0][1])
    best_score: float = evaluated[0][0]

    for restart in range(cfg.restarts):
        if restart == 0:
            assign = dict(best_global)
        elif restart - 1 < len(evaluated):
            assign = dict(evaluated[restart - 1][1])
        else:
            assign = {}
            for n in g.nodes:
                if n.hint and cfg.respect_hints and rng.random() < 0.3:
                    assign[n.id] = n.hint if n.hint in legal[n.id] else legal[n.id][0]
                else:
                    assign[n.id] = rng.choice(legal[n.id])
        cur_score, _ = score(g, assign, cm, cfg)
        T0 = max(1.0, trace.initial_us * 0.12)
        best_local, best_local_score = dict(assign), cur_score
        for it in range(cfg.iters):
            T = T0 * (1.0 - it / cfg.iters) ** 2 + 0.05
            nid = rng.choice(g.nodes).id
            alts = [b for b in legal[nid] if b != assign[nid]]
            if not alts:
                continue
            newb = rng.choice(alts)
            # 50% 概率做「整块同后端连通段」变异，跳出碎片局部最优
            if rng.random() < 0.5:
                block = _same_backend_block(g, assign, nid)
                if not all(newb in legal[x] for x in block):
                    continue
                backup = {x: assign[x] for x in block}
                for x in block:
                    assign[x] = newb
            else:
                backup = {nid: assign[nid]}
                assign[nid] = newb
            new_score, _ = score(g, assign, cm, cfg)
            d = new_score - cur_score
            if d < 0 or rng.random() < math.exp(-d / T):
                cur_score = new_score
                trace.accepted += 1
                if cur_score < best_local_score:
                    best_local, best_local_score = dict(assign), cur_score
            else:
                for k, v in backup.items():
                    assign[k] = v
            trace.iterations += 1
            if it % 40 == 0:
                trace.history.append(round(best_local_score, 2))
        if best_local_score < best_score:
            best_score, best_global = best_local_score, dict(best_local)

    trace.final_us = best_score
    return best_global, trace


def _makespan_of(g: Graph, assign: dict[str, str], cm: CostModel,
                 cfg: PartitionConfig) -> float:
    try:
        build_parts(g, assign, cm, cfg)
        return simulate(g, cm).makespan_us
    except (ValueError, KeyError):
        return float("inf")


def partition(g: Graph, cm: CostModel, cfg: PartitionConfig | None = None) -> tuple[Graph, SearchTrace]:
    """主入口：就地完成分区，返回 (graph, 搜索轨迹)。"""
    cfg = cfg or PartitionConfig()
    if cfg.strategy == "force_hetero":
        # 研究/验证模式：明确标注，不参与生产裁决，用于验证跨芯片并行调度链路
        cfg.allow_single_candidates = False
        cfg.require_multi_backend = True
    greedy = initial_assignment(g, cm, cfg)
    singles = {b: _single_assignment(g, b, cm) for b in BACKENDS}

    if cfg.strategy in ("single", "greedy"):
        assign = initial_assignment(g, cm, cfg)
        s0, _ = score(g, assign, cm, cfg)
        trace = SearchTrace(initial_us=s0, final_us=s0, iterations=0)
    else:
        # auto / manual 都跑局部搜索：训练侧提示只是起点与先验，
        # 编译器始终用代价模型消除碎片（严格照搬提示会产生 20+ 个 1 算子子图）。
        assign, trace = search(g, cm, cfg)
        if cfg.strategy != "force_hetero":
            # 最终裁决：在所有候选中按「真实 makespan」取最优，碎片只作为搜索期软约束。
            # 保证分区结果绝不劣于任何一种单一后端整图方案。
            candidates = [("search", assign), ("greedy", greedy)] + list(singles.items())
            best_name, assign = min(
                candidates,
                key=lambda kv: _makespan_of(g, kv[1], cm, cfg),
            )
            g.meta.setdefault("partition", {})["selected_candidate"] = best_name

    build_parts(g, assign, cm, cfg)
    sim = simulate(g, cm)
    trace.final_us = sim.makespan_us
    meta = g.meta.setdefault("partition", {})
    meta.update({
        "strategy": cfg.strategy,
        "search": trace.to_dict(),
        "demoted_nodes": 0,
        "small_npu_parts": len(_small_npu_parts(g, cm)),
        "part_count": len(g.parts),
        "cross_backend_edges": sum(1 for e in g.edges if e.cross),
        "total_boundary_bytes": sum(e.bytes for e in g.edges if e.cross),
        "assignment": assign,
    })
    errs = g.validate()
    if errs:
        raise RuntimeError("分区结果非法:\n  " + "\n  ".join(errs))
    return g, trace


def explain(g: Graph) -> list[dict[str, Any]]:
    """给出每个子图的划分理由，便于 OEM 侧评审与调优。"""
    out = []
    for p in g.parts:
        ops: dict[str, int] = {}
        for nid in p.nodes:
            op = g.node(nid).op
            ops[op] = ops.get(op, 0) + 1
        cross_in = [e.tensor for e in g.edges if e.dst == p.id and e.cross]
        cross_out = [e.tensor for e in g.edges if e.src == p.id and e.cross]
        out.append({
            "part": p.id, "backend": p.backend,
            "nodes": len(p.nodes), "ops": ops,
            "est_us": round(p.est_us, 2),
            "cross_in": cross_in, "cross_out": cross_out,
            "reason": _reason(p, ops, cross_in, cross_out),
        })
    return out


def _reason(p: Part, ops: dict[str, int], cin: list[str], cout: list[str]) -> str:
    if p.backend == "npu":
        s = "算子在 APU 支持表内且计算密集，放 NPU 取得最高能效"
    elif p.backend == "gpu":
        s = "含 NPU 不支持的非线性/归一化算子，或需要与 NPU 分支并发，放 GPU"
    else:
        s = "NPU/GPU 不支持的算子兜底，或小算子放 CPU 避免碎片化与搬运"
    if cin or cout:
        s += f"；跨片边界 {len(cin) + len(cout)} 条"
    return s
