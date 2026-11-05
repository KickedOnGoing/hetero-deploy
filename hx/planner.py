"""部署计划生成与收益判定（严密论证的代码化）。

对每一种切法给出可证伪的判定，而不是笼统地说"可以加速"：
  T_seq : 单一最快后端顺序执行全部算子的时间
  T_cp  : 分区后 DAG 的关键路径（纯计算，不含资源竞争）
  T_par : 离散事件模拟出的真实 makespan（含搬运/同步/提交/资源竞争）

  S     = T_seq / T_par          实际加速比
  S_max = T_seq / T_cp           理论加速上界
  T_par = T_cp + 资源竞争 + 搬运 + 同步

链式图（DAG 宽度 = 1）必然 T_cp ≈ T_seq，切分只增开销，S < 1：
工具链会明确判定「不建议算子级并行」，并改推跨帧流水。
"""
from __future__ import annotations

from typing import Any

from .cost import CostModel, SimResult, simulate, speedup_bound
from .ir import Graph
from .opset import BACKENDS
from .partition import PartitionConfig, _single_assignment, build_parts


def _fair_baselines(g: Graph, cm: CostModel) -> dict[str, float]:
    """用与分区方案**完全相同**的离散事件模拟计算单后端基线。

    朴素地把所有算子时间相加会漏掉「单后端方案自己也会因不支持算子而
    产生跨片搬运」的成本，从而人为夸大并行方案的劣势。这里把每种单后端
    解也 build_parts + simulate 一遍，得到可比基线。
    """
    saved_parts, saved_edges = g.parts, g.edges
    saved_nodes = [(n.part, n.backend) for n in g.nodes]
    cfg = PartitionConfig()
    out: dict[str, float] = {}
    for b in BACKENDS:
        try:
            a = _single_assignment(g, b, cm)
            build_parts(g, a, cm, cfg)
            out[b] = simulate(g, cm).makespan_us
        except (ValueError, KeyError):
            out[b] = cm.baseline_us(g, b)
    g.parts, g.edges = saved_parts, saved_edges
    for n, (p, bb) in zip(g.nodes, saved_nodes):
        n.part, n.backend = p, bb
    return out


def dag_metrics(g: Graph) -> dict[str, Any]:
    """DAG 形状指标：宽度、层级、可并行分支、链式判定。"""
    preds, succs = g.preds(), g.succs()
    level: dict[str, int] = {}
    for nid in g.topo():
        level[nid] = max((level[p] + 1 for p in preds[nid]), default=0)
    layers: dict[int, int] = {}
    for v in level.values():
        layers[v] = layers.get(v, 0) + 1
    width = max(layers.values()) if layers else 0
    depth = max(level.values()) + 1 if level else 0
    # 分叉点：被 >1 个节点消费的张量
    forks = [t.id for t in g.tensors.values()
             if t.kind == "activation" and t.producer and
             sum(1 for n in g.nodes if t.id in n.inputs) > 1]
    return {
        "nodes": len(g.nodes), "depth": depth, "max_width": width,
        "branch_forks": len(forks), "is_chain": width <= 1,
        "levels": {str(k): v for k, v in sorted(layers.items())},
    }


def pipeline_analysis(g: Graph, res: SimResult, depth: int) -> dict[str, Any]:
    """跨帧流水分析：延迟不降时，吞吐仍可能提升。"""
    per_backend = res.backend_busy_us
    bottleneck = max(per_backend, key=per_backend.get) if per_backend else "cpu"
    steady = max(per_backend.values()) if per_backend else res.makespan_us
    seq_single = res.total_compute_us + res.total_launch_us
    ideal_pipeline = seq_single / len(BACKENDS)
    return {
        "pipeline_depth": depth,
        "bottleneck_backend": bottleneck,
        "steady_frame_us": round(steady, 2),
        "steady_fps": round(1e6 / steady, 2) if steady else 0.0,
        "single_frame_latency_us": round(res.makespan_us, 2),
        "no_pipeline_fps": round(1e6 / res.makespan_us, 2) if res.makespan_us else 0.0,
        "ideal_balanced_fps": round(1e6 / ideal_pipeline, 2) if ideal_pipeline else 0.0,
        "note": "跨帧流水让不同帧占用不同后端，提高吞吐但不降低单帧延迟",
    }


def make_plan(g: Graph, cm: CostModel, *, pipeline_depth: int = 2) -> dict[str, Any]:
    res = simulate(g, cm)
    metrics = dag_metrics(g)
    fair = _fair_baselines(g, cm)
    baselines = {b: round(v, 2) for b, v in fair.items()}
    naive = {b: round(cm.baseline_us(g, b), 2) for b in BACKENDS}
    best_single = min(baselines, key=baselines.get)  # type: ignore[arg-type]
    best_baseline_us = fair[best_single]
    bound = speedup_bound(g, cm, res, "npu", baseline_us=best_baseline_us)
    pipe = pipeline_analysis(g, res, pipeline_depth)

    S = bound["latency_speedup"]
    cross = sum(1 for e in g.edges if e.cross)
    # ---- 判定 ----
    if metrics["is_chain"]:
        verdict = "NOT_RECOMMENDED"
        reason = ("该图为链式 DAG（最大宽度=1），不存在可并行的独立分支；"
                  "任何算子级切分只会引入跨片搬运与同步，单帧延迟必然变差。"
                  "建议：单后端整图编译 + 跨帧流水（仅提升吞吐）。")
    elif S >= 1.15:
        verdict = "RECOMMENDED"
        reason = (f"并行分支使关键路径缩短，模拟加速比 {S:.2f}×，"
                  f"搬运开销占比 {bound['transfer_overhead_ratio'] * 100:.1f}%。")
    elif S >= 1.0:
        verdict = "MARGINAL"
        reason = (f"加速比 {S:.2f}×，收益小于 15% 的工程噪声与热波动，"
                  f"需上板实测再决策；可尝试增大分支粒度以降低 {cross} 条跨片边界。")
    else:
        verdict = "NOT_RECOMMENDED"
        reason = (f"模拟加速比 {S:.2f}× < 1：跨片搬运/同步/提交开销超过了并行收益。"
                  f"建议减少分区数或改用单一后端。")

    plan = {
        "verdict": verdict,
        "reason": reason,
        "objective": {"latency": True, "throughput": True},
        "dag": metrics,
        "baselines_single_backend_us": baselines,
        "naive_sequential_baselines_us": naive,
        "best_single_backend": best_single,
        "_baseline_note": "基线为单后端方案的离散事件模拟（含其自身的降级搬运），与分区方案可比",
        "bound": bound,
        "simulation": res.summary(),
        "parallel_execution": {
            "observed": res.overlap_pairs > 0,
            "overlap_pairs": res.overlap_pairs,
            "dag_max_width": metrics["max_width"],
            "note": "若 observed=false，加速比来自『把算子放到更合适的后端』，"
                    "而非不同芯片同时执行；两者必须分开评估",
        },
        "pipeline": pipe,
        "cost_breakdown": {
            "compute_us": round(res.total_compute_us, 2),
            "transfer_us": round(res.total_transfer_us, 2),
            "launch_us": round(res.total_launch_us, 2),
            "resource_idle_us": round(max(0.0, res.makespan_us - res.critical_path_us), 2),
        },
        "constraints": {
            "cross_backend_edges": cross,
            "boundary_bytes": sum(e.bytes for e in g.edges if e.cross),
            "max_part_count": len(g.parts),
            "per_backend_serial": True,
            "thermal_note": "持续满载 2~3 分钟后 SoC 可能降频（文献实测最高 4.3× 衰减），"
                            "运行时需启用热感知调度与动态降级",
        },
    }
    g.meta["plan"] = plan
    return plan
