"""收益临界条件的敏感性分析（严密论证的数据来源）。

扫描三个决定性参数，回答「什么条件下分块并行才真的划算」：
  1. GPU/NPU 有效吞吐比  r = peak_npu / peak_gpu
  2. 跨片固定开销        fixed_cross_us（给定切分，开销如何吃掉收益）
  3. 模型规模            flops_scale（等效于增大输入分辨率/通道）

为保证曲线物理干净，本脚本区分两类扫描：
  - 比例扫描：auto 决策（演示「GPU 太弱就不拆」的自动裁决）；
  - 开销/规模扫描：固定一个已确定的切分（force_hetero），基线用
    「整图单后端（GPU，零跨片）」。这样扫描参数的变化只作用于分区方案，
    基线是稳定的参考，不会出现「基线也吃到开销增量」的反常。

输出 CSV + 两张图，并在控制台打印临界点。
用法：cd /workspace && python3 -m tests.sensitivity
"""
from __future__ import annotations

import copy
import csv
import json
import os
from typing import Any

import torch

from examples.model import build_model
from hx.capture import export
from hx.cost import CostModel, simulate
from hx.lower import lower
from hx.opset import DEFAULT_CHIP
from hx.partition import PartitionConfig, build_parts, partition
from hx.planner import make_plan

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "build", "analysis")
os.makedirs(OUT, exist_ok=True)


def base_graph():
    torch.manual_seed(0)
    cap = export(build_model(True), (torch.randn(1, 3, 32, 32),))
    return lower(cap)


# ---------------------------------------------------------------------------
# 比例扫描：auto 决策（DEFAULT_CHIP，NPU 主导），flops 放大 16x
# ---------------------------------------------------------------------------
def evaluate_auto(g0, *, gpu_peak=None, flops_scale=16.0,
                  strategy="auto") -> dict[str, Any]:
    g = copy.deepcopy(g0)
    if flops_scale != 1.0:
        for n in g.nodes:
            n.flops = int(n.flops * flops_scale)
    chip = copy.deepcopy(DEFAULT_CHIP)
    if gpu_peak is not None:
        chip["backends"]["gpu"]["peak_gflops"] = gpu_peak
    cm = CostModel(chip)
    cfg = PartitionConfig(strategy=strategy, iters=220, restarts=2, seed=7)
    partition(g, cm, cfg)
    plan = make_plan(g, cm)
    return {
        "parts": len(g.parts),
        "backends": len({p.backend for p in g.parts}),
        "overlap": plan["parallel_execution"]["observed"],
        "speedup": plan["bound"]["latency_speedup"],
        "upper_bound": plan["bound"]["theoretical_upper_bound_vs_baseline"],
        "verdict": plan["verdict"],
        "latency_us": plan["simulation"]["makespan_us"],
        "baseline_us": plan["bound"]["baseline_single_backend_us"],
        "transfer_ratio": plan["bound"]["transfer_overhead_ratio"],
    }


def sweep_ratio(g0) -> list[dict[str, Any]]:
    rows = []
    for gpu in (600, 1000, 1500, 2200, 3000, 4200, 6000, 9000, 14000):
        r = evaluate_auto(g0, gpu_peak=float(gpu))
        r["gpu_gflops"] = gpu
        r["ratio_npu_over_gpu"] = round(9000.0 / gpu, 2)
        rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# 固定切分：在均衡型芯片下用 force_hetero 产生一次确定切分，供开销/规模扫描复用
# ---------------------------------------------------------------------------
def fixed_assignment(g0) -> tuple[dict[str, str], dict[str, Any]]:
    chip = copy.deepcopy(DEFAULT_CHIP)
    # 均衡型：GPU 与 NPU 有效吞吐接近，便于演示「真并行」收益
    chip["backends"]["gpu"]["peak_gflops"] = 4200.0
    chip["backends"]["npu"]["peak_gflops"] = 6000.0
    chip["backends"]["gpu"]["layout"] = "NHWC"
    cm = CostModel(chip)
    g = copy.deepcopy(g0)
    cfg = PartitionConfig(strategy="force_hetero", respect_hints=False,
                          iters=300, restarts=3, seed=7)
    partition(g, cm, cfg)
    # 保留 force_hetero 的原始切分（含一支 CPU、4 parts、真实 overlap）。
    # 它天然 S<1：这正是「强制异构无收益窗口、auto 会拒绝它」的证据，
    # 与比例扫描里 auto 给出 2.3~2.6× 形成对照。曲线只用来观察
    # 「开销增大 S 降 / 规模放大 S 单调趋近上界」这两个单调性。
    assign = {n.id: n.backend for n in g.nodes}
    return assign, chip


def evaluate_fixed(g0, assign, chip, *, flops_scale=1.0,
                   baseline="gpu") -> dict[str, Any]:
    """固定切分下，扫描参数只作用于分区方案；基线 = 整图单后端（零跨片）。"""
    if flops_scale != 1.0:
        g = copy.deepcopy(g0)
        for n in g.nodes:
            n.flops = int(n.flops * flops_scale)
    else:
        g = copy.deepcopy(g0)
    cm = CostModel(chip)
    cfg = PartitionConfig(strategy="manual", respect_hints=False)
    build_parts(g, assign, cm, cfg)
    res = simulate(g, cm)
    T_par = res.makespan_us

    g2 = copy.deepcopy(g0)
    if flops_scale != 1.0:
        for n in g2.nodes:
            n.flops = int(n.flops * flops_scale)
    cm2 = CostModel(chip)
    base_a = {n.id: baseline for n in g2.nodes}
    build_parts(g2, base_a, cm2, cfg)
    T_seq = simulate(g2, cm2).makespan_us

    S = T_seq / T_par if T_par else 0.0
    ub = T_seq / res.critical_path_us if res.critical_path_us else 0.0
    comp_launch = res.total_compute_us + res.total_launch_us
    return {
        "T_par_us": round(T_par, 2),
        "T_seq_us": round(T_seq, 2),
        "parts": len(g.parts),
        "backends": len({p.backend for p in g.parts}),
        "overlap": res.overlap_pairs > 0,
        "speedup": round(S, 3),
        "upper_bound": round(ub, 3),
        "transfer_ratio": round(res.total_transfer_us / comp_launch, 4) if comp_launch else 0.0,
    }


def sweep_overhead(g0, assign, chip_ref) -> list[dict[str, Any]]:
    rows = []
    for c in (0, 5, 15, 35, 60, 100, 150, 250):
        chip = copy.deepcopy(chip_ref)
        chip["fixed_cross_us"] = float(c)
        r = evaluate_fixed(g0, assign, chip, flops_scale=16.0)
        r["fixed_cross_us"] = c
        rows.append(r)
    return rows


def sweep_scale(g0, assign, chip_ref) -> list[dict[str, Any]]:
    rows = []
    for s in (1, 2, 4, 8, 16, 32, 64):
        r = evaluate_fixed(g0, assign, chip_ref, flops_scale=float(s))
        r["flops_scale"] = s
        rows.append(r)
    return rows


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
def write_csv(name, rows):
    p = os.path.join(OUT, name)
    if not rows:
        return
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot(rows_ratio, rows_overhead, rows_scale):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib unavailable, skip plotting")
        return []
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.4))
    # 1) throughput ratio
    ax = axes[0]
    ax.plot([r["ratio_npu_over_gpu"] for r in rows_ratio],
            [r["speedup"] for r in rows_ratio], "o-", color="#c62828", label="S (actual)")
    ax.plot([r["ratio_npu_over_gpu"] for r in rows_ratio],
            [r["upper_bound"] for r in rows_ratio], "s--", color="#1565c0", label="S_max (bound)")
    ax.axhline(1.0, color="gray", lw=1)
    ax.set_xlabel("NPU/GPU effective throughput ratio r")
    ax.set_ylabel("speedup S = T_seq / T_par")
    ax.set_title("Gain vs backend balance (auto decision)")
    ax.legend(); ax.grid(alpha=.3)
    # 2) fixed cross overhead (fixed partition)
    ax = axes[1]
    ax.plot([r["fixed_cross_us"] for r in rows_overhead],
            [r["speedup"] for r in rows_overhead], "o-", color="#2e7d32", label="S")
    ax.plot([r["fixed_cross_us"] for r in rows_overhead],
            [r["upper_bound"] for r in rows_overhead], "s--", color="#1565c0", label="S_max")
    ax.axhline(1.0, color="gray", lw=1)
    ax.set_xlabel("cross-chip fixed overhead (us)")
    ax.set_ylabel("speedup S")
    ax.set_title("Fixed partition: overhead eats gain")
    ax.legend(); ax.grid(alpha=.3)
    # 3) model scale (fixed partition)
    ax = axes[2]
    ax.plot([r["flops_scale"] for r in rows_scale],
            [r["speedup"] for r in rows_scale], "o-", color="#6a1b9a", label="S")
    ax.plot([r["flops_scale"] for r in rows_scale],
            [r["upper_bound"] for r in rows_scale], "s--", color="#1565c0", label="S_max")
    ax.axhline(1.0, color="gray", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("compute scale (resolution / channels)")
    ax.set_ylabel("speedup S")
    ax.set_title("Scale amortizes fixed overhead")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout()
    p = os.path.join(OUT, "sensitivity.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return [p]


def main() -> int:
    g0 = base_graph()
    print(f"base graph: {len(g0.nodes)} nodes, "
          f"total FLOPs={sum(n.flops for n in g0.nodes):,}")
    assign, chip_ref = fixed_assignment(g0)
    print(f"fixed partition (force_hetero): "
          f"backends={sorted(set(assign.values()))}, nodes={len(assign)}")

    rr = sweep_ratio(g0)
    ro = sweep_overhead(g0, assign, chip_ref)
    rs = sweep_scale(g0, assign, chip_ref)
    write_csv("sweep_ratio.csv", rr)
    write_csv("sweep_overhead.csv", ro)
    write_csv("sweep_scale.csv", rs)
    imgs = plot(rr, ro, rs)

    print("\n--- throughput ratio (auto, flops x16) ---")
    for r in rr:
        print(f"  r={r['ratio_npu_over_gpu']:5.2f}  S={r['speedup']:5.2f}  "
              f"bound={r['upper_bound']:5.2f}  parts={r['parts']:2d}  "
              f"overlap={r['overlap']}  {r['verdict']}")
    print("\n--- cross-chip fixed overhead (fixed partition, flops x16) ---")
    for r in ro:
        print(f"  fixed={r['fixed_cross_us']:4d}us  S={r['speedup']:5.2f}  "
              f"bound={r['upper_bound']:5.2f}  transfer%={r['transfer_ratio']:.3f}  "
              f"T_par={r['T_par_us']}us T_seq={r['T_seq_us']}us")
    print("\n--- model scale (fixed partition) ---")
    for r in rs:
        print(f"  scale={r['flops_scale']:3d}x  S={r['speedup']:5.2f}  "
              f"bound={r['upper_bound']:5.2f}  parts={r['parts']:2d}  "
              f"overlap={r['overlap']}")

    hetero = [r["ratio_npu_over_gpu"] for r in rr
              if r["parts"] > 1 and r["backends"] >= 2]
    ov_below = [r["fixed_cross_us"] for r in ro if r["speedup"] < 1.0]
    summary = {
        "auto_hetero_ratio_window": [min(hetero), max(hetero)] if hetero else None,
        "max_auto_speedup": max(r["speedup"] for r in rr),
        "ratio_note": "auto 决策：某一后端明显更强时收敛到单后端整图，吞吐接近时才拆分并行；"
                      "曲线非单调源于启发式搜索在不同配置下的策略跳变",
        "scale_speedup_range": [rs[0]["speedup"], rs[-1]["speedup"]],
        "scale_threshold_x": next((r["flops_scale"] for r in rs if r["speedup"] >= 1.0), None),
        "scale_note": "固定切分含 CPU 分支，S 始终 <1（自动分区会拒绝它）；"
                      "但随规模放大搬运被摊薄，S 单调趋近该切分的理论上界",
        "fixed_overhead_knee_us": next(iter(ov_below), None),
        "fixed_overhead_note": "固定切分：S 随跨片固定开销单调下降；"
                              "knee 为 S 跌破 1 的开销（本切分在 0us 时已 <1）",
    }
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "ratio": rr, "overhead": ro, "scale": rs},
                  f, indent=2, ensure_ascii=False)
    print("\nSUMMARY:", json.dumps(summary, ensure_ascii=False))
    for p in imgs:
        print("image:", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
