"""一站式编译入口：训练模型 -> 部署包。

    from hx import compile_model
    result = compile_model(model, (torch.randn(1,3,224,224),), "build/demo")

全流程：抓取 -> 降级/重写 -> 分区搜索 -> 计划评估 -> 双重校验 -> 产物生成。
任何一步失败都会给出可操作的错误信息，不做静默降级。
"""
from __future__ import annotations

import dataclasses as dc
import json
import os
from typing import Any, Sequence

import torch

from .capture import CaptureResult, export
from .codegen import Codegen
from .cost import CostModel, simulate
from .ir import Graph
from .lower import lower
from .opset import load_chip
from .partition import PartitionConfig, explain, partition
from .planner import make_plan
from .verify import verify_numerics, verify_structure


@dc.dataclass
class CompileResult:
    graph: Graph
    capture: CaptureResult
    cost_model: CostModel
    plan: dict[str, Any]
    numerics: dict[str, Any]
    structure: dict[str, Any]
    files: dict[str, str]
    outdir: str

    def summary(self) -> dict[str, Any]:
        p = self.plan
        return {
            "verdict": p["verdict"],
            "reason": p["reason"],
            "nodes": len(self.graph.nodes),
            "parts": len(self.graph.parts),
            "backends": sorted({x.backend for x in self.graph.parts}),
            "cross_backend_edges": p["constraints"]["cross_backend_edges"],
            "latency_us": p["simulation"]["makespan_us"],
            "baseline_us": p["baselines_single_backend_us"],
            "speedup": p["bound"]["latency_speedup"],
            "theoretical_upper_bound": p["bound"]["theoretical_upper_bound_vs_baseline"],
            "steady_fps": p["pipeline"]["steady_fps"],
            "parallel_execution": p["parallel_execution"]["observed"],
            "overlap_pairs": p["parallel_execution"]["overlap_pairs"],
            "numerics_pass": self.numerics["pass"],
            "structure_pass": self.structure["pass"],
        }


def compile_model(
    model: torch.nn.Module,
    sample_inputs: Sequence[Any],
    outdir: str,
    *,
    strategy: str = "auto",
    chip: str | None = None,
    dynamic_shapes: Any = None,
    strict: bool = True,
    fold_bn: bool = True,
    eliminate_identity: bool = True,
    decompose_silu: bool = False,
    pipeline_depth: int = 2,
    iters: int = 600,
    restarts: int = 3,
    seed: int = 2026,
    respect_hints: bool = True,
    enforce_min_npu_group: bool = True,
    atol: float = 1e-5,
    rtol: float = 1e-4,
    calibration_inputs: Sequence[Any] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> CompileResult:
    os.makedirs(outdir, exist_ok=True)

    # 1) 抓取（训练侧零约束）
    cap = export(model, sample_inputs, dynamic_shapes=dynamic_shapes, strict=strict,
                 calibration_inputs=calibration_inputs, extra_meta=extra_meta)

    # 2) 降级 + 可部署化重写
    g = lower(cap, fold_bn=fold_bn, eliminate_identity=eliminate_identity,
              decompose_silu=decompose_silu)

    # 3) 分区
    chip_cfg = load_chip(chip)
    cm = CostModel(chip_cfg)
    cfg = PartitionConfig(strategy=strategy, iters=iters, restarts=restarts,
                          seed=seed, respect_hints=respect_hints,
                          enforce_min_npu_group=enforce_min_npu_group,
                          pipeline_depth=pipeline_depth)
    g, trace = partition(g, cm, cfg)

    # 4) 计划与判定
    plan = make_plan(g, cm, pipeline_depth=pipeline_depth)
    g.name = getattr(model, "__class__.__name__", "model")
    g.meta["chip"] = chip_cfg
    g.meta["strategy"] = strategy

    # 5) 校验（数值 + 结构），失败直接抛错
    numerics = verify_numerics(cap, g, atol=atol, rtol=rtol)
    structure = verify_structure(g)
    if not numerics["pass"]:
        raise RuntimeError("IR 数值校验失败: " + json.dumps(numerics, ensure_ascii=False))
    if not structure["pass"]:
        raise RuntimeError("分区结构校验失败: " + json.dumps(structure, ensure_ascii=False))

    # 6) 产物
    with torch.no_grad():
        ref = cap.ep.module()(*cap.sample_inputs)
    expected = tuple(ref) if isinstance(ref, (tuple, list)) else (ref,)
    cg = Codegen(g, cm, outdir)
    files = cg.run(cap.sample_inputs, expected)

    verification = {
        "numerics": numerics, "structure": structure,
        "partition_explain": explain(g),
        "search_trace": trace.to_dict(),
        "simulation": simulate(g, cm).summary(),
    }
    vp = os.path.join(outdir, "verification.json")
    with open(vp, "w", encoding="utf-8") as f:
        json.dump(verification, f, indent=2, ensure_ascii=False)
    files["verification"] = vp

    res = CompileResult(g, cap, cm, plan, numerics, structure, files, outdir)
    with open(os.path.join(outdir, "compile_summary.json"), "w", encoding="utf-8") as f:
        json.dump(res.summary(), f, indent=2, ensure_ascii=False)
    return res
