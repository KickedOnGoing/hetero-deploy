"""训练侧抓取：把任意 PyTorch 模型固化为可编译的 ExportedProgram。

「最大化训练自由度」的落地方式
--------------------------------
训练侧**不需要**为部署改写模型，可以继续使用：
  - Python 控制流（if / for / while，基于 shape 的分支在导出时被特化）；
  - 任意 nn.Module 嵌套、参数共享、自定义 autograd.Function；
  - 任意 dtype、动态 batch（通过 dynamic_shapes 声明）；
  - 任意算子（工具链在 lower 阶段做重写 / 分解 / 降级）。

训练侧只需提供两样东西：
  1. `sample_inputs`：一组代表性输入，用于形状特化；
  2. （可选）`dynamic_shapes`：需要动态化的维度；
  3. （可选）`calibration_inputs`：PTQ 校准集。

所有"可部署化"工作（BN 折叠、算子分解、layout 推导、分区、量化）
都由编译工具链完成，不污染训练代码。
"""
from __future__ import annotations

import dataclasses as dc
import json
import os
import time
import traceback
from typing import Any, Sequence

import torch

from . import markers  # noqa: F401 —— 确保自定义算子已注册

TORCH_DTYPE_MAP = {
    torch.float32: "f32", torch.float16: "f16", torch.bfloat16: "bf16",
    torch.int8: "i8", torch.uint8: "u8", torch.int32: "i32",
    torch.int64: "i64", torch.bool: "bool",
}


@dc.dataclass
class CaptureResult:
    ep: torch.export.ExportedProgram
    meta: dict[str, Any]
    sample_inputs: tuple[Any, ...]

    @property
    def graph(self):
        return self.ep.graph_module.graph

    def node_count(self) -> int:
        return sum(1 for n in self.graph.nodes if n.op == "call_function")

    def report(self) -> dict[str, Any]:
        return {
            "torch_version": torch.__version__,
            "op_nodes": self.node_count(),
            "inputs": len(self.sample_inputs),
            "dynamic_shapes": self.meta.get("dynamic_shapes_declared", False),
            "markers": self.meta.get("marker_count", 0),
            "hard_boundaries": self.meta.get("boundary_count", 0),
            "constraints": [str(c) for c in self.meta.get("range_constraints", [])],
        }

    def save(self, path: str) -> None:
        """保存 ExportedProgram + 元信息（.pt2 + .json）。"""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.export.save(self.ep, path)
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump({"meta": _jsonable(self.meta), "report": self.report()},
                      f, indent=2, ensure_ascii=False)


def export(
    model: torch.nn.Module,
    sample_inputs: Sequence[Any],
    *,
    dynamic_shapes: Any = None,
    strict: bool = True,
    calibration_inputs: Sequence[Any] | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> CaptureResult:
    """导出模型。失败时给出可操作的修复建议，而不是抛裸异常。"""
    model.eval()
    args = tuple(sample_inputs)
    t0 = time.time()
    try:
        ep = torch.export.export(model, args, dynamic_shapes=dynamic_shapes, strict=strict)
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        hint = []
        if "constraint violation" in msg.lower() or "specialize" in msg.lower():
            hint.append("存在数据依赖的动态控制流或动态 shape：请用 dynamic_shapes 声明动态维度，"
                        "或把依赖 tensor 数据的分支改为依赖 shape 的分支。")
        if "graph break" in msg.lower():
            hint.append("遇到无法追踪的 Python 构造：可用 torch._dynamo 允许的机制，"
                        "或把该段逻辑移到模型外部（工具链支持图外前/后处理）。")
        if "custom" in msg.lower() or "operator" in msg.lower():
            hint.append("自定义算子需要注册 FakeTensor/Meta 实现才能被 torch.export 捕获。")
        raise RuntimeError(
            "torch.export 抓取失败。\n"
            + ("\n".join("  · " + h for h in hint) if hint else "")
            + f"\n原始错误: {msg}\n{traceback.format_exc(limit=3)}"
        ) from e

    gm = ep.graph_module.graph
    marker_count = sum(1 for n in gm.nodes if "backend_hint" in str(n.target))
    boundary_count = sum(1 for n in gm.nodes if "partition_point" in str(n.target))
    op_names = sorted({str(n.target) for n in gm.nodes if n.op == "call_function"})

    meta: dict[str, Any] = {
        "producer": "hx-capture",
        "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "capture_ms": round((time.time() - t0) * 1000, 2),
        "torch_version": torch.__version__,
        "dynamic_shapes_declared": dynamic_shapes is not None,
        "marker_count": marker_count,
        "boundary_count": boundary_count,
        "op_set": op_names,
        "range_constraints": list(str(k) for k in getattr(ep, "range_constraints", {})),
        "input_shapes": [list(t.shape) if hasattr(t, "shape") else None for t in args],
        "input_dtypes": [str(t.dtype) if hasattr(t, "dtype") else None for t in args],
    }
    if calibration_inputs is not None:
        meta["calibration_count"] = len(calibration_inputs)
        meta["calibration_file"] = "calibration.pt"
    if extra_meta:
        meta.update(extra_meta)

    res = CaptureResult(ep=ep, meta=meta, sample_inputs=args)
    if calibration_inputs is not None:
        res.meta["_calibration_inputs"] = list(calibration_inputs)
    return res


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items() if k != "_calibration_inputs"}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    return str(x)
