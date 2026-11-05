"""HX：面向 MTK CPU/GPU/NPU 的算法异构部署 SDK 工具链（Python 编译侧）。

训练侧只需要：
    import hx
    pkg = hx.compile_model(model, (sample_input,), "build/demo")

可选（绝不强制）：
    hx.prefer(x, "npu")     # 后端软提示
    hx.boundary(x, "fuse")  # 硬切分边界
    hx.mark(module, "gpu")  # 模块级提示
"""
from .api import CompileResult, compile_model
from .capture import CaptureResult, export
from .cost import CostModel, simulate
from .ir import Graph, Node, Tensor
from .memory import plan_buffers, verify_buffer_plan
from .markers import mark, partition as boundary, prefer
from .opset import DEFAULT_CHIP, load_chip
from .partition import PartitionConfig, explain, partition
from .planner import make_plan
from .verify import HXInterpreter, verify_numerics, verify_structure

__version__ = "0.2.0"
__all__ = [
    "compile_model", "CompileResult", "export", "CaptureResult",
    "Graph", "Node", "Tensor", "CostModel", "simulate",
    "PartitionConfig", "partition", "explain", "make_plan",
    "plan_buffers", "verify_buffer_plan",
    "DEFAULT_CHIP", "load_chip",
    "prefer", "boundary", "mark",
    "HXInterpreter", "verify_numerics", "verify_structure",
]
