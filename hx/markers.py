"""训练侧可选标记（软提示 / 强制边界）。

**关键设计**：标记是可选的，且只是"提示"，不是约束。
- 不加任何标记：编译器完全按代价模型自动分区，训练侧零改动；
- 加了标记：仅改变分区搜索的先验，编译器在标记不可行时会自动忽略
  （例如把不支持的算子标记为 npu），绝不导致编译失败；
- 只有 `partition()` 是硬边界，用于训练侧明确知道必须切开的语义位置。

这样保证"最大化训练自由度"：训练代码用原生 PyTorch 任意写法，
部署决策 100% 留在编译期。
"""
from __future__ import annotations

from typing import Any

import torch

# ---------------------------------------------------------------------------
# 注册两个纯标识算子。它们不改变数值（返回输入本身），只在 FX 图里留痕。
# ---------------------------------------------------------------------------
try:
    _lib = torch.library.Library("hx", "DEF")  # noqa: F811
    _lib.define("backend_hint(Tensor x, str backend) -> Tensor")
    _lib.define("partition_point(Tensor x, str name) -> Tensor")
    _HX_LIB = _lib
except RuntimeError:
    # 同一进程重复导入时库已存在
    _HX_LIB = None


def _identity(x: torch.Tensor, *_: Any) -> torch.Tensor:
    return x


if _HX_LIB is not None:
    for _name in ("backend_hint", "partition_point"):
        # CompositeExplicitAutograd 让 eager CPU 也能执行（训练/推理都不受影响）
        _HX_LIB.impl(_name, _identity, "CompositeExplicitAutograd")
        _HX_LIB.impl(_name, _identity, "Meta")
else:  # 重复导入时补注册到已存在的库
    _impl_lib = torch.library.Library("hx", "FRAGMENT")
    for _name in ("backend_hint", "partition_point"):
        try:
            _impl_lib.impl(_name, _identity, "CompositeExplicitAutograd")
        except RuntimeError:
            pass


def prefer(x: torch.Tensor, backend: str) -> torch.Tensor:
    """软提示：建议从此处开始的下游算子优先放在指定后端。

    backend ∈ {"npu", "gpu", "cpu"}。编译器会校验算子能力，忽略非法提示。
    """
    assert backend in ("npu", "gpu", "cpu"), f"未知后端: {backend}"
    return torch.ops.hx.backend_hint(x, backend)


def partition(x: torch.Tensor, name: str = "") -> torch.Tensor:
    """硬边界：要求工具链在此处至少切开一个子图。

    用于训练侧明确知道存在语义/内存/精度边界的位置（如前后处理、
    分支汇合点）。这是训练侧唯一需要主动写的部署相关代码。
    """
    return torch.ops.hx.partition_point(x, name)


def mark(module: torch.nn.Module, backend: str) -> torch.nn.Module:
    """模块级软提示：给一个子模块的输出打上后端偏好。

    用法：`self.encoder = hx.mark(nn.Sequential(...), "npu")`
    仅影响分区搜索先验，不改变模型数值，训练/推理均可正常执行。
    """
    assert backend in ("npu", "gpu", "cpu")
    if getattr(module, "_hx_marked", False):
        return module
    orig_forward = module.forward

    def _wrapped(*args: Any, **kwargs: Any) -> Any:
        out = orig_forward(*args, **kwargs)
        if isinstance(out, torch.Tensor):
            return torch.ops.hx.backend_hint(out, backend)
        if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor):
            first = torch.ops.hx.backend_hint(out[0], backend)
            return type(out)((first, *out[1:]))
        return out

    module.forward = _wrapped  # type: ignore[method-assign]
    module._hx_marked = True  # type: ignore[attr-defined]
    module._hx_backend = backend  # type: ignore[attr-defined]
    return module


HINT_OPS = ("hx::backend_hint", "hx::partition_point")
