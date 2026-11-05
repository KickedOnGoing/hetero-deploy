"""数值与结构正确性校验。

在把产物交给 C++ 运行时或 MTK 编译器之前，先在 Python 侧用 IR 解释器
把整张图按分区后的拓扑序重新执行一遍，与原始 PyTorch 模型逐元素比对。
这能独立验证：抓取、降级、BN 折叠、identity 消除、分区都不改变语义。
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .capture import CaptureResult
from .ir import Graph


class HXInterpreter:
    """HX IR 的纯 PyTorch 参考实现（与 C++ 运行时算子集一一对应）。"""

    def __init__(self, g: Graph):
        self.g = g
        self.env: dict[str, torch.Tensor] = {}
        self.trace: list[dict[str, Any]] = []

    def _get(self, tid: str) -> Any:
        if tid in self.env:
            return self.env[tid]
        t = self.g.tensors[tid]
        if t.data is not None:
            return t.data
        # 标量属性可能以字面量出现
        return t.data

    def _shape(self, tid: str) -> tuple[int, ...]:
        return tuple(self.g.tensors[tid].shape)

    def run(self, inputs: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if len(inputs) != len(self.g.inputs):
            raise ValueError(f"需要 {len(self.g.inputs)} 个输入，实际 {len(inputs)}")
        for tid, v in zip(self.g.inputs, inputs):
            self.env[tid] = v
        for t in self.g.tensors.values():
            if t.data is not None and t.id not in self.env:
                self.env[t.id] = t.data

        for n in self.g.nodes:
            x = [self._get(t) for t in n.inputs]
            y = self._exec(n, x)
            outs = y if isinstance(y, (list, tuple)) else [y]
            if len(outs) != len(n.outputs):
                raise RuntimeError(f"{n.id} 输出数量不符")
            for tid, v in zip(n.outputs, outs):
                if not isinstance(v, torch.Tensor):
                    v = torch.tensor(v)
                self.env[tid] = v
            self.trace.append({"node": n.id, "op": n.op, "part": n.part,
                               "backend": n.backend})
        return [self.env[t] for t in self.g.outputs]

    def _exec(self, n, x: list[Any]) -> Any:
        op, a = n.op, n.attrs
        oshp = self._shape(n.outputs[0]) if n.outputs else ()
        if op == "conv2d":
            return F.conv2d(x[0], x[1], x[2] if len(x) > 2 and x[2] is not None else None,
                            stride=tuple(a.get("stride", [1, 1])),
                            padding=tuple(a.get("padding", [0, 0])),
                            dilation=tuple(a.get("dilation", [1, 1])),
                            groups=int(a.get("groups", 1)))
        if op == "conv_transpose2d":
            return F.conv_transpose2d(x[0], x[1], x[2] if len(x) > 2 else None,
                                      stride=tuple(a.get("stride", [1, 1])),
                                      padding=tuple(a.get("padding", [0, 0])),
                                      groups=int(a.get("groups", 1)))
        if op == "linear":
            return F.linear(x[0], x[1], x[2] if len(x) > 2 and x[2] is not None else None)
        if op == "relu":
            return F.relu(x[0])
        if op == "relu6":
            return F.relu6(x[0])
        if op == "silu":
            return F.silu(x[0])
        if op == "sigmoid":
            return torch.sigmoid(x[0])
        if op == "tanh":
            return torch.tanh(x[0])
        if op == "gelu":
            return F.gelu(x[0])
        if op == "sqrt":
            return torch.sqrt(x[0])
        if op == "exp":
            return torch.exp(x[0])
        if op == "log":
            return torch.log(x[0])
        if op == "neg":
            return -x[0]
        if op == "abs":
            return torch.abs(x[0])
        if op == "clip":
            return torch.clamp(x[0], min=float(a.get("min", -1)), max=float(a.get("max", 1)))
        if op in ("add", "mul", "sub", "div"):
            b = x[1] if len(x) > 1 else 0
            if isinstance(b, (int, float)):
                b = torch.tensor(b, dtype=x[0].dtype)
            if op == "add":
                return x[0] + b
            if op == "mul":
                return x[0] * b
            if op == "sub":
                return x[0] - b
            return x[0] / b
        if op == "pow":
            return torch.pow(x[0], float(a.get("exponent", 2)))
        if op == "batch_norm":
            return F.batch_norm(x[0], x[3], x[4], x[1], x[2], False,
                                0.0, float(a.get("eps", 1e-5)))
        if op == "layer_norm":
            return F.layer_norm(x[0], (oshp[-1],))
        if op == "softmax":
            return F.softmax(x[0], dim=int(a.get("dim", -1)))
        if op == "global_avg_pool":
            return torch.mean(x[0], dim=(-2, -1), keepdim=True) if x[0].dim() == 4 else x[0]
        if op == "avg_pool2d":
            k = tuple(a.get("kernel", [2, 2])); s = tuple(a.get("stride", k))
            return F.avg_pool2d(x[0], k, stride=s, padding=tuple(a.get("padding", [0, 0])))
        if op == "max_pool2d":
            k = tuple(a.get("kernel", [2, 2])); s = tuple(a.get("stride", k))
            return F.max_pool2d(x[0], k, stride=s, padding=tuple(a.get("padding", [0, 0])))
        if op == "concat":
            return torch.cat(x, dim=int(a.get("dim", 1)))
        if op in ("reshape", "flatten", "identity"):
            return x[0].reshape(oshp) if oshp else x[0]
        if op == "permute":
            return x[0].permute(list(a.get("dims", [0])))
        if op == "transpose":
            return x[0].transpose(int(a.get("dim0", 0)), int(a.get("dim1", 1)))
        if op == "mean":
            d = a.get("dim", -1)
            return torch.mean(x[0], dim=tuple(d) if isinstance(d, list) else int(d), keepdim=True)
        if op == "amax":
            d = a.get("dim", -1)
            return torch.amax(x[0], dim=tuple(d) if isinstance(d, list) else int(d), keepdim=True)
        if op == "where":
            return torch.where(x[0], x[1], x[2])
        if op == "pad":
            return F.pad(x[0], list(a.get("pad", [0, 0, 0, 0])))
        if op == "upsample":
            return F.interpolate(x[0], size=oshp[-2:], mode="nearest")
        if op == "matmul":
            return torch.matmul(x[0], x[1])
        raise NotImplementedError(f"IR 解释器未实现算子 {op}")


def verify_numerics(cap: CaptureResult, g: Graph, *, atol: float = 1e-5,
                    rtol: float = 1e-4) -> dict[str, Any]:
    """原模型 vs IR 解释器，逐输出比对。"""
    inp = cap.sample_inputs
    with torch.no_grad():
        ref = cap.ep.module()(*inp)
    ref = list(ref) if isinstance(ref, (tuple, list)) else [ref]
    got = HXInterpreter(g).run(list(inp))
    if len(ref) != len(got):
        raise RuntimeError(f"输出数量不一致: ref={len(ref)} got={len(got)}")
    details = []
    ok = True
    for i, (a, b) in enumerate(zip(ref, got)):
        a, b = a.detach().cpu().float(), b.detach().cpu().float()
        if a.shape != b.shape:
            details.append({"out": i, "error": f"形状不一致 {tuple(a.shape)} vs {tuple(b.shape)}"})
            ok = False
            continue
        diff = (a - b).abs()
        maxerr = float(diff.max()) if diff.numel() else 0.0
        meanerr = float(diff.mean()) if diff.numel() else 0.0
        passed = torch.allclose(a, b, atol=atol, rtol=rtol)
        ok &= passed
        details.append({"out": i, "shape": list(a.shape), "max_abs_err": maxerr,
                        "mean_abs_err": meanerr, "pass": bool(passed)})
    return {"pass": bool(ok), "atol": atol, "rtol": rtol, "details": details}


def verify_structure(g: Graph) -> dict[str, Any]:
    """分区结构不变量：商图无环、边界闭合、每个算子可达。"""
    errs = g.validate()
    # 子图商图环检测
    pmap = {n.id: n.part for n in g.nodes}
    deps: dict[int, set[int]] = {p.id: set() for p in g.parts}
    for n in g.nodes:
        for t in n.inputs:
            prod = g.tensors[t].producer
            if prod in pmap and pmap[prod] != n.part:
                deps[n.part].add(pmap[prod])
    # 拓扑排序检测环
    indeg = {k: len(v) for k, v in deps.items()}
    succ: dict[int, list[int]] = {k: [] for k in deps}
    for d, ps in deps.items():
        for p in ps:
            succ[p].append(d)
    q = [k for k, v in indeg.items() if v == 0]
    seen = 0
    while q:
        u = q.pop(0); seen += 1
        for v in succ[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if seen != len(deps):
        errs.append("子图商图存在环")
    # 边界闭合
    declared = {(e.src, e.dst, e.tensor) for e in g.edges}
    for p in g.parts:
        for t in p.inputs:
            prod = g.tensors[t].producer
            if prod in pmap and pmap[prod] != p.id and (pmap[prod], p.id, t) not in declared:
                errs.append(f"子图 {p.id} 输入 {t} 缺少跨片边声明")
    part_order = [next((pmap[n] for n in g.topo() if n in pmap), -1) for p in []]
    return {
        "pass": not errs, "errors": errs,
        "parts": len(g.parts), "nodes": len(g.nodes),
        "cross_edges": len(g.edges),
        "subgraph_dag_acyclic": seen == len(deps),
    }
