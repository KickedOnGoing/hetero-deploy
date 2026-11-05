"""FX 图降级到 HX IR，并执行"可部署化重写"。

这一层是训练自由度与部署约束之间的**隔离层**：
训练图里出现什么（BN、SiLU、GELU、动态控制流、hint 标记）都无所谓，
这里统一变成后端能力表能理解的一组标准算子。
"""
from __future__ import annotations

import math
from typing import Any

import torch

from .capture import CaptureResult, TORCH_DTYPE_MAP
from .ir import Graph, Node, Tensor
from .opset import estimate_flops

# ---------------------------------------------------------------------------
# ATen -> HX IR 映射
# ---------------------------------------------------------------------------
_UNARY = {
    "aten.relu.default": "relu", "aten.relu_.default": "relu",
    "aten.relu6.default": "relu6", "aten.silu.default": "silu",
    "aten.silu_.default": "silu", "aten.sigmoid.default": "sigmoid",
    "aten.sigmoid_.default": "sigmoid", "aten.tanh.default": "tanh",
    "aten.tanh_.default": "tanh", "aten.gelu.default": "gelu",
    "aten.sqrt.default": "sqrt", "aten.sqrt_.default": "sqrt",
    "aten.exp.default": "exp", "aten.log.default": "log",
    "aten.neg.default": "neg", "aten.abs.default": "abs",
    "aten.hardswish.default": "silu", "aten.hardswish_.default": "silu",
}
_BINARY = {
    "aten.add.Tensor": "add", "aten.mul.Tensor": "mul",
    "aten.sub.Tensor": "sub", "aten.div.Tensor": "div",
    "aten.pow.Tensor_Scalar": "pow",
}
_POOL = {
    "aten.avg_pool2d.default": "avg_pool2d",
    "aten.max_pool2d.default": "max_pool2d",
    "aten.max_pool2d_with_indices.default": "max_pool2d",
}


def _d(v: Any) -> Any:
    """把 SymInt / FakeTensor 元信息转成 Python 原生值。"""
    if isinstance(v, (list, tuple)):
        return [_d(x) for x in v]
    if isinstance(v, torch.Tensor):
        return v
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return v
    if isinstance(v, str) or v is None:
        return v
    if hasattr(v, "node") and hasattr(v.node, "hint") and v.node.hint is not None:
        return int(v.node.hint)
    try:
        return int(v)
    except Exception:  # noqa: BLE001
        return str(v)


def _spec(val: Any, default_shape: tuple[int, ...] = ()) -> tuple[tuple[int, ...], str]:
    if hasattr(val, "shape") and hasattr(val, "dtype"):
        shp = tuple(int(_d(x)) for x in val.shape)
        return shp, TORCH_DTYPE_MAP.get(val.dtype, "f32")
    return default_shape, "f32"


class Lowerer:
    def __init__(self, cap: CaptureResult, *, fold_bn: bool = True,
                 eliminate_identity: bool = True, decompose_silu: bool = False):
        self.cap = cap
        self.ep = cap.ep
        self.g = Graph("hx_model")
        self.env: dict[str, Any] = {}
        self.hint_of: dict[str, str] = {}
        self.markers: list[tuple[str, str]] = []
        self.boundaries: set[str] = set()
        self.cur_hint: str | None = None
        self.fold_bn = fold_bn
        self.eliminate_identity = eliminate_identity
        self.decompose_silu = decompose_silu
        self.stats: dict[str, int] = {
            "aten_nodes": 0, "unsupported": 0, "hints": 0,
            "bn_folded": 0, "identity_removed": 0, "silu_decomposed": 0,
        }

    # ---------------- 张量工具 ----------------
    def _tid(self, base: str, idx: int = 0) -> str:
        return base if idx == 0 else f"{base}:{idx}"

    def _mk_tensor(self, tid: str, val: Any, kind: str,
                   data: torch.Tensor | None = None, producer: str = "") -> Tensor:
        shp, dt = _spec(val)
        t = Tensor(id=tid, dtype=dt, shape=shp, kind=kind, data=data, producer=producer)
        self.g.add_tensor(t)
        return t

    def _resolve(self, x: Any) -> Any:
        """把 FX 参数解析成 tensor id / 嵌套结构 / 常量。"""
        if isinstance(x, torch.fx.Node):
            return self.env.get(x.name)
        if isinstance(x, (list, tuple)):
            return type(x)(self._resolve(v) for v in x)
        return _d(x)

    def _flat(self, x: Any) -> list[str]:
        if isinstance(x, str):
            return [x]
        if isinstance(x, (list, tuple)):
            out: list[str] = []
            for v in x:
                out.extend(self._flat(v))
            return out
        return []

    def _hint_for(self, inputs: list[str]) -> str | None:
        for t in inputs:
            h = self.hint_of.get(t)
            if h:
                return h
        return None

    # ---------------- 主流程 ----------------
    def run(self) -> Graph:
        ep = self.ep
        sig = {getattr(s.arg, "name", None): s for s in ep.graph_signature.input_specs}
        state = dict(ep.state_dict)
        consts = dict(getattr(ep, "constants", {}) or {})

        # 1) placeholder：区分用户输入 / 参数 / 缓冲 / 常量
        for n in ep.graph_module.graph.nodes:
            if n.op != "placeholder":
                continue
            s = sig.get(n.name)
            kind = getattr(s, "kind", None)
            kname = str(kind).split(".")[-1] if kind is not None else "USER_INPUT"
            val = n.meta.get("val")
            if kname in ("PARAMETER", "BUFFER", "CONSTANT"):
                target = getattr(s, "target", None)
                raw = state.get(target, consts.get(target)) if target else None
                if raw is None:
                    raw = getattr(n.meta.get("val"), "constant", None)
                if raw is None:
                    raise RuntimeError(f"无法取得占位符 {n.name} 的权重数据 (target={target})")
                t = self._mk_tensor(n.name, val if val is not None else raw,
                                    "weight" if kname == "PARAMETER" else "constant",
                                    raw.detach().cpu().contiguous())
                self.env[n.name] = t.id
                self.g.meta.setdefault("weights", []).append(t.id)
            else:
                t = self._mk_tensor(n.name, val, "input")
                self.g.inputs.append(t.id)
                self.env[n.name] = t.id

        # 2) call_function
        for n in ep.graph_module.graph.nodes:
            if n.op == "call_function":
                self.stats["aten_nodes"] += 1
                self._lower_call(n)

        # 3) output
        for n in ep.graph_module.graph.nodes:
            if n.op == "output":
                outs = self._flat(self._resolve(n.args[0]))
                if not outs:
                    raise RuntimeError("图没有输出张量")
                self.g.outputs = outs
                for o in outs:
                    self.g.tensors[o].kind = "output"

        # 4) 重写
        if self.fold_bn:
            self._fold_batch_norm()
        if self.eliminate_identity:
            self._eliminate_identity()
        if self.decompose_silu:
            self._decompose_silu()
        self._propagate_hints_backward()

        # 5) 计算 FLOPs + 收集 hint 元数据
        for node in self.g.nodes:
            node.flops = estimate_flops(node, self.g)
        self.g.meta["capture"] = self.cap.meta
        self.g.meta["lower_stats"] = dict(self.stats)
        self.g.meta["hard_boundaries"] = sorted(self.boundaries)
        self.g.meta["hints"] = {k: v for k, v in self.hint_of.items() if v}
        return self.g

    # ---------------- 算子降级 ----------------
    def _lower_call(self, n: torch.fx.Node) -> None:
        target = str(n.target)
        args = n.args
        val = n.meta.get("val")
        resolved = [self._resolve(a) for a in args]

        # --- 自定义标记 ---
        if "backend_hint" in target or "partition_point" in target:
            src = self._flat(resolved[0])
            if not src:
                return
            tid = src[0]
            if "backend_hint" in target:
                backend = str(_d(args[1]))
                self.hint_of[tid] = backend
                self.markers.append((tid, backend))
                self.cur_hint = backend
                self.stats["hints"] += 1
            else:
                self.boundaries.add(tid)
            self.env[n.name] = tid  # identity：不产生 IR 节点
            return

        if "getitem" in target or target.endswith(".__getitem__"):
            seq = resolved[0]
            idx = int(_d(args[1]))
            self.env[n.name] = seq[idx] if isinstance(seq, (list, tuple)) else seq
            return

        # 收集所有参数位里的张量 id（常量 / 标量 / list[int] 会被 _flat 忽略）
        ins = []
        for rv in resolved:
            ins.extend(self._flat(rv))
        ins = list(dict.fromkeys(x for x in ins if isinstance(x, str)))

        # 输出张量（可能是 tuple）
        if isinstance(val, (tuple, list)):
            out_ids = [self._tid(n.name, i) for i in range(len(val))]
            for oid, v in zip(out_ids, val):
                self._mk_tensor(oid, v, "activation", producer=n.id if False else n.name)
            self.env[n.name] = out_ids
        else:
            out_ids = [self._tid(n.name, 0)]
            self._mk_tensor(out_ids[0], val, "activation", producer=n.name)
            self.env[n.name] = out_ids[0]

        op, attrs, real_ins, real_outs = self._map(target, resolved, args, ins, out_ids, val)
        if op is None:
            self.stats["unsupported"] += 1
            raise RuntimeError(
                f"不支持的 ATen 算子: {target}\n"
                f"  节点: {n.name}, args={args}\n"
                f"  处理方式：在 hx/lower.py 的映射表增加该算子，或调用 "
                f"ep.run_decompositions() 先分解，或用 hx.partition() 把它隔离到 CPU 子图。"
            )
        # 多输出算子（如 max_pool2d_with_indices）只保留第 0 项
        node = Node(
            id=n.name, op=op, inputs=real_ins, outputs=real_outs,
            attrs=attrs, hint=self._hint_for(real_ins), name=n.name,
        )
        self.g.add_node(node)
        for o in real_outs:
            self.g.tensors[o].producer = node.id
        h = self._hint_for(real_ins)
        for o in real_outs:
            if h:
                self.hint_of[o] = h

    def _map(self, target: str, r: list[Any], raw_args: tuple,
             ins: list[str], outs: list[str], val: Any):
        """返回 (op, attrs, inputs, outputs)。"""
        a: dict[str, Any] = {}

        # 纯 identity / 视图类
        if target in ("aten.clone.default", "aten.alias.default", "aten.lift_fresh_copy.default",
                      "aten._to_copy.default", "aten.detach.default", "aten.contiguous.default"):
            return "identity", {}, [r[0]] if r and isinstance(r[0], str) else ins[:1], outs[:1]
        if target in ("aten.view.default", "aten._unsafe_view.default",
                      "aten.reshape.default", "aten._reshape_copy.default"):
            a["shape"] = [int(_d(x)) for x in (r[1] if len(r) > 1 else [])]
            return "reshape", a, ins[:1], outs[:1]
        if target == "aten.flatten.using_ints":
            a["start_dim"] = int(_d(raw_args[1])) if len(raw_args) > 1 else 1
            return "flatten", a, ins[:1], outs[:1]
        if target in ("aten.permute.default", "aten.permute_copy.default"):
            a["dims"] = [int(_d(x)) for x in r[1]]
            return "permute", a, ins[:1], outs[:1]
        if target == "aten.transpose.int":
            a["dim0"] = int(_d(raw_args[1])); a["dim1"] = int(_d(raw_args[2]))
            return "transpose", a, ins[:1], outs[:1]
        if target in ("aten.t.default",):
            return "transpose", {"dim0": 0, "dim1": 1}, ins[:1], outs[:1]

        # 卷积 / 全连接
        if target in ("aten.conv2d.default", "aten.convolution.default",
                      "aten.conv_transpose2d.default"):
            if target == "aten.convolution.default":
                a.update(stride=_d(r[3]), padding=_d(r[4]), dilation=_d(r[5]),
                         groups=int(_d(r[8])) if len(r) > 8 else 1,
                         transposed=bool(_d(r[6])) if len(r) > 6 else False)
            else:
                a.update(stride=_d(r[3]) if len(r) > 3 else [1, 1],
                         padding=_d(r[4]) if len(r) > 4 else [0, 0],
                         dilation=_d(r[5]) if len(r) > 5 else [1, 1],
                         groups=int(_d(r[6])) if len(r) > 6 else 1)
            return ("conv_transpose2d" if target == "aten.conv_transpose2d.default"
                    else "conv2d"), a, ins[:3], outs[:1]
        if target in ("aten.linear.default", "aten.addmm.default", "aten.mm.default",
                      "aten.matmul.default"):
            if target == "aten.addmm.default":
                return "linear", {}, [ins[1], ins[2], ins[0]], outs[:1]
            return "linear", {}, ins[:3], outs[:1]

        # 逐元素
        if target in _UNARY:
            return _UNARY[target], {}, ins[:1], outs[:1]
        if target in _BINARY:
            return _BINARY[target], {}, ins[:2], outs[:1]
        if target in ("aten.hardtanh.default", "aten.clamp.default"):
            return "clip", {"min": _d(raw_args[1]) if len(raw_args) > 1 else -1,
                            "max": _d(raw_args[2]) if len(raw_args) > 2 else 1}, ins[:1], outs[:1]

        # 归一化
        if target == "aten.batch_norm.default":
            a = {"eps": float(_d(raw_args[7])) if len(raw_args) > 7 else 1e-5,
                 "training": bool(_d(raw_args[5])) if len(raw_args) > 5 else False}
            return "batch_norm", a, [ins[0], ins[1], ins[2], ins[3], ins[4]], outs[:1]
        if target == "aten.native_layer_norm.default":
            return "layer_norm", {"eps": 1e-5}, ins[:1], outs[:1]
        if target in ("aten.softmax.int", "aten._softmax.default", "aten.softmax.default"):
            return "softmax", {"dim": int(_d(raw_args[1])) if len(raw_args) > 1 else -1}, ins[:1], outs[:1]

        # 池化
        if target in _POOL or target == "aten.adaptive_avg_pool2d.default":
            if target == "aten.adaptive_avg_pool2d.default":
                osz = _d(raw_args[1])
                oshp, _ = _spec(val)
                if list(osz) == [1, 1] or (len(oshp) >= 2 and oshp[-1] == 1 and oshp[-2] == 1):
                    return "global_avg_pool", {}, ins[:1], outs[:1]
                ishp, _ = _spec(raw_args[0].meta.get("val")) if hasattr(raw_args[0], "meta") else ([], "f32")
                if len(ishp) == 4 and len(oshp) == 4:
                    kh = ishp[-2] // oshp[-2]; kw = ishp[-1] // oshp[-1]
                    return "avg_pool2d", {"kernel": [kh, kw], "stride": [kh, kw]}, ins[:1], outs[:1]
                return "global_avg_pool", {}, ins[:1], outs[:1]
            kernel = _d(raw_args[1]) if len(raw_args) > 1 else [2, 2]
            stride = _d(raw_args[2]) if len(raw_args) > 2 else kernel
            pad = _d(raw_args[3]) if len(raw_args) > 3 else [0, 0]
            return _POOL[target], {"kernel": kernel, "stride": stride, "padding": pad}, ins[:1], outs[:1]

        # 拼接 / 规约
        if target == "aten.cat.default":
            return "concat", {"dim": int(_d(raw_args[1])) if len(raw_args) > 1 else 1}, ins, outs[:1]
        if target == "aten.mean.dim":
            return "mean", {"dim": _d(raw_args[1])}, ins[:1], outs[:1]
        if target == "aten.amax.default":
            return "amax", {"dim": _d(raw_args[1])}, ins[:1], outs[:1]
        if target == "aten.where.self":
            return "where", {}, ins[:3], outs[:1]
        if target == "aten.constant_pad_nd.default":
            return "pad", {"pad": _d(raw_args[1])}, ins[:1], outs[:1]
        if target == "aten.upsample_nearest2d.vec":
            return "upsample", {"scale": _d(raw_args[1])}, ins[:1], outs[:1]

        return None, {}, [], []

    # ---------------- 重写 pass ----------------
    def _fold_batch_norm(self) -> None:
        """conv/linear 后紧接的 eval-mode BN 折叠进权重，消除一个算子与一次访存。"""
        g = self.g
        consumers: dict[str, list[Node]] = {}
        for n in g.nodes:
            for t in n.inputs:
                consumers.setdefault(t, []).append(n)
        remove: set[str] = set()
        for bn in list(g.nodes):
            if bn.op != "batch_norm" or bn.id in remove:
                continue
            if bn.attrs.get("training"):
                continue
            x, gamma_id, beta_id, mean_id, var_id = (bn.inputs + [None] * 5)[:5]
            conv = None
            for c in g.nodes:
                if c.id == g.tensors[x].producer and c.op in ("conv2d", "linear"):
                    conv = c
                    break
            if conv is None or len(consumers.get(conv.outputs[0], [])) != 1:
                continue
            try:
                w = g.tensors[conv.inputs[1]].data
                b = g.tensors[conv.inputs[2]].data if len(conv.inputs) > 2 else None
                gamma, beta = g.tensors[gamma_id].data, g.tensors[beta_id].data
                mean, var = g.tensors[mean_id].data, g.tensors[var_id].data
                eps = float(bn.attrs.get("eps", 1e-5))
                s = gamma / torch.sqrt(var + eps)
                if conv.op == "conv2d":
                    w2 = w * s.reshape([-1] + [1] * (w.dim() - 1))
                else:
                    w2 = w * s.reshape([-1, 1]) if w.dim() == 2 else w * s
                b0 = b if b is not None else torch.zeros_like(gamma)
                b2 = (b0 - mean) * s + beta
                g.tensors[conv.inputs[1]].data = w2.contiguous()
                if len(conv.inputs) > 2:
                    g.tensors[conv.inputs[2]].data = b2.contiguous()
                else:
                    tid = conv.id + "_bias"
                    g.add_tensor(Tensor(tid, dtype="f32", shape=tuple(b2.shape),
                                        kind="bias", data=b2.contiguous()))
                    conv.inputs.append(tid)
                    g.meta.setdefault("weights", []).append(tid)
            except Exception:  # noqa: BLE001 —— 折叠失败则保留 BN，交给 CPU 子图
                continue
            # 重连：conv 直接产出 bn 的输出张量
            old_out, new_out = conv.outputs[0], bn.outputs[0]
            conv.outputs = [new_out]
            g.tensors[new_out].producer = conv.id
            if old_out in g.tensors:
                del g.tensors[old_out]
            remove.add(bn.id)
            self.stats["bn_folded"] += 1
        if remove:
            g.nodes = [n for n in g.nodes if n.id not in remove]

    def _eliminate_identity(self) -> None:
        """消除 clone / alias / to_copy 等无计算节点，保持张量 id 不变。"""
        g = self.g
        alias: dict[str, str] = {}
        for n in list(g.nodes):
            if n.op == "identity" and n.inputs and n.outputs:
                alias[n.outputs[0]] = n.inputs[0]
        if not alias:
            return
        changed = True
        while changed:
            changed = False
            for k, v in list(alias.items()):
                if v in alias and alias[v] != k:
                    alias[k] = alias[v]; changed = True
        for n in g.nodes:
            n.inputs = [alias.get(t, t) for t in n.inputs]
        for t in g.tensors.values():
            if t.producer in alias:
                t.producer = alias[t.producer]
        g.inputs = [alias.get(t, t) for t in g.inputs]
        g.outputs = [alias.get(t, t) for t in g.outputs]
        drop = set()
        for n in list(g.nodes):
            if n.op == "identity":
                drop.add(n.outputs[0])
                g.nodes.remove(n)
                self.stats["identity_removed"] += 1
        for t in list(drop):
            if t in g.tensors and t not in g.outputs and t not in g.inputs:
                del g.tensors[t]

    def _propagate_hints_backward(self) -> None:
        """把 hx.mark(模块, backend) 的标记反向传播到该模块内部的算子。

        mark 在模块**输出**处留痕，正向传播只能影响下游；这里沿 producer
        链反向回溯，直到遇到分叉张量（被多个节点消费）为止，从而把整条
        分支的算子都打上软提示，同时不污染共享主干。提示是软的：
        partition 会先校验算子能力，非法提示自动忽略。
        """
        g = self.g
        if not self.markers:
            return
        consumers: dict[str, int] = {}
        for n in g.nodes:
            for t in n.inputs:
                consumers[t] = consumers.get(t, 0) + 1
        forks = {t for t, c in consumers.items() if c > 1}
        by_id = {n.id: n for n in g.nodes}
        for tid, backend in self.markers:
            queue, visited = [tid], set()
            while queue:
                t = queue.pop(0)
                if t in visited or t not in g.tensors:
                    continue
                visited.add(t)
                prod = g.tensors[t].producer
                node = by_id.get(prod)
                if node is None:
                    continue
                if node.hint and node.hint != backend:
                    continue
                node.hint = backend
                self.stats["hints"] += 0  # 计数由 marker 统计，避免重复
                for inp in node.inputs:
                    if inp in forks:      # 共享主干，不再向上污染
                        continue
                    queue.append(inp)

    def _decompose_silu(self) -> None:
        """可选：SiLU(x) = x * sigmoid(x)，让不支持 SiLU 的 NPU 也能承载。

        默认关闭：分解会增加算子数和一次中间张量写回，是否划算由分区器
        依据代价模型决定（CLI: --decompose-silu 开启）。
        """
        g = self.g
        new_nodes: list[Node] = []
        for n in list(g.nodes):
            if n.op != "silu":
                new_nodes.append(n)
                continue
            x, y = n.inputs[0], n.outputs[0]
            mid = n.id + "_sig"
            g.add_tensor(Tensor(mid, dtype=g.tensors[y].dtype,
                                shape=g.tensors[y].shape, kind="activation", producer=mid))
            s = Node(mid, "sigmoid", [x], [mid], hint=n.hint, name=n.name + "_sig")
            m = Node(n.id, "mul", [x, mid], [y], hint=n.hint, name=n.name + "_mul")
            new_nodes.extend([s, m])
            self.stats["silu_decomposed"] += 1
        g.nodes = new_nodes


def lower(cap: CaptureResult, **kw: Any) -> Graph:
    return Lowerer(cap, **kw).run()
