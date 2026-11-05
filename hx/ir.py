"""HX 中间表示：训练图与部署计划之间的唯一契约。

设计原则
--------
* IR 与后端无关：只描述算子、张量、依赖，不描述任何 MTK / LiteRT / NNAPI 细节。
* 可序列化：manifest.json 是 Python 编译器与 C++ 运行时之间的唯一交付物。
* 可校验：任何分区结果都必须通过 Graph.validate()，保证不重不漏、无环、边界闭合。
"""
from __future__ import annotations

import dataclasses as dc
import json
import math
from typing import Any, Iterable

DTYPE_BYTES: dict[str, int] = {
    "f32": 4, "f16": 2, "bf16": 2, "i8": 1, "u8": 1,
    "i16": 2, "i32": 4, "i64": 8, "bool": 1,
}

KINDS = ("input", "output", "weight", "bias", "constant", "activation")


@dc.dataclass
class Tensor:
    """一个张量。权重张量的 payload 落在 weights.bin，按 offset 索引。"""
    id: str
    dtype: str = "f32"
    shape: tuple[int, ...] = ()
    kind: str = "activation"
    offset: int = -1             # weights.bin 中的字节偏移；-1 表示运行时激活
    producer: str = ""           # 产出该张量的节点 id
    part: int = -1               # 归属子图（仅权重有意义）
    data: Any = None             # 仅编译期存在，不进 manifest

    @property
    def numel(self) -> int:
        if not self.shape:
            return 0
        return math.prod(int(d) for d in self.shape)

    @property
    def nbytes(self) -> int:
        return self.numel * DTYPE_BYTES.get(self.dtype, 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "dtype": self.dtype, "shape": list(self.shape),
            "kind": self.kind, "offset": self.offset, "bytes": self.nbytes,
            "producer": self.producer,
        }

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Tensor":
        return Tensor(
            id=d["id"], dtype=d.get("dtype", "f32"),
            shape=tuple(d.get("shape", ())), kind=d.get("kind", "activation"),
            offset=int(d.get("offset", -1)), producer=d.get("producer", ""),
        )


@dc.dataclass
class Node:
    """一个 IR 算子节点。inputs/outputs 均为 Tensor.id。"""
    id: str
    op: str
    inputs: list[str]
    outputs: list[str]
    attrs: dict[str, Any] = dc.field(default_factory=dict)
    flops: int = 0
    part: int = -1
    backend: str = ""
    hint: str | None = None       # 训练侧软提示，仅作搜索先验
    name: str = ""               # 原始 FX 节点名，便于溯源

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id, "op": self.op, "inputs": self.inputs,
            "outputs": self.outputs, "attrs": _jsonable(self.attrs),
            "flops": self.flops, "part": self.part,
            "backend": self.backend, "name": self.name,
        }
        if self.hint:
            d["hint"] = self.hint
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Node":
        return Node(
            id=d["id"], op=d["op"], inputs=list(d["inputs"]),
            outputs=list(d["outputs"]), attrs=d.get("attrs", {}),
            flops=int(d.get("flops", 0)), part=int(d.get("part", -1)),
            backend=d.get("backend", ""), hint=d.get("hint"),
            name=d.get("name", ""),
        )


@dc.dataclass
class Part:
    """一个编译产物：绑定到单一后端的连续子图。"""
    id: int
    backend: str
    nodes: list[str]
    inputs: list[str]            # 边界输入张量
    outputs: list[str]           # 边界输出张量
    est_us: float = 0.0
    artifact: str = ""
    compile_flags: dict[str, Any] = dc.field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "backend": self.backend, "nodes": self.nodes,
            "inputs": self.inputs, "outputs": self.outputs,
            "est_us": round(self.est_us, 3), "artifact": self.artifact,
            "compile_flags": self.compile_flags,
        }


@dc.dataclass
class Edge:
    """跨子图数据边。cross=True 表示需要一次跨后端搬运。"""
    src: int
    dst: int
    tensor: str
    src_backend: str
    dst_backend: str
    bytes: int
    cost_us: float = 0.0
    zero_copy: bool = False

    @property
    def cross(self) -> bool:
        return self.src_backend != self.dst_backend

    def to_dict(self) -> dict[str, Any]:
        return {
            "src": self.src, "dst": self.dst, "tensor": self.tensor,
            "src_backend": self.src_backend, "dst_backend": self.dst_backend,
            "bytes": self.bytes, "cost_us": round(self.cost_us, 3),
            "cross": self.cross, "zero_copy": self.zero_copy,
        }


class Graph:
    """完整计算图 + 分区结果。"""

    def __init__(self, name: str = "model"):
        self.name = name
        self.tensors: dict[str, Tensor] = {}
        self.nodes: list[Node] = []
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self.parts: list[Part] = []
        self.edges: list[Edge] = []
        self.meta: dict[str, Any] = {}

    # ---------- 构建 ----------
    def add_tensor(self, t: Tensor) -> Tensor:
        if t.id in self.tensors:
            raise ValueError(f"重复张量 id: {t.id}")
        self.tensors[t.id] = t
        return t

    def add_node(self, n: Node) -> Node:
        if n.id in {x.id for x in self.nodes}:
            raise ValueError(f"重复节点 id: {n.id}")
        self.nodes.append(n)
        return n

    def node(self, nid: str) -> Node:
        for n in self.nodes:
            if n.id == nid:
                return n
        raise KeyError(nid)

    def node_index(self) -> dict[str, int]:
        return {n.id: i for i, n in enumerate(self.nodes)}

    # ---------- 图分析 ----------
    def preds(self) -> dict[str, list[str]]:
        """节点级前驱（经张量 producer 反查）。"""
        p: dict[str, list[str]] = {n.id: [] for n in self.nodes}
        for n in self.nodes:
            for t in n.inputs:
                prod = self.tensors[t].producer
                if prod and prod in p and prod not in p[n.id]:
                    p[n.id].append(prod)
        return p

    def succs(self) -> dict[str, list[str]]:
        s: dict[str, list[str]] = {n.id: [] for n in self.nodes}
        for n, ps in self.preds().items():
            for p in ps:
                s[p].append(n)
        return s

    def topo(self) -> list[str]:
        """Kahn 拓扑序；有环立即抛错。"""
        preds, succs, indeg = self.preds(), self.succs(), {}
        for n, ps in preds.items():
            indeg[n] = len(ps)
        q = [n for n, d in indeg.items() if d == 0]
        out: list[str] = []
        while q:
            # LIFO = 深度优先：让同一条分支在拓扑序上连续，
            # 分区时才能把整条分支聚成一个 Part（BFS 会让并行分支交错切碎）。
            u = q.pop()
            out.append(u)
            for v in succs[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
        if len(out) != len(self.nodes):
            raise ValueError("图中存在环，无法分区")
        return out

    def validate(self) -> list[str]:
        """返回问题列表；空列表表示合法。"""
        errs: list[str] = []
        known = set(self.tensors)
        for n in self.nodes:
            for t in n.inputs + n.outputs:
                if t not in known:
                    errs.append(f"节点 {n.id} 引用未知张量 {t}")
            if n.part < 0:
                errs.append(f"节点 {n.id} 未分配子图")
            if n.backend not in ("cpu", "gpu", "npu"):
                errs.append(f"节点 {n.id} 后端非法: {n.backend}")
        for t in self.tensors.values():
            if t.kind in ("weight", "bias", "constant") and t.offset < 0 and t.data is None:
                errs.append(f"张量 {t.id} 缺少权重数据")
        for i in self.inputs + self.outputs:
            if i not in known:
                errs.append(f"图 IO 引用未知张量 {i}")
        try:
            self.topo()
        except ValueError as e:
            errs.append(str(e))
        # 分区完整性
        covered: dict[int, set[str]] = {}
        for n in self.nodes:
            covered.setdefault(n.part, set()).add(n.id)
        for p in self.parts:
            if p.id not in covered:
                errs.append(f"子图 {p.id} 无节点")
                continue
            if set(p.nodes) != covered[p.id]:
                errs.append(f"子图 {p.id} 节点集合与 Node.part 不一致")
            for t in p.inputs + p.outputs:
                if t not in known:
                    errs.append(f"子图 {p.id} 边界引用未知张量 {t}")
        if len(covered) != len(self.parts):
            errs.append("子图编号不连续或存在孤儿节点")
        return errs

    # ---------- 序列化 ----------
    def to_dict(self) -> dict[str, Any]:
        return {
            "version": "1.0",
            "producer": self.meta.get("producer", "hx-compile"),
            "graph": {
                "name": self.name,
                "inputs": [self.tensors[i].to_dict() for i in self.inputs],
                "outputs": [self.tensors[i].to_dict() for i in self.outputs],
            },
            "tensors": {k: v.to_dict() for k, v in self.tensors.items()},
            "nodes": [n.to_dict() for n in self.nodes],
            "parts": [p.to_dict() for p in self.parts],
            "edges": [e.to_dict() for e in self.edges],
            "weights_file": self.meta.get("weights_file", "weights.bin"),
            "weights_bytes": self.meta.get("weights_bytes", 0),
            "plan": self.meta.get("plan", {}),
            "chip": self.meta.get("chip", {}),
            "meta": self.meta,
        }

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    @staticmethod
    def load(d: dict[str, Any] | str) -> "Graph":
        if isinstance(d, str):
            with open(d, encoding="utf-8") as f:
                d = json.load(f)
        g = Graph(d["graph"]["name"])
        for k, v in d["tensors"].items():
            g.tensors[k] = Tensor.from_dict(v)
        g.nodes = [Node.from_dict(x) for x in d["nodes"]]
        g.inputs = [x["id"] for x in d["graph"]["inputs"]]
        g.outputs = [x["id"] for x in d["graph"]["outputs"]]
        for x in d.get("parts", []):
            g.parts.append(Part(
                id=x["id"], backend=x["backend"], nodes=x["nodes"],
                inputs=x["inputs"], outputs=x["outputs"],
                est_us=x.get("est_us", 0.0), artifact=x.get("artifact", ""),
                compile_flags=x.get("compile_flags", {}),
            ))
        g.meta = d.get("meta", {})
        return g


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, "item"):
        return x.item()
    return str(x)


def shape_of(t: Tensor) -> list[int]:
    return [int(d) for d in t.shape]
