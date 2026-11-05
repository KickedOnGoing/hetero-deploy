"""算子集、后端能力表、FLOPs 估算与目标芯片档案。

能力表是整个可行性判断的第一道闸门：算子在某后端不可用时，该后端在
分区搜索中直接不可选（退化为 CPU），这与 MTK Neuron Delegate 的实际行为
一致——不支持的算子会把图切碎，产生跨子图搬运。
"""
from __future__ import annotations

import json
import math
from typing import Any

from .ir import Graph, Node, Tensor

# ---------------------------------------------------------------------------
# 后端能力表：efficiency = 该算子在此后端能达到的峰值算力利用率
# 0 表示不支持（禁止把该算子绑定到此后端）
# ---------------------------------------------------------------------------
SUPPORT: dict[str, dict[str, float]] = {
    # ---- NPU / MTK APU(MDLA)：高吞吐、强约束 ----
    "npu": {
        "conv2d": 0.80, "linear": 0.65, "matmul": 0.60,
        "relu": 0.90, "relu6": 0.90, "clip": 0.85,
        "add": 0.75, "mul": 0.75, "sub": 0.75,
        "concat": 0.70, "avg_pool2d": 0.80, "max_pool2d": 0.80,
        "global_avg_pool": 0.80, "batch_norm": 0.70,
        "reshape": 0.95, "flatten": 0.95,
        # 以下为 APU 常见缺口，训练侧可自由使用，由工具链重写或降级
        "silu": 0.0, "sigmoid": 0.0, "tanh": 0.0, "gelu": 0.0,
        "softmax": 0.0, "layer_norm": 0.0, "transpose": 0.0,
        "permute": 0.0, "mean": 0.0, "amax": 0.0, "identity": 0.0,
        "conv_transpose2d": 0.0, "pad": 0.0, "upsample": 0.0,
    },
    # ---- GPU / Mali-Immortalis，OpenCL / Vulkan Compute ----
    "gpu": {
        "conv2d": 0.45, "linear": 0.70, "matmul": 0.70,
        "relu": 0.95, "relu6": 0.95, "clip": 0.95,
        "silu": 0.95, "sigmoid": 0.95, "tanh": 0.95, "gelu": 0.90,
        "softmax": 0.80, "layer_norm": 0.75,
        "add": 0.95, "mul": 0.95, "sub": 0.95,
        "concat": 0.85, "avg_pool2d": 0.75, "max_pool2d": 0.75,
        "global_avg_pool": 0.75, "batch_norm": 0.85,
        "transpose": 0.70, "permute": 0.70, "reshape": 0.95,
        "flatten": 0.95, "pad": 0.80, "upsample": 0.70,
        "mean": 0.80, "amax": 0.80, "identity": 0.95,
        "conv_transpose2d": 0.40,
    },
    # ---- CPU：全算子兜底，吞吐最低 ----
    "cpu": {op: 0.55 for op in (
        "conv2d", "conv_transpose2d", "linear", "matmul", "relu", "relu6",
        "clip", "silu", "sigmoid", "tanh", "gelu", "softmax", "layer_norm",
        "add", "mul", "sub", "concat", "avg_pool2d", "max_pool2d",
        "global_avg_pool", "batch_norm", "transpose", "permute", "reshape",
        "flatten", "pad", "upsample", "mean", "amax", "identity", "div",
        "sqrt", "exp", "log", "pow", "neg", "abs", "where",
    )},
}

BACKENDS = ("cpu", "gpu", "npu")
VIEW_OPS = {"reshape", "flatten", "transpose", "permute", "identity"}

# ---------------------------------------------------------------------------
# 目标芯片档案（可替换；所有数值来源与敏感性分析见 docs/01）
# ---------------------------------------------------------------------------
DEFAULT_CHIP: dict[str, Any] = {
    "name": "Dimensity-9300-class (可配置模板)",
    "backends": {
        # peak_gflops: 该后端 fp16 等效峰值；launch_us: 一次提交固定开销
        # mem_bw_gbps: 片内有效访存带宽，用于 roofline 访存下限（小算子不被估成 0）
        "cpu": {"peak_gflops": 180.0, "launch_us": 12.0, "layout": "NCHW",
                "mem_bw_gbps": 25.0, "slots": 1},
        "gpu": {"peak_gflops": 1400.0, "launch_us": 45.0, "layout": "NHWC",
                "mem_bw_gbps": 90.0, "slots": 1},
        "npu": {"peak_gflops": 9000.0, "launch_us": 90.0, "layout": "NHWC",
                "mem_bw_gbps": 150.0, "slots": 1},
    },
    # 跨后端搬运：固定同步开销 + 每 MB 传输（有效带宽 ≈ 55 GB/s）+ layout 转换
    "fixed_cross_us": 35.0,
    "per_mb_us": 18.0,
    "layout_convert_us": 20.0,
    "zero_copy_capable": True,
    # 同后端内不产生跨片搬运，但仍有 fence/事件同步开销
    "intra_part_sync_us": 3.0,
    # NPU 子图最小规模：小于该计算量不上 NPU，避免碎片化
    # 对应 MTK NEURON_FLAG_MIN_GROUP_SIZE / 学术界 adaptive window size
    "min_npu_nodes": 3,
    "min_part_flops": 2_000_000,
    # 并发资源：每个后端一个串行执行队列
    "backend_slots": {"cpu": 1, "gpu": 1, "npu": 1},
}

# ---------------------------------------------------------------------------
# FLOPs 估算（以输出张量形状为准，不依赖运行时 profiling）
# ---------------------------------------------------------------------------
def estimate_flops(node: Node, g: Graph) -> int:
    op = node.op
    out = g.tensors[node.outputs[0]] if node.outputs else None
    outn = out.numel if out else 0
    a = node.attrs

    if op in ("conv2d", "conv_transpose2d"):
        w = g.tensors.get(node.inputs[1])
        if w is None or len(w.shape) < 4:
            return outn * 9 * 2
        groups = int(a.get("groups", 1))
        cin_per_g = int(w.shape[1])
        kh, kw = int(w.shape[2]), int(w.shape[3])
        mac = outn * cin_per_g * kh * kw
        return mac * 2 + outn  # +bias

    if op in ("linear", "matmul"):
        k = 1
        if len(node.inputs) > 1 and node.inputs[1] in g.tensors:
            wt = g.tensors[node.inputs[1]]
            if wt.shape:
                k = int(wt.shape[-1])
        return 2 * outn * k + outn

    if op in ("batch_norm", "layer_norm", "softmax"):
        return outn * 6
    if op in ("silu", "gelu", "sigmoid", "tanh", "exp", "log", "sqrt"):
        return outn * 6
    if op in ("relu", "relu6", "clip", "add", "mul", "sub", "div", "pow",
              "neg", "abs", "where", "mean", "amax"):
        return outn * 2
    if op in ("avg_pool2d", "max_pool2d"):
        kh = int(a.get("kernel", [1, 1])[0])
        kw = int(a.get("kernel", [1, 1])[1])
        return outn * kh * kw
    if op == "global_avg_pool":
        # 需要读入整个输入
        tin = g.tensors.get(node.inputs[0]) if node.inputs else None
        return (tin.numel if tin else outn) * 2
    if op in ("concat", "transpose", "permute", "reshape", "flatten",
              "identity", "pad", "upsample"):
        return 0
    return outn * 2


def estimate_io_bytes(node: Node, g: Graph) -> int:
    """算子的最小访存量（输入 + 输出 + 权重，字节）。

    用于 roofline 的访存下限：element-wise / view 类算子 FLOPs 很低甚至为 0，
    若只按算力估时会得到 0 us，分区器会误以为"放哪都一样"，进而把小算子
    随意铺到 NPU 上制造碎片。真实硬件上这些算子受带宽约束，必须显式建模。
    """
    total = 0
    for tid in list(node.inputs) + list(node.outputs):
        t = g.tensors.get(tid)
        if t is not None:
            total += t.nbytes
    return total


def is_supported(op: str, backend: str) -> bool:
    return SUPPORT.get(backend, {}).get(op, 0.0) > 0.0


def efficiency(op: str, backend: str) -> float:
    return SUPPORT.get(backend, {}).get(op, 0.0)


def allowed_backends(node: Node) -> list[str]:
    return [b for b in BACKENDS if is_supported(node.op, b)]


def load_chip(path: str | None) -> dict[str, Any]:
    if not path:
        return json.loads(json.dumps(DEFAULT_CHIP))
    with open(path, encoding="utf-8") as f:
        user = json.load(f)
    chip = json.loads(json.dumps(DEFAULT_CHIP))
    for k, v in user.items():
        if k == "backends" and isinstance(v, dict):
            for b, bv in v.items():
                chip["backends"].setdefault(b, {}).update(bv)
        else:
            chip[k] = v
    return chip


def dump_default_chip(path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CHIP, f, indent=2, ensure_ascii=False)
