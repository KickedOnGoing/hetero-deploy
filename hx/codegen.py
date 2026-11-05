"""产物生成：manifest / 权重 / 子图 / 编译命令 / 缓冲需求 / 报告。

产物是 Python 编译器与 C++ 运行时、以及 MTK 生产工具链之间的契约。
本原型在沙箱内生成完整的、可被 C++ 直接执行的部署包；生产落地时，
`compile_commands.json` 里的命令把每个 Part 交给 NP8 Converter /
LiteRT Compiler Plugin / ncc-tflite 编译成硬件二进制。
"""
from __future__ import annotations

import json
import os
import struct
from typing import Any

import torch

from .cost import CostModel
from .ir import Graph
from .partition import explain

ALIGN = 64


def _align(n: int) -> int:
    return (n + ALIGN - 1) // ALIGN * ALIGN


def _json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


class Codegen:
    def __init__(self, g: Graph, cm: CostModel, outdir: str):
        self.g = g
        self.cm = cm
        self.out = outdir
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(os.path.join(outdir, "parts"), exist_ok=True)

    # ---------------- 权重 ----------------
    def write_weights(self) -> dict[str, Any]:
        """按 64B 对齐打包所有常量张量，回填 Tensor.offset。"""
        blobs: list[bytes] = []
        off = 0
        manifest: dict[str, Any] = {}
        ids = self.g.meta.get("weights", [])
        for tid in ids:
            t = self.g.tensors.get(tid)
            if t is None or t.data is None:
                continue
            raw = t.data
            if not isinstance(raw, torch.Tensor):
                raw = torch.tensor(raw)
            arr = raw.detach().cpu().contiguous()
            b = arr.numpy().tobytes()
            t.offset = off
            manifest[tid] = {"offset": off, "bytes": len(b),
                             "shape": list(arr.shape), "dtype": t.dtype}
            blobs.append(b)
            padded = _align(len(b))
            if padded > len(b):
                blobs.append(b"\x00" * (padded - len(b)))
            off += padded
        data = b"".join(blobs)
        with open(os.path.join(self.out, "weights.bin"), "wb") as f:
            f.write(data)
        self.g.meta["weights_bytes"] = len(data)
        self.g.meta["weights_file"] = "weights.bin"
        return manifest

    # ---------------- 主 manifest ----------------
    def write_manifest(self) -> str:
        p = os.path.join(self.out, "manifest.json")
        self.g.save(p)
        return p

    # ---------------- 子图 ----------------
    def write_parts(self) -> list[str]:
        paths = []
        for part in self.g.parts:
            nodes = [self.g.node(n).to_dict() for n in part.nodes]
            tens = {}
            for n in part.nodes:
                for t in self.g.node(n).inputs + self.g.node(n).outputs:
                    if t in self.g.tensors:
                        tens[t] = self.g.tensors[t].to_dict()
            obj = {
                "part": part.to_dict(),
                "nodes": nodes,
                "tensors": tens,
                "target": part.backend,
                "artifact": f"parts/part_{part.id}.hxbin",
            }
            p = os.path.join(self.out, "parts", f"part_{part.id}.json")
            _json(p, obj)
            paths.append(p)
        return paths

    # ---------------- 生产编译命令 ----------------
    def write_compile_commands(self) -> str:
        cmds: dict[str, Any] = {
            "_note": "生产落地命令模板。NNAPI 自 Android 15 起 deprecated 且将于 2027 移除，"
                     "不要作为新架构依赖；OEM 侧走 NeuroPilot NP8 + LiteRT NeuroPilot Accelerator。",
            "target_chip": self.cm.chip.get("name"),
            "npu": {
                "model_convert": "np8-converter --framework=pytorch --input=part_N.onnx "
                                 "--output=part_N.tflite --quantize=int8 --calib=calibration.pt",
                "aot_offline_compile": "ncc-tflite --arch=mdla3.0 --relax-fp32 "
                                       "part_N.tflite -o part_N.dla",
                "jit": "LiteRT CompiledModel(Accelerator.NPU) + libLiteRtCompilerPlugin_MediaTek.so",
                "runtime": "libLiteRtDispatch_MediaTek.so / neuronrt -a part_N.dla",
                "honor": "NEURON_FLAG_MIN_GROUP_SIZE 与本工具 min_npu_nodes 语义一致；"
                         "注意 float32 bias 要求与 INT8/FP16 量化一致性",
            },
            "gpu": {
                "runtime": "LiteRT CompiledModel(Accelerator.GPU) / OpenCL-Vulkan compute",
                "note": "GPU delegate 必须在调用 Interpreter 的同一线程初始化",
            },
            "cpu": {
                "runtime": "LiteRT XNNPACK / ARM NN，作为不支持算子的兜底与降级路径",
            },
            "parts": [],
        }
        for p in self.g.parts:
            cmds["parts"].append({
                "id": p.id, "backend": p.backend, "nodes": len(p.nodes),
                "artifact": f"parts/part_{p.id}.hxbin",
                "command": f"# 用上述 {p.backend} 流程编译 parts/part_{p.id}.json",
            })
        path = os.path.join(self.out, "compile_commands.json")
        _json(path, cmds)
        return path

    # ---------------- 缓冲需求握手表 ----------------
    def write_tensor_requirements(self) -> str:
        """对应 LiteRT Dispatch API 的 GetInput/OutputRequirements 握手。"""
        reqs = []
        for e in self.g.edges:
            t = self.g.tensors[e.tensor]
            src_l = self.cm.chip["backends"][e.src_backend].get("layout", "NCHW")
            dst_l = self.cm.chip["backends"][e.dst_backend].get("layout", "NCHW")
            reqs.append({
                "tensor": e.tensor, "from_part": e.src, "to_part": e.dst,
                "src_backend": e.src_backend, "dst_backend": e.dst_backend,
                "bytes": e.bytes, "dtype": t.dtype, "shape": list(t.shape),
                "alignment": 64 if e.src_backend == e.dst_backend else 128,
                "src_layout": src_l, "dst_layout": dst_l,
                "layout_conversion": src_l != dst_l,
                "zero_copy": e.zero_copy and src_l == dst_l,
                "buffer_type": "AHardwareBuffer / DMA-BUF" if e.cross else "host shared",
                "sync": "LiteRtEvent (sync fence fd)" if e.cross else "intra-part",
                "cost_us": round(e.cost_us, 3),
            })
        obj = {
            "_note": "零拷贝成立的前提是上下游后端要求相同的 buffer type / layout / stride。"
                     "若 NPU 只接受特定对齐的 AHWB 而 GPU 产出 GL texture，"
                     "则每帧都要转换一次，收益会被吃掉 —— 这是上板前必须验证的握手项。",
            "requirements": reqs,
        }
        path = os.path.join(self.out, "tensor_requirements.json")
        _json(path, obj)
        return path

    # ---------------- 报告 ----------------
    def write_report(self) -> str:
        obj = {
            "graph": self.g.name,
            "plan": self.g.meta.get("plan", {}),
            "partition": self.g.meta.get("partition", {}),
            "lower": self.g.meta.get("lower_stats", {}),
            "parts": explain(self.g),
            "edges": [e.to_dict() for e in self.g.edges],
            "chip": self.cm.chip,
        }
        path = os.path.join(self.out, "report.json")
        _json(path, obj)
        return path

    # ---------------- 测试向量 ----------------
    def write_test_vectors(self, inputs: tuple[Any, ...], expected: tuple[Any, ...]) -> str:
        desc: dict[str, Any] = {"inputs": [], "outputs": []}
        for i, t in enumerate(inputs):
            arr = t.detach().cpu().contiguous()
            fn = f"input_{i}.bin"
            with open(os.path.join(self.out, fn), "wb") as f:
                f.write(arr.numpy().tobytes())
            desc["inputs"].append({"file": fn, "shape": list(arr.shape),
                                   "dtype": str(arr.dtype).replace("torch.", ""),
                                   "bytes": arr.numel() * arr.element_size()})
        for i, t in enumerate(expected):
            arr = t.detach().cpu().contiguous() if hasattr(t, "detach") else torch.tensor(t)
            fn = f"expected_{i}.bin"
            with open(os.path.join(self.out, fn), "wb") as f:
                f.write(arr.numpy().tobytes())
            desc["outputs"].append({"file": fn, "shape": list(arr.shape),
                                    "dtype": str(arr.dtype).replace("torch.", ""),
                                    "bytes": arr.numel() * arr.element_size()})
        path = os.path.join(self.out, "io.json")
        _json(path, desc)
        return path

    def run(self, inputs: tuple[Any, ...] | None = None,
            expected: tuple[Any, ...] | None = None) -> dict[str, str]:
        self.write_weights()
        out = {
            "manifest": self.write_manifest(),
            "compile_commands": self.write_compile_commands(),
            "tensor_requirements": self.write_tensor_requirements(),
            "report": self.write_report(),
        }
        self.write_parts()
        if inputs is not None and expected is not None:
            out["io"] = self.write_test_vectors(inputs, expected)
        return out
