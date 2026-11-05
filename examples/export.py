"""端到端编译入口：生成 C++ 运行时可直接加载的部署包。

    cd /workspace && python3 -m examples.export
"""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from examples.model import build_chain, build_model  # noqa: E402
from hx import compile_model  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    torch.manual_seed(0)
    x = torch.randn(1, 3, 32, 32)
    x_large = torch.randn(1, 3, 96, 96)
    chip_balanced = os.path.join(ROOT, "examples", "chip_balanced.json")

    print("=" * 72)
    print("[1/3] 双分支网络 / 96x96 / NPU 主导芯片 / auto 自动分区")
    print("=" * 72)
    pkg = compile_model(build_model(True), (x_large,), os.path.join(ROOT, "build", "demo"),
                        strategy="auto", pipeline_depth=3)
    print(json.dumps(pkg.summary(), indent=2, ensure_ascii=False))
    for p in pkg.graph.parts:
        ops = [pkg.graph.node(n).op for n in p.nodes]
        print(f"  part {p.id} [{p.backend}] {len(p.nodes)} nodes: {ops}")

    print()
    print("=" * 72)
    print("[2/3] 纯链式反例：验证「不建议算子级并行」判定")
    print("=" * 72)
    chain = compile_model(build_chain(), (x,), os.path.join(ROOT, "build", "chain"),
                          strategy="auto")
    print(json.dumps(chain.summary(), indent=2, ensure_ascii=False))

    print()
    print("=" * 72)
    print("[3/3] 均衡型芯片 / manual 提示分区：验证真实跨芯片并行")
    print("=" * 72)
    par = compile_model(build_model(True), (x,), os.path.join(ROOT, "build", "parallel"),
                        strategy="force_hetero", chip=chip_balanced, pipeline_depth=3)
    print(json.dumps(par.summary(), indent=2, ensure_ascii=False))
    for p in par.graph.parts:
        ops = [par.graph.node(n).op for n in p.nodes]
        print(f"  part {p.id} [{p.backend}] {len(p.nodes)} nodes: {ops}")
    print("  part DAG 依赖:", [(e.src, e.dst, e.tensor) for e in par.graph.edges][:12])

    print("\n部署包:")
    for k, v in pkg.files.items():
        print(f"  {k:20s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
