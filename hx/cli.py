"""hx 命令行工具。

    python -m hx compile --model examples.model:build_model --shapes 1,3,64,64 \
        --out build/demo --strategy auto
    python -m hx chip --out chip.json
    python -m hx inspect --manifest build/demo/manifest.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from typing import Any

import torch

from .api import compile_model
from .ir import Graph
from .opset import dump_default_chip


def _load_obj(spec: str):
    if ":" in spec:
        mod, name = spec.split(":", 1)
    else:
        parts = spec.rsplit(".", 1)
        mod, name = parts[0], parts[1]
    m = importlib.import_module(mod)
    obj = getattr(m, name)
    return obj() if callable(obj) and not isinstance(obj, torch.nn.Module) else obj


def _shapes(s: str) -> tuple[torch.Tensor, ...]:
    outs = []
    for group in s.split(";"):
        dims = [int(x) for x in group.strip().split(",") if x.strip()]
        outs.append(torch.randn(*dims))
    return tuple(outs)


def cmd_compile(args: argparse.Namespace) -> int:
    model = _load_obj(args.model)
    inp = _shapes(args.shapes)
    res = compile_model(
        model, inp, args.out, strategy=args.strategy, chip=args.chip,
        pipeline_depth=args.pipeline_depth, iters=args.iters, restarts=args.restarts,
        seed=args.seed, fold_bn=not args.no_fold_bn,
        decompose_silu=args.decompose_silu,
        enforce_min_npu_group=not args.no_min_group,
        respect_hints=not args.ignore_hints,
    )
    print(json.dumps(res.summary(), indent=2, ensure_ascii=False))
    print(f"\n产物目录: {args.out}")
    return 0 if (res.numerics["pass"] and res.structure["pass"]) else 1


def cmd_chip(args: argparse.Namespace) -> int:
    dump_default_chip(args.out)
    print(f"已导出默认芯片档案: {args.out}")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    g = Graph.load(args.manifest)
    print(f"图: {g.name}  节点 {len(g.nodes)}  子图 {len(g.parts)}")
    for p in g.parts:
        print(f"  part {p.id:2d} [{p.backend:3s}] nodes={len(p.nodes):3d} "
              f"est={p.est_us:8.2f}us in={len(p.inputs)} out={len(p.outputs)}")
    print(f"跨片边: {sum(1 for e in g.edges if e.cross)}  校验: {g.validate() or 'OK'}")
    print("计划:", json.dumps(g.meta.get("plan", {}).get("verdict", "N/A")))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hx", description="异构部署编译工具链")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compile", help="编译模型为部署包")
    c.add_argument("--model", required=True, help="模块:工厂函数，如 examples.model:build_model")
    c.add_argument("--shapes", required=True, help="输入形状，多输入用 ; 分隔，如 1,3,64,64")
    c.add_argument("--out", required=True)
    c.add_argument("--strategy", default="auto",
                   choices=["auto", "greedy", "single", "manual", "force_hetero"])
    c.add_argument("--chip", default=None, help="芯片档案 JSON")
    c.add_argument("--pipeline-depth", type=int, default=2)
    c.add_argument("--iters", type=int, default=600)
    c.add_argument("--restarts", type=int, default=3)
    c.add_argument("--seed", type=int, default=2026)
    c.add_argument("--no-fold-bn", action="store_true")
    c.add_argument("--decompose-silu", action="store_true")
    c.add_argument("--no-min-group", action="store_true")
    c.add_argument("--ignore-hints", action="store_true")
    c.set_defaults(func=cmd_compile)

    d = sub.add_parser("chip", help="导出默认芯片档案")
    d.add_argument("--out", default="chip.json")
    d.set_defaults(func=cmd_chip)

    i = sub.add_parser("inspect", help="查看部署包")
    i.add_argument("--manifest", required=True)
    i.set_defaults(func=cmd_inspect)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
