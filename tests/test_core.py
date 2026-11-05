"""HX 工具链核心单元测试（可复现、无外部依赖）。

覆盖本轮深入优化引入/修复的关键性质：
  * roofline 访存下界真实生效（不再有 0us 死代码）
  * 离散事件模拟器确定性（与 edges 遍历顺序无关）+ 搬运按张量累加
  * IR 序列化往返完整（edges / slot / io_bytes 不丢失）
  * 缓冲复用静态可证明安全（含并行分支与 view 别名）
  * 分区搜索不劣于任一单后端方案，并接近小规模穷举最优

运行: python3 -m tests.test_core
"""
from __future__ import annotations

import copy
import itertools
import random
import unittest

from hx.cost import CostModel, simulate
from hx.ir import Edge, Graph, Node, Part, Tensor
from hx.memory import plan_buffers, verify_buffer_plan
from hx.opset import allowed_backends, load_chip
from hx.partition import (PartitionConfig, _single_assignment, build_parts,
                          partition)
from hx.verify import verify_structure


def _tensor(g: Graph, tid: str, shape, kind="activation", producer="",
            offset=-1) -> None:
    g.add_tensor(Tensor(id=tid, dtype="f32", shape=tuple(shape), kind=kind,
                        offset=offset, producer=producer))


def small_graph() -> Graph:
    """8 节点双分支小图，用于穷举对比（规模足够小，可枚举全部绑定）。"""
    g = Graph("small_branch")
    _tensor(g, "x", (1, 4, 8, 8), "input")
    for t in ("w1", "w2", "w3", "w4"):
        _tensor(g, t, (8, 4, 3, 3), "weight", offset=0)
    for t in ("b1", "b2", "b3", "b4"):
        _tensor(g, t, (8,), "bias", offset=0)
    outs = ["t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7"]
    for t in outs:
        _tensor(g, t, (1, 8, 8, 8), "activation")
    g.inputs = ["x"]
    g.outputs = ["t7"]

    specs = [
        ("n0", "conv2d", ["x", "w1", "b1"], "t0", 200_000, 300_000),
        ("n1", "relu", ["t0"], "t1", 5_000, 20_000),
        ("n2", "conv2d", ["t1", "w2", "b2"], "t2", 200_000, 300_000),
        ("n3", "silu", ["t1"], "t3", 10_000, 20_000),
        ("n4", "conv2d", ["t3", "w3", "b3"], "t4", 200_000, 300_000),
        ("n5", "concat", ["t2", "t4"], "t5", 1_000, 40_000),
        ("n6", "linear", ["t5", "w4", "b4"], "t6", 100_000, 150_000),
        ("n7", "relu", ["t6"], "t7", 5_000, 20_000),
    ]
    for nid, op, ins, out, flops, io in specs:
        g.add_node(Node(id=nid, op=op, inputs=ins, outputs=[out],
                        flops=flops, io_bytes=io, name=nid))
        g.tensors[out].producer = nid
    return g


class TestCostModel(unittest.TestCase):
    def setUp(self):
        self.cm = CostModel(load_chip(None))

    def test_roofline_not_zero(self):
        """flops=0 的 view 算子仍必须受带宽约束，不能被估成 0。"""
        g = small_graph()
        n = Node("v", "reshape", ["t0"], ["v0"], flops=0, io_bytes=1_000_000)
        t = self.cm.compute_us(n, "npu")
        self.assertGreater(t, 0.0)
        self.assertFalse(t == float("inf"))
        # 带宽翻倍，访存下界减半
        chip2 = load_chip(None)
        chip2["backends"]["npu"]["mem_bw_gbps"] *= 2
        t2 = CostModel(chip2).compute_us(n, "npu")
        self.assertAlmostEqual(t / t2, 2.0, places=3)

    def test_unsupported_is_inf(self):
        n = Node("s", "silu", ["t0"], ["o"], flops=100, io_bytes=100)
        self.assertEqual(self.cm.compute_us(n, "npu"), float("inf"))
        self.assertGreater(self.cm.compute_us(n, "gpu"), 0)

    def test_transfer_linear_in_bytes(self):
        self.assertAlmostEqual(
            self.cm.transfer_us(2_000_000, "npu", "gpu")
            - self.cm.transfer_us(1_000_000, "npu", "gpu"),
            self.cm.chip["per_mb_us"], places=5)

    def test_transfer_accumulates_per_tensor(self):
        """同一 (src,dst) 对上的两条跨片边必须分别计费，不能取 max。"""
        g = Graph("xfer")
        for tid in ("i", "a", "b", "o"):
            _tensor(g, tid, (1, 8, 8, 8),
                    "input" if tid == "i" else "activation")
        g.add_node(Node("n0", "relu", ["i"], ["a", "b"],
                        flops=1000, io_bytes=1000, part=0, backend="npu"))
        g.add_node(Node("n1", "relu", ["a"], ["o"],
                        flops=1000, io_bytes=1000, part=1, backend="gpu"))
        g.parts = [Part(0, "npu", ["n0"], [], ["a", "b"]),
                   Part(1, "gpu", ["n1"], ["a", "b"], ["o"])]
        for t in ("a", "b"):
            g.edges.append(Edge(0, 1, t, "npu", "gpu", g.tensors[t].nbytes))
        each = self.cm.edge_cost(g.edges[0])
        r = simulate(g, self.cm)
        self.assertEqual(r.transfer_edges, 2)
        self.assertAlmostEqual(r.total_transfer_us, 2 * each, places=5)
        self.assertAlmostEqual(r.dma_busy_us, 2 * each, places=5)


class TestSimulatorDeterminism(unittest.TestCase):
    def setUp(self):
        self.cm = CostModel(load_chip(None))
        self.g = small_graph()
        self.cfg = PartitionConfig()
        a = _single_assignment(self.g, "gpu", self.cm)
        # 制造一个跨片切分，让 DMA 队列真正参与
        for n in self.g.nodes:
            if n.op == "conv2d":
                a[n.id] = "npu"
        build_parts(self.g, a, self.cm, self.cfg)
        self.assertGreater(sum(1 for e in self.g.edges if e.cross), 0)

    def _sig(self, r):
        return (r.makespan_us, r.total_transfer_us, r.dma_busy_us,
                r.critical_path_us, r.steady_frame_us, r.overlap_pairs,
                r.transfer_edges,
                tuple((e.part, round(e.start_us, 6), round(e.end_us, 6))
                      for e in r.events))

    def test_repeated_runs_identical(self):
        sigs = {self._sig(simulate(self.g, self.cm)) for _ in range(10)}
        self.assertEqual(len(sigs), 1)

    def test_edge_order_does_not_change_result(self):
        base = self._sig(simulate(self.g, self.cm))
        for seed in range(6):
            rng = random.Random(seed)
            edges = list(self.g.edges)
            rng.shuffle(edges)
            self.g.edges = edges
            self.assertEqual(self._sig(simulate(self.g, self.cm)), base)

    def test_slots_reduce_steady_frame(self):
        """后端执行槽增加时，稳态帧间隔不应变大。"""
        g2 = copy.deepcopy(self.g)
        cm2 = CostModel(load_chip(None))
        cm2.chip["backends"]["npu"]["slots"] = 2
        self.assertLessEqual(simulate(g2, cm2).steady_frame_us,
                             simulate(self.g, self.cm).steady_frame_us + 1e-9)


class TestIRRoundTrip(unittest.TestCase):
    def test_edges_and_fields_survive_save_load(self):
        g = small_graph()
        cm = CostModel(load_chip(None))
        build_parts(g, _single_assignment(g, "gpu", cm), cm, PartitionConfig())
        plan_buffers(g)
        d = g.to_dict()
        g2 = Graph.load(d)
        self.assertEqual(len(g2.edges), len(g.edges))
        self.assertEqual([(e.src, e.dst, e.tensor) for e in g2.edges],
                         [(e.src, e.dst, e.tensor) for e in g.edges])
        self.assertEqual(g2.node("n0").io_bytes, g.node("n0").io_bytes)
        self.assertEqual(g2.tensors["t2"].slot, g.tensors["t2"].slot)
        self.assertTrue(verify_structure(g2)["pass"])


class TestMemoryPlan(unittest.TestCase):
    def test_plan_is_provably_safe(self):
        g = small_graph()
        plan = plan_buffers(g)
        self.assertEqual(verify_buffer_plan(g, plan), [])
        self.assertTrue(plan["verified"])
        self.assertGreaterEqual(plan["naive_activation_bytes"],
                                plan["peak_activation_bytes"])

    def test_alias_shares_root_slot(self):
        """view 输出与根张量必须同槽，否则运行时零拷贝别名会串缓冲。"""
        g = small_graph()
        # 先重连 t5 的下游，再追加 flatten 节点（否则循环会把 nf 的输入
        # 也改成 flat，造成自环）
        for n in g.nodes:
            if "t5" in n.inputs:
                n.inputs = ["flat" if t == "t5" else t for t in n.inputs]
        _tensor(g, "flat", (1, 8 * 8 * 8), "activation", producer="nf")
        g.add_node(Node(id="nf", op="flatten", inputs=["t5"], outputs=["flat"],
                        flops=0, io_bytes=1000, name="nf"))
        plan = plan_buffers(g)
        self.assertEqual(g.tensors["flat"].slot, g.tensors["t5"].slot)
        self.assertEqual(verify_buffer_plan(g, plan), [])


class TestPartitionQuality(unittest.TestCase):
    def test_auto_never_worse_than_single_and_near_exhaustive(self):
        cm = CostModel(load_chip(None))
        cfg = PartitionConfig(iters=400, restarts=3, seed=7)
        g = small_graph()
        g_auto, _ = partition(g, cm, cfg)
        auto_ms = simulate(g_auto, cm).makespan_us

        singles = {}
        for b in ("cpu", "gpu", "npu"):
            gg = small_graph()
            build_parts(gg, _single_assignment(gg, b, cm), cm, cfg)
            singles[b] = simulate(gg, cm).makespan_us
        self.assertLessEqual(auto_ms, min(singles.values()) + 1e-6,
                             f"auto={auto_ms} singles={singles}")

        # 穷举全部合法绑定，取真实 makespan 最优
        gg = small_graph()
        legal = {n.id: allowed_backends(n) or ["cpu"] for n in gg.nodes}
        ids = [n.id for n in gg.nodes]
        best = float("inf")
        for combo in itertools.product(*(legal[i] for i in ids)):
            a = dict(zip(ids, combo))
            try:
                build_parts(gg, a, cm, cfg)
                best = min(best, simulate(gg, cm).makespan_us)
            except (ValueError, KeyError):
                continue
        self.assertLess(best, float("inf"))
        # 启发式搜索应接近穷举最优（容差 25%，证明不是随机解）
        self.assertLessEqual(auto_ms, best * 1.25 + 1e-6,
                             f"auto={auto_ms} exhaustive_best={best}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
