"""编译期内存规划：张量生命周期分析 + 共享缓冲槽复用。

移动端推理的硬约束往往不是算力而是**峰值内存**。朴素做法是给每个激活张量
各分配一块缓冲，峰值等于全图激活张量之和；对 224×224 的多分支网络这轻易
达到几十 MB，且多帧流水时还要乘以在飞帧数。

本模块在编译期计算每个张量的生命周期，把**不会同时存活**的张量映射到同一个
缓冲槽（slot），运行时只需为每个 slot 分配一块内存。复用判定不用「拓扑序区间
不重叠」这种近似 —— 在并行分支上它是不安全的：拓扑序靠后的节点完全可能与
靠前的节点同时执行（不同后端并发）。这里改用**依赖偏序**：

    张量 Y 可复用槽 S，当且仅当 S 上所有旧张量的全部 consumer 都
    happens-before Y 的 producer（即都是 producer 的祖先节点）。

这样即使两个分支在不同芯片上同时跑，也不会出现「一边还在读、另一边已覆盖」
的竞态。该条件是可静态证明的充分条件，见 verify_buffer_plan()。

产物写入 manifest 的 `memory` 段与每个 Tensor 的 `slot` 字段；C++ 运行时据此
按 slot 共享缓冲。槽大小按 64B 对齐，权重常驻 weights.bin 不占激活池。
"""
from __future__ import annotations

from typing import Any

from .ir import Graph

ALIGN = 64

# 常驻张量：不参与复用（权重在 weights.bin，图 IO 需保留到执行结束）
_PERSISTENT_KINDS = ("weight", "bias", "constant", "input", "output")

# 与 C++ ops.cc::view() 对齐：这些算子是零拷贝视图，输出不占独立缓冲，
# 必须与输入（的根张量）规划到同一 slot，否则运行时别名会破坏复用规划。
_ALIAS_OPS = ("reshape", "flatten", "identity")


def _alias_roots(g: Graph) -> dict[str, str]:
    """返回 {view 输出张量: 其最终根张量}，解析多级 view 链。"""
    alias: dict[str, str] = {}
    for n in g.nodes:
        if n.op in _ALIAS_OPS and n.inputs and n.outputs:
            alias[n.outputs[0]] = n.inputs[0]

    def root(t: str) -> str:
        seen: set[str] = set()
        while t in alias and t not in seen:
            seen.add(t)
            t = alias[t]
        return t

    return {t: root(t) for t in alias}


def _align(n: int, align: int = ALIGN) -> int:
    return (n + align - 1) // align * align


def _ancestors(g: Graph) -> tuple[list[str], dict[str, int], list[int]]:
    """拓扑序 + 每个节点的祖先 bitmask（传递闭包，用位运算加速）。"""
    topo = g.topo()
    pos = {nid: i for i, nid in enumerate(topo)}
    preds = g.preds()
    masks = [0] * len(topo)
    for u in topo:
        m = 0
        for p in preds[u]:
            if p in pos:
                m |= masks[pos[p]] | (1 << pos[p])
        masks[pos[u]] = m
    return topo, pos, masks


def plan_buffers(g: Graph, align: int = ALIGN) -> dict[str, Any]:
    """计算缓冲槽分配，并把 slot 写回 Tensor。返回可序列化的规划结果。"""
    _, pos, masks = _ancestors(g)

    consumers: dict[str, list[str]] = {tid: [] for tid in g.tensors}
    for n in g.nodes:
        for t in n.inputs:
            if t in consumers:
                consumers[t].append(n.id)

    persistent = {tid for tid, t in g.tensors.items() if t.kind in _PERSISTENT_KINDS}
    persistent |= set(g.inputs) | set(g.outputs)

    # view 算子（reshape/flatten/identity）在运行时是零拷贝别名：输出与输入
    # 共享同一块内存。必须把其 consumer 归并到根张量，并把输出规划到根所在
    # slot，否则 C++ 的 y.storage = x.storage 会写进错误的缓冲。
    all_roots = _alias_roots(g)
    roots = {t: r for t, r in all_roots.items() if t not in persistent}
    alias_tensors = set(roots)
    for t, r in roots.items():
        if r != t:
            consumers.setdefault(r, []).extend(consumers.get(t, []))
            consumers[t] = []

    # 只为需要独立缓冲的运行时激活张量分配槽（view 输出不占）
    live = [tid for tid, t in g.tensors.items()
            if tid not in persistent and tid not in alias_tensors and t.producer in pos]

    def cons_mask(tid: str) -> int:
        m = 0
        for c in consumers.get(tid, []):
            if c in pos:
                m |= 1 << pos[c]
        return m

    # 大张量优先，减少槽扩容带来的浪费；同尺寸按 birth 稳定排序
    items = sorted(live, key=lambda tid: (
        pos[g.tensors[tid].producer], -g.tensors[tid].nbytes, tid))

    slots: list[dict[str, Any]] = []          # {bytes, release_mask, tensors}
    tensor_slots: dict[str, int] = {}

    for tid in items:
        t = g.tensors[tid]
        need = _align(t.nbytes, align)
        prod = t.producer
        prod_anc = masks[pos[prod]]          # 严格 happens-before producer 的节点集
        cm = cons_mask(tid)                  # 本张量被读完所需的节点集

        chosen = -1
        for si, s in enumerate(slots):
            if s["bytes"] < need:
                continue
            # 槽上所有旧张量都已被读完，才允许覆盖
            if s["release_mask"] & ~prod_anc == 0:
                chosen = si
                break
        if chosen < 0:
            chosen = len(slots)
            slots.append({"bytes": need, "release_mask": 0, "tensors": []})
        s = slots[chosen]
        s["bytes"] = max(s["bytes"], need)
        s["release_mask"] |= cm
        s["tensors"].append(tid)
        tensor_slots[tid] = chosen
        t.slot = chosen

    for tid in persistent:
        g.tensors[tid].slot = -1
    # view 输出与根张量共槽，运行时零拷贝别名才与规划一致
    for t, r in roots.items():
        g.tensors[t].slot = g.tensors[r].slot
        tensor_slots[t] = g.tensors[t].slot
    # 常驻的 view 输出（如恰好是图输出）必须独占，运行时走真实拷贝
    for t in set(all_roots) - set(roots):
        g.tensors[t].slot = -1

    naive_activation = sum(_align(g.tensors[t].nbytes, align) for t in live)
    peak_activation = sum(s["bytes"] for s in slots)
    weights_bytes = sum(g.tensors[t].nbytes for t in g.tensors
                        if g.tensors[t].kind in ("weight", "bias", "constant"))
    io_bytes = sum(_align(g.tensors[t].nbytes, align)
                   for t in set(g.inputs) | set(g.outputs) if t in g.tensors)

    plan = {
        "align": align,
        "slot_count": len(slots),
        "reusable_tensors": len(live),
        "persistent_tensors": len(persistent),
        "alias_tensors": sorted(roots),
        "alias_count": len(roots),
        "naive_activation_bytes": naive_activation,
        "peak_activation_bytes": peak_activation,
        "activation_saving_ratio": round(
            1.0 - peak_activation / naive_activation, 4) if naive_activation else 0.0,
        "weights_bytes": weights_bytes,
        "io_persistent_bytes": io_bytes,
        "naive_total_bytes": weights_bytes + io_bytes + naive_activation,
        "peak_total_bytes": weights_bytes + io_bytes + peak_activation,
        "slots": [{"id": i, "bytes": s["bytes"], "tensors": s["tensors"]}
                  for i, s in enumerate(slots)],
        "tensor_slots": tensor_slots,
        "_note": "复用条件基于依赖偏序（祖先集），在并行分支上同样安全；"
                 "view 类算子为零拷贝别名，与其根张量共槽、不单独占缓冲，"
                 "故 naive 只统计需要独立缓冲的张量；"
                 "多帧流水时每帧独立一套 slot 池，峰值需乘以在飞帧数。",
    }
    # 自校验：规划本身必须可被证明无并发读写冲突
    plan["verified"] = not verify_buffer_plan(g, plan)
    return plan


def verify_buffer_plan(g: Graph, plan: dict[str, Any]) -> list[str]:
    """静态证明：同一 slot 内任意两个张量不会并发读写。

    对 slot 内按 producer 拓扑序排列的相邻张量，检查前一个（及其之前所有）
    张量的 consumer 是否全部 happens-before 后一个张量的 producer。
    返回错误列表，空列表表示规划安全。
    """
    _, pos, masks = _ancestors(g)
    consumers: dict[str, list[str]] = {tid: [] for tid in g.tensors}
    for n in g.nodes:
        for t in n.inputs:
            if t in consumers:
                consumers[t].append(n.id)

    errs: list[str] = []
    for s in plan.get("slots", []):
        release = 0
        members = sorted(s["tensors"], key=lambda t: pos.get(g.tensors[t].producer, -1))
        for tid in members:
            t = g.tensors[tid]
            prod = t.producer
            if prod not in pos:
                errs.append(f"slot {s['id']} 含无 producer 的张量 {tid}")
                continue
            prod_anc = masks[pos[prod]]
            if release & ~prod_anc:
                errs.append(
                    f"slot {s['id']}: 张量 {tid} 的 producer 与仍在读该槽的旧张量并发")
            for c in consumers.get(tid, []):
                if c in pos:
                    release |= 1 << pos[c]
    # view 别名：输出必须与根张量同槽（根为常驻时输出独占，运行时改走拷贝）
    for t, r in _alias_roots(g).items():
        ts, rs = g.tensors[t].slot, g.tensors[r].slot
        if ts >= 0 and ts != rs:
            errs.append(f"view 别名 {t}(slot {ts}) 与根 {r}(slot {rs}) 不一致")
    return errs
