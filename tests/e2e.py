"""端到端验证：核心单测 → 编译三场景 → C++ 运行时数值/调度/内存校验。

这是「无人值守」复现入口：一条命令验证全链路。
    cd /workspace && python3 -m tests.e2e

前置：C++ 运行时已编译（runtime/build/hx_run 存在）。
"""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HX_RUN = os.path.join(ROOT, "runtime", "build", "hx_run")
PKGS = ["demo", "parallel", "chain"]


def _run(cmd: list[str], cwd: str = ROOT) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def main() -> int:
    if not os.path.exists(HX_RUN):
        print("[fail] C++ 运行时未编译，请先：")
        print("       cd runtime && cmake -S . -B build && cmake --build build -j8")
        return 1
    all_ok = True

    print("== 步骤 1 / 5：核心单元测试（代价模型 / 模拟器 / 内存 / 分区）==")
    r = _run([sys.executable, "-m", "tests.test_core"])
    tail = (r.stdout + r.stderr).strip().splitlines()
    ok = r.returncode == 0 and any("OK" in x for x in tail[-3:])
    all_ok &= ok
    print(f"   {'PASS' if ok else 'FAIL'}  {tail[-1] if tail else ''}")

    print("== 步骤 2 / 5：编译三场景（训练侧零约束 → 分区 → 部署包）==")
    r = _run([sys.executable, "-m", "examples.export"])
    ok = r.returncode == 0
    all_ok &= ok
    print(f"   {'PASS' if ok else 'FAIL'}  demo(96x96,auto) / chain(链式) / parallel(force_hetero)")
    if not ok:
        print(r.stdout[-2000:]); print(r.stderr[-2000:]); return 1

    print("== 步骤 3 / 5：C++ 运行时数值 + 调度 + 缓冲复用 ==")
    for m in PKGS:
        p = os.path.join(ROOT, "build", m, "manifest.json")
        r = _run([HX_RUN, "--manifest", p, "--frames", "1", "--depth", "1",
                  "--scale", "20"])
        out = r.stdout + r.stderr
        passed = ("PASS" in out) and ("max_abs_err=0.00" in out) and (r.returncode == 0)
        reuse_ok = any("reuse=" in ln and "slots=" in ln for ln in out.splitlines())
        passed = passed and reuse_ok
        all_ok &= passed
        print(f"   {m:8s}: {'PASS' if passed else 'FAIL'} (rc={r.returncode})")
        for line in out.splitlines():
            if "wall=" in line or "numerics_ok" in line or "memory:" in line:
                print("      " + line.strip())

    print("== 步骤 4 / 5：跨帧流水（depth=3, 5 帧）数值正确性 ==")
    for m in ("demo", "parallel"):
        p = os.path.join(ROOT, "build", m, "manifest.json")
        r = _run([HX_RUN, "--manifest", p, "--frames", "5", "--depth", "3",
                  "--scale", "20"])
        out = r.stdout + r.stderr
        passed = ("PASS" in out) and (r.returncode == 0)
        all_ok &= passed
        print(f"   {m:8s}: {'PASS' if passed else 'FAIL'}")

    print("== 步骤 5 / 5：跨芯片并行证据 ==")
    for m in PKGS:
        p = os.path.join(ROOT, "build", m, "manifest.json")
        r = _run([HX_RUN, "--manifest", p, "--frames", "1", "--depth", "1",
                  "--scale", "1"])
        for line in (r.stdout + r.stderr).splitlines():
            if "overlap_pairs" in line:
                print(f"   {m:8s}: {line.strip()}")
                break

    print("== 结论 ==")
    print("ALL PASS ✅" if all_ok else "SOME FAILED ❌")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
