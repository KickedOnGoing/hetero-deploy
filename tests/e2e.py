"""端到端验证：编译三场景 → C++ 运行时执行 → 数值 / 调度校验。

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


def _run(cmd: list[str], cwd: str) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def main() -> int:
    if not os.path.exists(HX_RUN):
        print("[fail] C++ 运行时未编译，请先：")
        print("       cd runtime && cmake -S . -B build && cmake --build build -j8")
        return 1

    print("== 步骤 1 / 3：编译三场景（训练侧零约束 → 分区 → 部署包）==")
    r = _run([sys.executable, "-m", "examples.export"], cwd=ROOT)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr); return 1
    print("        ok：demo(96x96,auto) / chain(链式) / parallel(force_hetero)")

    print("== 步骤 2 / 3：C++ 运行时数值 + 调度校验 ==")
    all_ok = True
    for m in PKGS:
        p = os.path.join(ROOT, "build", m, "manifest.json")
        r = _run([HX_RUN, "--manifest", p, "--frames", "1", "--depth", "1",
                  "--scale", "20"], cwd=ROOT)
        out = r.stdout + r.stderr
        passed = ("PASS" in out) and ("max_abs_err=0.00" in out) and (r.returncode == 0)
        all_ok = all_ok and passed
        print(f"   {m:8s}: {'PASS' if passed else 'FAIL'} (rc={r.returncode})")
        for line in out.splitlines():
            if "wall=" in line or "numerics_ok" in line or "overlap_pairs" in line:
                print("      " + line.strip())

    print("== 步骤 3 / 3：跨芯片并行证据 ==")
    for m in PKGS:
        p = os.path.join(ROOT, "build", m, "manifest.json")
        r = _run([HX_RUN, "--manifest", p, "--frames", "1", "--depth", "1",
                  "--scale", "1"], cwd=ROOT)
        out = r.stdout + r.stderr
        for line in out.splitlines():
            if "overlap_pairs" in line:
                print(f"   {m:8s}: {line.strip()}")
                break

    print("== 结论 ==")
    print("ALL PASS ✅" if all_ok else "SOME FAILED ❌")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
