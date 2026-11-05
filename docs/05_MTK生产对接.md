# 生产对接路线：从原型到天玑真机

本文件把 `docs/01_可行性论证.md` 第 8 节的路线具体化，给出可直接排期的步骤、
接口形状与验收口径。所有性能数字都来自**目标机型上板 profiling**，不是本沙箱模板。

---

## 0. 工具链在流水线中的位置

```
PyTorch 训练模型
      │ torch.export（训练侧零约束）
      ▼
HX IR (fx 降级) ──► 分区搜索 ──► 部署计划 (manifest + 子图)
      │                              │
      │ hx_compile                   │ 每个 Part 的 TFLite 子图
      ▼                              ▼
 HX 运行时 (C++ SDK)          NP Converter / ncc-tflite --arch=mdla3.0
  (DAG 调度 + 同步 + 流水)            │
      │                              ▼
      │                        .dla / LiteRT Dispatch context
      └────────── 派发执行 ──────► Neuron Runtime / LiteRT GPU
```

HX 不替代 MTK 的编译器，而是**在它之上做图分区与运行时编排**：每个 `Part`
最终仍交给 MTK 自己的编译产物执行。

---

## 1. Phase 0 — 校准（没有它，任何加速比数字都无意义）

在目标天玑机型（如 9300/9400 类）上测出，写入 `hx/opset.py` 的 `DEFAULT_CHIP`
或独立 `chip.json`：

| 字段 | 含义 | 测量方法 |
|---|---|---|
| `backends.*.peak_gflops` | fp16 等效峰值 | 标准算子 micro-benchmark |
| `backends.*.launch_us` | 一次提交固定开销 | 空/小算子 invoke 多次取均值 |
| `backends.*.layout` | 偏好 layout | 查 SDK 文档 |
| `fixed_cross_us` | 跨片同步固定开销 | 两个子图乒乓 invoke 测增量 |
| `per_mb_us` | 跨片搬运带宽倒数 | 不同尺寸张量搬运计时 |
| `layout_convert_us` | layout 转换固定开销 | 实测 |
| `min_npu_nodes` / `min_part_flops` | 最小 NPU 子图规模 | 扫不同子图规模取拐点 |

> 对应 MTK `NEURON_FLAG_MIN_GROUP_SIZE` 与本工具链的碎片抑制参数。

校准后 `python3 -m examples.export` 与 `tests/sensitivity.py` 产出的加速比才可信。

---

## 2. Phase 1 — 单后端基线

整图走 LiteRT `CompiledModel(Accelerator.NPU)` 或 NP8 Converter，建立
**单后端基线 makespan**。这是 `docs/01` 里 `baselines_single_backend_us` 的真机来源，
也是一切「该不该拆」裁决的参照。

---

## 3. Phase 2 — 分块与异构执行

1. `hx-compile --model model.pt --input sample.pt --chip chip.json --strategy auto`
   产出 `manifest.json` + 各 `Part` 的 TFLite 子图 + `tensor_requirements.json`；
2. NPU 子图用 `ncc-tflite --arch=mdla3.0` 编译为 `.dla`；
3. 运行时按 `manifest` 创建后端，DAG 就绪驱动提交；
4. **异步执行 + fence 同步**：用 LiteRT Dispatch `async_interface` 与 `LiteRtEvent`
   （sync fence fd）实现跨加速器同步，而不是 CPU 轮询；
5. **零拷贝共享张量**：`RegisterTensorBuffer` / `AttachInput` 用 `AHardwareBuffer` /
   DMA-BUF，避免跨芯片物理拷贝；仅当上下游 layout 不一致时按 `layout_convert_us` 转换。

> `manifest` 的 `plan` 字段已给出 `verdict`、预测 makespan、跨片边数与搬运占比，
> 上板后应与真机 profile 核对。

---

## 4. Phase 3 — 跨帧流水与自适应

- **跨帧流水**：`Options.pipeline_depth > 1` 时第 k 帧在 NPU、第 k-1 帧在 GPU，
  只提吞吐不降单帧延迟（见 `docs/01` 第 3 节基准）；
- **热感知调度**：监听 SoC 温度/频率，持续满载触发降频时按 `backend_priority`
  把负载迁到更稳的后端或降级到单后端；
- **能力探测与降级**：任一后端 `execute()` 返回 `false` 时，运行时 100% 回退到 CPU
  （见 `ADAPTERS.md` 第 1 节错误契约）。

---

## 5. Phase 4 — 规模化

- 多模型 / 多任务统一调度（按 `backend_busy_us` 负载均衡）；
- 模型 OTA：编译产物按 SoC 型号分发（`chip.json` 维度）；
- A/B 与回放：保留 `timeline.json` 时间线用于回归比对。

---

## 6. 量化与一致性

| 步骤 | 工具 | 说明 |
|---|---|---|
| 校准 | `hx.compile_model(..., calibration_inputs=...)` | 提供校准集触发 PTQ |
| 量化 | NP Converter / LiteRT quantizer | INT8/FP16 子图 |
| 校验 | `verify_numerics` | 逐子图与端到端比对 PyTorch（fp32 参考） |
| 边界声明 | `tensor_requirements.json` | 混合精度转换点显式声明 |

---

## 7. 验收标准（建议）

1. 逐子图与端到端数值误差 ≤ 量化容差；
2. 单帧延迟相对单后端基线改善 ≥ 15%（否则不拆，工具链已自动判定 `MARGINAL/NOT_RECOMMENDED`）；
3. 持续 30 分钟满载无热降频导致的性能塌方（Phase 3 兜底）；
4. 任一后端初始化失败 100% 可降级到 CPU；
5. 跨片边界数量、layout 转换次数在上板 profile 中可核对，与 `tensor_requirements.json` 一致；
6. `hx_bench` 跨帧流水吞吐提升符合 `plan.pipeline.steady_fps` 预测。

---

## 8. 明确不依赖 NNAPI

自 Android 15 起 NNAPI deprecated，Google 宣布 2027 移除。本方案走 **LiteRT
Compiler Plugin / Dispatch API**，对 NNAPI 零依赖，规避生命周期风险。
