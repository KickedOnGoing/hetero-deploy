# 后端适配器：从参考后端到真机 NPU / GPU

本目录承载「运行时 SDK 如何对接 MTK 真机 SDK」的实现骨架。
**调度器（`runtime/src/runtime.cc`）完全不感知具体芯片 SDK**——它只认
`IBackend` 接口与 `manifest.json`。因此生产落地时你只需要实现「把一个 `Part`
交给某后端执行」，其余（缓冲池、fence、DAG 拓扑、跨帧流水、profiling）全部复用。

---

## 1. 接口契约（不可改）

定义在 `runtime/include/hx/hx_backend.h`：

```cpp
class IBackend {
public:
    virtual ~IBackend() = default;
    virtual Backend kind() const = 0;                 // CPU / GPU / NPU
    virtual const char* name() const = 0;             // 调试用
    virtual double launch_us() const = 0;             // 一次提交固定开销
    virtual bool execute(const PartDesc& part,
                         const Model& model,
                         ITensorStore& store,
                         std::string* err) = 0;       // 执行一个 Part
};
```

要点：

- `launch_us()` 必须回填芯片档案里该后端的 `launch_us`（提交/握手固定开销）。调度器的离散事件模拟直接读取它。
- `execute()` **只负责一个 Part 内的算子**；Part 之间的依赖、缓冲、同步由调度器统一编排。
- `store` 是帧内张量仓库（`ITensorStore`），`execute` 通过它按 `TensorDesc.id` 取输入、写输出。跨片张量已经在上一帧/上游 Part 写好后就绪。
- 出错时把人类可读原因写进 `err` 并返回 `false`；调度器会据此触发降级。

---

## 2. 参考后端（已实现，可直接编译运行）

`runtime/src/backend.cc` 中的 `create_reference_backend`：

- **真实执行 IR 算子**（`src/ops.cc`）：卷积、GEMM、逐元素、归一化、池化、concat 等，数值严格正确，因此端到端可与 PyTorch 比对。
- **按代价模型注入时长**：`Options.time_scale` 把 `PartDesc.cost_us` 压成 `sleep`，让并发时序在 CI 里可观测、可复现，又不拖慢测试。
- 它验证的是**编译 → 分区 → 调度 → 同步 → 流水 → 校验**这条链路的完整性与正确性，不是目标芯片性能数字。

> 没有 MTK SDK 的任何环境（如本沙箱）也能完整编译运行，正是靠这个参考后端。

---

## 3. Neuron（NPU）适配骨架

### 对接对象
- **公开路径**：LiteRT `CompilerPlugin` + `Dispatch API`，加载 MTK 提供的
  `libLiteRtDispatch_MediaTek.so`；或
- **OEM/NDA 路径**：NeuroPilot Neuron SDK，`ncc-tflite --arch=mdla3.0` 离线编译出 `.dla`，
  用 `neuronrt` 执行、Neuron Delegate 派发。

### 实现步骤
1. `cmake -DHX_ENABLE_MTK=ON`，链接 `neuronrt` / LiteRT Dispatch 与目标机 NNAPI；
2. 在 `create_mtk_neuron_backend()` 中：
   - 把 `Part` 对应的 **TFLite 子图**（由 `hx_compile` 用 NP Converter 或 `ncc-tflite`
     生成 `.dla` / context binary）编译进 `NeuronModel`；
   - `execute()`：从 `store` 取出输入张量，**零拷贝**共享 `AHardwareBuffer` / DMA-BUF，
     调用 `neuronrt` 异步 `invoke`，用 `LiteRtEvent`（sync fence fd）等待完成；
3. `launch_us()` 填 APU 实测提交开销；不支持的算子已由编译期能力表降级（不会到这一步才失败）。

### 关键约束（来自平台事实）
- APU 要求 **float32 bias、不能用混合精度整型 bias**（社区实测踩坑）；
- 过度分区实测带来最高 **+28% 延迟**——所以编译期有 `min_npu_nodes` / `min_part_flops`
  与软惩罚抑制碎片；
- 持续满载 2.5 分钟后热降频最高 **4.3× 衰减**——运行时需热感知动态降级。

---

## 4. LiteRT GPU 适配骨架

用 `LiteRT CompiledModel(Accelerator.GPU)` 或 OpenCL / Vulkan Compute 后端：

1. `cmake -DHX_ENABLE_LITERT_GPU=ON`；
2. `create_litert_gpu_backend()` 把 `Part` 的 TFLite 子图交给 GPU delegate；
3. `execute()` 同样走 `store` 取放张量、异步执行、fence 同步。
GPU 的优势在于非线性/归一化（SiLU/GELU/Softmax/LayerNorm）吞吐远高于 APU，
适合承载 APU 能力表外的算子或需与 NPU 分支并发的旁路。

---

## 5. 共享缓冲与零拷贝

`hx_compile` 产出 `tensor_requirements.json`，声明每个跨片张量的
**buffer type / layout / stride / align**（NHWC vs NCHW、AHWB vs GL texture 等）。

- 参考后端用统一 `f32` 向量，跳过转换；
- **真机必须实现协商**：`execute()` 时校验 `store` 中张量是否满足该后端 buffer 要求，
  不满足则按 `chip.json` 的 `layout_convert_us` 转换一次。若 NPU 只吃特定对齐 AHWB、
  而 GPU 产出 GL texture，每帧都要转换——这是零拷贝失效的真实代价，已在代价模型里计入。

---

## 6. 量化 / 混合精度

- INT8 / FP16 子图：权重在编译期由 `hx_compile` 用校准集做 PTQ 量化并固化进 `weights.bin`；
- `ITensorStore` 的张量带 `dtype` 字段，`execute()` 按子图 `dtype` 执行；
- 边界处若有混合精度，编译期在 `tensor_requirements.json` 显式声明转换点，
  运行期按声明执行；数值校验（编译期逐子图比对 PyTorch）会捕获漂移。

---

## 7. 编译开关

```cmake
option(HX_ENABLE_MTK       "Link MTK NeuroPilot / LiteRT Dispatch backend" OFF)
option(HX_ENABLE_LITERT_GPU "Link LiteRT GPU backend"                     OFF)
```

未开启时 `create_mtk_neuron_backend` / `create_litert_gpu_backend` 返回 `nullptr`，
`Engine::load` 自动回退到参考后端——**本工程在任何环境都能编译运行**。
