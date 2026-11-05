// 可插拔后端接口。
//
// 生产落地时只需实现 IBackend::execute：把 Part 交给
//   - NPU: LiteRT Dispatch API (libLiteRtDispatch_MediaTek.so) / Neuron Runtime (.dla)
//   - GPU: LiteRT CompiledModel(Accelerator.GPU) / OpenCL-Vulkan compute
//   - CPU: XNNPACK / ARM NN
// 调度器、缓冲池、fence、profiling 全部复用，不感知具体 SDK。
#pragma once
#include <memory>
#include <string>

#include "hx_types.h"

namespace hx {

class IBackend {
public:
    virtual ~IBackend() = default;
    virtual Backend kind() const = 0;
    virtual const char* name() const = 0;
    virtual double launch_us() const = 0;
    virtual bool execute(const PartDesc& part, const Model& model,
                         ITensorStore& store, std::string* err) = 0;
};

// 原型后端：真实执行算子（保证数值正确）+ 按代价模型注入时间延迟（保证调度可观测）。
// time_scale > 1 时按倍率压缩 sleep，便于在 CI 中快速验证并发时序。
std::unique_ptr<IBackend> create_reference_backend(Backend b, double time_scale,
                                                   const Model* model);

// 生产对接点（见 runtime/backends/）。未启用 HX_ENABLE_MTK 时返回 nullptr，
// 运行时自动回退到参考后端，因此本工程在无 MTK SDK 的环境也能完整编译运行。
std::unique_ptr<IBackend> create_mtk_neuron_backend(const Model* model, std::string* err);
std::unique_ptr<IBackend> create_litert_gpu_backend(const Model* model, std::string* err);

}  // namespace hx
