// 后端实现。
//
// 本原型提供「参考后端」：真实执行 IR 算子（数值与 PyTorch 对齐），
// 并按 manifest 里的代价模型注入执行时长，从而在无 MTK 硬件的环境下
// 完整验证调度器、跨片同步、跨帧流水与 profiling。
//
// 生产构建定义 HX_ENABLE_MTK 后，把 NPU/GPU 后端替换成下方对接骨架即可，
// 调度层代码零改动。
#include "hx/hx_backend.h"

#include <cmath>
#include <thread>

#include "ops.hpp"

namespace hx {
namespace {

class ReferenceBackend : public IBackend {
public:
    ReferenceBackend(Backend b, double scale, const Model* model)
        : b_(b), scale_(scale > 0 ? scale : 1.0), model_(model) {}

    Backend kind() const override { return b_; }
    const char* name() const override { return backend_name(b_); }

    double launch_us() const override {
        if (!model_) return 0.0;
        try {
            return model_->chip.at("backends").at(backend_name(b_))
                        .at("launch_us").as_num(0.0);
        } catch (...) { return 0.0; }
    }

    bool execute(const PartDesc& part, const Model& model,
                 ITensorStore& store, std::string* err) override {
        // 1) 注入后端执行时长（真实硬件上是 APU/GPU 的计算，这里用受控 sleep 模拟）
        store.delay(part.est_us / scale_);
        // 2) 真实执行算子，保证数值正确、可校验（吞吐压测时可跳过）
        if (store.compute_enabled()) {
            for (const auto& nid : part.nodes) {
                const NodeDesc* n = model.node(nid);
                if (!n) { if (err) *err = "part references unknown node: " + nid; return false; }
                if (!run_node(*n, model, store, err)) return false;
            }
        }
        // 3) 标记输出张量的所有权（对应 AHardwareBuffer + fence 的归属移交）
        for (const auto& tid : part.outputs)
            if (store.exists(tid)) store.get(tid).owner = b_;
        return true;
    }

private:
    Backend b_;
    double scale_;
    const Model* model_;
};

}  // namespace

std::unique_ptr<IBackend> create_reference_backend(Backend b, double time_scale,
                                                   const Model* model) {
    return std::make_unique<ReferenceBackend>(b, time_scale, model);
}

// ---------------------------------------------------------------------------
// 生产对接骨架：OEM 侧拿到 NeuroPilot NP8 / LiteRT SDK 后在此实现。
// 接口形状与真实 API 对齐，见 runtime/backends/ADAPTERS.md。
// ---------------------------------------------------------------------------
std::unique_ptr<IBackend> create_mtk_neuron_backend(const Model*, std::string* err) {
    if (err) *err = "MTK Neuron backend not built (define HX_ENABLE_MTK and link "
                    "libLiteRtDispatch_MediaTek.so / NeuronSDK); using reference backend";
    return nullptr;
}

std::unique_ptr<IBackend> create_litert_gpu_backend(const Model*, std::string* err) {
    if (err) *err = "LiteRT GPU backend not built (define HX_ENABLE_LITERT_GPU); "
                    "using reference backend";
    return nullptr;
}

}  // namespace hx
