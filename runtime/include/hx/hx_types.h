// 部署包在 C++ 侧的镜像结构。字段与 manifest.json 一一对应。
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "json.hpp"

namespace hx {

enum class Backend : uint8_t { CPU = 0, GPU = 1, NPU = 2, Unknown = 3 };

inline Backend parse_backend(const std::string& s) {
    if (s == "cpu") return Backend::CPU;
    if (s == "gpu") return Backend::GPU;
    if (s == "npu") return Backend::NPU;
    return Backend::Unknown;
}
inline const char* backend_name(Backend b) {
    switch (b) {
        case Backend::CPU: return "cpu";
        case Backend::GPU: return "gpu";
        case Backend::NPU: return "npu";
        default: return "unknown";
    }
}
inline int backend_index(Backend b) {
    return b == Backend::Unknown ? -1 : static_cast<int>(b);
}

enum class DType : uint8_t { F32 = 0, F16 = 1, I8 = 2, U8 = 3, I32 = 4, I64 = 5 };

inline DType parse_dtype(const std::string& s) {
    if (s == "f16" || s == "bf16") return DType::F16;
    if (s == "i8") return DType::I8;
    if (s == "u8") return DType::U8;
    if (s == "i32") return DType::I32;
    if (s == "i64") return DType::I64;
    return DType::F32;
}
inline size_t dtype_size(DType d) {
    switch (d) {
        case DType::F16: return 2;
        case DType::I8: case DType::U8: return 1;
        case DType::I32: case DType::I64: return 4;
        default: return 4;
    }
}

struct TensorDesc {
    std::string id;
    DType dtype = DType::F32;
    std::vector<int64_t> shape;
    std::string kind;           // input|output|weight|bias|constant|activation
    int64_t offset = -1;        // weights.bin 字节偏移
    size_t bytes = 0;
    std::string producer;
    int slot = -1;                  // 共享缓冲槽；-1 = 常驻/独占

    size_t numel() const {
        if (shape.empty()) return 0;
        size_t n = 1;
        for (auto d : shape) n *= static_cast<size_t>(d < 0 ? 0 : d);
        return n;
    }
};

struct NodeDesc {
    std::string id;
    std::string op;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    JValue attrs;
    int64_t flops = 0;
    int part = -1;
    Backend backend = Backend::CPU;
    std::string name;
};

struct PartDesc {
    int id = -1;
    Backend backend = Backend::CPU;
    std::vector<std::string> nodes;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    double est_us = 0.0;
    std::string artifact;
};

struct EdgeDesc {
    int src = -1, dst = -1;
    std::string tensor;
    Backend src_backend = Backend::CPU, dst_backend = Backend::CPU;
    size_t bytes = 0;
    double cost_us = 0.0;
    bool cross = false;
    bool zero_copy = false;
};

struct Model {
    std::string name;
    std::vector<TensorDesc> tensors;
    std::vector<NodeDesc> nodes;
    std::vector<PartDesc> parts;
    std::vector<EdgeDesc> edges;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::string weights_file;
    size_t weights_bytes = 0;
    // 编译期内存规划（manifest.memory）：按 slot 共享缓冲，避免 O(全图激活) 峰值
    std::vector<size_t> slot_bytes;
    size_t memory_naive_bytes = 0;
    size_t memory_peak_bytes = 0;
    JValue plan;
    JValue chip;
    JValue raw;

    // O(1) 索引：算子执行时 get(tensor)/node() 是热点，线性查找会放大开销
    std::unordered_map<std::string, size_t> tensor_idx;
    std::unordered_map<std::string, size_t> node_idx;

    void build_index() {
        tensor_idx.clear();
        node_idx.clear();
        for (size_t i = 0; i < tensors.size(); ++i) tensor_idx[tensors[i].id] = i;
        for (size_t i = 0; i < nodes.size(); ++i) node_idx[nodes[i].id] = i;
    }

    const TensorDesc* tensor(const std::string& id) const {
        auto it = tensor_idx.find(id);
        return it == tensor_idx.end() ? nullptr : &tensors[it->second];
    }
    const NodeDesc* node(const std::string& id) const {
        auto it = node_idx.find(id);
        return it == node_idx.end() ? nullptr : &nodes[it->second];
    }
};

// 运行时张量：存储可共享（reshape / flatten 为零拷贝视图），
// owner 记录最后一次写入它的后端 —— 对应生产环境 AHardwareBuffer + fence 的所有权语义。
struct TensorBuf {
    TensorDesc desc;
    std::shared_ptr<std::vector<float>> storage;
    size_t offset = 0;          // 以 float 元素计
    Backend owner = Backend::CPU;
    bool ready = false;

    float* data() { return storage ? storage->data() + offset : nullptr; }
    const float* data() const { return storage ? storage->data() + offset : nullptr; }
    size_t numel() const { return desc.numel(); }
};

// 后端执行时的张量仓库抽象。
class ITensorStore {
public:
    virtual ~ITensorStore() = default;
    virtual TensorBuf& get(const std::string& id) = 0;
    virtual TensorBuf& alloc(const std::string& id, const TensorDesc& desc) = 0;
    virtual bool exists(const std::string& id) const = 0;
    virtual const float* weights() const = 0;
    virtual size_t weights_size() const = 0;
    virtual void delay(double virtual_us) = 0;   // 后端延迟注入（原型）
    virtual bool compute_enabled() const = 0;    // false = 只跑调度，不跑算子（吞吐压测）
};

}  // namespace hx
