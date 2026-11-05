// HX 运行时 SDK 公共 API（C++17）。
//
//   auto engine = hx::Engine::load("build/demo/manifest.json");
//   engine->set_input(0, ptr, n);
//   auto rep = engine->run(hx::Options{.frames = 5, .pipeline_depth = 3, .time_scale = 20});
//
// 设计要点：
//  * 部署包驱动：manifest.json + weights.bin 就是全部输入，不链接 PyTorch；
//  * 三个后端各一条串行执行队列（真实硬件约束），DAG 就绪驱动提交；
//  * 支持跨帧流水，多帧 in-flight 时不同后端自然重叠；
//  * 共享缓冲 + owner/fence 语义，跨后端边按 manifest 的 cost_us 计入；
//  * 全链路 profiling，输出可核对的时间线。
#pragma once
#include <memory>
#include <string>
#include <vector>

#include "hx_types.h"

namespace hx {

struct Options {
    int frames = 1;                 // 总帧数
    int pipeline_depth = 1;         // 同时在飞的帧数（跨帧流水）
    double time_scale = 1.0;        // 模拟延迟压缩倍率，1.0 = 真实估计时长
    bool simulate_transfer = true;  // 计入跨片搬运/同步开销
    bool profile = true;
    bool zero_copy = true;          // 共享缓冲，不做物理拷贝
    bool skip_compute = false;      // 只跑调度（吞吐压测），不执行算子、不校验数值
    std::string timeline_path;      // 非空则写出 JSON 时间线
};

struct PartTiming {
    int frame = -1;
    int part = -1;
    std::string backend;
    double start_us = 0.0;
    double end_us = 0.0;
    double virtual_us = 0.0;        // 该 Part 的模型估计时长
    std::string kind = "compute";   // compute | transfer
};

struct RunReport {
    bool ok = false;
    std::string error;
    double wall_us = 0.0;           // 真实墙钟（受 time_scale 影响）
    double virtual_wall_us = 0.0;   // 按 time_scale 还原后的等效时长
    double predicted_us = 0.0;      // manifest.plan.simulation.makespan_us
    double max_abs_err = 0.0;
    bool numerics_ok = false;
    size_t completed_parts = 0;
    double backend_busy_us[3] = {0.0, 0.0, 0.0};
    int overlap_pairs = 0;          // 不同后端在时间上重叠的 Part 对数
    std::vector<PartTiming> timeline;

    std::string gantt() const;
    JValue to_json() const;
    std::string brief() const;
};

class Engine {
public:
    static std::unique_ptr<Engine> load(const std::string& manifest_path);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    const Model& model() const;
    const std::string& dir() const;

    // 输入/输出：fp32 连续内存，按 manifest 的 graph.inputs/outputs 顺序。
    bool set_input(int idx, const float* data, size_t n);
    const float* output(int idx, size_t* n) const;

    RunReport run(const Options& opt);
    std::string summary() const;

private:
    Engine();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// 工具：读取二进制测试向量
bool read_bin_f32(const std::string& path, std::vector<float>& out);
bool write_bin_f32(const std::string& path, const float* data, size_t n);

}  // namespace hx
