// 跨帧流水基准：验证「不同帧占用不同后端」带来的吞吐提升，
// 并对比 manifest 里代价模型的单帧预测。
#include <algorithm>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "hx/hx_runtime.h"

using namespace hx;

int main(int argc, char** argv) {
    std::string manifest = "build/demo/manifest.json";
    int frames = 8;
    // 默认放大虚拟延迟（scale<1 等价放大 sleep），以降低 us 级 sleep 精度噪声占比，
    // 让跨帧流水的吞吐趋势稳定可复现；绝对值已缩放，仅反映趋势。
    double scale = 0.05;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (k == "--manifest") manifest = next();
        else if (k == "--frames") frames = std::atoi(next());
        else if (k == "--scale") scale = std::atof(next());
    }
    try {
        auto engine = Engine::load(manifest);
        JValue io = json_load_file(engine->dir() + "/io.json");
        int idx = 0;
        for (const auto& v : io.at("inputs").arr) {
            std::vector<float> d;
            read_bin_f32(engine->dir() + "/" + v.at("file").as_str(), d);
            engine->set_input(idx++, d.data(), d.size());
        }
        std::cout << "# model=" << engine->model().name
                  << " parts=" << engine->model().parts.size()
                  << " frames=" << frames << " time_scale=" << scale << "\n";
        std::cout << "depth,wall_us,virtual_wall_us,throughput_fps,overlap_pairs,completed_parts,ok\n";
        for (int depth : {1, 4, 8, 12, 16, 24}) {
            Options o;
            o.frames = frames;
            o.pipeline_depth = depth;
            o.time_scale = scale;
            o.profile = true;
            o.skip_compute = true;   // 纯调度压测：时间线只由虚拟延迟决定
            RunReport r = engine->run(o);
            const double fps = r.wall_us > 0 ? frames * 1e6 / r.wall_us : 0.0;
            std::cout << depth << "," << std::fixed << std::setprecision(1) << r.wall_us
                      << "," << r.virtual_wall_us << "," << std::setprecision(2) << fps
                      << "," << r.overlap_pairs << "," << r.completed_parts << ","
                      << (r.ok ? "true" : "false") << "\n";
            if (!r.ok) std::cerr << "  error: " << r.error << "\n";
        }
        std::cout << "# predicted single-frame makespan_us = "
                  << engine->model().plan.at("simulation").at("makespan_us").as_num(0.0)
                  << " (scale-invariant)\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }
}
