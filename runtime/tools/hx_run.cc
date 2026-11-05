// 端到端 runner：加载部署包 -> 灌入测试向量 -> 多后端并发执行 -> 数值校验 + 时间线
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "hx/hx_runtime.h"

using namespace hx;

namespace {
struct Args {
    std::string manifest = "build/demo/manifest.json";
    std::string timeline;
    int frames = 1;
    int depth = 1;
    double scale = 20.0;
    bool transfer = true;
    double atol = 2e-4;
};

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() -> std::string {
            return (i + 1 < argc) ? argv[++i] : std::string();
        };
        if (k == "--manifest") a.manifest = next();
        else if (k == "--timeline") a.timeline = next();
        else if (k == "--frames") a.frames = std::atoi(next().c_str());
        else if (k == "--depth") a.depth = std::atoi(next().c_str());
        else if (k == "--scale") a.scale = std::atof(next().c_str());
        else if (k == "--atol") a.atol = std::atof(next().c_str());
        else if (k == "--no-transfer") a.transfer = false;
        else if (k == "--help" || k == "-h") {
            std::cout << "usage: hx_run --manifest PATH [--frames N] [--depth D] "
                         "[--scale S] [--timeline PATH] [--no-transfer] [--atol E]\n";
            std::exit(0);
        }
    }
    return a;
}
}  // namespace

int main(int argc, char** argv) {
    Args a = parse(argc, argv);
    try {
        auto engine = Engine::load(a.manifest);
        std::cout << "=== deployed package ===\n" << engine->summary() << "\n\n";

        // 读取测试向量
        JValue io = json_load_file(engine->dir() + "/io.json");
        int idx = 0;
        for (const auto& v : io.at("inputs").arr) {
            std::vector<float> data;
            const std::string p = engine->dir() + "/" + v.at("file").as_str();
            if (!read_bin_f32(p, data)) {
                std::cerr << "cannot read input: " << p << "\n";
                return 2;
            }
            engine->set_input(idx++, data.data(), data.size());
        }
        std::vector<std::vector<float>> expected;
        for (const auto& v : io.at("outputs").arr) {
            std::vector<float> data;
            const std::string p = engine->dir() + "/" + v.at("file").as_str();
            if (!read_bin_f32(p, data)) {
                std::cerr << "cannot read expected output: " << p << "\n";
                return 2;
            }
            expected.push_back(std::move(data));
        }

        Options opt;
        opt.frames = a.frames;
        opt.pipeline_depth = a.depth;
        opt.time_scale = a.scale;
        opt.simulate_transfer = a.transfer;
        opt.timeline_path = a.timeline;
        RunReport rep = engine->run(opt);

        // 数值校验（取第 0 帧输出）
        double maxerr = 0.0;
        bool num_ok = true;
        for (size_t i = 0; i < expected.size(); ++i) {
            size_t n = 0;
            const float* got = engine->output((int)i, &n);
            if (!got || n != expected[i].size()) { num_ok = false; continue; }
            for (size_t k = 0; k < n; ++k)
                maxerr = std::max(maxerr, (double)std::fabs(got[k] - expected[i][k]));
        }
        num_ok &= (maxerr <= a.atol);
        rep.max_abs_err = maxerr;
        rep.numerics_ok = num_ok;

        std::cout << "=== run report ===\n" << rep.brief() << "\n";
        std::cout << "=== execution timeline (N=NPU G=GPU C=CPU ==transfer) ===\n";
        std::cout << rep.gantt() << "\n";
        if (!a.timeline.empty())
            std::cout << "timeline json: " << a.timeline << "\n";
        if (!rep.ok) {
            std::cerr << "RUNTIME FAILED: " << rep.error << "\n";
            return 3;
        }
        if (!num_ok) {
            std::cerr << "NUMERICS FAILED: max_abs_err=" << maxerr
                      << " atol=" << a.atol << "\n";
            return 4;
        }
        std::cout << "PASS: runtime + numerics + scheduling\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FATAL: " << e.what() << "\n";
        return 1;
    }
}
