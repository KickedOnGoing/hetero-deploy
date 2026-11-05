// HX 运行时：部署包加载、共享缓冲、DAG 调度、跨帧流水、profiling。
#include "hx/hx_runtime.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>
#include <unordered_map>

#include "hx/hx_backend.h"
#include "ops.hpp"

namespace hx {
namespace {

using Clock = std::chrono::steady_clock;

double now_us(const Clock::time_point& t0) {
    return std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
}

// ------------------------------------------------------------ manifest 解析
Model parse_model(const JValue& j) {
    Model m;
    m.raw = j;
    const JValue& g = j.at("graph");
    m.name = g.at("name").as_str("model");
    for (const auto& v : g.at("inputs").arr) m.inputs.push_back(v.at("id").as_str());
    for (const auto& v : g.at("outputs").arr) m.outputs.push_back(v.at("id").as_str());

    for (const auto& kv : j.at("tensors").obj) {
        const JValue& v = kv.second;
        TensorDesc t;
        t.id = kv.first;
        t.dtype = parse_dtype(v.at("dtype").as_str("f32"));
        t.shape = v.at("shape").as_int_vec();
        t.kind = v.at("kind").as_str("activation");
        t.offset = v.at("offset").as_int(-1);
        t.bytes = (size_t)v.at("bytes").as_int(0);
        t.producer = v.at("producer").as_str("");
        m.tensors.push_back(std::move(t));
    }
    for (const auto& v : j.at("nodes").arr) {
        NodeDesc n;
        n.id = v.at("id").as_str();
        n.op = v.at("op").as_str();
        n.inputs = v.at("inputs").as_str_vec();
        n.outputs = v.at("outputs").as_str_vec();
        n.attrs = v.at("attrs");
        n.flops = v.at("flops").as_int(0);
        n.part = (int)v.at("part").as_int(-1);
        n.backend = parse_backend(v.at("backend").as_str("cpu"));
        n.name = v.at("name").as_str("");
        m.nodes.push_back(std::move(n));
    }
    for (const auto& v : j.at("parts").arr) {
        PartDesc p;
        p.id = (int)v.at("id").as_int(-1);
        p.backend = parse_backend(v.at("backend").as_str("cpu"));
        p.nodes = v.at("nodes").as_str_vec();
        p.inputs = v.at("inputs").as_str_vec();
        p.outputs = v.at("outputs").as_str_vec();
        p.est_us = v.at("est_us").as_num(0.0);
        p.artifact = v.at("artifact").as_str("");
        m.parts.push_back(std::move(p));
    }
    if (j.has("edges")) {
        for (const auto& v : j.at("edges").arr) {
            EdgeDesc e;
            e.src = (int)v.at("src").as_int(-1);
            e.dst = (int)v.at("dst").as_int(-1);
            e.tensor = v.at("tensor").as_str();
            e.src_backend = parse_backend(v.at("src_backend").as_str("cpu"));
            e.dst_backend = parse_backend(v.at("dst_backend").as_str("cpu"));
            e.bytes = (size_t)v.at("bytes").as_int(0);
            e.cost_us = v.at("cost_us").as_num(0.0);
            e.cross = v.at("cross").as_bool(false);
            e.zero_copy = v.at("zero_copy").as_bool(false);
            m.edges.push_back(std::move(e));
        }
    }
    m.weights_file = j.at("weights_file").as_str("weights.bin");
    m.weights_bytes = (size_t)j.at("weights_bytes").as_int(0);
    if (j.has("plan")) m.plan = j.at("plan");
    if (j.has("chip")) m.chip = j.at("chip");
    return m;
}

std::string dirname_of(const std::string& p) {
    size_t k = p.find_last_of("/\\");
    return k == std::string::npos ? "." : p.substr(0, k);
}

// JSON 序列化（写出时间线）
std::string dump_json(const JValue& v) {
    switch (v.type) {
        case JValue::T::Null: return "null";
        case JValue::T::Bool: return v.b ? "true" : "false";
        case JValue::T::Num: {
            std::ostringstream os; os << std::setprecision(10) << v.num; return os.str();
        }
        case JValue::T::Str: {
            std::string o = "\"";
            for (char c : v.str) {
                switch (c) {
                    case '"': o += "\\\""; break;
                    case '\\': o += "\\\\"; break;
                    case '\n': o += "\\n"; break;
                    case '\r': o += "\\r"; break;
                    case '\t': o += "\\t"; break;
                    default: o.push_back(c);
                }
            }
            return o + "\"";
        }
        case JValue::T::Arr: {
            std::string o = "[";
            for (size_t i = 0; i < v.arr.size(); ++i) {
                if (i) o += ",";
                o += dump_json(v.arr[i]);
            }
            return o + "]";
        }
        case JValue::T::Obj: {
            std::string o = "{";
            for (size_t i = 0; i < v.obj.size(); ++i) {
                if (i) o += ",";
                JValue k; k.type = JValue::T::Str; k.str = v.obj[i].first;
                o += dump_json(k) + ":" + dump_json(v.obj[i].second);
            }
            return o + "}";
        }
    }
    return "null";
}

// ------------------------------------------------------------ 帧内张量仓库
// 所有张量在帧启动时一次性预分配：运行期不新增张量条目，因此同一帧的多个后端
// 线程可以并发访问各自子图的张量而不需要全局锁（DAG 依赖已保证读写有序）。
class FrameState : public ITensorStore {
public:
    FrameState(const Model& m, std::shared_ptr<std::vector<float>> weights,
               const std::vector<std::vector<float>>& inputs, bool compute)
        : m_(m), weights_(std::move(weights)), compute_(compute) {
        for (const auto& d : m_.tensors) {
            TensorBuf& t = tensors_[d.id];
            t.desc = d;
            if (d.offset >= 0 && (d.kind == "weight" || d.kind == "bias" || d.kind == "constant")) {
                t.storage = weights_;
                t.offset = (size_t)(d.offset / 4);   // 原型统一 f32
                t.ready = true;
            } else {
                t.storage = std::make_shared<std::vector<float>>(d.numel(), 0.0f);
                t.offset = 0;
                t.ready = false;
            }
            t.owner = Backend::CPU;
        }
        for (size_t i = 0; i < m_.inputs.size() && i < inputs.size(); ++i) {
            TensorBuf& t = get(m_.inputs[i]);
            const auto& src = inputs[i];
            std::copy(src.begin(), src.begin() + std::min(src.size(), t.numel()), t.data());
            t.ready = true;
        }
    }

    bool exists(const std::string& id) const override { return tensors_.count(id) > 0; }

    TensorBuf& get(const std::string& id) override {
        auto it = tensors_.find(id);
        if (it == tensors_.end())
            throw std::runtime_error("FrameState::get unknown tensor: " + id);
        return it->second;
    }

    TensorBuf& alloc(const std::string& id, const TensorDesc&) override {
        return get(id);   // 已预分配
    }

    const float* weights() const override { return weights_ ? weights_->data() : nullptr; }
    size_t weights_size() const override { return weights_ ? weights_->size() : 0; }
    bool compute_enabled() const override { return compute_; }

    void delay(double virtual_us) override {
        if (virtual_us <= 0) return;
        std::this_thread::sleep_for(std::chrono::duration<double, std::micro>(virtual_us));
    }

private:
    const Model& m_;
    std::shared_ptr<std::vector<float>> weights_;
    bool compute_ = true;
    std::unordered_map<std::string, TensorBuf> tensors_;
};

// ------------------------------------------------------------ 调度任务
struct Task {
    enum class Kind { Compute, Transfer } kind = Kind::Compute;
    int frame = -1;
    int part = -1;      // Compute
    int src = -1, dst = -1;  // Transfer
    std::vector<int> edges;
};

struct FrameCtx {
    std::unique_ptr<FrameState> store;
    std::vector<int> indeg;
    int remaining = 0;
};

}  // namespace

// ---------------------------------------------------------------- Engine
struct Engine::Impl {
    std::string dir;
    std::string manifest_path;
    Model model;
    std::shared_ptr<std::vector<float>> weights;
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> outputs;
    std::array<std::unique_ptr<IBackend>, 3> backends;
};

Engine::Engine() : impl_(std::make_unique<Impl>()) {}
Engine::~Engine() = default;

const Model& Engine::model() const { return impl_->model; }
const std::string& Engine::dir() const { return impl_->dir; }

std::unique_ptr<Engine> Engine::load(const std::string& manifest_path) {
    auto e = std::unique_ptr<Engine>(new Engine());
    auto& I = *e->impl_;
    I.manifest_path = manifest_path;
    I.dir = dirname_of(manifest_path);
    I.model = parse_model(json_load_file(manifest_path));

    const std::string wp = I.dir + "/" + I.model.weights_file;
    I.weights = std::make_shared<std::vector<float>>(
        std::max<size_t>(1, I.model.weights_bytes / 4), 0.0f);
    std::ifstream f(wp, std::ios::binary);
    if (f) {
        f.read(reinterpret_cast<char*>(I.weights->data()),
               (std::streamsize)(I.model.weights_bytes));
    } else {
        throw std::runtime_error("cannot open weights file: " + wp);
    }
    I.inputs.resize(I.model.inputs.size());

    for (int i = 0; i < 3; ++i) {
        Backend b = static_cast<Backend>(i);
        I.backends[i] = create_reference_backend(b, 1.0, &I.model);
    }
    return e;
}

bool Engine::set_input(int idx, const float* data, size_t n) {
    if (idx < 0 || idx >= (int)impl_->inputs.size()) return false;
    auto& v = impl_->inputs[idx];
    v.assign(data, data + n);
    return true;
}

const float* Engine::output(int idx, size_t* n) const {
    if (idx < 0 || idx >= (int)impl_->outputs.size()) { if (n) *n = 0; return nullptr; }
    if (n) *n = impl_->outputs[idx].size();
    return impl_->outputs[idx].data();
}

RunReport Engine::run(const Options& opt) {
    RunReport rep;
    auto& I = *impl_;
    Model& m = I.model;
    const int P = (int)m.parts.size();
    const int F = std::max(1, opt.frames);
    const double scale = opt.time_scale > 0 ? opt.time_scale : 1.0;
    if (P == 0) { rep.error = "manifest has no parts"; return rep; }

    // 后端使用参考实现，但按本次 time_scale 重新创建以注入正确延迟
    for (int i = 0; i < 3; ++i)
        I.backends[i] = create_reference_backend(static_cast<Backend>(i), scale, &m);

    // 子图商图依赖
    std::vector<std::set<int>> succ(P);
    std::map<std::pair<int, int>, std::vector<int>> edge_map;
    for (size_t ei = 0; ei < m.edges.size(); ++ei) {
        const auto& e = m.edges[ei];
        if (e.src == e.dst || e.src < 0 || e.dst < 0) continue;
        edge_map[{e.src, e.dst}].push_back((int)ei);
        succ[e.src].insert(e.dst);
    }
    std::vector<int> indeg0(P, 0);
    for (int p = 0; p < P; ++p) for (int d : succ[p]) ++indeg0[d];

    // 每帧独立的张量仓库（避免跨帧写冲突）
    std::vector<FrameCtx> frames(F);

    // 每后端一条队列：锁只保护各自的 deque，热点执行路径无全局竞争。
    struct Q { std::mutex m; std::condition_variable cv; std::deque<Task> d; bool stop = false; };
    Q queues[4];                       // 0..2 = cpu/gpu/npu, 3 = DMA/transfer
    std::atomic<int> finished{0};
    std::mutex meta_mtx;               // 保护帧元数据 / 计数 / 时间线 / 错误
    std::vector<std::string> errors;
    int next_frame = 0;
    const int total = F * P;
    auto t0 = Clock::now();

    auto qidx = [](Backend b) { int k = backend_index(b); return k < 0 ? 0 : k; };

    auto push_task = [&](int qi, Task t) {
        { std::lock_guard<std::mutex> lk(queues[qi].m); queues[qi].d.push_back(std::move(t)); }
        queues[qi].cv.notify_one();
    };
    auto stop_all = [&]() {
        for (int k = 0; k < 4; ++k) {
            std::lock_guard<std::mutex> lk(queues[k].m);
            queues[k].stop = true;
            queues[k].cv.notify_all();
        }
    };

    // 启动一帧：预分配张量并把它所有 root part 投入各自后端队列
    auto launch_frame = [&](int f) {
        frames[f].store = std::make_unique<FrameState>(m, I.weights, I.inputs, !opt.skip_compute);
        frames[f].indeg = indeg0;
        frames[f].remaining = P;
        for (int p = 0; p < P; ++p)
            if (indeg0[p] == 0)
                push_task(qidx(m.parts[p].backend),
                          Task{Task::Kind::Compute, f, p, -1, -1, {}});
    };

    // 一条 (src->dst) 边就绪：若跨后端则走 transfer 队列，否则直接唤醒 dst
    auto edge_done = [&](int f, int dst) {
        bool ready;
        {
            std::lock_guard<std::mutex> lk(meta_mtx);
            ready = (--frames[f].indeg[dst] == 0);
        }
        if (ready)
            push_task(qidx(m.parts[dst].backend),
                      Task{Task::Kind::Compute, f, dst, -1, -1, {}});
    };

    std::vector<std::thread> workers;
    for (int qi = 0; qi < 4; ++qi) {
        workers.emplace_back([&, qi]() {
            while (true) {
                Task t;
                {
                    std::unique_lock<std::mutex> lk(queues[qi].m);
                    queues[qi].cv.wait(lk, [&] { return queues[qi].stop || !queues[qi].d.empty(); });
                    if (queues[qi].d.empty()) {
                        if (queues[qi].stop) break;
                        continue;
                    }
                    t = std::move(queues[qi].d.front());
                    queues[qi].d.pop_front();
                }
                const double s = now_us(t0);
                std::string err;
                bool ok = true;
                std::string bname;
                double virtual_us = 0.0;
                if (t.kind == Task::Kind::Compute) {
                    const PartDesc& p = m.parts[t.part];
                    bname = backend_name(p.backend);
                    virtual_us = p.est_us;
                    ok = I.backends[qidx(p.backend)]->execute(p, m, *frames[t.frame].store, &err);
                } else {
                    bname = "dma";
                    double cost = 0.0;
                    for (int ei : t.edges) cost = std::max(cost, m.edges[ei].cost_us);
                    virtual_us = cost;
                    if (opt.simulate_transfer) frames[t.frame].store->delay(cost / scale);
                    for (int ei : t.edges) {
                        const auto& e = m.edges[ei];
                        if (frames[t.frame].store->exists(e.tensor))
                            frames[t.frame].store->get(e.tensor).owner = e.dst_backend;
                    }
                }
                const double e_us = now_us(t0);
                {
                    std::lock_guard<std::mutex> lk(meta_mtx);
                    PartTiming pt;
                    pt.frame = t.frame;
                    pt.part = t.kind == Task::Kind::Compute ? t.part : -1;
                    pt.backend = bname;
                    pt.start_us = s;
                    pt.end_us = e_us;
                    pt.virtual_us = virtual_us;
                    pt.kind = t.kind == Task::Kind::Compute ? "compute" : "transfer";
                    rep.timeline.push_back(std::move(pt));
                    if (!ok) {
                        errors.push_back(err.empty() ? "backend execute failed" : err);
                        stop_all();
                    }
                }

                if (t.kind == Task::Kind::Compute) {
                    for (int d : succ[t.part]) {
                        auto it = edge_map.find({t.part, d});
                        bool need_transfer = false;
                        if (it != edge_map.end())
                            for (int ei : it->second)
                                need_transfer |= (opt.simulate_transfer && m.edges[ei].cross);
                        if (need_transfer) {
                            Task tt; tt.kind = Task::Kind::Transfer;
                            tt.frame = t.frame; tt.src = t.part; tt.dst = d;
                            tt.edges = it->second;
                            push_task(3, std::move(tt));
                        } else {
                            edge_done(t.frame, d);
                        }
                    }
                    int rem;
                    {
                        std::lock_guard<std::mutex> lk(meta_mtx);
                        rem = --frames[t.frame].remaining;
                    }
                    int done = ++finished;
                    if (rem == 0 && next_frame < F) {
                        int nf = next_frame++;
                        launch_frame(nf);
                    }
                    if (done >= total || !errors.empty())
                        stop_all();
                } else {
                    edge_done(t.frame, t.dst);
                }
            }
        });
    }

    {
        const int depth = std::max(1, opt.pipeline_depth);
        for (int f = 0; f < std::min(depth, F); ++f) {
            int nf = next_frame++;
            launch_frame(nf);
        }
    }
    for (auto& w : workers) w.join();

    // ---- 汇总 ----
    if (!rep.timeline.empty()) {
        double mn = rep.timeline[0].start_us, mx = rep.timeline[0].end_us;
        for (const auto& p : rep.timeline) {
            mn = std::min(mn, p.start_us);
            mx = std::max(mx, p.end_us);
            if (p.kind != "compute") continue;
            int k = p.backend == "cpu" ? 0 : (p.backend == "gpu" ? 1 : (p.backend == "npu" ? 2 : -1));
            if (k >= 0) rep.backend_busy_us[k] += p.end_us - p.start_us;
        }
        rep.wall_us = mx - mn;
        for (size_t i = 0; i < rep.timeline.size(); ++i)
            for (size_t j = i + 1; j < rep.timeline.size(); ++j) {
                const auto& a = rep.timeline[i];
                const auto& b = rep.timeline[j];
                if (a.backend == b.backend || a.kind != "compute" || b.kind != "compute") continue;
                if (a.start_us < b.end_us && b.start_us < a.end_us) ++rep.overlap_pairs;
            }
    }
    rep.virtual_wall_us = rep.wall_us * scale;
    try {
        rep.predicted_us = m.plan.at("simulation").at("makespan_us").as_num(0.0);
    } catch (...) { rep.predicted_us = 0.0; }
    rep.completed_parts = (size_t)finished.load();
    rep.ok = errors.empty() && finished.load() == total;
    if (!errors.empty()) rep.error = errors.front();
    else if (finished.load() != total) rep.error = "scheduler did not finish all parts";

    // 缓存第 0 帧输出
    I.outputs.clear();
    if (!frames.empty() && frames[0].store) {
        for (const auto& oid : m.outputs) {
            if (!frames[0].store->exists(oid)) { I.outputs.emplace_back(); continue; }
            TensorBuf& t = frames[0].store->get(oid);
            I.outputs.emplace_back(t.data(), t.data() + t.numel());
        }
    }
    if (!opt.timeline_path.empty()) {
        std::ofstream f(opt.timeline_path, std::ios::trunc);
        if (f) f << dump_json(rep.to_json()) << "\n";
    }
    return rep;
}

std::string Engine::summary() const {
    const auto& m = impl_->model;
    std::ostringstream os;
    os << "model=" << m.name << " nodes=" << m.nodes.size()
       << " parts=" << m.parts.size() << " tensors=" << m.tensors.size();
    for (const auto& p : m.parts)
        os << "\n  part " << p.id << " [" << backend_name(p.backend) << "] "
           << p.nodes.size() << " ops, est=" << std::fixed << std::setprecision(1)
           << p.est_us << "us";
    return os.str();
}

// ---------------------------------------------------------------- 报告
std::string RunReport::gantt() const {
    if (timeline.empty()) return "(empty timeline)";
    std::vector<const PartTiming*> v;
    for (const auto& t : timeline) v.push_back(&t);
    std::sort(v.begin(), v.end(), [](const PartTiming* a, const PartTiming* b) {
        return a->start_us < b->start_us;
    });
    double mn = v.front()->start_us, mx = mn;
    for (auto* p : v) mx = std::max(mx, p->end_us);
    const double span = std::max(1.0, mx - mn);
    const int W = 56;
    std::ostringstream os;
    os << std::fixed << std::setprecision(1);
    os << "frame part backend  " << std::left << std::setw(W) << "timeline" << "  us\n";
    for (auto* p : v) {
        int a = (int)((p->start_us - mn) / span * W);
        int b = (int)((p->end_us - mn) / span * W);
        a = std::max(0, std::min(W - 1, a));
        b = std::max(a + 1, std::min(W, b));
        std::string bar(W, ' ');
        for (int i = a; i < b; ++i)
            bar[i] = p->backend == "npu" ? 'N' : (p->backend == "gpu" ? 'G' :
                     (p->backend == "cpu" ? 'C' : '='));
        os << std::setw(5) << p->frame << " " << std::setw(4) << p->part << " "
           << std::left << std::setw(7) << p->backend << " " << bar
           << "  " << std::setw(8) << (p->end_us - p->start_us) << "\n";
    }
    os << "legend: N=NPU G=GPU C=CPU ==transfer\n";
    return os.str();
}

JValue RunReport::to_json() const {
    JValue v; v.type = JValue::T::Obj;
    auto put = [&](const char* k, JValue x) { v.obj.emplace_back(k, std::move(x)); };
    auto num = [](double x) { JValue j; j.type = JValue::T::Num; j.num = x; return j; };
    auto str = [](const std::string& s) { JValue j; j.type = JValue::T::Str; j.str = s; return j; };
    auto boolean = [](bool b) { JValue j; j.type = JValue::T::Bool; j.b = b; return j; };
    put("ok", boolean(ok));
    put("error", str(error));
    put("wall_us", num(wall_us));
    put("virtual_wall_us", num(virtual_wall_us));
    put("predicted_us", num(predicted_us));
    put("max_abs_err", num(max_abs_err));
    put("numerics_ok", boolean(numerics_ok));
    put("completed_parts", num((double)completed_parts));
    put("overlap_pairs", num((double)overlap_pairs));
    JValue busy; busy.type = JValue::T::Obj;
    busy.obj.emplace_back("cpu", num(backend_busy_us[0]));
    busy.obj.emplace_back("gpu", num(backend_busy_us[1]));
    busy.obj.emplace_back("npu", num(backend_busy_us[2]));
    put("backend_busy_us", std::move(busy));
    JValue tl; tl.type = JValue::T::Arr;
    for (const auto& p : timeline) {
        JValue e; e.type = JValue::T::Obj;
        e.obj.emplace_back("frame", num(p.frame));
        e.obj.emplace_back("part", num(p.part));
        e.obj.emplace_back("backend", str(p.backend));
        e.obj.emplace_back("kind", str(p.kind));
        e.obj.emplace_back("start_us", num(p.start_us));
        e.obj.emplace_back("end_us", num(p.end_us));
        e.obj.emplace_back("virtual_us", num(p.virtual_us));
        tl.arr.push_back(std::move(e));
    }
    put("timeline", std::move(tl));
    return v;
}

std::string RunReport::brief() const {
    std::ostringstream os;
    os << std::fixed << std::setprecision(2);
    os << "ok=" << (ok ? "true" : "false")
       << " wall=" << wall_us << "us"
       << " predicted=" << predicted_us << "us"
       << " parts=" << completed_parts
       << " overlap_pairs=" << overlap_pairs << "\n";
    os << "backend busy(us): cpu=" << backend_busy_us[0]
       << " gpu=" << backend_busy_us[1] << " npu=" << backend_busy_us[2] << "\n";
    os << "numerics_ok=" << (numerics_ok ? "true" : "false")
       << " max_abs_err=" << max_abs_err << "\n";
    if (!error.empty()) os << "error: " << error << "\n";
    return os.str();
}

// ---------------------------------------------------------------- 二进制 IO
bool read_bin_f32(const std::string& path, std::vector<float>& out) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return false;
    std::streamsize n = f.tellg();
    f.seekg(0);
    out.resize((size_t)n / sizeof(float));
    if (n > 0) f.read(reinterpret_cast<char*>(out.data()), n);
    return true;
}
bool write_bin_f32(const std::string& path, const float* data, size_t n) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(data), (std::streamsize)(n * sizeof(float)));
    return true;
}

}  // namespace hx
