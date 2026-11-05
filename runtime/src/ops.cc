// HX IR 算子的 C++ 参考实现。
// 与 hx/verify.py::HXInterpreter 语义一致，用于端到端数值校验。
// 布局统一 NCHW / fp32；生产环境由各后端按自己的 layout 需求转换。
#include "ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace hx {
namespace {

using Shape = std::vector<int64_t>;

size_t prod(const Shape& s, size_t a = 0, size_t b = 0) {
    if (b == 0) b = s.size();
    size_t v = 1;
    for (size_t i = a; i < b && i < s.size(); ++i)
        v *= static_cast<size_t>(s[i] < 0 ? 0 : s[i]);
    return v;
}

int64_t ai(const JValue& attrs, const char* k, int64_t def) {
    return attrs.has(k) ? attrs.at(k).as_int(def) : def;
}
double ad(const JValue& attrs, const char* k, double def) {
    return attrs.has(k) ? attrs.at(k).as_num(def) : def;
}
Shape avec2(const JValue& attrs, const char* k, int64_t def) {
    if (!attrs.has(k)) return Shape{def, def};
    const JValue& v = attrs.at(k);
    if (v.type == JValue::T::Num) return Shape{v.as_int(def), v.as_int(def)};
    Shape r = v.as_int_vec();
    if (r.empty()) return Shape{def, def};
    if (r.size() == 1) return Shape{r[0], r[0]};
    return Shape{r[0], r[1]};
}
Shape avecN(const JValue& attrs, const char* k, const Shape& def) {
    if (!attrs.has(k)) return def;
    Shape r = attrs.at(k).as_int_vec();
    return r.empty() ? def : r;
}
long norm_dim(long d, size_t rank) {
    if (d < 0) d += static_cast<long>(rank);
    return std::max(0L, std::min(d, static_cast<long>(rank) - 1));
}

const TensorDesc& TD(const Model& m, const std::string& id) {
    const TensorDesc* t = m.tensor(id);
    if (!t) throw std::runtime_error("unknown tensor id in manifest: " + id);
    return *t;
}
TensorBuf& OUT(ITensorStore& s, const Model& m, const std::string& id) {
    return s.alloc(id, TD(m, id));
}

float bat(const TensorBuf& b, size_t i, const Shape& xs) {
    const float* p = b.data();
    size_t n = b.numel();
    if (n == 0) return 0.0f;
    if (n == 1) return p[0];
    if (n == prod(xs)) return p[i];
    if (xs.size() >= 2 && n == static_cast<size_t>(xs[1])) {
        size_t inner = prod(xs, 2);
        return p[(i / (inner ? inner : 1)) % n];
    }
    if (!xs.empty() && n == static_cast<size_t>(xs.back())) return p[i % n];
    return p[i % n];
}

void unary(TensorBuf& x, TensorBuf& y, const std::function<float(float)>& f) {
    float* d = y.data();
    const float* s = x.data();
    for (size_t i = 0, n = y.numel(); i < n; ++i) d[i] = f(s[i]);
    y.ready = true;
}
void binary(TensorBuf& a, TensorBuf& b, TensorBuf& y,
            const std::function<float(float, float)>& f) {
    float* d = y.data();
    const float* pa = a.data();
    for (size_t i = 0, n = y.numel(); i < n; ++i) d[i] = f(pa[i], bat(b, i, y.desc.shape));
    y.ready = true;
}

// ---------------------------------------------------------------- conv2d
void conv2d(const NodeDesc& n, const Model& m, ITensorStore& s) {
    TensorBuf& x = s.get(n.inputs[0]);
    TensorBuf& w = s.get(n.inputs[1]);
    TensorBuf* bp = (n.inputs.size() > 2) ? &s.get(n.inputs[2]) : nullptr;
    TensorBuf& y = OUT(s, m, n.outputs[0]);
    const Shape& xs = x.desc.shape;
    const Shape& ws = w.desc.shape;
    const Shape& ys = y.desc.shape;
    if (xs.size() != 4 || ws.size() != 4 || ys.size() != 4)
        throw std::runtime_error("conv2d expects NCHW / OIHW, got invalid rank");
    const int N = (int)xs[0], H = (int)xs[2], W = (int)xs[3];
    const int O = (int)ws[0], Cg = (int)ws[1], kH = (int)ws[2], kW = (int)ws[3];
    const int OH = (int)ys[2], OW = (int)ys[3];
    int groups = std::max(1, (int)ai(n.attrs, "groups", 1));
    if (O % groups) throw std::runtime_error("conv2d: O not divisible by groups");
    Shape st = avec2(n.attrs, "stride", 1);
    Shape pd = avec2(n.attrs, "padding", 0);
    Shape dl = avec2(n.attrs, "dilation", 1);
    const int OPG = O / groups;
    const float* px = x.data();
    const float* pw = w.data();
    const float* pb = bp ? bp->data() : nullptr;
    float* py = y.data();
    // 以 (N, O, OH) 为并行粒度：比 (N, O) 多 OH 倍任务，多核利用率更高。
    const int64_t jobs = static_cast<int64_t>(N) * O * OH;

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t job = 0; job < jobs; ++job) {
        const int nn = static_cast<int>(job / (O * OH));
        const int oo = static_cast<int>((job / OH) % O);
        const int oh = static_cast<int>(job % OH);
        const int g = oo / OPG;
        {
            for (int ow = 0; ow < OW; ++ow) {
                float acc = pb ? pb[oo] : 0.0f;
                for (int cg = 0; cg < Cg; ++cg) {
                    const int ic = g * Cg + cg;
                    for (int kh = 0; kh < kH; ++kh) {
                        const int ih = oh * (int)st[0] - (int)pd[0] + kh * (int)dl[0];
                        if (ih < 0 || ih >= H) continue;
                        for (int kw = 0; kw < kW; ++kw) {
                            const int iw = ow * (int)st[1] - (int)pd[1] + kw * (int)dl[1];
                            if (iw < 0 || iw >= W) continue;
                            const size_t xi = ((size_t)nn * xs[1] + ic) * H * W
                                            + (size_t)ih * W + iw;
                            const size_t wi = (((size_t)oo * Cg + cg) * kH + kh) * kW + kw;
                            acc += px[xi] * pw[wi];
                        }
                    }
                }
                py[(((size_t)nn * O + oo) * OH + oh) * OW + ow] = acc;
            }
        }
    }
    y.ready = true;
}

// ---------------------------------------------------------------- linear
void linear(const NodeDesc& n, const Model& m, ITensorStore& s) {
    TensorBuf& x = s.get(n.inputs[0]);
    TensorBuf& w = s.get(n.inputs[1]);
    TensorBuf* bp = (n.inputs.size() > 2) ? &s.get(n.inputs[2]) : nullptr;
    TensorBuf& y = OUT(s, m, n.outputs[0]);
    const Shape& xs = x.desc.shape;
    const Shape& ws = w.desc.shape;
    if (xs.empty() || ws.size() != 2) throw std::runtime_error("linear: bad shapes");
    const int K = (int)xs.back();
    const size_t M = prod(xs) / (size_t)K;
    const int Nout = (int)(ws[1] == K ? ws[0] : ws[1]);
    const bool transposed_w = !(ws[1] == K && ws[0] == Nout);
    const float* px = x.data();
    const float* pw = w.data();
    const float* pb = bp ? bp->data() : nullptr;
    float* py = y.data();
    for (size_t mm = 0; mm < M; ++mm) {
        for (int oo = 0; oo < Nout; ++oo) {
            float acc = pb ? pb[oo] : 0.0f;
            for (int kk = 0; kk < K; ++kk) {
                const float wv = transposed_w ? pw[(size_t)kk * Nout + oo]
                                              : pw[(size_t)oo * K + kk];
                acc += px[mm * K + kk] * wv;
            }
            py[mm * Nout + oo] = acc;
        }
    }
    y.ready = true;
}

// ---------------------------------------------------------------- pooling
void pool(const NodeDesc& n, const Model& m, ITensorStore& s, bool max_mode, bool global) {
    TensorBuf& x = s.get(n.inputs[0]);
    TensorBuf& y = OUT(s, m, n.outputs[0]);
    const Shape& xs = x.desc.shape;
    if (xs.size() != 4) { unary(x, y, [](float v) { return v; }); return; }
    const int N = (int)xs[0], C = (int)xs[1], H = (int)xs[2], W = (int)xs[3];
    const int OH = (int)y.desc.shape[2], OW = (int)y.desc.shape[3];
    Shape ker = global ? Shape{H, W} : avec2(n.attrs, "kernel", 2);
    Shape st = global ? Shape{H, W} : avec2(n.attrs, "stride", ker[0]);
    Shape pd = global ? Shape{0, 0} : avec2(n.attrs, "padding", 0);
    const float* px = x.data();
    float* py = y.data();
    for (int nn = 0; nn < N; ++nn)
        for (int cc = 0; cc < C; ++cc)
            for (int oh = 0; oh < OH; ++oh)
                for (int ow = 0; ow < OW; ++ow) {
                    float v = max_mode ? -std::numeric_limits<float>::max() : 0.0f;
                    int cnt = 0;
                    for (int kh = 0; kh < (int)ker[0]; ++kh)
                        for (int kw = 0; kw < (int)ker[1]; ++kw) {
                            const int ih = global ? kh : oh * (int)st[0] - (int)pd[0] + kh;
                            const int iw = global ? kw : ow * (int)st[1] - (int)pd[1] + kw;
                            if (ih < 0 || ih >= H || iw < 0 || iw >= W) continue;
                            const float t = px[(((size_t)nn * C + cc) * H + ih) * W + iw];
                            if (max_mode) { v = std::max(v, t); ++cnt; }
                            else { v += t; ++cnt; }
                        }
                    if (!max_mode && cnt > 0) v /= static_cast<float>(cnt);
                    py[(((size_t)nn * C + cc) * OH + oh) * OW + ow] =
                        (cnt == 0) ? 0.0f : v;
                }
    y.ready = true;
}

// ---------------------------------------------------------------- concat
void concat(const NodeDesc& n, const Model& m, ITensorStore& s) {
    TensorBuf& y = OUT(s, m, n.outputs[0]);
    const Shape& ys = y.desc.shape;
    long dim = norm_dim((long)ai(n.attrs, "dim", 1), ys.size());
    const size_t outer = prod(ys, 0, (size_t)dim);
    const size_t after = prod(ys, (size_t)dim + 1);
    float* py = y.data();
    size_t filled = 0;
    for (const auto& id : n.inputs) {
        TensorBuf& x = s.get(id);
        const size_t chunk = (dim < (long)x.desc.shape.size())
            ? (size_t)x.desc.shape[dim] * after : after;
        const float* px = x.data();
        for (size_t o = 0; o < outer; ++o)
            std::memcpy(py + o * (y.numel() / (outer ? outer : 1)) + filled,
                        px + o * chunk, chunk * sizeof(float));
        filled += chunk;
    }
    y.ready = true;
}

// ---------------------------------------------------------------- permute
void permute(const NodeDesc& n, const Model& m, ITensorStore& s, Shape dims) {
    TensorBuf& x = s.get(n.inputs[0]);
    TensorBuf& y = OUT(s, m, n.outputs[0]);
    const Shape& xs = x.desc.shape;
    const Shape& ys = y.desc.shape;
    if (dims.empty()) {
        dims.resize(xs.size());
        for (size_t i = 0; i < xs.size(); ++i) dims[i] = (int64_t)i;
        long d0 = norm_dim((long)ai(n.attrs, "dim0", 0), xs.size());
        long d1 = norm_dim((long)ai(n.attrs, "dim1", 1), xs.size());
        std::swap(dims[d0], dims[d1]);
    }
    for (auto& d : dims) d = norm_dim((long)d, xs.size());
    Shape istr(xs.size(), 1), ostr(ys.size(), 1);
    for (long i = (long)xs.size() - 2; i >= 0; --i) istr[i] = istr[i + 1] * xs[i + 1];
    for (long i = (long)ys.size() - 2; i >= 0; --i) ostr[i] = ostr[i + 1] * ys[i + 1];
    const float* px = x.data();
    float* py = y.data();
    for (size_t o = 0, total = y.numel(); o < total; ++o) {
        size_t rem = o, src = 0;
        for (size_t d = 0; d < ys.size(); ++d) {
            const int64_t coord = static_cast<int64_t>(rem / ostr[d]);
            rem %= ostr[d];
            src += static_cast<size_t>(coord) * istr[dims[d]];
        }
        py[o] = px[src];
    }
    y.ready = true;
}

// ---------------------------------------------------------------- softmax / norm
void softmax(const NodeDesc& n, const Model& m, ITensorStore& s) {
    TensorBuf& x = s.get(n.inputs[0]);
    TensorBuf& y = OUT(s, m, n.outputs[0]);
    const Shape& xs = x.desc.shape;
    long dim = norm_dim((long)ai(n.attrs, "dim", -1), xs.size());
    const size_t outer = prod(xs, 0, (size_t)dim);
    const size_t ds = (size_t)xs[dim];
    const size_t inner = prod(xs, (size_t)dim + 1);
    const float* px = x.data();
    float* py = y.data();
    for (size_t o = 0; o < outer; ++o)
        for (size_t i = 0; i < inner; ++i) {
            float mx = -std::numeric_limits<float>::max();
            for (size_t d = 0; d < ds; ++d)
                mx = std::max(mx, px[(o * ds + d) * inner + i]);
            float sum = 0.0f;
            for (size_t d = 0; d < ds; ++d) {
                float e = std::exp(px[(o * ds + d) * inner + i] - mx);
                py[(o * ds + d) * inner + i] = e;
                sum += e;
            }
            for (size_t d = 0; d < ds; ++d) py[(o * ds + d) * inner + i] /= sum;
        }
    y.ready = true;
}

void layer_norm(const NodeDesc& n, const Model& m, ITensorStore& s) {
    TensorBuf& x = s.get(n.inputs[0]);
    TensorBuf& y = OUT(s, m, n.outputs[0]);
    const Shape& xs = x.desc.shape;
    const size_t D = xs.empty() ? 1 : (size_t)xs.back();
    const size_t rows = x.numel() / D;
    const float eps = (float)ad(n.attrs, "eps", 1e-5);
    const float* px = x.data();
    float* py = y.data();
    for (size_t r = 0; r < rows; ++r) {
        float mean = 0.0f, var = 0.0f;
        for (size_t d = 0; d < D; ++d) mean += px[r * D + d];
        mean /= (float)D;
        for (size_t d = 0; d < D; ++d) { float t = px[r * D + d] - mean; var += t * t; }
        var /= (float)D;
        const float inv = 1.0f / std::sqrt(var + eps);
        for (size_t d = 0; d < D; ++d) py[r * D + d] = (px[r * D + d] - mean) * inv;
    }
    y.ready = true;
}

void reduce(const NodeDesc& n, const Model& m, ITensorStore& s, bool is_max) {
    TensorBuf& x = s.get(n.inputs[0]);
    TensorBuf& y = OUT(s, m, n.outputs[0]);
    const Shape& xs = x.desc.shape;
    Shape dims;
    if (n.attrs.has("dim")) {
        const JValue& v = n.attrs.at("dim");
        if (v.type == JValue::T::Num) dims = {v.as_int(-1)};
        else dims = v.as_int_vec();
    }
    if (dims.empty()) dims = {(int64_t)xs.size() - 1};
    // 本原型支持 keepdim 的单维规约；多维规约按最后一维处理
    long dim = norm_dim((long)dims[0], xs.size());
    const size_t outer = prod(xs, 0, (size_t)dim);
    const size_t ds = (size_t)xs[dim];
    const size_t inner = prod(xs, (size_t)dim + 1);
    const float* px = x.data();
    float* py = y.data();
    for (size_t o = 0; o < outer; ++o)
        for (size_t i = 0; i < inner; ++i) {
            float v = is_max ? -std::numeric_limits<float>::max() : 0.0f;
            for (size_t d = 0; d < ds; ++d) {
                float t = px[(o * ds + d) * inner + i];
                v = is_max ? std::max(v, t) : v + t;
            }
            py[o * inner + i] = is_max ? v : v / (float)ds;
        }
    y.ready = true;
}

// ---------------------------------------------------------------- view / misc
void view(const NodeDesc& n, const Model& m, ITensorStore& s) {
    TensorBuf& x = s.get(n.inputs[0]);
    TensorBuf& y = OUT(s, m, n.outputs[0]);
    if (x.desc.slot >= 0 && y.desc.slot == x.desc.slot) {
        // 编译期已把 view 输出与其根张量规划到同一共享槽，零拷贝别名成立，
        // 且不会破坏 slot 复用规划。
        y.storage = x.storage;
        y.offset = x.offset;
    } else {
        // 输出为常驻/独占缓冲（或旧版 manifest 无 slot 规划）：必须真实拷贝，
        // 否则把常驻缓冲指向可复用槽会被后续写入覆盖。
        const size_t cnt = std::min(x.numel(), y.numel());
        std::copy(x.data(), x.data() + cnt, y.data());
    }
    y.owner = x.owner;
    y.ready = true;
}

void batch_norm(const NodeDesc& n, const Model& m, ITensorStore& s) {
    TensorBuf& x = s.get(n.inputs[0]);
    TensorBuf& gamma = s.get(n.inputs[1]);
    TensorBuf& beta = s.get(n.inputs[2]);
    TensorBuf& mean = s.get(n.inputs[3]);
    TensorBuf& var = s.get(n.inputs[4]);
    TensorBuf& y = OUT(s, m, n.outputs[0]);
    const Shape& xs = x.desc.shape;
    const size_t C = xs.size() >= 2 ? (size_t)xs[1] : 1;
    const size_t inner = prod(xs, 2);
    const float eps = (float)ad(n.attrs, "eps", 1e-5);
    const float* px = x.data();
    float* py = y.data();
    for (size_t nn = 0; nn < (xs.empty() ? 1 : (size_t)xs[0]); ++nn)
        for (size_t c = 0; c < C; ++c) {
            const float s_inv = 1.0f / std::sqrt(var.data()[c] + eps);
            const float b = beta.data()[c] - mean.data()[c] * gamma.data()[c] * s_inv;
            const float a = gamma.data()[c] * s_inv;
            for (size_t i = 0; i < inner; ++i)
                py[(nn * C + c) * inner + i] = a * px[(nn * C + c) * inner + i] + b;
        }
    y.ready = true;
}

void upsample(const NodeDesc& n, const Model& m, ITensorStore& s) {
    TensorBuf& x = s.get(n.inputs[0]);
    TensorBuf& y = OUT(s, m, n.outputs[0]);
    const Shape& xs = x.desc.shape;
    const Shape& ys = y.desc.shape;
    if (xs.size() != 4 || ys.size() != 4) { view(n, m, s); return; }
    const int N = (int)ys[0], C = (int)ys[1], OH = (int)ys[2], OW = (int)ys[3];
    const int IH = (int)xs[2], IW = (int)xs[3];
    const float* px = x.data();
    float* py = y.data();
    for (int nn = 0; nn < N; ++nn)
        for (int cc = 0; cc < C; ++cc)
            for (int oh = 0; oh < OH; ++oh)
                for (int ow = 0; ow < OW; ++ow) {
                    int ih = (int)((long)oh * IH / OH), iw = (int)((long)ow * IW / OW);
                    py[(((size_t)nn * C + cc) * OH + oh) * OW + ow] =
                        px[(((size_t)nn * C + cc) * IH + ih) * IW + iw];
                }
    y.ready = true;
}

void pad(const NodeDesc& n, const Model& m, ITensorStore& s) {
    TensorBuf& x = s.get(n.inputs[0]);
    TensorBuf& y = OUT(s, m, n.outputs[0]);
    Shape p = avecN(n.attrs, "pad", Shape{0, 0, 0, 0});
    while (p.size() < 4) p.insert(p.begin(), 0);
    const Shape& ys = y.desc.shape;
    const Shape& xs = x.desc.shape;
    if (ys.size() != 4 || xs.size() != 4) { view(n, m, s); return; }
    std::fill(y.data(), y.data() + y.numel(), 0.0f);
    const int N = (int)ys[0], C = (int)ys[1], IH = (int)xs[2], IW = (int)xs[3];
    const int OH = (int)ys[2], OW = (int)ys[3];
    const int pl = (int)p[0], pt = (int)p[2];
    for (int nn = 0; nn < N; ++nn)
        for (int cc = 0; cc < C; ++cc)
            for (int h = 0; h < IH; ++h)
                std::memcpy(y.data() + (((size_t)nn * C + cc) * OH + h + pt) * OW + pl,
                            x.data() + (((size_t)nn * C + cc) * IH + h) * IW,
                            (size_t)IW * sizeof(float));
    y.ready = true;
}

}  // namespace

bool run_node(const NodeDesc& n, const Model& m, ITensorStore& s, std::string* err) {
    try {
        const std::string& op = n.op;
        if (op == "conv2d" || op == "conv_transpose2d") { conv2d(n, m, s); return true; }
        if (op == "linear" || op == "matmul") { linear(n, m, s); return true; }
        if (op == "relu") { unary(s.get(n.inputs[0]), OUT(s, m, n.outputs[0]),
                                  [](float v) { return v > 0 ? v : 0.0f; }); return true; }
        if (op == "relu6") { unary(s.get(n.inputs[0]), OUT(s, m, n.outputs[0]),
                                   [](float v) { return std::min(std::max(v, 0.0f), 6.0f); }); return true; }
        if (op == "silu") { unary(s.get(n.inputs[0]), OUT(s, m, n.outputs[0]),
                                  [](float v) { return v / (1.0f + std::exp(-v)); }); return true; }
        if (op == "sigmoid") { unary(s.get(n.inputs[0]), OUT(s, m, n.outputs[0]),
                                     [](float v) { return 1.0f / (1.0f + std::exp(-v)); }); return true; }
        if (op == "tanh") { unary(s.get(n.inputs[0]), OUT(s, m, n.outputs[0]),
                                  [](float v) { return std::tanh(v); }); return true; }
        if (op == "gelu") { unary(s.get(n.inputs[0]), OUT(s, m, n.outputs[0]),
                                  [](float v) { return 0.5f * v * (1.0f + std::erf(v / 1.41421356f)); }); return true; }
        if (op == "sqrt" || op == "exp" || op == "log" || op == "neg" || op == "abs" || op == "clip") {
            TensorBuf& x = s.get(n.inputs[0]);
            TensorBuf& y = OUT(s, m, n.outputs[0]);
            const float lo = (float)ad(n.attrs, "min", -1), hi = (float)ad(n.attrs, "max", 1);
            std::function<float(float)> f;
            if (op == "sqrt") f = [](float v) { return std::sqrt(v); };
            else if (op == "exp") f = [](float v) { return std::exp(v); };
            else if (op == "log") f = [](float v) { return std::log(v); };
            else if (op == "neg") f = [](float v) { return -v; };
            else if (op == "abs") f = [](float v) { return std::abs(v); };
            else f = [lo, hi](float v) { return std::min(std::max(v, lo), hi); };
            unary(x, y, f);
            return true;
        }
        if (op == "add" || op == "mul" || op == "sub" || op == "div" || op == "pow") {
            TensorBuf& a = s.get(n.inputs[0]);
            TensorBuf* b = (n.inputs.size() > 1) ? &s.get(n.inputs[1]) : nullptr;
            TensorBuf& y = OUT(s, m, n.outputs[0]);
            const float p = (float)ad(n.attrs, "exponent", 2);
            std::function<float(float, float)> f;
            if (op == "add") f = [](float u, float v) { return u + v; };
            else if (op == "mul") f = [](float u, float v) { return u * v; };
            else if (op == "sub") f = [](float u, float v) { return u - v; };
            else if (op == "div") f = [](float u, float v) { return v == 0 ? 0.0f : u / v; };
            else f = [p](float u, float) { return std::pow(u, p); };
            if (b) binary(a, *b, y, f);
            else unary(a, y, [f, p](float u) { return f(u, p); });
            return true;
        }
        if (op == "global_avg_pool") { pool(n, m, s, false, true); return true; }
        if (op == "avg_pool2d") { pool(n, m, s, false, false); return true; }
        if (op == "max_pool2d") { pool(n, m, s, true, false); return true; }
        if (op == "concat") { concat(n, m, s); return true; }
        if (op == "reshape" || op == "flatten" || op == "identity") { view(n, m, s); return true; }
        if (op == "permute") {
            Shape d = avecN(n.attrs, "dims", Shape());
            permute(n, m, s, d);
            return true;
        }
        if (op == "transpose") { permute(n, m, s, Shape()); return true; }
        if (op == "softmax") { softmax(n, m, s); return true; }
        if (op == "layer_norm") { layer_norm(n, m, s); return true; }
        if (op == "batch_norm") { batch_norm(n, m, s); return true; }
        if (op == "mean") { reduce(n, m, s, false); return true; }
        if (op == "amax") { reduce(n, m, s, true); return true; }
        if (op == "upsample") { upsample(n, m, s); return true; }
        if (op == "pad") { pad(n, m, s); return true; }
        if (op == "where") {
            TensorBuf& c = s.get(n.inputs[0]);
            TensorBuf& a = s.get(n.inputs[1]);
            TensorBuf& b = s.get(n.inputs[2]);
            TensorBuf& y = OUT(s, m, n.outputs[0]);
            for (size_t i = 0; i < y.numel(); ++i) y.data()[i] = c.data()[i] != 0 ? a.data()[i] : b.data()[i];
            y.ready = true;
            return true;
        }
        if (err) *err = "unsupported IR op: " + op;
        return false;
    } catch (const std::exception& e) {
        if (err) *err = std::string("op '") + n.op + "' (" + n.id + ") failed: " + e.what();
        return false;
    }
}

}  // namespace hx
