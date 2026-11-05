// 轻量 JSON 解析器（header-only）。
// 只依赖标准库，避免生产 SDK 引入第三方依赖；支持 manifest 所需的全部类型。
#pragma once
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace hx {

struct JValue {
    enum class T { Null, Bool, Num, Str, Arr, Obj };
    T type = T::Null;
    bool b = false;
    double num = 0.0;
    std::string str;
    std::vector<JValue> arr;
    // 用 vector 而非 map：C++17 起 vector 允许元素类型不完整，map 不行。
    std::vector<std::pair<std::string, JValue>> obj;

    bool is_null() const { return type == T::Null; }
    bool has(const std::string& k) const {
        if (type != T::Obj) return false;
        for (const auto& kv : obj) if (kv.first == k) return true;
        return false;
    }
    const JValue& at(const std::string& k) const {
        if (type != T::Obj) throw std::runtime_error("JSON: at() on non-object, key=" + k);
        for (const auto& kv : obj) if (kv.first == k) return kv.second;
        throw std::runtime_error("JSON: missing key '" + k + "'");
    }
    const JValue& at_or(const std::string& k, const JValue& d) const {
        return has(k) ? at(k) : d;
    }
    const JValue& index(size_t i) const {
        if (type != T::Arr || i >= arr.size())
            throw std::runtime_error("JSON: index out of range");
        return arr[i];
    }
    size_t size() const {
        return type == T::Arr ? arr.size() : (type == T::Obj ? obj.size() : 0);
    }
    std::string as_str(const std::string& def = "") const {
        return type == T::Str ? str : def;
    }
    double as_num(double def = 0.0) const {
        return type == T::Num ? num : def;
    }
    int64_t as_int(int64_t def = 0) const {
        return type == T::Num ? static_cast<int64_t>(num) : def;
    }
    bool as_bool(bool def = false) const {
        return type == T::Bool ? b : def;
    }
    std::vector<std::string> as_str_vec() const {
        std::vector<std::string> out;
        if (type == T::Arr) for (const auto& v : arr) out.push_back(v.as_str());
        return out;
    }
    std::vector<int64_t> as_int_vec() const {
        std::vector<int64_t> out;
        if (type == T::Arr) for (const auto& v : arr) out.push_back(v.as_int());
        return out;
    }
};

class JParser {
public:
    explicit JParser(const std::string& s) : s_(s) {}

    JValue parse() {
        skip();
        JValue v = value();
        skip();
        if (p_ != s_.size()) fail("trailing content");
        return v;
    }

private:
    const std::string& s_;
    size_t p_ = 0;

    [[noreturn]] void fail(const std::string& msg) const {
        throw std::runtime_error("JSON parse error at byte " + std::to_string(p_) + ": " + msg);
    }
    void skip() {
        while (p_ < s_.size() && std::isspace(static_cast<unsigned char>(s_[p_]))) ++p_;
    }
    char peek() const { return p_ < s_.size() ? s_[p_] : '\0'; }

    JValue value() {
        skip();
        if (p_ >= s_.size()) fail("unexpected end");
        char c = s_[p_];
        if (c == '{') return object();
        if (c == '[') return array();
        if (c == '"') { JValue v; v.type = JValue::T::Str; v.str = string(); return v; }
        if (c == 't') { lit("true"); JValue v; v.type = JValue::T::Bool; v.b = true; return v; }
        if (c == 'f') { lit("false"); JValue v; v.type = JValue::T::Bool; v.b = false; return v; }
        if (c == 'n') { lit("null"); return JValue{}; }
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return number();
        fail("unexpected character");
    }

    void lit(const char* t) {
        size_t n = std::string(t).size();
        if (s_.compare(p_, n, t) != 0) fail(std::string("expected ") + t);
        p_ += n;
    }

    JValue object() {
        JValue v; v.type = JValue::T::Obj;
        ++p_;  // {
        skip();
        if (peek() == '}') { ++p_; return v; }
        while (true) {
            skip();
            if (peek() != '"') fail("expected object key");
            std::string k = string();
            skip();
            if (peek() != ':') fail("expected ':'");
            ++p_;
            v.obj.emplace_back(std::move(k), value());
            skip();
            char c = peek();
            if (c == ',') { ++p_; continue; }
            if (c == '}') { ++p_; break; }
            fail("expected ',' or '}'");
        }
        return v;
    }

    JValue array() {
        JValue v; v.type = JValue::T::Arr;
        ++p_;  // [
        skip();
        if (peek() == ']') { ++p_; return v; }
        while (true) {
            v.arr.push_back(value());
            skip();
            char c = peek();
            if (c == ',') { ++p_; continue; }
            if (c == ']') { ++p_; break; }
            fail("expected ',' or ']'");
        }
        return v;
    }

    static void utf8(std::string& out, unsigned cp) {
        if (cp < 0x80) out.push_back(static_cast<char>(cp));
        else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }

    std::string string() {
        ++p_;  // opening quote
        std::string out;
        while (p_ < s_.size()) {
            char c = s_[p_++];
            if (c == '"') return out;
            if (c != '\\') { out.push_back(c); continue; }
            if (p_ >= s_.size()) fail("bad escape");
            char e = s_[p_++];
            switch (e) {
                case '"': out.push_back('"'); break;
                case '\\': out.push_back('\\'); break;
                case '/': out.push_back('/'); break;
                case 'b': out.push_back('\b'); break;
                case 'f': out.push_back('\f'); break;
                case 'n': out.push_back('\n'); break;
                case 'r': out.push_back('\r'); break;
                case 't': out.push_back('\t'); break;
                case 'u': {
                    if (p_ + 4 > s_.size()) fail("bad unicode escape");
                    unsigned cp = 0;
                    for (int i = 0; i < 4; ++i) {
                        char h = s_[p_++];
                        cp <<= 4;
                        if (h >= '0' && h <= '9') cp |= h - '0';
                        else if (h >= 'a' && h <= 'f') cp |= h - 'a' + 10;
                        else if (h >= 'A' && h <= 'F') cp |= h - 'A' + 10;
                        else fail("bad hex digit");
                    }
                    utf8(out, cp);
                    break;
                }
                default: fail("unknown escape");
            }
        }
        fail("unterminated string");
    }

    JValue number() {
        size_t start = p_;
        if (peek() == '-') ++p_;
        while (p_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_]))) ++p_;
        if (peek() == '.') { ++p_; while (p_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_]))) ++p_; }
        if (peek() == 'e' || peek() == 'E') {
            ++p_;
            if (peek() == '+' || peek() == '-') ++p_;
            while (p_ < s_.size() && std::isdigit(static_cast<unsigned char>(s_[p_]))) ++p_;
        }
        JValue v; v.type = JValue::T::Num;
        v.num = std::strtod(s_.substr(start, p_ - start).c_str(), nullptr);
        return v;
    }
};

inline JValue json_parse(const std::string& s) { return JParser(s).parse(); }

inline JValue json_load_file(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open JSON file: " + path);
    std::string s;
    char buf[8192];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
    std::fclose(f);
    return json_parse(s);
}

}  // namespace hx
