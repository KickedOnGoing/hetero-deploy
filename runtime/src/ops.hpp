#pragma once
#include <string>

#include "hx/hx_types.h"

namespace hx {
// 执行单个 IR 算子。C++ 参考实现与 Python 侧 HXInterpreter 一一对应。
bool run_node(const NodeDesc& n, const Model& m, ITensorStore& store, std::string* err);
}  // namespace hx
