# HX — 面向 MTK CPU/GPU/NPU 的算法分块异构部署 SDK 工具链

> 目标：为手机 OEM 提供一套 **「训练用 PyTorch → 编译期自动分块 → 运行时跨 CPU/GPU/NPU 并行执行」** 的工具链与运行时 SDK。
> 核心主张：把"算法分块、不同块并行在不同芯片"做成**可证伪的工程决策**，而不是口号。

---

## 0. 一句话结论

**条件可行**：分块异构并行在「图有可并行分支、且跨片搬运/同步开销被规模摊薄」时成立；
在「链式图、或芯片间有效吞吐接近、或小模型」时必然不划算。本工具链的价值在于——
**编译期就给出加速比、理论上界、搬运开销占比与明确裁决（RECOMMENDED / MARGINAL / NOT_RECOMMENDED），而不是让工程师上板碰运气。**

务必区分三种常被混为一谈的收益：

| 收益类型 | 是否需要真并行 | 本项目实测 |
|---|---|---|
| **A. 后端适配**（把算子放到最合适的芯片） | 否，串行链也有 | 主要收益来源 |
| **B. 跨芯片并行**（分叉分支同时占用不同后端） | **是** | 96×96 样例 `overlap_pairs=1` |
| **C. 跨帧流水**（不同帧占用不同后端，提吞吐） | 是（跨帧） | 230 → 1667 fps |

---

## 1. 目录结构

```
workspace/
├── hx/                  # Python 编译侧（训练侧只依赖它）
│   ├── capture.py       # torch.export 抓取（训练侧零约束）
│   ├── lower.py         # FX → HX IR 降级 / 可部署化重写（BN 折叠、视图消除…）
│   ├── ir.py            # 中间表示与校验
│   ├── opset.py         # 后端能力表 + 芯片档案 DEFAULT_CHIP（可替换）
│   ├── cost.py          # 代价模型 + 离散事件模拟
│   ├── partition.py     # 并行感知分区搜索（模拟退火 + 多重启）
│   ├── planner.py       # 收益判定（S / S_max / 流水线 / 基线）
│   ├── codegen.py       # 部署包生成（manifest / 子图 / 权重 / 需求）
│   ├── verify.py        # 数值 + 结构双重校验
│   └── api.py           # 一站式入口 compile_model
├── runtime/             # C++17 运行时 SDK（不链接 PyTorch）
│   ├── include/hx/      # 公共 API：Engine / IBackend / Options
│   ├── src/             # 调度器、参考后端、算子库
│   ├── backends/        # 真机 NPU/GPU 适配骨架（ADAPTERS.md）
│   └── tools/           # hx_run（执行）/ hx_bench（流水基准）
├── examples/            # 样例网络 + 端到端导出脚本
├── tests/               # 敏感性分析 / 端到端验证
└── docs/                # 论证与对接文档
```

---

## 2. 快速开始

### 2.1 训练侧（零侵入）

```python
import torch, hx

model = MyNet()                      # 任意 nn.Module，可用 Python 控制流 / 自定义算子
pkg = hx.compile_model(
    model,
    (torch.randn(1, 3, 224, 224),),  # 代表性输入（形状特化）
    "build/demo",
    strategy="auto",                  # auto | greedy | single | force_hetero(研究)
)
print(pkg.summary())                 # 加速比 / 判定 / 跨片边数
```

可选标记（**绝不强制**，非法提示自动忽略）：

```python
import hx
x = hx.prefer(x, "npu")              # 后端软提示
x = hx.boundary(x, "fuse")           # 硬切分边界
hx.mark(module, "gpu")               # 模块级提示（沿 producer 链反向传播）
```

### 2.2 运行时（C++，部署侧）

```bash
cd runtime && cmake -S . -B build && cmake --build build -j8
./build/hx_run --manifest build/demo/manifest.json --frames 1 --depth 1 --scale 20
./build/hx_bench --manifest build/demo/manifest.json --frames 8 --scale 1
```

---

## 3. 构建

| 组件 | 命令 | 依赖 |
|---|---|---|
| Python 工具链 | `pip3 install torch` | PyTorch ≥ 2.x（`torch.export` / `torch.fx`） |
| C++ 运行时 | `cd runtime && cmake -S . -B build && cmake --build build -j8` | C++17、Threads、OpenMP（可选） |
| 真机 NPU/GPU | `cmake -DHX_ENABLE_MTK=ON` / `-DHX_ENABLE_LITERT_GPU=ON` | MTK NeuroPilot / LiteRT SDK（NDA） |

> 未开启 MTK/LiteRT 开关时，运行时回退到**参考后端**（真实执行 IR 算子 + 代价模型注入时长），
> 因此本工程在**任何环境都能完整编译运行**并验证全链路。

---

## 4. 端到端验证（可复现）

```bash
python3 -m examples.export        # 三场景编译 + 分区 + 校验 + 部署包
python3 -m tests.sensitivity      # 收益临界条件扫描（写入 build/analysis/）
```

实测结论（详见 `docs/01_可行性论证.md` 第 3 节）：

| 场景 | 策略 | 子图 | 加速比 | 真并行 | 判定 |
|---|---|---|---|---|---|
| 96×96 双分支 | `auto` | 4（NPU/GPU/NPU/GPU） | **2.21×** | 是 | `RECOMMENDED` |
| 32×32 双分支 | `auto` | 1（整图 GPU） | 1.00× | 否 | `MARGINAL`（整图最优） |
| 纯链式 | `auto` | 1（NPU） | 1.00× | 否 | `NOT_RECOMMENDED` |
| 32×32 强制异构 | `force_hetero` | 4 | 0.43× | 是 | `NOT_RECOMMENDED`（仅验证链路） |

C++ 运行时在 `build/*/manifest.json` 上实测 **与 PyTorch 逐元素误差 0**，`overlap_pairs` 与编译期预测一致。

---

## 5. 文档

| 文件 | 内容 |
|---|---|
| `docs/01_可行性论证.md` | 平台事实、形式化模型、收益判据、实测、敏感性、风险、生产路线 |
| `docs/05_MTK生产对接.md` | 真机落地 Phase 0–4、LiteRT/Neuron 对接、量化、验收标准 |
| `runtime/backends/ADAPTERS.md` | `IBackend` 契约、NPU/GPU 适配骨架、零拷贝与量化 |

---

## 6. 已知边界（诚实声明）

- 沙箱无 MTK 硬件与 NDA SDK，NPU/GPU 后端为**参考后端**：验证的是**链路正确性**，
  不是目标芯片性能数字；
- 芯片档案、算子能力表是**模板**，必须由 OEM 用上板 profiling 回填（`hx/opset.py: DEFAULT_CHIP`）；
- `force_hetero` 是**验证/研究模式**，明确不参与生产裁决；
- 生产对接接口骨架见 `docs/05_MTK生产对接.md` 与 `runtime/backends/ADAPTERS.md`。

---

## 7. 核心设计原则

1. **训练 / 部署解耦**：部署决策 100% 留在编译期，训练代码不写任何部署约束；
2. **同后端独占串行**：每后端一条串行队列，评价函数建模真实资源竞争；
3. **跨后端分叉强切**：公共主干若与某分支合并会悄悄吃掉并行机会；
4. **碎片抑制**：软惩罚而非强制降级（避免计算密集算子瞬间慢一个数量级）；
5. **可证伪裁决**：编译期给出加速比、上界、搬运占比与明确建议。
