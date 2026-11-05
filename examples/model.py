"""示例网络：刻意使用多种"部署不友好"写法，验证训练侧自由度。

包含：
  - Python 动态控制流（if x.shape[-1] >= 32），导出时被自动特化；
  - BatchNorm（工具链自动折叠进 conv）；
  - SiLU / GELU（APU 能力表外的算子，工具链自动放 GPU/CPU）；
  - 两个独立并行分支（这是"不同块并行在不同芯片"能成立的结构前提）；
  - 可选软提示 hx.mark（只是先验，不加也完全能编译）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import hx


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 3, stride: int = 1):
        super().__init__(
            nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class HeteroNet(nn.Module):
    def __init__(self, use_hints: bool = True):
        super().__init__()
        self.stem = ConvBNReLU(3, 32)
        # 分支 A：卷积密集、只含 APU 友好算子 -> 倾向 NPU
        self.branch_a = nn.Sequential(
            ConvBNReLU(32, 64),
            ConvBNReLU(64, 64),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        # 分支 B：含 SiLU 等 APU 不支持的激活 -> 倾向 GPU
        self.branch_b = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, 64), nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 10), nn.Sigmoid(),
        )
        if use_hints:
            hx.mark(self.branch_a, "npu")
            hx.mark(self.branch_b, "gpu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.stem(x)
        a = self.branch_a(z)
        b = self.branch_b(z)
        # 基于 shape 的 Python 控制流：训练侧完全自由，导出时特化
        if x.shape[-1] >= 32:
            fused = torch.cat([a, b], dim=1)
        else:
            fused = torch.cat([b, a], dim=1)
        return self.head(fused)


class ChainNet(nn.Module):
    """纯链式反例：用于验证工具链会明确判定「不建议算子级并行」。"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(3, 32), ConvBNReLU(32, 64),
            ConvBNReLU(64, 64), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.net(x)


def build_model(use_hints: bool = True) -> HeteroNet:
    torch.manual_seed(7)
    return HeteroNet(use_hints=use_hints).eval()


def build_chain() -> ChainNet:
    torch.manual_seed(7)
    return ChainNet().eval()
