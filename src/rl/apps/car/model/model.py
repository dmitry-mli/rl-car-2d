from dataclasses import dataclass
from typing import Sequence, Optional, Dict

import torch
import torch.nn as nn
from torch import Tensor

from rl.apps.car.utils.device import get_module_device


class VisionConvolutionalLayer(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            max_pool_kernel_size: Optional[Dict] = None,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.convolution = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.optional_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.optional_max_pool = nn.MaxPool2d(**max_pool_kernel_size) if max_pool_kernel_size else nn.Identity()

    def forward(self, value: Tensor) -> Tensor:
        value = self.convolution(value)
        value = self.batch_norm(value)
        value = self.relu(value)
        value = self.optional_dropout(value)
        value = self.optional_max_pool(value)
        return value


class VisionModel(nn.Module):
    def __init__(
            self,
            dimensions: Sequence[int],
            dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dimensions) - 1):
            not_last = i < len(dimensions) - 2
            self.layers.append(VisionConvolutionalLayer(
                in_channels=dimensions[i],
                out_channels=dimensions[i + 1],
                kernel_size=3,
                max_pool_kernel_size={"kernel_size": 2, "stride": 2, "padding": 0} if not_last else None,
                dropout=dropout if not_last else 0.0,
            ))

    def forward(self, value: Tensor) -> Tensor:
        for layer in self.layers:
            value = layer(value)
        return value


class DecisionLinearLayer(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            residual: bool,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()
        self.optional_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.optional_adjust_dimensions = (
            nn.Linear(in_features, out_features)
            if residual and in_features != out_features
            else nn.Identity()
        )
        self.residual = residual

    def forward(self, value: Tensor) -> Tensor:
        identity = value

        value = self.linear(value)
        value = self.batch_norm(value)
        value = self.relu(value)
        value = self.optional_dropout(value)

        if self.residual:
            identity = self.optional_adjust_dimensions(identity)
            value = value + identity
        return value


class DecisionModel(nn.Module):
    def __init__(
            self,
            dimensions: Sequence[int],
            residual: bool,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(dimensions) - 1):
            not_last = i < len(dimensions) - 2
            self.layers.append(DecisionLinearLayer(
                in_features=dimensions[i],
                out_features=dimensions[i + 1],
                residual=residual,
                dropout=dropout if not_last else 0.0,
            ))

    def forward(self, value: Tensor) -> Tensor:
        for layer in self.layers:
            value = layer(value)
        return value


@dataclass
class SelfDrivingCarModelParams:
    vision_dimensions: Sequence[int]
    vision_dropout: float
    decision_dimensions: Sequence[int]
    decision_residual: bool
    decision_dropout: float
    state_path: Optional[str] = None


class SelfDrivingCarModel(nn.Module):
    def __init__(self, params: SelfDrivingCarModelParams):
        super().__init__()
        self.vision = VisionModel(params.vision_dimensions, params.vision_dropout)
        self.decision = DecisionModel(params.decision_dimensions, params.decision_residual, params.decision_dropout)
        if params.state_path:
            self.load_state_dict(torch.load(params.state_path))

    def forward(self, value: Tensor) -> Tensor:
        value = value.to(get_module_device(self))

        value = self.vision(value)
        value = torch.flatten(value, len(value.size()) - 3)

        return self.decision(value)
