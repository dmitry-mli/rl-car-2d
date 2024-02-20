import os
from types import SimpleNamespace
from typing import TypeVar

from torch import nn

T = TypeVar("T")


def get_device() -> str:
    return os.environ.get("DEVICE")


def to_device(value: T) -> T:
    return value.to(get_device())


def get_module_device(module: nn.Module) -> str:
    return next(module.parameters(), SimpleNamespace(device="cpu")).device
