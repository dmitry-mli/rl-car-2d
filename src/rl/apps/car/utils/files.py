import os
import shutil
from typing import Iterable

import torch
from torch import nn


def move_files(filenames: Iterable[str], target_directory: str):
    os.makedirs(target_directory, exist_ok=True)
    for filename in filenames:
        shutil.move(filename, target_directory)


def save_file(path: str, filename: str, content: str) -> str:
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)
    with open(full_path, "w") as file:
        file.write(content)
    return full_path


def save_state(path: str, filename: str, module: nn.Module) -> str:
    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, filename)
    torch.save(module.state_dict(), full_path)
    return full_path
