import math
from typing import List, Iterable

import numpy as np

from rl.apps.car.common.types import Vector


def advance(position: Vector, angle: float, units: int) -> Vector:
    return (
        position[0] + math.cos(math.radians(angle)) * units,
        position[1] - math.sin(math.radians(angle)) * units,
    )


def distance(first: Vector, second: Vector) -> float:
    return np.linalg.norm(np.array(first) - np.array(second))


def normalize_min_max(values: Iterable[float]) -> List[float]:
    min_, max_ = min(values), max(values)
    if min_ == max_:
        return [0.5 for _ in values]
    else:
        return [(value - min_) / (max_ - min_) for value in values]
