from typing import Union, Tuple

Radius = int  # Degrees
Shape = str  # Either of: ─ │ ┌ ┐ └ ┘ ├ ┤ ┬ ┴ ┼
AngleDegrees = float
Vector = Union[Tuple[int, int], Tuple[float, float]]
Rectangle = Union[Tuple[int, int, int, int], Tuple[float, float, float, float]]
