import math
from typing import Sequence, Optional, Iterator

from rl.apps.car.common.types import AngleDegrees, Vector, Rectangle


def constraint_position(position: Vector, constraint: Rectangle) -> Vector:
    x, y = position
    constraint_x, constraint_y, constraint_width, constraint_height = constraint
    return (
        max(constraint_x, min(constraint_x + constraint_width, x)),
        max(constraint_y, min(constraint_y + constraint_height, y)),
    )


def rotate(
        position: Vector,
        angle_degrees: AngleDegrees,
        center: Vector,
        rotate_center: Vector = None,
) -> Vector:
    start_center_x, start_center_y = center
    start_x, start_y = position
    dx = start_x - start_center_x
    dy = -(start_y - start_center_y)

    sin, cos = math.sin(math.radians(angle_degrees)), math.cos(math.radians(angle_degrees))
    rotated_dx = dx * cos - dy * sin
    rotated_dy = dx * sin + dy * cos

    end_center_x, end_center_y = rotate_center or center
    end_x = end_center_x + rotated_dx
    end_y = end_center_y - rotated_dy
    return end_x, end_y


def rotate_polygon(
        corners: Sequence[Vector],
        angle_degrees: AngleDegrees,
        center: Vector,
        rotate_center: Vector = None,
) -> Sequence[Vector]:
    return tuple(rotate(corner, angle_degrees, center, rotate_center) for corner in corners)


def rectangle_to_polygon(rectangle: Rectangle) -> Sequence[Vector]:
    (x, y, width, height) = rectangle
    return (
        (x, y),  # Top left
        (x + width, y),  # Top right
        (x, y + height),  # Bottom left
        (x + width, y + height),  # Bottom right
    )


def bound_rectangle(corners: Sequence[Vector]) -> Optional[Rectangle]:
    if not corners:
        return None

    min_x, min_y = corners[0]
    max_x, max_y = corners[0]

    for x, y in corners:
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)

    return min_x, min_y, max_x - min_x, max_y - min_y


def extend_rectangle(rectangle: Rectangle, units: int) -> Rectangle:
    x, y, width, height = rectangle
    return x - units, y - units, width + units * 2, height + units * 2


def polygon_contains(corners: Sequence[Vector], position: Vector) -> bool:
    result = False
    x, y = position

    for i in range(len(corners)):
        this_corner_x, this_corner_y = corners[i]
        that_corner_x, that_corner_y = corners[(i + 1) % len(corners)]

        if min(this_corner_y, that_corner_y) < y <= max(this_corner_y, that_corner_y):
            if x <= max(this_corner_x, that_corner_x):
                if that_corner_y != this_corner_y:
                    x_cross = (y - that_corner_y) * (this_corner_x - that_corner_x) / (
                            this_corner_y - that_corner_y) + that_corner_x
                else:
                    x_cross = that_corner_x

                if x <= x_cross:
                    result = not result

    return result


def rectangle_contains(rectangle: Rectangle, position: Vector) -> bool:
    x, y, width, height = rectangle
    position_x, position_y = position
    return x <= position_x <= x + width and y <= position_y <= y + height


def iterate_line(start: Vector, end: Vector) -> Iterator[Vector]:
    start_x, start_y = start
    end_x, end_y = end
    dx = end_x - start_x
    dy = end_y - start_y
    steps = int(max(abs(dx), abs(dy)))

    for step in range(steps + 1):
        t = step / steps
        x = start_x + t * (end_x - start_x)
        y = start_y + t * (end_y - start_y)
        yield round(x), round(y)


def iterate_polygon_perimeter(corners: Sequence[Vector]) -> Iterator[Vector]:
    for i in range(len(corners)):
        start = corners[i]
        end = corners[(i + 1) % len(corners)]
        for point in iterate_line(start, end):
            if not polygon_contains(corners, point):
                yield point


def iterate_polygon_area(corners: Sequence[Vector]) -> Iterator[Vector]:
    min_x = min([corner[0] for corner in corners])
    max_x = max([corner[0] for corner in corners])
    min_y = min([corner[1] for corner in corners])
    max_y = max([corner[1] for corner in corners])

    for x in range(math.floor(min_x), math.ceil(max_x + 1)):
        for y in range(math.floor(min_y), math.ceil(max_y + 1)):
            if polygon_contains(corners, (x, y)):
                yield x, y
