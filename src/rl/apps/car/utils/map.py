from typing import List, Optional

from convolution.main import Position
from rl.apps.car.common.constants import SIDE, MARGIN, PAD, HALF, ROAD_MAP
from rl.apps.car.common.types import Vector, Shape
from rl.apps.car.utils.math_util import distance
from rl.apps.car.utils.shapes import rotate, rectangle_contains


def is_unit_vector(vector: Vector) -> bool:
    x, y = vector
    return x in (0, 1, -1) and y in (0, 1, -1)


def is_left(vector: Vector) -> bool:
    return vector == (-1, 0)


def is_right(vector: Vector) -> bool:
    return vector == (1, 0)


def is_up(vector: Vector) -> bool:
    return vector == (0, -1)


def is_down(vector: Vector) -> bool:
    return vector == (0, 1)


def get_tile(position: Vector) -> Vector:
    x, y = position
    tile_col, tile_row = int((x - MARGIN) / SIDE), int((y - MARGIN) / SIDE)
    return tile_col, tile_row


def get_tile_position(tile: Vector) -> Vector:
    tile_col, tile_row = tile
    return MARGIN + tile_col * SIDE, MARGIN + tile_row * SIDE


def get_shape(start: Vector, end: Vector) -> Shape:
    assert is_unit_vector(start)
    assert is_unit_vector(end)

    if start == end and (is_left(start) or is_right(start)):
        return "─"
    elif start == end and (is_up(start) or is_down(start)):
        return "│"
    elif (is_up(start) and is_right(end)) or (is_left(start) and is_down(end)):
        return "┌"
    elif (is_up(start) and is_left(end)) or (is_right(start) and is_down(end)):
        return "┐"
    elif (is_down(start) and is_right(end)) or (is_left(start) and is_up(end)):
        return "└"
    elif (is_down(start) and is_left(end)) or (is_right(start) and is_up(end)):
        return "┘"
    else:
        raise ValueError("Invalid start and end vectors")


def get_adjacent_tiles(tile: Vector) -> List[Vector]:
    tile_col, tile_row = tile
    shape = ROAD_MAP[tile_row][tile_col]
    if shape == "─":
        result = [
            (tile_col - 1, tile_row),
            (tile_col + 1, tile_row),
        ]
    elif shape == "│":
        result = [
            (tile_col, tile_row - 1),
            (tile_col, tile_row + 1),
        ]
    elif shape == "┌":
        result = [
            (tile_col + 1, tile_row),
            (tile_col, tile_row + 1),
        ]
    elif shape == "┐":
        result = [
            (tile_col - 1, tile_row),
            (tile_col, tile_row + 1),
        ]
    elif shape == "└":
        result = [
            (tile_col + 1, tile_row),
            (tile_col, tile_row - 1),
        ]
    elif shape == "┘":
        result = [
            (tile_col - 1, tile_row),
            (tile_col, tile_row - 1),
        ]
    elif shape == "┤":
        result = [
            (tile_col - 1, tile_row),
            (tile_col, tile_row - 1),
            (tile_col, tile_row + 1),
        ]
    elif shape == "├":
        result = [
            (tile_col + 1, tile_row),
            (tile_col, tile_row - 1),
            (tile_col, tile_row + 1),
        ]
    elif shape == "┬":
        result = [
            (tile_col - 1, tile_row),
            (tile_col + 1, tile_row),
            (tile_col, tile_row + 1),
        ]
    elif shape == "┴":
        result = [
            (tile_col - 1, tile_row),
            (tile_col + 1, tile_row),
            (tile_col, tile_row - 1),
        ]
    elif shape == "┼":
        result = [
            (tile_col - 1, tile_row),
            (tile_col + 1, tile_row),
            (tile_col, tile_row - 1),
            (tile_col, tile_row + 1),
        ]
    else:
        result = []
    return result


def road_next_tile(position: Position, shape: Shape) -> Optional[Vector]:
    tile_col, tile_row = get_tile(position)
    tile_x, tile_y = get_tile_position((tile_col, tile_row))

    def next_for_horizontal(position_: Vector) -> Optional[Vector]:
        _, y = position_
        if rectangle_contains((tile_x, tile_y + PAD, SIDE, SIDE - PAD * 2), position_):
            if y < (tile_y + SIDE / 2):  # Driving top lane towards left
                return tile_col - 1, tile_row
            else:  # Driving bottom lane towards right
                return tile_col + 1, tile_row
        else:
            return None

    def next_for_vertical(position_: Vector) -> Optional[Vector]:
        x, _ = position_
        if rectangle_contains((tile_x + PAD, tile_y, SIDE - PAD * 2, SIDE), position_):
            if x < (tile_x + SIDE / 2):  # Driving left lane towards bottom
                return tile_col, tile_row + 1
            else:  # Driving right lane towards top
                return tile_col, tile_row - 1
        else:
            return None

    def next_for_L_curved(
            position_: Vector,
            left_turn_result: Vector,
            right_turn_result: Vector,
    ) -> Optional[Vector]:
        x, y = position_
        curvature_center = (tile_x + SIDE - PAD, tile_y + PAD)
        curvature_radius = distance(position_, curvature_center)

        if rectangle_contains((tile_x + PAD, tile_y, SIDE - PAD * 2, PAD), position_):
            # Out of curvature on the top
            turning_left = x < (tile_x + SIDE / 2)
        elif rectangle_contains((tile_x + SIDE - PAD, tile_y + PAD, PAD, SIDE - PAD * 2), position_):
            # Out of curvature on the right
            turning_left = y >= (tile_y + SIDE / 2)
        elif 0 <= curvature_radius <= (HALF - PAD) * 2:
            # Inside the curvature
            turning_left = distance((x, y), curvature_center) > (HALF - PAD)
        else:
            return None

        return left_turn_result if turning_left else right_turn_result

    if shape == "─":
        return next_for_horizontal(position)
    elif shape == "│":
        return next_for_vertical(position)
    elif shape == "└":
        return next_for_L_curved(position, (tile_col + 1, tile_row), (tile_col, tile_row - 1))
    elif shape == "┘":
        projected = rotate(position, -90, (tile_x + SIDE / 2, tile_y + SIDE / 2))
        return next_for_L_curved(projected, (tile_col, tile_row - 1), (tile_col - 1, tile_row))
    elif shape == "┐":
        projected = rotate(position, -180, (tile_x + SIDE / 2, tile_y + SIDE / 2))
        return next_for_L_curved(projected, (tile_col - 1, tile_row), (tile_col, tile_row + 1))
    elif shape == "┌":
        projected = rotate(position, -270, (tile_x + SIDE / 2, tile_y + SIDE / 2))
        return next_for_L_curved(projected, (tile_col, tile_row + 1), (tile_col + 1, tile_row))
    else:
        return None
