import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Tuple

from rl.apps.car.common.constants import CAR_MAX_TURN, CAR_MIN_TURN, CAR_MAX_SPEED, CAR_MIN_SPEED, \
    CAR_TURN_DEGREES_PER_FRAME, CAR_SPEED_PIXELS_PER_FRAME, MARGIN, ACTION_AREA, SIDE, ROAD_MAP
from rl.apps.car.common.types import Vector, AngleDegrees, Rectangle, Shape
from rl.apps.car.environment.driver import DriverState
from rl.apps.car.utils.map import get_tile, road_next_tile, get_adjacent_tiles, get_tile_position, get_shape
from rl.apps.car.utils.math_util import advance
from rl.apps.car.utils.shapes import constraint_position, rectangle_contains
from rl.apps.car.utils.vectors import left, right

_STALE_COUNTER = 100


class Action(Enum):
    NONE = 0
    LEFT = 1
    RIGHT = 2
    ACCELERATION = 3
    DECELERATION = 4


class Blink(Enum):
    NONE = 0
    LEFT = 1
    RIGHT = 2


@dataclass
class CrossroadEvent:
    blink: Blink
    blink_area: Rectangle

    tile: Vector
    next_tile: Vector

    trajectory: Shape
    in_direction: Vector
    out_direction: Vector


@dataclass
class Events:
    crossroad: Optional[CrossroadEvent] = None


@dataclass
class CarState:
    position: Vector
    angle: AngleDegrees  # °, counterclockwise

    turn: int
    speed: int

    decelerating: bool = False
    steps_stopped: int = 0
    events: Events = field(default_factory=Events)
    driver: DriverState = field(default_factory=DriverState)


@dataclass
class CarObservation:
    turn: int
    speed: int


def reset_car(seed: CarState) -> Tuple[CarState, CarObservation]:
    state = _to_state(seed)
    observation = _to_observation(state)
    return state, observation


def step_car(previous: CarState, action: Action) -> Tuple[CarState, CarObservation, float, bool]:
    state = _to_state(previous, action)
    observation = _to_observation(state)
    reward = _to_reward(state)
    done = _to_done(state)
    return state, observation, reward, done


def _to_state(state: CarState, action: Optional[Action] = None) -> CarState:
    result: CarState = copy.deepcopy(state)

    if action is not None:
        # Action
        if action == Action.LEFT:
            result.turn = min(CAR_MAX_TURN, result.turn + 1)
        elif action == Action.RIGHT:
            result.turn = max(CAR_MIN_TURN, result.turn - 1)
        elif action == Action.ACCELERATION:
            result.speed = min(CAR_MAX_SPEED, result.speed + 1)
        elif action == Action.DECELERATION:
            result.speed = max(CAR_MIN_SPEED, result.speed - 1)

        # Experience
        if result.speed:
            result.angle = (result.angle + result.turn * CAR_TURN_DEGREES_PER_FRAME) % 360
            result.position = advance(result.position, result.angle, result.speed * CAR_SPEED_PIXELS_PER_FRAME)
            result.position = constraint_position(result.position, (MARGIN, MARGIN, ACTION_AREA[0], ACTION_AREA[1]))

        if result.position != state.position:
            result.steps_stopped = 0
        else:
            result.steps_stopped += 1

        result.decelerating = result.speed < state.speed

    # Events
    if not result.events.crossroad or not rectangle_contains(result.events.crossroad.blink_area, result.position):
        result.events.crossroad = _check_crossroad_event(state.driver, result.position)

    return result


def _to_observation(state: CarState) -> CarObservation:
    return CarObservation(state.turn, state.speed)


def _to_reward(state: CarState) -> float:
    return state.speed * 2


def _to_done(state: CarState) -> bool:
    return state.steps_stopped >= _STALE_COUNTER


def _check_crossroad_event(driver: DriverState, position: Vector) -> Optional[CrossroadEvent]:
    result = None
    tile_col, tile_row = get_tile(position)
    if next_tile := road_next_tile(position, ROAD_MAP[tile_row][tile_col]):
        current_tile_col, current_tile_row = get_tile(position)
        next_tile_col, next_tile_row = next_tile
        next_tile_options = [
            option
            for option in get_adjacent_tiles((next_tile_col, next_tile_row))
            if option != (current_tile_col, current_tile_row)
        ]
        if len(next_tile_options) > 1:  # Has turns
            next_tile_option = driver.choose(next_tile_options)
            (next_tile_option_col, next_tile_option_row) = next_tile_option
            next_tile_option_direction = (next_tile_option_col - next_tile_col, next_tile_option_row - next_tile_row)
            car_direction = (next_tile_col - current_tile_col, next_tile_row - current_tile_row)

            if car_direction == next_tile_option_direction:
                blink = Blink.NONE
            elif left(car_direction) == next_tile_option_direction:
                blink = Blink.LEFT
            elif right(car_direction) == next_tile_option_direction:
                blink = Blink.RIGHT
            else:
                blink = None

            trajectory = get_shape(car_direction, next_tile_option_direction)

            if blink:
                current_tile_x, current_tile_y = get_tile_position((current_tile_col, current_tile_row))
                next_tile_x, next_tile_y = get_tile_position((next_tile_col, next_tile_row))

                blink_area = (
                    area_x := min(current_tile_x, next_tile_x),
                    area_y := min(current_tile_y, next_tile_y),
                    max(current_tile_x + SIDE, next_tile_x + SIDE) - area_x,
                    max(current_tile_y + SIDE, next_tile_y + SIDE) - area_y,
                )
                result = CrossroadEvent(
                    blink=blink,
                    blink_area=blink_area,
                    tile=next_tile,
                    next_tile=next_tile_option,
                    trajectory=trajectory,
                    in_direction=car_direction,
                    out_direction=next_tile_option_direction,
                )
    return result
