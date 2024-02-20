from dataclasses import dataclass
from typing import Tuple, Optional

import pygame
from pygame import Surface

from rl.apps.car.common.constants import CANVAS_AREA, OBSERVATION_INPUT_AREA, OBSERVATION_DOWNSCALE_RATIO, \
    OBSERVATION_OUTPUT_AREA, CAR_LENGTH, CAR_WIDTH, DARK_GRAY, CENTERLINE
from rl.apps.car.environment.car import CarState, CarObservation, reset_car, Action, step_car
from rl.apps.car.helpers.canvas import draw_state, get_background
from rl.apps.car.utils.map import road_next_tile, get_tile
from rl.apps.car.utils.shapes import rectangle_to_polygon, rotate_polygon, extend_rectangle, bound_rectangle, \
    iterate_polygon_perimeter


@dataclass
class State:
    car: CarState
    view: Surface
    background: Surface


@dataclass
class Observation:
    car: CarObservation
    view: Surface


def reset_environment(seed: CarState) -> Tuple[State, Observation]:
    # Dependencies
    car_state, car_observation = reset_car(seed)

    # Reconcile
    state = _to_state(car_state)
    observation = _to_observation(state, car_observation)
    return state, observation


def step_environment(previous: State, action: Action) -> Tuple[State, Observation, float, float]:
    # Dependencies
    car_state, car_observation, car_reward, car_done = step_car(previous.car, action)

    # Reconcile
    state = _to_state(car_state, previous.car)
    observation = _to_observation(state, car_observation)
    reward = _to_reward(state, car_reward)
    done = _to_done(state, car_done)
    return state, observation, reward, done


def _to_state(car_state: CarState, previous_car: Optional[CarState] = None) -> State:
    return State(
        car=car_state,
        view=draw_state(pygame.Surface(CANVAS_AREA), car_state, previous_car),
        background=get_background(),
    )


def _to_observation(state: State, car_observation: CarObservation) -> Observation:
    # Select bounded car view
    car_x, car_y = state.car.position
    observation_input_width, observation_input_height = OBSERVATION_INPUT_AREA
    camera_x, camera_y = car_x - observation_input_width / 8 * 1, car_y - observation_input_height / 2
    corners = rectangle_to_polygon((camera_x, camera_y, observation_input_width, observation_input_height))
    rotated_corners = rotate_polygon(corners, state.car.angle, (car_x, car_y))
    bounded_car_view = state.view.subsurface(extend_rectangle(bound_rectangle(rotated_corners), 1))

    # Downscale
    downscaled_bounded_car_view = pygame.transform.scale(
        bounded_car_view,
        (bounded_car_view.get_width() * OBSERVATION_DOWNSCALE_RATIO,
         bounded_car_view.get_height() * OBSERVATION_DOWNSCALE_RATIO),
    )

    # Rotate car position
    rotated_downscaled_bounded_car_view = pygame.transform.rotate(downscaled_bounded_car_view, 90 - state.car.angle)

    # Unbounded
    downscaled_width, downscaled_height = OBSERVATION_OUTPUT_AREA
    bound_margin_x = (rotated_downscaled_bounded_car_view.get_width() - downscaled_width) / 2
    bound_margin_y = (rotated_downscaled_bounded_car_view.get_height() - downscaled_height) / 2
    rotated_downscaled_car_view = rotated_downscaled_bounded_car_view.subsurface(
        (bound_margin_x, bound_margin_y, downscaled_width, downscaled_height))
    return Observation(
        car=car_observation,
        view=rotated_downscaled_car_view,
    )


def _to_reward(state: State, car_reward: float) -> float:
    return car_reward


def _to_done(state: State, car_done: bool) -> bool:
    if car_done:
        return True

    # Is hit?
    car_x, car_y = state.car.position
    corners = rectangle_to_polygon((car_x - CAR_LENGTH / 2, car_y - CAR_WIDTH / 2, CAR_LENGTH, CAR_WIDTH))
    rotated_corners = rotate_polygon(corners, state.car.angle, state.car.position)

    for corner in iterate_polygon_perimeter(rotated_corners):
        color = state.background.get_at(corner)
        if color in [DARK_GRAY, CENTERLINE]:
            return True

        corner_tile = get_tile(corner)
        if state.car.events.crossroad and corner_tile == state.car.events.crossroad.tile:
            # When on crossroad, must drive by the trajectory
            if road_next_tile(corner, state.car.events.crossroad.trajectory) != state.car.events.crossroad.next_tile:
                return True
    return False


def _verify_always_fails(car: CarState) -> bool:
    def would_fail(action: Optional[Action]) -> bool:
        car_state, _, _, car_done = step_car(car, action)
        return _to_done(_to_state(car_state), car_done)

    if all([
        would_fail(None),
        would_fail(Action.LEFT),
        would_fail(Action.RIGHT),
        would_fail(Action.ACCELERATION),
        would_fail(Action.DECELERATION),
    ]):
        print()
        return True
    return False
