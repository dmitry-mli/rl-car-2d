import copy
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional, List

from rl.apps.car.common.constants import SIDE, MARGIN
from rl.apps.car.environment.car import Action, CarState
from rl.apps.car.environment.environment import State, Observation, reset_environment, step_environment
from rl.apps.car.helpers.canvas import draw_car

_DRAW_RESET_CARS = True

_RESET_CAR_FACTORIES_LONG = [  # Clockwise
    lambda: CarState(position=(MARGIN + 2.7 * SIDE, MARGIN + 2 * SIDE), angle=90, speed=1, turn=0),
    ##############################
    lambda: CarState(position=(MARGIN + 2.5 * SIDE, MARGIN + 0.5 * SIDE), angle=225, speed=1, turn=0),
    lambda: CarState(position=(MARGIN + 2.8 * SIDE, MARGIN + 0.80 * SIDE), angle=45, speed=1, turn=0),
    ##############################
    lambda: CarState(position=(MARGIN + 5.5 * SIDE, MARGIN + 1.7 * SIDE), angle=0, speed=1, turn=0),
    lambda: CarState(position=(MARGIN + 5.5 * SIDE, MARGIN + 1.3 * SIDE), angle=180, speed=1, turn=0),
    ##############################
    lambda: CarState(position=(MARGIN + 7 * SIDE, MARGIN + 0.7 * SIDE), angle=0, speed=1, turn=0),
    lambda: CarState(position=(MARGIN + 7 * SIDE, MARGIN + 0.3 * SIDE), angle=180, speed=1, turn=0),
    ##############################
    lambda: CarState(position=(MARGIN + 7.8 * SIDE, MARGIN + 1.2 * SIDE), angle=135, speed=1, turn=0),
    lambda: CarState(position=(MARGIN + 7.5 * SIDE, MARGIN + 1.5 * SIDE), angle=315, speed=1, turn=0),
    ##############################
    lambda: CarState(position=(MARGIN + 8.5 * SIDE, MARGIN + 1.5 * SIDE), angle=135, speed=1, turn=0),
    lambda: CarState(position=(MARGIN + 8.2 * SIDE, MARGIN + 1.8 * SIDE), angle=315, speed=1, turn=0),
    ##############################
    lambda: CarState(position=(MARGIN + 8 * SIDE, MARGIN + 3.7 * SIDE), angle=0, speed=1, turn=0),
    ##############################
    lambda: CarState(position=(MARGIN + 7 * SIDE, MARGIN + 3.3 * SIDE), angle=180, speed=1, turn=0),
    ##############################
    lambda: CarState(position=(MARGIN + 4 * SIDE, MARGIN + 3.7 * SIDE), angle=0, speed=1, turn=0),
    ##############################
    lambda: CarState(position=(MARGIN + 5.7 * SIDE, MARGIN + 5 * SIDE), angle=90, speed=1, turn=0),
    ##############################
    lambda: CarState(position=(MARGIN + 4 * SIDE, MARGIN + 3.3 * SIDE), angle=180, speed=1, turn=0),
    ##############################
    lambda: CarState(position=(MARGIN + 1 * SIDE, MARGIN + 3.7 * SIDE), angle=0, speed=1, turn=0),
    ##############################
    lambda: CarState(position=(MARGIN + 2.7 * SIDE, MARGIN + 4.7 * SIDE), angle=90, speed=1, turn=0),
    ##############################
    lambda: CarState(position=(MARGIN + 2.3 * SIDE, MARGIN + 2.0 * SIDE), angle=270, speed=1, turn=0),
]

_RESET_CAR_FACTORIES_SHORT = [  # Clockwise
    lambda: CarState(position=(MARGIN + 8.5 * SIDE, MARGIN + 1.5 * SIDE), angle=135, speed=1, turn=0),
    lambda: CarState(position=(MARGIN + 8.2 * SIDE, MARGIN + 1.8 * SIDE), angle=315, speed=1, turn=0),
    ##############################
    lambda: CarState(position=(MARGIN + 4 * SIDE, MARGIN + 3.7 * SIDE), angle=0, speed=1, turn=0),
    lambda: CarState(position=(MARGIN + 4 * SIDE, MARGIN + 3.3 * SIDE), angle=180, speed=1, turn=0),
]

_RESET_CAR_FACTORIES = _RESET_CAR_FACTORIES_SHORT


class RlEnvironmentMode(Enum):
    ORDERED_WITH_CRASH_REPLAY = 1


@dataclass
class RlEnvironmentHistoryItem:
    action: Optional[Action]
    state: State
    observation: Observation
    reward: Optional[float]


class RlEnvironment:
    def __init__(
            self,
            mode: RlEnvironmentMode,
            total_resets: int,
    ):
        self.mode = mode
        self.total_resets = total_resets

        self.reset_index = 0
        self.history: List[RlEnvironmentHistoryItem] = []

    def reset(self) -> Tuple[State, Observation]:
        state, observation = reset_environment(self._pick_reset_car())

        self.history.clear()
        self.history.append(RlEnvironmentHistoryItem(None, state, observation, None))
        self.reset_index += 1
        return state, observation

    def step(self, action: Action) -> Tuple[State, Observation, float, float]:
        state, observation, reward, done = step_environment(self.history[-1].state, action)

        self.history.append(RlEnvironmentHistoryItem(action, state, observation, reward))
        if _DRAW_RESET_CARS:
            for reset_car_factory in _RESET_CAR_FACTORIES:
                draw_car(state.view, reset_car_factory())
        return state, observation, reward, done

    def _pick_reset_car(self) -> CarState:
        if self.mode == RlEnvironmentMode.ORDERED_WITH_CRASH_REPLAY:
            if self.total_resets % len(_RESET_CAR_FACTORIES) != 0:
                raise ValueError(
                    f"When using {self.mode.name}, total resets (currently {self.total_resets})"
                    f"must be divisible by reset car factories (currently {len(_RESET_CAR_FACTORIES)})"
                )

            resets_per_car_state = self.total_resets // len(_RESET_CAR_FACTORIES)
            car_state_index: int = self.reset_index // resets_per_car_state
            previous_car_state_index = (self.reset_index - 1) // resets_per_car_state
            if car_state_index == previous_car_state_index:
                result = self._get_car_before_crash()
            else:
                result = _RESET_CAR_FACTORIES[car_state_index]()
        else:
            raise NotImplementedError

        return result

    def _get_car_before_crash(self, steps_into_past: int = 10) -> CarState:
        result = copy.deepcopy(self.history[0].state.car)
        skipped = set()
        for item in reversed(self.history):
            if item.state.car.position not in skipped:
                result = copy.deepcopy(item.state.car)
                if steps_into_past == 0:
                    break
                else:
                    skipped.add(item.state.car.position)
                    steps_into_past -= 1
        return result
