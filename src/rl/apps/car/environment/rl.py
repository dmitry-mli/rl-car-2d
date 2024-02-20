import copy
import random
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Optional, List

from rl.apps.car.common.constants import SIDE, MARGIN
from rl.apps.car.environment.car import Action, CarState
from rl.apps.car.environment.environment import State, Observation, reset_environment, step_environment
from rl.apps.car.helpers.canvas import draw_car

_DRAW_RESET_CARS = False
_RESET_CAR_STATES = [  # Clockwise
    CarState(position=(MARGIN + 2.7 * SIDE, MARGIN + 2 * SIDE), angle=90, speed=1, turn=0),
    ##############################
    CarState(position=(MARGIN + 2.5 * SIDE, MARGIN + 0.5 * SIDE), angle=225, speed=1, turn=0),
    CarState(position=(MARGIN + 2.8 * SIDE, MARGIN + 0.80 * SIDE), angle=45, speed=1, turn=0),
    ##############################
    CarState(position=(MARGIN + 5.5 * SIDE, MARGIN + 1.7 * SIDE), angle=0, speed=1, turn=0),
    CarState(position=(MARGIN + 5.5 * SIDE, MARGIN + 1.3 * SIDE), angle=180, speed=1, turn=0),
    ##############################
    CarState(position=(MARGIN + 7 * SIDE, MARGIN + 0.7 * SIDE), angle=0, speed=1, turn=0),
    CarState(position=(MARGIN + 7 * SIDE, MARGIN + 0.3 * SIDE), angle=180, speed=1, turn=0),
    ##############################
    CarState(position=(MARGIN + 7.8 * SIDE, MARGIN + 1.2 * SIDE), angle=135, speed=1, turn=0),
    CarState(position=(MARGIN + 7.5 * SIDE, MARGIN + 1.5 * SIDE), angle=315, speed=1, turn=0),
    ##############################
    CarState(position=(MARGIN + 8.5 * SIDE, MARGIN + 1.5 * SIDE), angle=135, speed=1, turn=0),
    CarState(position=(MARGIN + 8.2 * SIDE, MARGIN + 1.8 * SIDE), angle=315, speed=1, turn=0),
    ##############################
    CarState(position=(MARGIN + 8 * SIDE, MARGIN + 3.7 * SIDE), angle=0, speed=1, turn=0),
    ##############################
    CarState(position=(MARGIN + 7 * SIDE, MARGIN + 3.3 * SIDE), angle=180, speed=1, turn=0),
    ##############################
    CarState(position=(MARGIN + 4 * SIDE, MARGIN + 3.7 * SIDE), angle=0, speed=1, turn=0),
    ##############################
    CarState(position=(MARGIN + 5.7 * SIDE, MARGIN + 5 * SIDE), angle=90, speed=1, turn=0),
    ##############################
    CarState(position=(MARGIN + 4 * SIDE, MARGIN + 3.3 * SIDE), angle=180, speed=1, turn=0),
    ##############################
    CarState(position=(MARGIN + 1 * SIDE, MARGIN + 3.7 * SIDE), angle=0, speed=1, turn=0),
    ##############################
    CarState(position=(MARGIN + 2.7 * SIDE, MARGIN + 4.7 * SIDE), angle=90, speed=1, turn=0),
    ##############################
    CarState(position=(MARGIN + 2.3 * SIDE, MARGIN + 2.0 * SIDE), angle=270, speed=1, turn=0),
]


class RlEnvironmentResetMode(Enum):
    RANDOM = 1
    RANDOM_ONCE = 2
    RANDOM_THEN_BEFORE_CRASH = 3


@dataclass
class RlEnvironmentHistoryItem:
    action: Optional[Action]
    state: State
    observation: Observation
    reward: Optional[float]


class RlEnvironment:
    def __init__(self, reset_mode: RlEnvironmentResetMode):
        self.reset_mode = reset_mode
        self.history: List[RlEnvironmentHistoryItem] = []

    def reset(self) -> Tuple[State, Observation]:
        state, observation = reset_environment(self._pick_reset_seed())

        self.history.clear()
        self.history.append(RlEnvironmentHistoryItem(None, state, observation, None))
        return state, observation

    def step(self, action: Action) -> Tuple[State, Observation, float, float]:
        state, observation, reward, done = step_environment(self.history[-1].state, action)

        self.history.append(RlEnvironmentHistoryItem(action, state, observation, reward))
        if _DRAW_RESET_CARS:
            for reset_car in _RESET_CAR_STATES:
                draw_car(state.view, reset_car)
        return state, observation, reward, done

    def _pick_reset_seed(self) -> CarState:
        if self.reset_mode == RlEnvironmentResetMode.RANDOM:
            result = copy.deepcopy(random.choice(_RESET_CAR_STATES))
        elif self.reset_mode == RlEnvironmentResetMode.RANDOM_ONCE:
            result = copy.deepcopy(
                self.history[0].state.car
                if self.history
                else random.choice(_RESET_CAR_STATES)
            )
        elif self.reset_mode == RlEnvironmentResetMode.RANDOM_THEN_BEFORE_CRASH:
            reset_position_margin = 10
            skipped = set()
            for item in reversed(self.history):
                if item.state.car.position not in skipped:
                    reset_car = item.state.car
                    if reset_position_margin == 0:
                        break
                    else:
                        skipped.add(item.state.car.position)
                        reset_position_margin -= 1

            result = (
                reset_car
                if self.history
                else random.choice(_RESET_CAR_STATES)
            )
        else:
            raise NotImplementedError

        return result
