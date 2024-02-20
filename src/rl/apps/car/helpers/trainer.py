import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pygame
import torch
from cattr import unstructure
from pygame import Surface
from torch import Tensor

from rl.apps.car.common.constants import CAR_MIN_SPEED, CAR_MAX_SPEED, CAR_MIN_TURN, CAR_MAX_TURN
from rl.apps.car.environment.car import Action
from rl.apps.car.environment.environment import Observation
from rl.apps.car.environment.rl import RlEnvironmentResetMode, RlEnvironment
from rl.apps.car.helpers.display import Display
from rl.apps.car.helpers.keyboard import Keyboard
from rl.apps.car.model.model import SelfDrivingCarModelParams, SelfDrivingCarModel
from rl.apps.car.model.rl import RlModel
from rl.apps.car.utils.device import to_device
from rl.apps.car.utils.files import move_files, save_state, save_file
from rl.apps.car.utils.tee import capture_stdout
from rl.apps.car.utils.timestamp import timestamp


class Metrics:
    def __init__(self):
        self.max_return = 0.
        self.max_loss = 0.

    def update(self, return_: float, loss: float):
        self.max_return = max(self.max_return, return_)
        self.max_loss = max(self.max_loss, loss)


@dataclass
class HyperParams:
    dry_run: bool
    epochs: int
    learning_rate: float
    weight_decay: float
    max_batches: int
    max_episodes: int
    environment_reset_mode: RlEnvironmentResetMode
    model: SelfDrivingCarModelParams
    epoch_state_return_threshold: int

    def to_label(self, metrics: Metrics, took: float):
        result = " | ".join((
            f"ret {metrics.max_return:5.0f}",
            f"loss {metrics.max_loss:5.0f}",
            f"took {took / 60:5.1f}m",
            f"e {self.epochs:4}",
            f"b {self.max_batches:3}",
            f"ep {self.max_episodes:5}",
            f"lr {self.learning_rate:5.0e}",
            f"wd {self.weight_decay:5.0e}",
            f"v {self.model.vision_dimensions}",
            f"v_dr {self.model.vision_dropout:3}",
            f"d {self.model.decision_dimensions}",
            f"d_dr {self.model.decision_dropout:3}",
            f"d_rsdl {self.model.decision_residual}",
            timestamp(),
        ))
        return result


class Trainer:
    def __init__(self, path: str):
        self._keyboard = Keyboard()
        self._display = Display(self._keyboard)
        self._out_path = os.path.join(path, timestamp())

    def run_hyper_params_list(self, hyper_param_list: List[HyperParams]):
        hyper_params_list_metrics = Metrics()
        for index, hyper_params in enumerate(hyper_param_list, start=1):
            hyper_params_metrics = Metrics()
            start = time.time()
            with capture_stdout() as buffer:  # Duplicates all print() output also into buffer
                print(
                    f"{timestamp()}: Starting hyper params {index}/{len(hyper_param_list)}. "
                    f"Params: {json.dumps(unstructure(hyper_params))}"
                )
                out_filepaths = self._run_hyper_params(hyper_params, hyper_params_metrics, hyper_params_list_metrics)

                label = hyper_params.to_label(hyper_params_metrics, took := (time.time() - start))
                print(
                    f"{timestamp()}: Finished hyper params {index}/{len(hyper_param_list)}. "
                    f"Took: {took:0.1f}s. "
                    f"Label: '{label}'"
                )

            if not hyper_params.dry_run:
                log_filepath = save_file(self._out_path, f"log.txt", buffer.get())
                move_files(out_filepaths + [log_filepath], os.path.join(self._out_path, label))

    def _run_hyper_params(
            self,
            hyper_params: HyperParams,
            hyper_params_metrics: Metrics,
            hyper_param_list_metrics: Metrics,
    ) -> List[str]:
        file_paths = []
        model = RlModel(
            module=to_device(SelfDrivingCarModel(hyper_params.model)),
            dry_run=hyper_params.dry_run,
            learning_rate=hyper_params.learning_rate,
            weight_decay=hyper_params.weight_decay,
        )
        for epoch in range(hyper_params.epochs):
            epoch_start = time.time()
            epoch_observations: List[List[Tensor]] = []
            epoch_actions: List[List[int]] = []
            epoch_rewards: List[List[float]] = []
            epoch_weights: List[List[float]] = []

            environment = RlEnvironment(hyper_params.environment_reset_mode)

            for batch in range(hyper_params.max_batches):
                batch_observations: List[Tensor] = []
                batch_actions: List[int] = []
                batch_rewards: List[float] = []

                state, observation = environment.reset()

                for episode in range(hyper_params.max_episodes):
                    self._keyboard.step()
                    self._display.step(state, observation, epoch, batch, episode)

                    observation = self._to_observation_tensors(observation)
                    batch_observations += [observation.clone()]

                    human_action = self._get_human_action()
                    action = model.act(observation) if human_action is None else human_action
                    batch_actions += [action]

                    state, observation, reward, batch_done = environment.step(Action(action))
                    batch_rewards += [reward]

                    if batch_done or self._keyboard.is_pressed([pygame.K_b, pygame.K_e, pygame.K_s]):
                        break

                epoch_observations += [batch_observations]
                epoch_actions += [batch_actions]
                epoch_rewards += [batch_rewards]
                epoch_weights += [self._compute_batch_weights(batch_observations, batch_actions, batch_rewards)]
                if self._keyboard.is_pressed([pygame.K_e, pygame.K_s]):
                    break

            epoch_return = float(np.mean([sum(batch) for batch in epoch_rewards]))
            epoch_loss = model.backprop(
                [el for batch in epoch_observations for el in batch],
                [el for batch in epoch_actions for el in batch],
                [el for batch in epoch_weights for el in batch],
            )
            hyper_params_metrics.update(epoch_return, epoch_loss)
            hyper_param_list_metrics.update(epoch_return, epoch_loss)

            print(" | ".join((
                f"{timestamp()}",
                f"epoch {epoch :4}",
                f"return {epoch_return :5.0f} -> {hyper_params_metrics.max_return:5.0f} -> {hyper_param_list_metrics.max_return:5.0f}",
                f"loss {epoch_loss :5.0f} -> {hyper_params_metrics.max_loss:5.0f} -> {hyper_param_list_metrics.max_loss:5.0f}",
                f"took {(time.time() - epoch_start) :5.1f}s",
            )))
            if not hyper_params.dry_run and epoch_return >= hyper_params.epoch_state_return_threshold:
                filename = f"state_epoch{epoch}_return{epoch_return:.0f}.pth"
                file_paths.append(save_state(self._out_path, filename, model.model))
            if self._keyboard.is_pressed([pygame.K_s]):
                break
        return file_paths

    @staticmethod
    def _compute_batch_weights(
            observations: List[Tensor],
            actions: List[int],
            rewards: List[float],
    ) -> List[float]:
        assert len(observations) == len(actions) and len(actions) == len(rewards)

        def get_penalty(observation_: Tensor, action_: int) -> Optional[float]:
            if torch.equal(observation_, observations[-1]) and action_ == actions[-1]:
                return -CAR_MAX_SPEED * 2
            elif torch.equal(observation_, observations[-2]) and action_ == actions[-2]:
                return -CAR_MAX_SPEED
            else:
                return None

        weights = []
        for index, (observation, action, reward) in enumerate(zip(observations, actions, rewards)):
            penalty = get_penalty(observation, action)
            if penalty is None:
                weights.append(reward)
            else:
                weights.append(penalty)
        return weights

    @staticmethod
    def _to_observation_tensors(observation: Observation) -> Tensor:
        def surface_to_tensor(surface: Surface) -> Tensor:
            flat = np.frombuffer(pygame.image.tostring(surface, "RGB"), dtype=np.uint8)
            shape = (surface.get_height(), surface.get_width(), 3)
            as_array = np.transpose(flat.reshape(shape), (2, 1, 0))  # (3, W, H)
            return torch.from_numpy(as_array.copy())

        visual = surface_to_tensor(observation.view)
        visual_normalized = visual / 255.0
        _, height, width = visual.shape

        speed = torch.full((1, width, height), observation.car.speed)
        speed_normalized = (speed - CAR_MIN_SPEED) / (CAR_MAX_SPEED - CAR_MIN_SPEED)

        turn = torch.full((1, width, height), observation.car.turn)
        turn_normalized = (turn - CAR_MIN_TURN) / (CAR_MAX_TURN - CAR_MIN_TURN)

        result = torch.cat([visual_normalized, speed_normalized, turn_normalized], dim=0)
        return result

    def _get_human_action(self) -> Optional[int]:
        result: Optional[Action] = None
        if self._keyboard.is_pressed([pygame.K_UP]):
            result = Action.ACCELERATION
        elif self._keyboard.is_pressed([pygame.K_DOWN]):
            result = Action.DECELERATION
        elif self._keyboard.is_pressed([pygame.K_LEFT]):
            result = Action.LEFT
        elif self._keyboard.is_pressed([pygame.K_RIGHT]):
            result = Action.RIGHT
        elif self._keyboard.is_pressed([pygame.K_n]):
            result = Action.NONE
        return result.value if result else None
