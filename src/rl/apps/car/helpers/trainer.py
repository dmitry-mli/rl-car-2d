import csv
import io
import json
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Collection

import numpy as np
import pygame
import torch
from cattr import unstructure
from pygame import Surface
from torch import Tensor

from rl.apps.car.common.constants import CAR_MIN_SPEED, CAR_MAX_SPEED, CAR_MIN_TURN, CAR_MAX_TURN
from rl.apps.car.environment.car import Action
from rl.apps.car.environment.environment import Observation
from rl.apps.car.environment.rl import RlEnvironmentMode, RlEnvironment
from rl.apps.car.helpers.display import Display
from rl.apps.car.helpers.keyboard import Keyboard
from rl.apps.car.model.model import SelfDrivingCarModelParams, SelfDrivingCarModel
from rl.apps.car.model.rl import RlModel
from rl.apps.car.utils.device import to_device
from rl.apps.car.utils.files import move_files, save_state, save_file
from rl.apps.car.utils.tee import capture_stdout
from rl.apps.car.utils.timestamp import get_timestamp


class Metrics:
    def __init__(self):
        self.max_reward = 0.
        self.max_loss = 0.
        self.improvements = 0
        self.took = 0.

    def update(self, reward: float, loss: float, improvements: int, took: float):
        self.max_reward = max(self.max_reward, reward)
        self.max_loss = max(self.max_loss, loss)
        self.improvements = max(self.improvements, improvements)
        self.took += took


class HyperParamsOutput(dict):
    def to_label(self, fields: Collection[str]) -> str:
        return " | ".join([f"{key} {self[key]}" for key in fields])

    @staticmethod
    def to_csv(outputs: List["HyperParamsOutput"]) -> str:
        with io.StringIO() as result:
            writer = csv.writer(result)
            writer.writerow(outputs[0].keys())
            for output in outputs:
                writer.writerow(output.values())
            return result.getvalue()


@dataclass
class HyperParams:
    dry_run: bool
    epochs: int
    learning_rate: float
    weight_decay: float
    max_batches: int
    max_episodes: int
    environment_mode: RlEnvironmentMode
    model: SelfDrivingCarModelParams
    epoch_state_reward_threshold: int

    def to_output(self, metrics: Metrics, timestamp: str) -> HyperParamsOutput:
        return HyperParamsOutput({
            "ts": timestamp,
            "ret": f"{metrics.max_reward:5.0f}",
            "imp": f"{metrics.improvements:5.0f}",
            "loss": f"{metrics.max_loss:5.0f}",
            "took": f"{metrics.took:5.1f}s",
            "e": f"{self.epochs:4}",
            "b": f"{self.max_batches:3}",
            "ep": f"{self.max_episodes:5}",
            "lr": f"{self.learning_rate:5.0e}",
            "wd": f"{self.weight_decay:5.0e}",
            "v": f"{self.model.vision_dimensions}",
            "v_dr": f"{self.model.vision_dropout:3}",
            "d": f"{self.model.decision_dimensions}",
            "d_dr": f"{self.model.decision_dropout:3}",
            "d_rsdl": f"{self.model.decision_residual}",
        })


class Trainer:
    def __init__(self, path: str):
        self._keyboard = Keyboard()
        self._display = Display(self._keyboard)
        self._out_path = os.path.join(path, get_timestamp())

    def run_hyper_params_list(self, hyper_param_list: List[HyperParams]):
        hyper_params_list_metrics = Metrics()
        outputs: List[HyperParamsOutput] = []

        for index, hyper_params in enumerate(hyper_param_list, start=1):
            hyper_params_metrics = Metrics()
            with capture_stdout() as buffer:  # Duplicates all print() output also into buffer
                print(
                    f"{get_timestamp()}: Starting hyper params {index}/{len(hyper_param_list)}. "
                    f"Params: {json.dumps(unstructure(hyper_params))}"
                )
                out_filepaths = self._run_hyper_params(hyper_params, hyper_params_metrics, hyper_params_list_metrics)

                output = hyper_params.to_output(hyper_params_metrics, get_timestamp())
                outputs.append(output)
                label = output.to_label(["ret", "imp", "ts"])
                print(f"{get_timestamp()}: Finished hyper params {index}/{len(hyper_param_list)}. Label: '{label}'")

            if not hyper_params.dry_run:
                # Session
                log_filepath = save_file(self._out_path, f"log.txt", buffer.get())
                move_files(out_filepaths + [log_filepath], os.path.join(self._out_path, label))

                # Summary
                save_file(self._out_path, f"summary.csv", HyperParamsOutput.to_csv(outputs))

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
        improvements = 0
        for epoch in range(hyper_params.epochs):
            epoch_start = time.time()
            epoch_observations: List[List[Tensor]] = []
            epoch_actions: List[List[int]] = []
            epoch_rewards: List[List[float]] = []
            epoch_weights: List[List[float]] = []

            environment = RlEnvironment(
                mode=hyper_params.environment_mode,
                total_resets=hyper_params.max_batches,
            )

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

            epoch_reward = float(np.mean([sum(batch) for batch in epoch_rewards]))
            epoch_loss = model.backprop(
                [el for batch in epoch_observations for el in batch],
                [el for batch in epoch_actions for el in batch],
                [el for batch in epoch_weights for el in batch],
            )
            epoch_took = time.time() - epoch_start

            improvements += 1 if (epoch_reward > hyper_params_metrics.max_reward) else 0
            epoch_metrics = Metrics()
            epoch_metrics.update(epoch_reward, epoch_loss, improvements, epoch_took)
            hyper_params_metrics.update(epoch_reward, epoch_loss, improvements, epoch_took)
            hyper_param_list_metrics.update(epoch_reward, epoch_loss, improvements, epoch_took)

            print(" | ".join((
                f"{get_timestamp()}",
                f"epoch {epoch :4}",
                f"imp {epoch_metrics.improvements :3.0f} -> {hyper_params_metrics.improvements:3.0f} -> {hyper_param_list_metrics.improvements:3.0f}",
                f"reward {epoch_metrics.max_reward :5.0f} -> {hyper_params_metrics.max_reward:5.0f} -> {hyper_param_list_metrics.max_reward:5.0f}",
                f"loss {epoch_metrics.max_loss :5.0f} -> {hyper_params_metrics.max_loss:5.0f} -> {hyper_param_list_metrics.max_loss:5.0f}",
                f"took {epoch_metrics.took:5.1f}s -> {hyper_params_metrics.took:5.1f}s -> {hyper_param_list_metrics.took:5.1f}s",
            )))
            if not hyper_params.dry_run and epoch_reward >= hyper_params.epoch_state_reward_threshold:
                filename = f"state_epoch{epoch}_reward{epoch_reward:.0f}.pth"
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
