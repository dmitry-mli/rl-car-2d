import os
from typing import List, Sequence

import pygame

from rl.apps.car.common.constants import OBSERVATION_INPUT_SIDE
from rl.apps.car.environment.car import Action
from rl.apps.car.environment.rl import RlEnvironmentMode
from rl.apps.car.helpers.trainer import HyperParams, Trainer
from rl.apps.car.model.model import SelfDrivingCarModelParams


def run_training_plan():
    def visual_activation_flat(image_dimensions: Sequence[int]) -> int:
        side = OBSERVATION_INPUT_SIDE / (2 ** (len(image_dimensions) - 2))
        return int(image_dimensions[-1] * side * side)

    hyper_param_list: List[HyperParams] = [
        HyperParams(
            dry_run=False,
            epochs=epoch,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_batches=max_batches,
            max_episodes=max_episodes,
            environment_mode=environment_mode,
            model=SelfDrivingCarModelParams(
                vision_dimensions=vision_dimensions,
                vision_dropout=vision_dropout,
                decision_dimensions=[
                    visual_activation_flat(vision_dimensions),
                    *decision_hiddens,
                    len(Action),
                ],
                decision_residual=decision_residual,
                decision_dropout=decision_dropout,
                # state_path="../../../../resources/rl/apps/car/out/2024-01-01T12-45-42/ret   167 | loss     4 | took  26.9m | lr 5e-05 | wd 0e+00 | e  500 | max_b  50 | max_ep 10000 | drpt 0.4 | rsdl     1 | vis_dim [5, 8, 10, 12, 16, 32] | dec_dim [2048, 1024, 512, 256, 128, 5] | 2024-01-01T16-48-55/state_epoch487_return167.pth",
            ),
            epoch_state_return_threshold=100,
        )
        # Control
        for attempt in range(3)

        # Hyperparams
        for epoch in [300]
        for learning_rate in [
            # 5e-04,
            # 1e-04,
            5e-05,
            1e-05,
        ]
        for weight_decay in [
            # 0,
            1e-5,
        ]
        for max_batches in [60]
        for max_episodes in [10000]

        # Environment
        for environment_mode in [RlEnvironmentMode.ORDERED_WITH_CRASH_REPLAY]

        # Model
        for vision_dropout in [
            0.0,
            # 0.3,
            # 0.4,
            # 0.5,
        ]
        for decision_dropout in [
            # 0.0,
            0.3,
            # 0.4,
            # 0.5,
        ]
        for decision_residual in [
            # False,
            True,
        ]
        for vision_dimensions in [
            # [5, 8, 12, 16, 32, 64],
            [5, 8, 10, 12, 16, 32], # default
            # [5, 8, 10, 12, 16],
            # [5, 8, 10, 16, 32],
        ]  # each hidden reduces w and h by 2
        for decision_hiddens in [
            # [128],
            # [256, 128],
            # [512, 256, 128],
            [1024, 512, 256, 128],  # default
        ]
    ]

    trainer = Trainer(path="../../../../resources/rl/apps/car/out")
    trainer.run_hyper_params_list(hyper_param_list)


os.environ["SDL_VIDEO_WINDOW_POS"] = "0,0"  # Open window in top left corner
pygame.init()
run_training_plan()
pygame.quit()
