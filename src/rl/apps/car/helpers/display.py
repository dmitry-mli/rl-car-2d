import os

import pygame
from pygame import Surface

from rl.apps.car.common.constants import DISPLAY_AREA, CANVAS_TO_DISPLAY_OFFSET, \
    CANVAS_AREA, FRAMES_PER_SECOND
from rl.apps.car.environment.environment import State, Observation
from rl.apps.car.helpers.canvas import draw_observation, draw_stats
from rl.apps.car.helpers.keyboard import Keyboard


class Display:
    def __init__(self, keyboard: Keyboard):
        self._display: Surface = pygame.display.set_mode(DISPLAY_AREA)
        self._clock = pygame.time.Clock()
        self._keyboard = keyboard
        self._render = os.environ.get("DISPLAY_RENDER", "false").lower() == "true"
        self._fast_render = os.environ.get("FAST_RENDER", "false").lower() == "true"

    def step(
            self,
            state: State,
            observation: Observation,
            epoch: int,
            batch: int,
            episode: int,
    ):
        if self._render:
            # Compose visible surface
            canvas: Surface = pygame.Surface(CANVAS_AREA)
            canvas.blit(state.view, (0, 0))
            draw_stats(canvas, state.car, self._fast_render, epoch, batch, episode)
            if not self._fast_render:
                draw_observation(canvas, observation.view)

            self._display.blit(canvas, CANVAS_TO_DISPLAY_OFFSET)
            pygame.display.flip()

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                exit(0)

        # Handle keyboard inputs
        if self._keyboard.is_pressed([pygame.K_r]):
            self._render = not self._render
        if self._keyboard.is_pressed([pygame.K_f]):
            self._fast_render = not self._fast_render

        # Tick
        if self._render and not self._fast_render:
            self._clock.tick(FRAMES_PER_SECOND)
