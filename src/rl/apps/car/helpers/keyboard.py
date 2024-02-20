import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

import pygame


@dataclass
class _KeyState:
    remaining: int
    last_pressed: float
    timeout: int


class Keyboard:
    def __init__(self):
        self._timeouts = {
            # Actions
            pygame.K_UP: 0.25,
            pygame.K_DOWN: 0.25,
            pygame.K_LEFT: 0,
            pygame.K_RIGHT: 0,
            pygame.K_n: 0,

            # Configuration
            pygame.K_r: 1,  # Toggle rendering
            pygame.K_f: 1,  # Toggle fast rendering

            # Training
            pygame.K_e: 1,  # Finish episode
            pygame.K_b: 1,  # Finish batch
            pygame.K_s: 1,  # Finish session
        }
        self._last_pressed = defaultdict(float)
        self._pressed_keys = []

    def step(self):
        now = time.time()
        pressed_keys = pygame.key.get_pressed()
        self._pressed_keys.clear()

        for key, timeout in self._timeouts.items():
            if pressed_keys[key] and now > (self._last_pressed[key] + timeout):
                self._pressed_keys.append(key)
                self._last_pressed[key] = now

    def is_pressed(self, either_key: Iterable[int]) -> bool:
        return any(key in self._pressed_keys for key in either_key)
