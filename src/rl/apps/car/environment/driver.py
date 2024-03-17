from dataclasses import dataclass, field
from random import Random
from typing import TypeVar, Sequence

T = TypeVar("T")
_SEED = 11


@dataclass
class DriverState:
    random: Random = field(default_factory=lambda: Random(_SEED))

    def choose(self, options: Sequence[T]) -> T:
        return self.random.choice(options)
