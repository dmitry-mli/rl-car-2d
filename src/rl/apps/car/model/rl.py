from typing import List

import torch
from torch import nn, Tensor
from torch.distributions import Categorical
from torch.optim import Adam

from rl.apps.car.utils.device import to_device


class RlModel:
    def __init__(
            self,
            module: nn.Module,
            dry_run: bool,
            learning_rate: float,
            weight_decay: float,
    ):
        self.model = to_device(module)
        self.dry_run = dry_run
        self.optimizer = Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def act(self, observations: Tensor) -> int:
        result = self._get_policy(observations.unsqueeze(0), False).sample().item()
        return result

    def backprop(self, observations: List[Tensor], actions: List[float], weights: List[float]) -> float:
        loss = self._compute_loss(
            observations=torch.stack(observations),
            actions=torch.flatten(torch.as_tensor(actions, dtype=torch.int32)),
            weights=torch.flatten(torch.as_tensor(weights, dtype=torch.float32))
        )
        if not self.dry_run:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return float(loss)

    def _get_policy(self, observations: Tensor, training: bool) -> Categorical:
        self.model.train(training)
        logits = self.model(observations)
        return Categorical(logits=logits)

    def _compute_loss(self, observations: Tensor, actions: Tensor, weights: Tensor) -> Tensor:
        policy_output = self._get_policy(observations, True)
        log_p = policy_output.log_prob(to_device(actions))
        log_p_loss = -(log_p * to_device(weights)).mean()
        return log_p_loss
