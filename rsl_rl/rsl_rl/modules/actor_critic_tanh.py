# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
#
# This file adds a tanh-bounded policy head for CAT-style action output.

from __future__ import annotations

import torch

from .actor_critic import ActorCritic


class ActorCriticTanh(ActorCritic):
    def _bounded_mean(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.actor(observations))

    def update_distribution(self, observations):
        mean = self._bounded_mean(observations)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = torch.distributions.Normal(mean, std)

    def act_inference(self, observations):
        return self._bounded_mean(observations)
