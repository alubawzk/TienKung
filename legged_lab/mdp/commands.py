# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.commands import UniformVelocityCommand
from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.utils import configclass


@configclass
class SuddenStopVelocityCommandCfg(UniformVelocityCommandCfg):
    """Extends UniformVelocityCommandCfg with a sudden-stop probability.

    rel_sudden_stop_envs applies only on mid-episode resamples (timer-triggered)
    for environments that were previously moving. It is independent of and stacks
    on top of rel_standing_envs.

    Episode-reset resamples are unaffected because _was_moving is cleared on reset.
    """

    class_type: type = None  # set after class definition below

    rel_sudden_stop_envs: float = 0.0
    """Probability of forcing zero command when a moving env is resampled mid-episode.
    Range [0, 1]. Default 0 means no extra stops beyond rel_standing_envs.
    """

    dead_zone_vel: float = 0.0
    """Commands whose L2 norm is below this threshold are snapped to zero.
    Applied every step after heading/standing post-processing. Default 0 disables.
    """


class SuddenStopVelocityCommand(UniformVelocityCommand):
    """UniformVelocityCommand with sudden-stop sampling.

    On every mid-episode resample (timer expiry), envs that were actively
    moving are given an additional ``rel_sudden_stop_envs`` probability of
    having their command forced to zero, simulating an abrupt stop command.

    The ``rel_standing_envs`` probability from the parent class is applied
    to ALL resamples (including episode start); ``rel_sudden_stop_envs``
    is applied only mid-episode to previously-moving envs.
    """

    cfg: SuddenStopVelocityCommandCfg

    def __init__(self, cfg: SuddenStopVelocityCommandCfg, env):
        super().__init__(cfg, env)
        # Tracks which envs had non-zero command at the previous step.
        # Initialised to False so episode-start resamples don't trigger sudden-stop.
        self._was_moving = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------
    # CommandTerm overrides
    # ------------------------------------------------------------------

    def reset(self, env_ids: Sequence[int]):
        """Reset: clear _was_moving so episode-start resamples skip sudden-stop."""
        self._was_moving[env_ids] = False
        return super().reset(env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        # Let the parent sample vx, vy, wz, heading, is_standing_env normally.
        super()._resample_command(env_ids)

        # Apply extra sudden-stop only to envs that were moving before this resample.
        if self.cfg.rel_sudden_stop_envs > 0.0:
            was_moving = self._was_moving[env_ids]
            sudden_stop = was_moving & (
                torch.rand(len(env_ids), device=self.device) < self.cfg.rel_sudden_stop_envs
            )
            # Force standing for those envs (OR-assign so parent's is_standing_env is preserved).
            self.is_standing_env[env_ids] = self.is_standing_env[env_ids] | sudden_stop

    def _update_command(self):
        """Post-process commands and update _was_moving for next resample."""
        super()._update_command()
        # Dead zone: snap to zero when linear speed (vx, vy) is below threshold.
        # Angular velocity (wz) is excluded so pure turning commands are not affected.
        if self.cfg.dead_zone_vel > 0.0:
            low_speed = torch.norm(self.vel_command_b[:, :2], dim=-1) < self.cfg.dead_zone_vel
            self.vel_command_b[low_speed] = 0.0
        # Record moving state AFTER dead zone and standing zeroing.
        self._was_moving = torch.norm(self.vel_command_b, dim=-1) > 0.01


SuddenStopVelocityCommandCfg.class_type = SuddenStopVelocityCommand
