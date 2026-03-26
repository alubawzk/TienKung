# cat_traverse_flow_mimic_env.py
#
# CatTraverseFlowMimicEnv — RL TRAINING env where the policy outputs 29
# absolute joint position targets (matching PLANNER_JOINT_NAMES order).
#
# The env's compute_observations() returns flat tensors compatible with
# FlowMimicActorCritic (TransformerTeacherPolicy):
#
#   actor  obs: (N, SEQ_LEN * D_ACTOR_HISTORY  + D_ACTOR_FUTURE)  = (N, 2230)
#   critic obs: (N, SEQ_LEN * D_CRITIC_HISTORY + D_CRITIC_FUTURE) = (N, 3490)
#
# The history buffers are maintained as (N, SEQ_LEN, D) tensors, updated
# after every physics step.
#
# ─────────────────────────────────────────────────────────────────────────────
# Obs layout per history frame:
#
#   actor_history frame (D_ACTOR_HISTORY = 160 dims):
#     [ 0:  3]  angular velocity (pelvis body frame)
#     [ 3:  6]  projected gravity vector
#     [ 6: 29]  joint_pos - default_joint_pos  (obs_joint_ids, 23 joints)
#     [29: 52]  joint_vel - default_joint_vel  (obs_joint_ids, 23 joints)
#     [52: 64]  last planner action — first 12 joints (leg joints)
#     [64: 76]  motor_targets — first 12 planner joint targets
#     [76: 79]  velocity command (nav frame)
#     [79: 80]  foot_height_target
#     [80: 84]  gait_phase (cos+sin × 2 feet)
#     [84:160]  actor PF obs (76 dims): delay_gf_nav + actor_bf_nav + actor_sdf
#
#   critic_history frame (D_CRITIC_HISTORY = 286 dims):
#     extends actor frame with privileged info (lin_vel, body positions,
#     domain-rand scales, contact forces, …)
#
#   actor_future / critic_future (D_ACTOR_FUTURE = D_CRITIC_FUTURE = 630 dims):
#     current undelayed PF obs in world frame + body positions & velocities
#
# !! The layouts above are best-effort reconstructions.  Use _match_dim()
# !! safety padding/truncation to survive minor dimension mismatches.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.buffers import DelayBuffer

from .cat_traverse_env import CatTraverseEnv
from .cat_traverse_g1_wbc_cfg import CatTraverseG1WbcEnvCfg, PLANNER_JOINT_NAMES
from .flow_mimic_policy import (
    D_ACTOR_HISTORY,
    D_ACTOR_FUTURE,
    D_CRITIC_HISTORY,
    D_CRITIC_FUTURE,
    SEQ_LEN,
    NUM_ACTIONS,
)


class CatTraverseFlowMimicEnv(CatTraverseEnv):
    """CAT traversal env for RL training with TransformerTeacherPolicy.

    Policy outputs 29 absolute joint position targets (PLANNER_JOINT_NAMES
    order).  The env assembles the structured obs expected by
    FlowMimicActorCritic and maintains SEQ_LEN-frame history buffers.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, cfg: CatTraverseG1WbcEnvCfg, headless: bool):
        self._planner_joint_ids: list[int] = []
        self._actor_history_buf: torch.Tensor | None = None
        self._critic_history_buf: torch.Tensor | None = None
        super().__init__(cfg, headless)

    # ------------------------------------------------------------------
    # Buffer initialisation
    # ------------------------------------------------------------------

    def init_buffers(self):
        # 1. Run base CAT init.
        super().init_buffers()

        # 2. Override num_actions to 29 (planner joints).
        if self.num_actions != NUM_ACTIONS:
            self.num_actions = NUM_ACTIONS
            max_delay = self.cfg.domain_rand.action_delay.params["max_delay"]
            self.action_buffer = DelayBuffer(max_delay, self.num_envs, device=self.device)
            self.action_buffer.compute(
                torch.zeros(
                    self.num_envs, NUM_ACTIONS,
                    dtype=torch.float, device=self.device, requires_grad=False,
                )
            )
            if self.cfg.domain_rand.action_delay.enable:
                time_lags = torch.randint(
                    low=self.cfg.domain_rand.action_delay.params["min_delay"],
                    high=max_delay + 1,
                    size=(self.num_envs,),
                    dtype=torch.int,
                    device=self.device,
                )
                self.action_buffer.set_time_lag(
                    time_lags, torch.arange(self.num_envs, device=self.device)
                )
            # Reinit obs / noise buffers with the correct action dim.
            self.init_obs_buffer()

        # 3. Resolve planner joint IDs (IsaacLab indices in planner order).
        self._resolve_planner_joint_ids()

        # 4. Allocate history buffers.
        self._actor_history_buf = torch.zeros(
            self.num_envs, SEQ_LEN, D_ACTOR_HISTORY, device=self.device
        )
        self._critic_history_buf = torch.zeros(
            self.num_envs, SEQ_LEN, D_CRITIC_HISTORY, device=self.device
        )

    def _resolve_planner_joint_ids(self):
        """Resolve IsaacLab joint indices in planner_joint_names order."""
        joint_cfg = SceneEntityCfg(
            name="robot",
            joint_names=list(PLANNER_JOINT_NAMES),
            preserve_order=True,
        )
        joint_cfg.resolve(self.scene)
        self._planner_joint_ids = [
            int(jid.item()) if hasattr(jid, "item") else int(jid)
            for jid in joint_cfg.joint_ids
        ]

    # ------------------------------------------------------------------
    # Observation builders
    # ------------------------------------------------------------------

    def _build_actor_history_frame(self) -> torch.Tensor:
        """Build one actor history frame: (N, D_ACTOR_HISTORY=160)."""
        field = self._ensure_cat_state()

        ang_vel      = self.robot.data.root_ang_vel_b                      # (N, 3)
        proj_grav    = self.robot.data.projected_gravity_b                 # (N, 3)
        joint_pos    = (
            self.robot.data.joint_pos - self.robot.data.default_joint_pos
        )[:, self._obs_joint_ids]                                          # (N, 23)
        joint_vel    = (
            self.robot.data.joint_vel - self.robot.data.default_joint_vel
        )[:, self._obs_joint_ids]                                          # (N, 23)

        # Last action: first 12 planner joints (leg joints).
        last_act_raw = self.action_buffer._circular_buffer.buffer[:, -1, :]  # (N, 29)
        last_act     = last_act_raw[:, :12]                                # (N, 12)
        motor_tgt    = self._motor_targets[:, self._planner_joint_ids[:12]]# (N, 12)

        command      = field["command_actor_nav"][:, :3] * self.obs_scales.commands  # (N, 3)
        foot_height  = self._foot_height_target                            # (N, 1)
        gait_phase   = torch.cat(
            (torch.cos(self._gait_phase), torch.sin(self._gait_phase)), dim=-1
        )                                                                  # (N, 4)
        actor_pf     = self._flatten_pf_groups(
            field["delay_gf_nav"], field["actor_bf_nav"], field["actor_sdf"]
        )                                                                  # (N, pf_dim)

        frame = torch.cat([
            ang_vel,      # 3
            proj_grav,    # 3
            joint_pos,    # 23
            joint_vel,    # 23
            last_act,     # 12
            motor_tgt,    # 12
            command,      # 3
            foot_height,  # 1
            gait_phase,   # 4
            actor_pf,     # pf_dim
        ], dim=-1)

        return _match_dim(frame, D_ACTOR_HISTORY, "actor_history_frame")

    def _build_critic_history_frame(self) -> torch.Tensor:
        """Build one critic history frame: (N, D_CRITIC_HISTORY=286)."""
        field = self._ensure_cat_state()

        ang_vel      = self.robot.data.root_ang_vel_b
        proj_grav    = self.robot.data.projected_gravity_b
        joint_pos    = (
            self.robot.data.joint_pos - self.robot.data.default_joint_pos
        )[:, self._obs_joint_ids]
        joint_vel    = (
            self.robot.data.joint_vel - self.robot.data.default_joint_vel
        )[:, self._obs_joint_ids]
        last_act_raw = self.action_buffer._circular_buffer.buffer[:, -1, :]
        last_act     = last_act_raw[:, :12]
        motor_tgt    = self._motor_targets[:, self._planner_joint_ids[:12]]
        command      = field["command_current_world"][:, :3] * self.obs_scales.commands
        foot_height  = self._foot_height_target
        gait_phase   = torch.cat(
            (torch.cos(self._gait_phase), torch.sin(self._gait_phase)), dim=-1
        )
        critic_pf    = self._flatten_pf_groups(
            field["current_gf_world"], field["current_bf_world"], field["current_sdf"]
        )
        lin_vel      = self.robot.data.root_lin_vel_b * self.obs_scales.lin_vel
        head_pos_w   = self._group(field["positions_w"], "head").reshape(self.num_envs, -1)
        head_vel_w   = self._group(field["velocities_w"], "head").reshape(self.num_envs, -1)
        pelv_pos_w   = self._group(field["positions_w"], "pelvis").reshape(self.num_envs, -1)
        tors_pos_w   = self._group(field["positions_w"], "torso").reshape(self.num_envs, -1)
        feet_pos_w   = self._group(field["positions_w"], "feet").reshape(self.num_envs, -1)
        feet_vel_w   = self._group(field["velocities_w"], "feet").reshape(self.num_envs, -1)
        hands_pos_w  = self._group(field["positions_w"], "hands").reshape(self.num_envs, -1)
        hands_vel_w  = self._group(field["velocities_w"], "hands").reshape(self.num_envs, -1)
        knees_pos_w  = self._group(field["positions_w"], "knees").reshape(self.num_envs, -1)
        shlds_pos_w  = self._group(field["positions_w"], "shoulders").reshape(self.num_envs, -1)
        torso_nav_rpy = field["torso_nav_rpy"][:, :2]
        gait_mask    = self._gait_mask
        net_cf       = self.contact_sensor.data.net_forces_w_history
        feet_contact = (
            torch.max(torch.norm(net_cf[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5
        ).float()
        kp_scale     = self._kp_scale
        kd_scale     = self._kd_scale
        rfi_scale    = self._rfi_lim_scale[:, self._planner_joint_ids[:12]]

        frame = torch.cat([
            ang_vel, proj_grav,
            joint_pos, joint_vel,
            last_act, motor_tgt,
            command, foot_height, gait_phase,
            critic_pf,
            lin_vel,
            head_pos_w, head_vel_w,
            pelv_pos_w, tors_pos_w,
            feet_pos_w, feet_vel_w,
            hands_pos_w, hands_vel_w,
            knees_pos_w, shlds_pos_w,
            torso_nav_rpy,
            gait_mask,
            feet_contact,
            kp_scale, kd_scale,
            rfi_scale,
        ], dim=-1)

        return _match_dim(frame, D_CRITIC_HISTORY, "critic_history_frame")

    def _build_future_obs(self) -> torch.Tensor:
        """Build future/goal obs: (N, D_ACTOR_FUTURE=630).

        This is the same for both actor and critic (same D_AF = D_CF = 630).
        """
        field = self._ensure_cat_state()

        full_pf     = self._flatten_pf_groups(
            field["current_gf_world"], field["current_bf_world"], field["current_sdf"]
        )
        head_pos_w  = self._group(field["positions_w"], "head").reshape(self.num_envs, -1)
        head_vel_w  = self._group(field["velocities_w"], "head").reshape(self.num_envs, -1)
        pelv_pos_w  = self._group(field["positions_w"], "pelvis").reshape(self.num_envs, -1)
        tors_pos_w  = self._group(field["positions_w"], "torso").reshape(self.num_envs, -1)
        feet_pos_w  = self._group(field["positions_w"], "feet").reshape(self.num_envs, -1)
        feet_vel_w  = self._group(field["velocities_w"], "feet").reshape(self.num_envs, -1)
        hands_pos_w = self._group(field["positions_w"], "hands").reshape(self.num_envs, -1)
        hands_vel_w = self._group(field["velocities_w"], "hands").reshape(self.num_envs, -1)
        knees_pos_w = self._group(field["positions_w"], "knees").reshape(self.num_envs, -1)
        shlds_pos_w = self._group(field["positions_w"], "shoulders").reshape(self.num_envs, -1)

        future = torch.cat([
            full_pf,
            head_pos_w, head_vel_w,
            pelv_pos_w, tors_pos_w,
            feet_pos_w, feet_vel_w,
            hands_pos_w, hands_vel_w,
            knees_pos_w, shlds_pos_w,
        ], dim=-1)

        return _match_dim(future, D_ACTOR_FUTURE, "future_obs")

    # ------------------------------------------------------------------
    # Observations  (override base)
    # ------------------------------------------------------------------

    def compute_observations(self):
        """Return (actor_obs, critic_obs) with flow_mimic obs layout.

        actor  obs shape: (N, SEQ_LEN * D_ACTOR_HISTORY  + D_ACTOR_FUTURE)
        critic obs shape: (N, SEQ_LEN * D_CRITIC_HISTORY + D_CRITIC_FUTURE)
        """
        future = self._build_future_obs()  # (N, 630) — shared by actor & critic

        actor_obs = torch.cat([
            self._actor_history_buf.reshape(self.num_envs, -1),   # (N, 1600)
            future,                                                # (N, 630)
        ], dim=-1)  # (N, 2230)
        actor_obs = torch.nan_to_num(actor_obs, nan=0.0, posinf=0.0, neginf=0.0)

        critic_obs = torch.cat([
            self._critic_history_buf.reshape(self.num_envs, -1),  # (N, 2860)
            future,                                                # (N, 630)
        ], dim=-1)  # (N, 3490)
        critic_obs = torch.nan_to_num(critic_obs, nan=0.0, posinf=0.0, neginf=0.0)

        # Populate cat_traverse extras for logging (same as base env).
        if self.latest_field_cache:
            self.extras["cat_traverse"] = {
                "field_dir":       self.latest_field_cache["field_dir"],
                "obstacle_asset":  self.latest_field_cache["obstacle_asset"],
                "min_sdf":         self.latest_field_cache["current_sdf"].amin(dim=1),
                "min_sdf_delay":   self.latest_field_cache["delay_sdf"].amin(dim=1),
                "command_nav":     self.latest_field_cache["command_current_nav"],
                "policy_command_nav": self.latest_field_cache["command_actor_nav"],
                "stop_timestep":   self.latest_field_cache["stop_timestep"],
            }

        return actor_obs, critic_obs

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor):
        super().reset(env_ids)
        if len(env_ids) == 0:
            return
        if self._actor_history_buf is not None:
            self._actor_history_buf[env_ids] = 0.0
        if self._critic_history_buf is not None:
            self._critic_history_buf[env_ids] = 0.0

    # ------------------------------------------------------------------
    # Step  (training: actions are 29-dim absolute joint targets)
    # ------------------------------------------------------------------

    def step(self, actions: torch.Tensor):
        # ---- 1. Delay & clip ------------------------------------------------
        delayed_actions = self.action_buffer.compute(actions)
        clipped_actions = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(
            self.device
        )

        # ---- 2. Write 29 absolute joint targets into _motor_targets ---------
        self._motor_targets.copy_(self.robot.data.default_joint_pos)
        soft_lower = self.robot.data.soft_joint_pos_limits[:, self._planner_joint_ids, 0]
        soft_upper = self.robot.data.soft_joint_pos_limits[:, self._planner_joint_ids, 1]
        joint_targets = torch.maximum(clipped_actions, soft_lower)
        joint_targets = torch.minimum(joint_targets, soft_upper)
        self._motor_targets[:, self._planner_joint_ids] = joint_targets

        # ---- 3. Physics step ------------------------------------------------
        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            if self.cfg.dm_rand.enable_rfi:
                rfi_noise = (2.0 * torch.rand_like(self._rfi_lim_scale) - 1.0) * self._rfi_lim_scale
            else:
                rfi_noise = torch.zeros_like(self._rfi_lim_scale)
            self._set_joint_effort_target(rfi_noise)
            self.robot.set_joint_position_target(self._motor_targets)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        if not self.headless:
            self.sim.render()

        # ---- 4. Post-step bookkeeping ---------------------------------------
        self.episode_length_buf += 1
        self._update_gait_phase()
        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self._cat_state_step = -1
        self.latest_field_cache = {}
        self.reset_buf, self.time_out_buf = self.check_reset()
        reward_buf = self.reward_manager.compute(self.step_dt)
        reward_buf = torch.clamp(reward_buf, min=0.0, max=10000.0)
        self._last_joint_vel.copy_(self.robot.data.joint_vel)

        # ---- 5. Update history buffers with current state -------------------
        actor_frame  = self._build_actor_history_frame()   # (N, D_AH)
        critic_frame = self._build_critic_history_frame()  # (N, D_CH)

        self._actor_history_buf  = torch.roll(self._actor_history_buf,  shifts=-1, dims=1)
        self._actor_history_buf[:, -1, :]  = actor_frame

        self._critic_history_buf = torch.roll(self._critic_history_buf, shifts=-1, dims=1)
        self._critic_history_buf[:, -1, :] = critic_frame

        # ---- 6. Reset terminated envs --------------------------------------
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(env_ids)

        # ---- 7. Observations -----------------------------------------------
        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        return actor_obs, reward_buf, self.reset_buf, self.extras


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def _match_dim(tensor: torch.Tensor, target: int, name: str = "") -> torch.Tensor:
    """Pad (with zeros) or truncate last dim to exactly *target* dims."""
    actual = tensor.shape[-1]
    if actual == target:
        return tensor
    import warnings
    if actual < target:
        warnings.warn(
            f"[CatTraverseFlowMimicEnv] {name}: dim={actual} < {target}, "
            f"padding {target - actual} zeros.",
            stacklevel=2,
        )
        pad = torch.zeros(*tensor.shape[:-1], target - actual,
                          device=tensor.device, dtype=tensor.dtype)
        return torch.cat([tensor, pad], dim=-1)
    warnings.warn(
        f"[CatTraverseFlowMimicEnv] {name}: dim={actual} > {target}, "
        f"truncating {actual - target} dims.",
        stacklevel=2,
    )
    return tensor[..., :target]
