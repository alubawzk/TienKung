# cat_traverse_flow_mimic_env.py
#
# Hierarchical env for training a high-level RL policy that outputs 14-body
# poses, which are fed as actor_future[frame_0] into a frozen
# TransformerTeacherPolicy (flow_mimic.pt) to generate 29-dim joint targets.
#
# Data flow:
#   current state (cat_traverse obs)
#       │
#       ▼
#   RL Policy (trained)
#       │  actions: (N, 14*9=126)  — 14 body poses in anchor frame
#       │  body order: see transformer_teacher_dims.md
#       ▼
#   actor_future (N, 630):
#       frame_0[:126] = RL policy output
#       frames 1-4   = zeros  (variable-horizon N=1 masking)
#   actor_history (N, 10, 160): maintained rolling buffer, format matches
#       flow_mimic.pt training (see _build_actor_history_frame)
#       │
#       ▼
#   flow_mimic.pt  (FROZEN, TransformerTeacherPolicy)
#       │  joint_targets: (N, 29)
#       ▼
#   robot.set_joint_position_target()
#
# ──────────────────────────────────────────────────────────────────────────────
# actor_history per frame (160 dims) — strict match to flow_mimic.pt training:
#
#   [0:29]    command_joint_pos  = last flow_mimic targets  (reference)
#   [29:58]   command_joint_vel  = (last - prev) / dt
#   [58:61]   motion_anchor_pos_b = RL output pelvis pos  (body 0, dims [0:3])
#   [61:67]   motion_anchor_ori_b = RL output pelvis 6D   (body 0, dims [3:9])
#   [67:70]   base_lin_vel
#   [70:73]   base_ang_vel
#   [73:102]  joint_pos_rel      (29 joints, PLANNER order)
#   [102:131] joint_vel          (29 joints, PLANNER order)
#   [131:160] last_action        = last flow_mimic targets  (same as cmd)
#
# RL Policy obs (~150 dims, cat_traverse style):
#   ang_vel(3) + proj_gravity(3) + joint_pos_rel(23) + joint_vel(23)
#   + last_fm_leg_targets(12) + command(4) + foot_height(1) + gait_phase(4)
#   + actor_pf_obs(77)
# ──────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import os

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer

from .cat_traverse_env import CatTraverseEnv
from .cat_traverse_flow_mimic_cfg import CatTraverseFlowMimicEnvCfg
from .cat_traverse_g1_wbc_cfg import PLANNER_JOINT_NAMES
from .flow_mimic_policy import (
    D_ACTOR_FUTURE,
    D_ACTOR_HISTORY,
    SEQ_LEN,
    TransformerTeacherPolicy,
    load_flow_mimic,
)

# ── Architecture constants ───────────────────────────────────────────────────
N_TRACKED_BODIES: int = 14   # bodies tracked in actor_future
BODY_POSE_DIM:    int = 9    # pos(3) + 6D_rot(6) per body
RL_ACTION_DIM:    int = N_TRACKED_BODIES * BODY_POSE_DIM   # 126
FM_LEG_JOINTS:    int = 12   # first 12 joints of PLANNER_JOINT_NAMES are legs
FM_JOINT_DIM:     int = 29   # total joints output by flow_mimic.pt


class CatTraverseFlowMimicEnv(CatTraverseEnv):
    """CAT traversal env: RL policy (126-dim body poses) → frozen flow_mimic.pt
    → joint targets → robot.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, cfg: CatTraverseFlowMimicEnvCfg, headless: bool):
        self._flow_mimic_joint_ids:      list[int] = []
        self._actor_history_buf:         torch.Tensor | None = None
        self._last_flow_mimic_targets:   torch.Tensor | None = None
        self._prev_flow_mimic_targets:   torch.Tensor | None = None
        self._flow_mimic_policy:         TransformerTeacherPolicy | None = None
        super().__init__(cfg, headless)

    # ------------------------------------------------------------------
    # Buffer initialisation
    # ------------------------------------------------------------------

    def init_buffers(self):
        # 1. Base CAT env init (field_loader, probes, domain_rand, obs_buffer).
        #    At this point num_actions = len(G1_ACTION_JOINT_NAMES) = 12.
        #    init_obs_buffer() is called inside here — our override handles it.
        super().init_buffers()

        # 2. Override num_actions to RL_ACTION_DIM (126) and rebuild action buffer.
        self.num_actions = RL_ACTION_DIM
        max_delay = self.cfg.domain_rand.action_delay.params["max_delay"]
        self.action_buffer = DelayBuffer(max_delay, self.num_envs, device=self.device)
        self.action_buffer.compute(
            torch.zeros(
                self.num_envs, RL_ACTION_DIM,
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
            self.action_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))

        # 3. Resolve joint IDs in PLANNER_JOINT_NAMES order (= flow_mimic.pt order).
        self._resolve_flow_mimic_joint_ids()

        # 4. Allocate actor_history buffer and flow_mimic target tracking.
        self._actor_history_buf = torch.zeros(
            self.num_envs, SEQ_LEN, D_ACTOR_HISTORY, device=self.device
        )
        default_29 = self.robot.data.default_joint_pos[:, self._flow_mimic_joint_ids]
        self._last_flow_mimic_targets = default_29.clone()
        self._prev_flow_mimic_targets = default_29.clone()

        # 5. Rebuild obs buffer now that _last_flow_mimic_targets is initialised.
        self.init_obs_buffer()

        # 6. Load and freeze flow_mimic.pt.
        self._load_flow_mimic_policy()

    def init_obs_buffer(self):
        """Override: build noise vector matching our obs layout (not base cat obs).

        Our actor_obs layout:
          ang_vel(3) + proj_gravity(3) + joint_pos(obs_joint_dim)
          + joint_vel(obs_joint_dim) + last_fm_leg_targets(12)
          + command(4) + foot_height(1) + gait_phase(4) + actor_pf_obs(...)
        """
        if self.add_noise:
            # We need _obs_joint_ids already resolved (done before super().init_buffers()
            # calls us). Use a zero placeholder for _last_flow_mimic_targets if not yet ready.
            if self._last_flow_mimic_targets is None:
                _placeholder = torch.zeros(self.num_envs, FM_JOINT_DIM, device=self.device)
                self._last_flow_mimic_targets = _placeholder
                actor_obs, _ = self.compute_current_observations()
                self._last_flow_mimic_targets = None
            else:
                actor_obs, _ = self.compute_current_observations()

            obs_joint_dim = len(self._obs_joint_ids)
            noise_vec = torch.zeros_like(actor_obs[0])
            ns = self.cfg.noise.noise_scales

            cursor = 0
            noise_vec[cursor: cursor + 3]             = ns.ang_vel          * self.obs_scales.ang_vel
            cursor += 3
            noise_vec[cursor: cursor + 3]             = ns.projected_gravity * self.obs_scales.projected_gravity
            cursor += 3
            noise_vec[cursor: cursor + obs_joint_dim] = ns.joint_pos        * self.obs_scales.joint_pos
            cursor += obs_joint_dim
            noise_vec[cursor: cursor + obs_joint_dim] = ns.joint_vel        * self.obs_scales.joint_vel
            cursor += obs_joint_dim
            # last_fm_leg_targets (12), command (4), foot_height (1), gait_phase (4),
            # actor_pf_obs — all no noise, noise_vec already zero.
            self.noise_scale_vec = noise_vec

        self.actor_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.actor_obs_history_length,
            batch_size=self.num_envs,
            device=self.device,
        )
        self.critic_obs_buffer = CircularBuffer(
            max_len=self.cfg.robot.critic_obs_history_length,
            batch_size=self.num_envs,
            device=self.device,
        )

    def _resolve_flow_mimic_joint_ids(self):
        """Resolve IsaacLab joint indices in PLANNER_JOINT_NAMES order."""
        joint_cfg = SceneEntityCfg(
            name="robot",
            joint_names=list(PLANNER_JOINT_NAMES),
            preserve_order=True,
        )
        joint_cfg.resolve(self.scene)
        self._flow_mimic_joint_ids = [
            int(jid.item()) if hasattr(jid, "item") else int(jid)
            for jid in joint_cfg.joint_ids
        ]

    def _load_flow_mimic_policy(self):
        """Load flow_mimic.pt and freeze all parameters."""
        pt_path = self.cfg.flow_mimic_pt_path
        if not os.path.isabs(pt_path):
            import legged_lab as _ll
            root = os.path.dirname(os.path.dirname(os.path.abspath(_ll.__file__)))
            pt_path = os.path.join(root, pt_path)
        if not os.path.isfile(pt_path):
            raise FileNotFoundError(
                f"[CatTraverseFlowMimicEnv] flow_mimic.pt not found: '{pt_path}'"
            )
        self._flow_mimic_policy = load_flow_mimic(pt_path, device=self.device)
        self._flow_mimic_policy.eval()
        for p in self._flow_mimic_policy.parameters():
            p.requires_grad_(False)
        print(f"[CatTraverseFlowMimicEnv] Loaded frozen flow_mimic from {pt_path}")

    # ------------------------------------------------------------------
    # actor_history builder  (matches flow_mimic.pt training format exactly)
    # ------------------------------------------------------------------

    def _build_actor_history_frame(self, actions: torch.Tensor) -> torch.Tensor:
        """Build one 160-dim history frame for flow_mimic.pt.

        Args:
            actions: RL policy output this step (N, 126), clipped.
                     actions[:, 0:3]  = desired pelvis position (anchor frame)
                     actions[:, 3:9]  = desired pelvis 6D rotation (anchor frame)
        """
        # ── command: last flow_mimic targets as "reference motion" ────────────
        cmd_pos = self._last_flow_mimic_targets                               # (N, 29)
        cmd_vel = (self._last_flow_mimic_targets
                   - self._prev_flow_mimic_targets) / self.step_dt            # (N, 29)

        # ── motion anchor: desired pelvis pose from RL policy ─────────────────
        anchor_pos = actions[:, 0:3]    # (N, 3)  pelvis position offset
        anchor_ori = actions[:, 3:9]    # (N, 6)  pelvis 6D rotation

        # ── robot proprioception ──────────────────────────────────────────────
        lin_vel  = self.robot.data.root_lin_vel_b                             # (N, 3)
        ang_vel  = self.robot.data.root_ang_vel_b                             # (N, 3)
        jpos_rel = (
            self.robot.data.joint_pos - self.robot.data.default_joint_pos
        )[:, self._flow_mimic_joint_ids]                                      # (N, 29)
        jvel = self.robot.data.joint_vel[:, self._flow_mimic_joint_ids]       # (N, 29)

        frame = torch.cat([
            cmd_pos,     # 29  [0:29]
            cmd_vel,     # 29  [29:58]
            anchor_pos,  #  3  [58:61]
            anchor_ori,  #  6  [61:67]
            lin_vel,     #  3  [67:70]
            ang_vel,     #  3  [70:73]
            jpos_rel,    # 29  [73:102]
            jvel,        # 29  [102:131]
            cmd_pos,     # 29  [131:160]  last_action = same reference
        ], dim=-1)  # (N, 160)

        return torch.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)

    # ------------------------------------------------------------------
    # RL Policy observations  (override base cat obs)
    # ------------------------------------------------------------------

    def compute_current_observations(self):
        """Returns (actor_obs, critic_obs) for the RL policy.

        Reuses cat_traverse obs style but replaces last_action / motor_targets
        with last flow_mimic leg joint targets (12 dims).
        """
        field = self._ensure_cat_state()
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        # ── proprioception ────────────────────────────────────────────────────
        ang_vel = self.robot.data.root_ang_vel_b * self.obs_scales.ang_vel
        proj_grav = self.robot.data.projected_gravity_b * self.obs_scales.projected_gravity
        joint_pos = (
            (self.robot.data.joint_pos - self.robot.data.default_joint_pos)
            [:, self._obs_joint_ids] * self.obs_scales.joint_pos
        )
        joint_vel = (
            (self.robot.data.joint_vel - self.robot.data.default_joint_vel)
            [:, self._obs_joint_ids] * self.obs_scales.joint_vel
        )

        # Last flow_mimic leg targets as action reference (12 dims).
        # Fallback to zeros during init before targets are set.
        if self._last_flow_mimic_targets is not None:
            last_fm = self._last_flow_mimic_targets[:, :FM_LEG_JOINTS]        # (N, 12)
        else:
            last_fm = torch.zeros(self.num_envs, FM_LEG_JOINTS, device=self.device)

        # ── gait / contact ────────────────────────────────────────────────────
        gait_phase = torch.cat(
            (torch.cos(self._gait_phase), torch.sin(self._gait_phase)), dim=-1
        )                                                                      # (N, 4)
        feet_contact = (
            torch.max(
                torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1),
                dim=1,
            )[0] > 0.5
        ).float()                                                              # (N, 2)
        root_lin_vel_b = self.robot.data.root_lin_vel_b * self.obs_scales.lin_vel

        # ── PF obs ────────────────────────────────────────────────────────────
        actor_pf = self._flatten_pf_groups(
            field["delay_gf_nav"], field["actor_bf_nav"], field["actor_sdf"]
        )
        critic_pf = self._flatten_pf_groups(
            field["current_gf_world"], field["current_bf_world"], field["current_sdf"]
        )

        # ── body positions (critic privileged) ───────────────────────────────
        head_pos_w  = self._group(field["positions_w"], "head").reshape(self.num_envs, -1)
        head_vel_w  = self._group(field["velocities_w"], "head").reshape(self.num_envs, -1)
        pelv_pos_w  = self._group(field["positions_w"], "pelvis").reshape(self.num_envs, -1)
        tors_pos_w  = self._group(field["positions_w"], "torso").reshape(self.num_envs, -1)
        feet_pos_w  = self._group(field["positions_w"], "feet").reshape(self.num_envs, -1)
        feet_vel_w  = self._group(field["velocities_w"], "feet").reshape(self.num_envs, -1)
        hand_pos_w  = self._group(field["positions_w"], "hands").reshape(self.num_envs, -1)
        hand_vel_w  = self._group(field["velocities_w"], "hands").reshape(self.num_envs, -1)
        rfi_leg = self._rfi_lim_scale[:, self._flow_mimic_joint_ids[:FM_LEG_JOINTS]]

        # ── actor obs ─────────────────────────────────────────────────────────
        actor_obs = torch.cat([
            ang_vel,                                               # 3
            proj_grav,                                             # 3
            joint_pos,                                             # 23
            joint_vel,                                             # 23
            last_fm,                                               # 12
            field["command_actor_nav"] * self.obs_scales.commands, # 4
            self._foot_height_target,                              # 1
            gait_phase,                                            # 4
            actor_pf,                                              # 77
        ], dim=-1)                                                 # ≈ 150

        # ── critic obs (actor + privileged) ──────────────────────────────────
        critic_obs = torch.cat([
            ang_vel,
            proj_grav,
            joint_pos,
            joint_vel,
            last_fm,
            field["command_current_world"] * self.obs_scales.commands,
            self._foot_height_target,
            gait_phase,
            critic_pf,
            root_lin_vel_b,
            head_pos_w, head_vel_w,
            pelv_pos_w, tors_pos_w,
            feet_pos_w, feet_vel_w,
            hand_pos_w, hand_vel_w,
            field["torso_nav_rpy"],
            self._gait_mask,
            feet_contact,
            self._kp_scale,
            self._kd_scale,
            rfi_leg,
            field["torso_ang_vel_nav"],
        ], dim=-1)

        return actor_obs, critic_obs

    def compute_observations(self):
        actor_obs, critic_obs = super().compute_observations()
        actor_obs = torch.nan_to_num(actor_obs, nan=0.0, posinf=0.0, neginf=0.0)
        critic_obs = torch.nan_to_num(critic_obs, nan=0.0, posinf=0.0, neginf=0.0)
        return actor_obs, critic_obs

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor):
        super().reset(env_ids)
        if len(env_ids) == 0:
            return
        # Clear history buffer.
        if self._actor_history_buf is not None:
            self._actor_history_buf[env_ids] = 0.0
        # Reset flow_mimic target tracking to default pose.
        if self._last_flow_mimic_targets is not None:
            default_29 = self.robot.data.default_joint_pos[env_ids][:, self._flow_mimic_joint_ids]
            self._last_flow_mimic_targets[env_ids]  = default_29
            self._prev_flow_mimic_targets[env_ids] = default_29

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, actions: torch.Tensor):
        # ---- 1. Delay & clip RL policy output (N, 126) ----------------------
        delayed = self.action_buffer.compute(actions)
        clipped = torch.clamp(delayed, -self.clip_actions, self.clip_actions).to(self.device)

        # ---- 2. Construct actor_future (N, 630) -----------------------------
        # frame_0 = RL policy prediction; frames 1-4 = 0 (variable-horizon N=1)
        actor_future = torch.zeros(self.num_envs, D_ACTOR_FUTURE, device=self.device)
        actor_future[:, :RL_ACTION_DIM] = clipped

        # ---- 3. Get rolling actor_history (N, 10, 160) ----------------------
        actor_history = self._actor_history_buf  # (N, SEQ_LEN, 160)

        # ---- 4. Frozen flow_mimic.pt inference → joint_targets (N, 29) ------
        with torch.no_grad():
            joint_targets = self._flow_mimic_policy._actor_forward(actor_history, actor_future)
        joint_targets = torch.nan_to_num(joint_targets, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- 5. Clamp to soft joint limits ----------------------------------
        soft_lo = self.robot.data.soft_joint_pos_limits[:, self._flow_mimic_joint_ids, 0]
        soft_hi = self.robot.data.soft_joint_pos_limits[:, self._flow_mimic_joint_ids, 1]
        joint_targets = torch.clamp(joint_targets, soft_lo, soft_hi)

        # ---- 6. Write to motor targets and step physics ---------------------
        self._motor_targets.copy_(self.robot.data.default_joint_pos)
        self._motor_targets[:, self._flow_mimic_joint_ids] = joint_targets

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

        # ---- 7. Post-step bookkeeping ----------------------------------------
        self.episode_length_buf += 1
        self._update_gait_phase()
        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # ---- 8. Update flow_mimic target history ----------------------------
        self._prev_flow_mimic_targets.copy_(self._last_flow_mimic_targets)
        self._last_flow_mimic_targets.copy_(joint_targets)

        # ---- 9. Append new frame to actor_history buffer --------------------
        # Uses clipped RL output for motion_anchor fields (pelvis body 0).
        new_frame = self._build_actor_history_frame(clipped)   # (N, 160)
        self._actor_history_buf = torch.roll(self._actor_history_buf, shifts=-1, dims=1)
        self._actor_history_buf[:, -1, :] = new_frame

        # ---- 10. Reset, rewards, observations --------------------------------
        self._cat_state_step = -1
        self.latest_field_cache = {}
        self.reset_buf, self.time_out_buf = self.check_reset()
        reward_buf = self.reward_manager.compute(self.step_dt)
        reward_buf = torch.clamp(reward_buf, min=0.0, max=10_000.0)
        self._last_joint_vel.copy_(self.robot.data.joint_vel)

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(env_ids)

        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        return actor_obs, reward_buf, self.reset_buf, self.extras
