# cat_traverse_flow_mimic_env.py
#
# Hierarchical env for training a high-level RL policy that outputs 14-body
# DELTA poses, which are composed with the current body poses and fed as
# actor_future[frame_0] into a frozen TransformerTeacherPolicy (flow_mimic.pt)
# to generate 29-dim joint targets.
#
# Data flow:
#   current state (cat_traverse obs, ~192 dims)
#       │
#       ▼
#   RL Policy (trained)
#       │  actions: (N, 14*9=126)  — delta_pos(3) + delta_rot6d(6) per body
#       │  body order: see TRACKED_BODY_NAMES below
#       ▼
#   _compose_delta_pose():
#       current_body_pose_anchor = _get_current_tracked_body_pose_anchor()
#       target_pos   = current_pos + delta_pos * delta_pos_scale
#       R_target     = R_current @ gram_schmidt(delta_rot6d)
#       rot6d_target = matrix_to_rot6d(R_target)   ← explicit column concat
#       composed_pose: (N, 14, 9)  [absolute pos+rot6d in anchor frame]
#       │
#       ▼
#   actor_future (N, 630):
#       frame_0[:126] = composed_pose.view(N, 126)   ← absolute pose
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
#   [58:61]   motion_anchor_pos_b = composed pelvis pos  (body 0, dims [0:3])
#   [61:67]   motion_anchor_ori_b = composed pelvis rot6d (body 0, dims [3:9])
#   [67:70]   base_lin_vel
#   [70:73]   base_ang_vel
#   [73:102]  joint_pos_rel      (29 joints, PLANNER order)
#   [102:131] joint_vel          (29 joints, PLANNER order)
#   [131:160] last_action        = last flow_mimic targets  (same as cmd)
#
# RL Policy obs (~192 dims, cat_traverse style):
#   ang_vel(3) + proj_gravity(3) + joint_pos_rel(23) + joint_vel(23)
#   + last_fm_leg_targets(12) + command(4) + foot_height(1) + gait_phase(4)
#   + actor_pf_obs(77) + body_pos_flat(42)
#
# Anchor frame = pelvis local coordinate system, updated every step:
#   p_body_a = R_wa @ (p_body_w - p_pelvis_w)   where R_wa = R_pelvis_w^T
#   R_body_a = R_wa @ R_body_w
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
from .rot_utils import gram_schmidt, matrix_to_rot6d, quat_to_matrix

# ── Architecture constants ────────────────────────────────────────────────────
N_TRACKED_BODIES: int = 14   # bodies tracked in actor_future
BODY_POSE_DIM:    int = 9    # delta_pos(3) + delta_rot6d(6) per body
RL_ACTION_DIM:    int = N_TRACKED_BODIES * BODY_POSE_DIM   # 126
FM_LEG_JOINTS:    int = 12   # first 12 joints of PLANNER_JOINT_NAMES are legs
FM_JOINT_DIM:     int = 29   # total joints output by flow_mimic.pt

# 14 tracked body names in actor_future order (must match flow_mimic.pt training)
TRACKED_BODY_NAMES: list[str] = [
    "pelvis",                  # 0  — anchor origin; pos_a ≈ [0,0,0]
    "left_hip_roll_link",      # 1
    "left_knee_link",          # 2
    "left_ankle_roll_link",    # 3
    "right_hip_roll_link",     # 4
    "right_knee_link",         # 5
    "right_ankle_roll_link",   # 6
    "torso_link",              # 7
    "left_shoulder_roll_link", # 8
    "left_elbow_link",         # 9
    "left_wrist_yaw_link",     # 10
    "right_shoulder_roll_link",# 11
    "right_elbow_link",        # 12
    "right_wrist_yaw_link",    # 13
]


class CatTraverseFlowMimicEnv(CatTraverseEnv):
    """CAT traversal env: RL policy (126-dim delta poses) → frozen flow_mimic.pt
    → joint targets → robot.

    The RL policy outputs delta_pos(3) + delta_rot6d(6) per tracked body (9 dims
    × 14 bodies = 126 dims total).  These deltas are composed with the current
    14-body poses in the pelvis anchor frame before being passed to flow_mimic.pt
    as absolute body poses.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, cfg: CatTraverseFlowMimicEnvCfg, headless: bool):
        self._flow_mimic_joint_ids:      list[int] = []
        self._tracked_body_ids:          list[int] = []
        self._actor_history_buf:         torch.Tensor | None = None
        self._last_flow_mimic_targets:   torch.Tensor | None = None
        self._prev_flow_mimic_targets:   torch.Tensor | None = None
        self._flow_mimic_policy:         TransformerTeacherPolicy | None = None
        self._step_counter:              int = 0
        super().__init__(cfg, headless)

    # ------------------------------------------------------------------
    # Buffer initialisation
    # ------------------------------------------------------------------

    def init_buffers(self):
        # 1. Base CAT env init (field_loader, probes, domain_rand, obs_buffer).
        #    At this point num_actions = len(G1_ACTION_JOINT_NAMES) = 12.
        #    init_obs_buffer() is called inside here — our override handles it
        #    using zero placeholders for unresolved IDs.
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

        # 3. Resolve joint IDs in PLANNER_JOINT_NAMES order (= flow_mimic.pt order)
        #    and body IDs for 14 tracked bodies.
        self._resolve_flow_mimic_joint_ids()
        self._resolve_tracked_body_ids()

        # 4. Allocate actor_history buffer and flow_mimic target tracking.
        self._actor_history_buf = torch.zeros(
            self.num_envs, SEQ_LEN, D_ACTOR_HISTORY, device=self.device
        )
        default_29 = self.robot.data.default_joint_pos[:, self._flow_mimic_joint_ids]
        self._last_flow_mimic_targets = default_29.clone()
        self._prev_flow_mimic_targets = default_29.clone()

        # 5. Rebuild obs buffer now that all IDs are initialised.
        self.init_obs_buffer()

        # 6. Load and freeze flow_mimic.pt.
        self._load_flow_mimic_policy()

    def init_obs_buffer(self):
        """Override: build noise vector matching our obs layout.

        Our actor_obs layout (~192 dims):
          ang_vel(3) + proj_gravity(3) + joint_pos(obs_joint_dim)
          + joint_vel(obs_joint_dim) + last_fm_leg_targets(12)
          + command(4) + foot_height(1) + gait_phase(4) + actor_pf_obs(77)
          + body_pos_flat(42)  ← 14 tracked bodies × pos[3] in anchor frame
        """
        if self.add_noise:
            # Use placeholders for tensors not yet initialised on the first call
            # (which happens inside super().init_buffers()).
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
            # last_fm_leg_targets(12), command(4), foot_height(1), gait_phase(4),
            # actor_pf_obs(77), body_pos_flat(42) — all zero noise.
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

    def _resolve_tracked_body_ids(self):
        """Resolve IsaacLab body indices for the 14 tracked bodies.

        Order must strictly match TRACKED_BODY_NAMES (= flow_mimic.pt training
        body order).  Prints resolved names on startup for verification.
        """
        body_cfg = SceneEntityCfg(
            name="robot",
            body_names=list(TRACKED_BODY_NAMES),
            preserve_order=True,
        )
        body_cfg.resolve(self.scene)
        self._tracked_body_ids = [
            int(bid.item()) if hasattr(bid, "item") else int(bid)
            for bid in body_cfg.body_ids
        ]
        print(
            f"[CatTraverseFlowMimicEnv] Tracked body IDs resolved "
            f"({len(self._tracked_body_ids)} bodies): {TRACKED_BODY_NAMES}"
        )

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
    # Anchor-frame body pose helpers
    # ------------------------------------------------------------------

    def _get_current_tracked_body_pose_anchor(self):
        """Return 14 tracked bodies' pos and rotation in the pelvis anchor frame.

        Anchor frame definition (updated every step):
            origin    = current pelvis world position
            axes      = current pelvis rotation (R_pelvis_w)
            R_wa      = R_pelvis_w^T   (world → anchor transform)

        Returns:
            current_body_pos_a : (N, 14, 3)  positions in anchor frame
            current_body_rot_a : (N, 14, 3, 3)  rotation matrices in anchor frame
        """
        body_pos_w  = self.robot.data.body_pos_w[:, self._tracked_body_ids, :]   # (N, 14, 3)
        body_quat_w = self.robot.data.body_quat_w[:, self._tracked_body_ids, :]  # (N, 14, 4) (w,x,y,z)

        pelvis_pos_w  = self.robot.data.root_pos_w    # (N, 3)
        pelvis_quat_w = self.robot.data.root_quat_w   # (N, 4)  (w, x, y, z)

        # World → anchor rotation
        R_pelvis_w = quat_to_matrix(pelvis_quat_w)          # (N, 3, 3)
        R_wa = R_pelvis_w.transpose(-1, -2)                 # (N, 3, 3)  R_wa = R_aw^T

        # Body positions in anchor frame
        diff = body_pos_w - pelvis_pos_w.unsqueeze(1)        # (N, 14, 3)
        # einsum "nij,nbj->nbi":  R_wa[n] @ diff[n,b]  for each body b
        current_body_pos_a = torch.einsum("nij,nbj->nbi", R_wa, diff)   # (N, 14, 3)

        # Body rotations in anchor frame
        R_body_w = quat_to_matrix(body_quat_w)               # (N, 14, 3, 3)
        # einsum "nij,nbjk->nbik":  R_wa[n] @ R_body_w[n,b]
        current_body_rot_a = torch.einsum("nij,nbjk->nbik", R_wa, R_body_w)  # (N, 14, 3, 3)

        return current_body_pos_a, current_body_rot_a

    def _compose_delta_pose(self, clipped: torch.Tensor) -> torch.Tensor:
        """Compose delta policy output with current body poses → absolute pose.

        Args:
            clipped: (N, 126) clipped RL policy output.
                     Reshaped to (N, 14, 9):
                       [..., 0:3]  delta_pos   (meters, in anchor frame)
                       [..., 3:9]  delta_rot6d (columns of R_delta, unitless)

        Returns:
            composed_pose: (N, 14, 9) absolute pos+rot6d in anchor frame.
        """
        N = clipped.shape[0]
        actions_r   = clipped.view(N, N_TRACKED_BODIES, BODY_POSE_DIM)  # (N, 14, 9)
        delta_pos   = actions_r[..., 0:3]   # (N, 14, 3)  unit: meters
        delta_rot6d = actions_r[..., 3:9]   # (N, 14, 6)

        # Read current body poses in anchor frame
        current_pos_a, current_rot_a = self._get_current_tracked_body_pose_anchor()
        # current_pos_a : (N, 14, 3)
        # current_rot_a : (N, 14, 3, 3)

        # Position: add scaled delta
        target_pos_a = current_pos_a + delta_pos * self.cfg.delta_pos_scale  # (N, 14, 3)

        # Rotation: right-multiply current by delta (stays in SO(3))
        R_delta  = gram_schmidt(delta_rot6d)          # (N, 14, 3, 3)
        R_target = current_rot_a @ R_delta            # (N, 14, 3, 3)

        # Convert back to rot6d via explicit column concat (column-vector convention)
        rot6d_target = matrix_to_rot6d(R_target)      # (N, 14, 6)

        # Concatenate → absolute pose in anchor frame
        composed_pose = torch.cat([target_pos_a, rot6d_target], dim=-1)  # (N, 14, 9)
        return composed_pose

    # ------------------------------------------------------------------
    # actor_history builder  (matches flow_mimic.pt training format exactly)
    # ------------------------------------------------------------------

    def _build_actor_history_frame(self, composed_pose: torch.Tensor) -> torch.Tensor:
        """Build one 160-dim history frame for flow_mimic.pt.

        Args:
            composed_pose: (N, 14, 9) absolute pos+rot6d in anchor frame,
                           as computed by _compose_delta_pose().
                           composed_pose[:, 0, :]  = pelvis anchor pose (body 0).
                             [:, 0, 0:3]  pelvis position in anchor frame (≈ [0,0,0])
                             [:, 0, 3:9]  pelvis 6D rotation in anchor frame (≈ identity col)
        """
        # ── command: last flow_mimic targets as "reference motion" ────────────
        cmd_pos = self._last_flow_mimic_targets                               # (N, 29)
        cmd_vel = (self._last_flow_mimic_targets
                   - self._prev_flow_mimic_targets) / self.step_dt            # (N, 29)

        # ── motion anchor: composed pelvis pose (body 0 of composed_pose) ─────
        anchor_pos = composed_pose[:, 0, 0:3]   # (N, 3)  pelvis pos in anchor frame
        anchor_ori = composed_pose[:, 0, 3:9]   # (N, 6)  pelvis rot6d in anchor frame

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
    # Debug logging (play mode)
    # ------------------------------------------------------------------

    def _debug_log_step(
        self,
        delta_pos:     torch.Tensor,   # (N, 14, 3)
        delta_rot6d:   torch.Tensor,   # (N, 14, 6)
        R_delta:       torch.Tensor,   # (N, 14, 3, 3)
        composed_pose: torch.Tensor,   # (N, 14, 9)
        joint_targets: torch.Tensor,   # (N, 29)
    ) -> None:
        """Print key intermediate values every 50 steps (play mode validation)."""
        if self._step_counter % 50 != 0:
            return

        # 1. Policy delta output statistics
        print(f"[DBG step={self._step_counter}] "
              f"delta_pos    mean={delta_pos[0].mean():.4f}  std={delta_pos[0].std():.4f}")
        print(f"[DBG step={self._step_counter}] "
              f"delta_rot6d  mean={delta_rot6d[0].mean():.4f}  std={delta_rot6d[0].std():.4f}")

        # 2. Gram-Schmidt determinants (expect ≈ 1.0 for all 14 bodies)
        det = torch.linalg.det(R_delta[0])   # (14,)
        print(f"[DBG step={self._step_counter}] "
              f"R_delta det  min={det.min():.4f}  max={det.max():.4f}  (expect ≈ 1.0)")

        # 3. Pelvis composed pose in anchor frame (pos ≈ [0,0,0])
        pelv = composed_pose[0, 0]
        print(f"[DBG step={self._step_counter}] "
              f"composed pelvis  pos_a={pelv[:3].tolist()}  rot6d={pelv[3:].tolist()}")

        # 4. Flow_mimic output joint target range
        print(f"[DBG step={self._step_counter}] "
              f"joint_targets  min={joint_targets[0].min():.4f}  max={joint_targets[0].max():.4f}")

        # 5. NaN / Inf detection
        for name, t in [
            ("delta_pos",     delta_pos),
            ("delta_rot6d",   delta_rot6d),
            ("composed_pose", composed_pose),
            ("joint_targets", joint_targets),
        ]:
            if torch.isnan(t).any() or torch.isinf(t).any():
                print(f"[WARN step={self._step_counter}] NaN/Inf detected in {name}!")

    # ------------------------------------------------------------------
    # RL Policy observations  (override base cat obs)
    # ------------------------------------------------------------------

    def compute_current_observations(self):
        """Returns (actor_obs, critic_obs) for the RL policy.

        Actor obs layout (~192 dims):
          ang_vel(3) + proj_gravity(3) + joint_pos(23) + joint_vel(23)
          + last_fm_leg_targets(12) + command(4) + foot_height(1) + gait_phase(4)
          + actor_pf_obs(77) + body_pos_flat(42)

        body_pos_flat encodes the current 14-body positions in the pelvis anchor
        frame, giving the policy the context needed to output meaningful deltas.
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

        # ── body positions in anchor frame (actor obs augmentation, 42 dims) ──
        # Guard: during the first init_obs_buffer() call _tracked_body_ids is [].
        if self._tracked_body_ids:
            current_body_pos_a, _ = self._get_current_tracked_body_pose_anchor()
            body_pos_flat = current_body_pos_a.reshape(self.num_envs, -1)      # (N, 42)
        else:
            body_pos_flat = torch.zeros(
                self.num_envs, N_TRACKED_BODIES * 3, device=self.device
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
            body_pos_flat,                                         # 42  ← NEW
        ], dim=-1)                                                 # ≈ 192

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
        self._step_counter += 1

        # ---- 1. Delay & clip RL policy output (N, 126) ----------------------
        delayed = self.action_buffer.compute(actions)
        clipped = torch.clamp(delayed, -self.clip_actions, self.clip_actions).to(self.device)

        # ---- 2. Compose delta → absolute pose in anchor frame ---------------
        # composed_pose: (N, 14, 9)  absolute pos+rot6d in pelvis anchor frame
        composed_pose = self._compose_delta_pose(clipped)
        composed_pose = torch.nan_to_num(composed_pose, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- 3. Construct actor_future (N, 630) from composed absolute pose --
        # frame_0[:126] = composed absolute pose; frames 1-4 = 0
        actor_future = torch.zeros(self.num_envs, D_ACTOR_FUTURE, device=self.device)
        actor_future[:, :RL_ACTION_DIM] = composed_pose.view(self.num_envs, -1)

        # ---- 4. Get rolling actor_history (N, 10, 160) ----------------------
        actor_history = self._actor_history_buf  # (N, SEQ_LEN, 160)

        # ---- 5. Frozen flow_mimic.pt inference → joint_targets (N, 29) ------
        with torch.no_grad():
            joint_targets = self._flow_mimic_policy._actor_forward(actor_history, actor_future)
        joint_targets = torch.nan_to_num(joint_targets, nan=0.0, posinf=0.0, neginf=0.0)

        # ---- 6. Optional debug logging (play mode) ---------------------------
        if self.num_envs == 1:
            N = clipped.shape[0]
            actions_r   = clipped.view(N, N_TRACKED_BODIES, BODY_POSE_DIM)
            R_delta = gram_schmidt(actions_r[..., 3:9])
            self._debug_log_step(
                delta_pos     = actions_r[..., 0:3],
                delta_rot6d   = actions_r[..., 3:9],
                R_delta       = R_delta,
                composed_pose = composed_pose,
                joint_targets = joint_targets,
            )

        # ---- 7. Clamp to soft joint limits ----------------------------------
        soft_lo = self.robot.data.soft_joint_pos_limits[:, self._flow_mimic_joint_ids, 0]
        soft_hi = self.robot.data.soft_joint_pos_limits[:, self._flow_mimic_joint_ids, 1]
        joint_targets = torch.clamp(joint_targets, soft_lo, soft_hi)

        # ---- 8. Write to motor targets and step physics ---------------------
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

        # ---- 9. Post-step bookkeeping ----------------------------------------
        self.episode_length_buf += 1
        self._update_gait_phase()
        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        # ---- 10. Update flow_mimic target history ---------------------------
        self._prev_flow_mimic_targets.copy_(self._last_flow_mimic_targets)
        self._last_flow_mimic_targets.copy_(joint_targets)

        # ---- 11. Append new frame to actor_history buffer -------------------
        # Uses composed_pose (absolute anchor frame) for motion_anchor fields.
        new_frame = self._build_actor_history_frame(composed_pose)   # (N, 160)
        self._actor_history_buf = torch.roll(self._actor_history_buf, shifts=-1, dims=1)
        self._actor_history_buf[:, -1, :] = new_frame

        # ---- 12. Reset, rewards, observations --------------------------------
        self._cat_state_step = -1
        self.latest_field_cache = {}
        self.reset_buf, self.time_out_buf = self.check_reset()
        reward_buf = self.reward_manager.compute(self.step_dt)
        reward_buf = torch.clamp(reward_buf, min=-10_000.0, max=10_000.0)
        self._last_joint_vel.copy_(self.robot.data.joint_vel)

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(env_ids)

        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        return actor_obs, reward_buf, self.reset_buf, self.extras
