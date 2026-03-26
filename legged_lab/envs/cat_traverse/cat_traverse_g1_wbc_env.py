# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
#
# cat_traverse_g1_wbc_env.py
#
# G1 WBC environment: the RL policy outputs 8 planner control signals
# (target_vel, mode, movement_direction[3], facing_direction[3]) which are
# forwarded to planner_sonic.onnx.  The planner returns a predicted joint
# trajectory; we use the first frame as the joint position target for every
# decimation sub-step.
#
# Data flow:
#   IsaacSim state
#       -> Policy (8 outputs)
#       -> planner_sonic.onnx
#       -> joint_pos_target [num_envs, 29]
#       -> robot.set_joint_position_target(...)

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.buffers import DelayBuffer

from .cat_traverse_env import CatTraverseEnv
from .cat_traverse_g1_wbc_cfg import CatTraverseG1WbcEnvCfg, PLANNER_JOINT_NAMES

# Alias for other modules that need to reference the joint name list.
PLANNER_JOINT_NAMES_REF = PLANNER_JOINT_NAMES

try:
    import onnxruntime as ort

    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False


class CatTraverseG1WbcEnv(CatTraverseEnv):
    """CAT traversal environment with planner-in-the-loop whole-body control.

    The policy outputs 8 floats interpreted as planner control signals:
      actions[:, 0]   -> target_vel
      actions[:, 1]   -> mode  (rounded & clamped to int [0, 26])
      actions[:, 2:5] -> movement_direction (L2-normalized)
      actions[:, 5:8] -> facing_direction   (L2-normalized)

    These are fed per-env into planner_sonic.onnx (batch=1).  The planner
    outputs mujoco_qpos [1, 64, 36]; we take qpos[0, 0, 7:36] (first predicted
    frame, 29 joint angles) as the joint position target.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, cfg: CatTraverseG1WbcEnvCfg, headless: bool):
        # Declare WBC-specific attributes BEFORE super().__init__ calls
        # init_buffers / reset internally.
        self._planner_session = None
        self._planner_context_buf: torch.Tensor | None = None
        self._planner_seeds: list[int] = []
        self._planner_joint_ids: list[int] = []
        super().__init__(cfg, headless)

    # ------------------------------------------------------------------
    # Buffer initialisation
    # ------------------------------------------------------------------

    def init_buffers(self):
        # 1. Run base CAT init (resolves joints, field, domain-rand buffers …).
        super().init_buffers()

        # 2. Override action dimension: WBC policy emits 8 planner signals,
        #    not joint-delta actions.  Rebuild the action buffer to match.
        if self.num_actions != 8:
            self.num_actions = 8
            max_delay = self.cfg.domain_rand.action_delay.params["max_delay"]
            self.action_buffer = DelayBuffer(max_delay, self.num_envs, device=self.device)
            self.action_buffer.compute(
                torch.zeros(
                    self.num_envs, 8, dtype=torch.float, device=self.device, requires_grad=False
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
            # Reinitialise obs / noise buffers with the correct action dim.
            self.init_obs_buffer()

        # 3. Resolve planner joint IDs (IsaacLab indices in planner order).
        self._resolve_planner_joint_ids()

        # 4. 4-frame qpos context buffer for the planner: [num_envs, 4, 36].
        self._planner_context_buf = torch.zeros(
            self.num_envs, 4, 36, dtype=torch.float32, device=self.device
        )

        # 5. Per-env random seed counters for the planner.
        self._planner_seeds = list(range(self.num_envs))

        # 6. Load ONNX inference session.
        if not _ORT_AVAILABLE:
            raise ImportError(
                "onnxruntime is required for CatTraverseG1WbcEnv. "
                "Install it with: pip install onnxruntime"
            )
        self._load_planner_session()

    def _resolve_planner_joint_ids(self):
        """Resolve IsaacLab joint indices in planner_joint_names order."""
        joint_cfg = SceneEntityCfg(
            name="robot",
            joint_names=list(self.cfg.planner.planner_joint_names),
            preserve_order=True,
        )
        joint_cfg.resolve(self.scene)
        self._planner_joint_ids = [
            int(jid.item()) if hasattr(jid, "item") else int(jid)
            for jid in joint_cfg.joint_ids
        ]

    def _load_planner_session(self):
        onnx_path = self.cfg.planner.onnx_path
        if not os.path.isabs(onnx_path):
            # Resolve relative to TienKung project root (parent of legged_lab).
            import legged_lab as _ll

            tienkung_dir = os.path.dirname(os.path.dirname(os.path.abspath(_ll.__file__)))
            onnx_path = os.path.join(tienkung_dir, onnx_path)

        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(
                f"planner_sonic.onnx not found at '{onnx_path}'. "
                "Check cfg.planner.onnx_path."
            )

        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 1
        sess_options.intra_op_num_threads = 1
        self._planner_session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

    # ------------------------------------------------------------------
    # Planner context buffer helpers
    # ------------------------------------------------------------------

    def _get_current_qpos_frame(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Build a [*, 36] qpos tensor: [root_pos(3), root_quat(4), joints(29)].

        Positions are expressed in env-local coordinates so each environment is
        handled independently regardless of its world offset.
        """
        if env_ids is None:
            root_pos = self.robot.data.root_pos_w - self.scene.env_origins
            root_quat = self.robot.data.root_quat_w
            joint_pos = self.robot.data.joint_pos[:, self._planner_joint_ids]
        else:
            root_pos = self.robot.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
            root_quat = self.robot.data.root_quat_w[env_ids]
            joint_pos = self.robot.data.joint_pos[env_ids][:, self._planner_joint_ids]
        return torch.cat([root_pos, root_quat, joint_pos], dim=-1)  # [*, 36]

    def _update_planner_context(self, qpos_frame: torch.Tensor):
        """Shift the 4-frame ring buffer and append the latest qpos frame."""
        self._planner_context_buf = torch.roll(self._planner_context_buf, shifts=-1, dims=1)
        self._planner_context_buf[:, -1, :] = qpos_frame

    def _reset_planner_context(self, env_ids: torch.Tensor):
        """Fill all 4 context frames with the current static pose for reset envs."""
        if len(env_ids) == 0:
            return
        qpos_frame = self._get_current_qpos_frame(env_ids)  # [|env_ids|, 36]
        self._planner_context_buf[env_ids] = qpos_frame.unsqueeze(1).expand(-1, 4, -1)

    # ------------------------------------------------------------------
    # Planner input construction
    # ------------------------------------------------------------------

    def _build_planner_inputs(
        self,
        i: int,
        target_vel: torch.Tensor,
        mode_int: torch.Tensor,
        move_dir: torch.Tensor,
        face_dir: torch.Tensor,
    ) -> dict:
        """Assemble the 11-tensor input dict for env index *i*.

        Shapes required by planner_sonic.onnx:
          context_mujoco_qpos       float32  [1, 4, 36]
          target_vel                float32  [1]
          mode                      int64    [1]
          movement_direction        float32  [1, 3]
          facing_direction          float32  [1, 3]
          random_seed               int64    [1]
          has_specific_target       int64    [1, 1]
          specific_target_positions float32  [1, 4, 3]
          specific_target_headings  float32  [1, 4]
          allowed_pred_num_tokens   int64    [1, 11]
          height                    float32  [1]
        """
        return {
            "context_mujoco_qpos": self._planner_context_buf[i : i + 1].cpu().numpy(),
            "target_vel": target_vel[i : i + 1].squeeze(-1).cpu().numpy(),
            "mode": mode_int[i : i + 1].squeeze(-1).cpu().numpy().astype(np.int64),
            "movement_direction": move_dir[i : i + 1].cpu().numpy(),
            "facing_direction": face_dir[i : i + 1].cpu().numpy(),
            "random_seed": np.array([self._planner_seeds[i]], dtype=np.int64),
            "has_specific_target": np.zeros((1, 1), dtype=np.int64),
            "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
            "specific_target_headings": np.zeros((1, 4), dtype=np.float32),
            "allowed_pred_num_tokens": np.ones((1, 11), dtype=np.int64),
            "height": np.array([self.cfg.planner.default_height], dtype=np.float32),
        }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, env_ids: torch.Tensor):
        super().reset(env_ids)
        # Re-seed planner random counters for reset environments.
        for eid in env_ids.tolist():
            self._planner_seeds[eid] = 0
        self._reset_planner_context(env_ids)

    # ------------------------------------------------------------------
    # Step  (override base CatTraverseEnv.step)
    # ------------------------------------------------------------------

    def step(self, actions: torch.Tensor):
        # ---- 1. Delay buffer ------------------------------------------------
        delayed_actions = self.action_buffer.compute(actions)
        clipped_actions = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(
            self.device
        )

        # ---- 2. Parse planner control signals from policy output ------------
        target_vel = clipped_actions[:, 0:1]           # [N, 1]  float
        mode_raw = clipped_actions[:, 1:2]             # [N, 1]  float -> int
        move_dir_raw = clipped_actions[:, 2:5]         # [N, 3]
        face_dir_raw = clipped_actions[:, 5:8]         # [N, 3]

        move_dir = F.normalize(move_dir_raw, dim=-1)   # [N, 3]  L2-normalized
        face_dir = F.normalize(face_dir_raw, dim=-1)   # [N, 3]  L2-normalized
        mode_int = mode_raw.round().clamp(0, 26).long()  # [N, 1]  int64

        # ---- 3. Update 4-frame context buffer with current robot state ------
        self._update_planner_context(self._get_current_qpos_frame())

        # ---- 4. Per-env planner inference -----------------------------------
        joint_targets = torch.zeros(
            self.num_envs, 29, device=self.device, dtype=torch.float32
        )
        for i in range(self.num_envs):
            ort_inputs = self._build_planner_inputs(i, target_vel, mode_int, move_dir, face_dir)
            mujoco_qpos, _num_pred_frames = self._planner_session.run(None, ort_inputs)
            # mujoco_qpos: [1, 64, 36] — take frame 0, joint indices [7:36]
            joint_targets[i] = torch.from_numpy(mujoco_qpos[0, 0, 7:36]).to(self.device)
            self._planner_seeds[i] += 1

        # ---- 5. Write joint targets into _motor_targets ---------------------
        self._motor_targets.copy_(self.robot.data.default_joint_pos)
        self._motor_targets[:, self._planner_joint_ids] = joint_targets

        # ---- 6. Physics step (identical to base CatTraverseEnv) -------------
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

        # ---- 7. Post-step bookkeeping (identical to base) -------------------
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

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset(env_ids)

        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        return actor_obs, reward_buf, self.reset_buf, self.extras
