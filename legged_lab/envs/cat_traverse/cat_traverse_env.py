# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
#
# This file is part of the CAT IsaacLab task scaffold for Click-and-Traverse.

from __future__ import annotations

import math
import re
from copy import deepcopy
from pathlib import Path

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.buffers import CircularBuffer

from legged_lab.envs.base.base_env import BaseEnv

from .cat_traverse_cfg import CatTraverseEnvCfg
from .field_loader import CatTraverseFieldLoader, resolve_field_dir


EPS = 1.0e-6
GROUP_ORDER = ("head", "pelvis", "torso", "feet", "hands", "knees", "shoulders")
PHASE_INIT_LR = torch.tensor([0.0, math.pi], dtype=torch.float32)
PHASE_INIT_RL = torch.tensor([math.pi, 0.0], dtype=torch.float32)
STANCE_PHASE = torch.tensor([0.0, 0.0], dtype=torch.float32)


def quat_to_matrix(quat_wxyz: torch.Tensor) -> torch.Tensor:
    quat_wxyz = quat_wxyz / torch.linalg.norm(quat_wxyz, dim=-1, keepdim=True).clamp_min(EPS)
    w, x, y, z = quat_wxyz.unbind(dim=-1)

    ww = w * w
    xx = x * x
    yy = y * y
    zz = z * z
    wx = w * x
    wy = w * y
    wz = w * z
    xy = x * y
    xz = x * z
    yz = y * z

    return torch.stack(
        [
            torch.stack((ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)), dim=-1),
            torch.stack((2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)), dim=-1),
            torch.stack((2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz), dim=-1),
        ],
        dim=-2,
    )


class CatTraverseEnv(BaseEnv):
    """IsaacLab CAT traversal environment with PF delay and obstacle import."""

    def __init__(self, cfg: CatTraverseEnvCfg, headless):
        cfg = deepcopy(cfg)
        if cfg.scene.mesh_obstacle is not None and cfg.scene.mesh_obstacle.enable:
            if not cfg.scene.mesh_obstacle.source_path:
                cfg.scene.mesh_obstacle.source_path = str(Path(cfg.field.path) / "obs.obj")
            # PF grids are indexed in the field's canonical world frame. The exported
            # obstacle mesh vertices are stored in the same frame shifted by -origin,
            # so the imported mesh must be translated back by field.origin to align
            # with sdf/gf/bf sampling and the robot reset frame.
            if tuple(cfg.scene.mesh_obstacle.pos) == (0.0, 0.0, 0.0):
                cfg.scene.mesh_obstacle.pos = tuple(float(v) for v in cfg.field.origin)
        if getattr(cfg.scene, "filtered_contact_sensors", None) is None:
            cfg.scene.filtered_contact_sensors = self._build_pair_contact_sensors(cfg)

        self.cfg: CatTraverseEnvCfg
        self._probe_specs: list[tuple[str, tuple[str, ...], tuple[tuple[float, float, float], ...]]] = []
        self._probe_slices: dict[str, slice] = {}
        self._probe_body_ids: list[int] = []
        self._probe_offsets_local: torch.Tensor | None = None
        self._torso_body_id: int | None = None
        self._gait_phase: torch.Tensor | None = None
        self._gait_phase_dt: torch.Tensor | None = None
        self._gait_mask: torch.Tensor | None = None
        self._foot_height_target: torch.Tensor | None = None
        self._motor_targets: torch.Tensor | None = None
        self._delayed_root_pos_w: torch.Tensor | None = None
        self._delayed_root_quat_w: torch.Tensor | None = None
        self._current_command_world: torch.Tensor | None = None
        self._last_command_world: torch.Tensor | None = None
        self._stop_timestep: torch.Tensor | None = None
        self._last_joint_vel: torch.Tensor | None = None
        self._obs_joint_ids: list[int] = []
        self._default_joint_stiffness: torch.Tensor | None = None
        self._default_joint_damping: torch.Tensor | None = None
        self._default_joint_effort_limit: torch.Tensor | None = None
        self._kp_scale: torch.Tensor | None = None
        self._kd_scale: torch.Tensor | None = None
        self._rfi_lim_scale: torch.Tensor | None = None
        self._cat_state_step: int = -1
        self.latest_field_cache: dict[str, torch.Tensor] = {}
        super().__init__(cfg, headless)

    @staticmethod
    def _build_pair_contact_sensors(cfg: CatTraverseEnvCfg) -> dict[str, ContactSensorCfg]:
        left_foot, right_foot = cfg.probes.feet_body_names
        left_shin, right_shin = cfg.probes.shin_body_names
        history_length = 3
        update_period = cfg.sim.dt
        return {
            "cat_left_pair_contact": ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{left_foot}",
                history_length=history_length,
                update_period=update_period,
                filter_prim_paths_expr=[
                    f"{{ENV_REGEX_NS}}/Robot/{right_foot}",
                    f"{{ENV_REGEX_NS}}/Robot/{right_shin}",
                ],
            ),
            "cat_right_pair_contact": ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/{right_foot}",
                history_length=history_length,
                update_period=update_period,
                filter_prim_paths_expr=[
                    f"{{ENV_REGEX_NS}}/Robot/{left_foot}",
                    f"{{ENV_REGEX_NS}}/Robot/{left_shin}",
                ],
            ),
        }

    def init_buffers(self):
        self.field_loader = CatTraverseFieldLoader(
            self.cfg.field.path,
            dx=self.cfg.field.dx,
            origin=self.cfg.field.origin,
            device=self.device,
        )
        self._resolve_cat_entities()
        self._resolve_obs_joint_ids()
        self._init_domain_rand_buffers()
        self.latest_field_cache = {}
        self._cat_state_step = -1
        self._gait_phase = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)
        self._gait_phase_dt = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)
        self._gait_mask = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)
        self._foot_height_target = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)
        self._motor_targets = torch.zeros_like(self.robot.data.default_joint_pos)
        self._delayed_root_pos_w = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self._delayed_root_quat_w = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float32)
        self._delayed_root_quat_w[:, 0] = 1.0
        self._current_command_world = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float32)
        self._last_command_world = torch.zeros_like(self._current_command_world)
        self._stop_timestep = torch.full(
            (self.num_envs,),
            int(self.cfg.command.stop_timestep_reset),
            device=self.device,
            dtype=torch.long,
        )
        self._last_joint_vel = torch.zeros_like(self.robot.data.joint_vel)
        super().init_buffers()
        self._motor_targets.copy_(self.robot.data.default_joint_pos)

    def _resolve_obs_joint_ids(self):
        if not getattr(self.cfg.robot, "obs_joint_names", None):
            self._obs_joint_ids = list(range(self.robot.data.default_joint_pos.shape[1]))
            return
        obs_joint_cfg = SceneEntityCfg(name="robot", joint_names=self.cfg.robot.obs_joint_names, preserve_order=True)
        obs_joint_cfg.resolve(self.scene)
        self._obs_joint_ids = [
            int(joint_id.item()) if hasattr(joint_id, "item") else int(joint_id)
            for joint_id in obs_joint_cfg.joint_ids
        ]

    def _joint_matches_any_pattern(self, joint_name: str, patterns: list[str] | tuple[str, ...] | str) -> bool:
        if isinstance(patterns, str):
            patterns = [patterns]
        return any(re.fullmatch(pattern, joint_name) for pattern in patterns)

    def _resolve_joint_property_from_actuators(self, property_name: str, default_value: float = 0.0) -> torch.Tensor:
        joint_names = list(self.robot.data.joint_names)
        values = torch.full((len(joint_names),), float(default_value), device=self.device, dtype=torch.float32)
        actuators = getattr(self.cfg.scene.robot, "actuators", {}) or {}
        for actuator_cfg in actuators.values():
            joint_patterns = getattr(actuator_cfg, "joint_names_expr", None)
            property_cfg = getattr(actuator_cfg, property_name, None)
            if joint_patterns is None or property_cfg is None:
                continue
            joint_patterns = joint_patterns if isinstance(joint_patterns, (list, tuple)) else [joint_patterns]
            matched_joint_ids = [
                joint_id
                for joint_id, joint_name in enumerate(joint_names)
                if self._joint_matches_any_pattern(joint_name, joint_patterns)
            ]
            if not matched_joint_ids:
                continue
            if isinstance(property_cfg, dict):
                for joint_id in matched_joint_ids:
                    joint_name = joint_names[joint_id]
                    for pattern, pattern_value in property_cfg.items():
                        if re.fullmatch(pattern, joint_name):
                            values[joint_id] = float(pattern_value)
                            break
            else:
                values[matched_joint_ids] = float(property_cfg)
        return values

    def _init_domain_rand_buffers(self):
        joint_count = self.robot.data.default_joint_pos.shape[1]
        self._default_joint_stiffness = self._resolve_joint_property_from_actuators("stiffness", default_value=0.0)
        self._default_joint_damping = self._resolve_joint_property_from_actuators("damping", default_value=0.0)
        self._default_joint_effort_limit = self._resolve_joint_property_from_actuators(
            "effort_limit_sim", default_value=0.0
        )
        self._kp_scale = torch.ones(self.num_envs, 1, device=self.device, dtype=torch.float32)
        self._kd_scale = torch.ones(self.num_envs, 1, device=self.device, dtype=torch.float32)
        self._rfi_lim_scale = torch.zeros(self.num_envs, joint_count, device=self.device, dtype=torch.float32)

    def _write_joint_property_to_sim(self, method_name: str, values: torch.Tensor, env_ids: torch.Tensor):
        if len(env_ids) == 0 or not hasattr(self.robot, method_name):
            return
        method = getattr(self.robot, method_name)
        try:
            method(values, env_ids=env_ids)
        except TypeError:
            method(values, None, env_ids)

    def _set_joint_effort_target(self, efforts: torch.Tensor, env_ids: torch.Tensor | None = None):
        if not hasattr(self.robot, "set_joint_effort_target"):
            return
        method = getattr(self.robot, "set_joint_effort_target")
        try:
            if env_ids is None:
                method(efforts)
            else:
                method(efforts, env_ids=env_ids)
        except TypeError:
            if env_ids is None:
                method(efforts, None, None)
            else:
                method(efforts, None, env_ids)

    def _reset_domain_randomization(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return

        num_reset_envs = len(env_ids)
        if self.cfg.dm_rand.enable_pd:
            kp_scale = torch.empty(num_reset_envs, 1, device=self.device, dtype=torch.float32).uniform_(
                *self.cfg.dm_rand.kp_range
            )
            kd_scale = torch.empty(num_reset_envs, 1, device=self.device, dtype=torch.float32).uniform_(
                *self.cfg.dm_rand.kd_range
            )
        else:
            kp_scale = torch.ones(num_reset_envs, 1, device=self.device, dtype=torch.float32)
            kd_scale = torch.ones(num_reset_envs, 1, device=self.device, dtype=torch.float32)
        self._kp_scale[env_ids] = kp_scale
        self._kd_scale[env_ids] = kd_scale

        default_effort_limit = self._default_joint_effort_limit.unsqueeze(0).expand(num_reset_envs, -1)
        if self.cfg.dm_rand.enable_rfi:
            rfi_noise_scale = torch.empty_like(default_effort_limit).uniform_(*self.cfg.dm_rand.rfi_lim_range)
            rfi_lim_scale = self.cfg.dm_rand.rfi_lim * rfi_noise_scale * default_effort_limit
        else:
            rfi_lim_scale = torch.zeros_like(default_effort_limit)
        self._rfi_lim_scale[env_ids] = rfi_lim_scale

        stiffness = self._default_joint_stiffness.unsqueeze(0) * kp_scale
        damping = self._default_joint_damping.unsqueeze(0) * kd_scale
        self._write_joint_property_to_sim("write_joint_stiffness_to_sim", stiffness, env_ids)
        self._write_joint_property_to_sim("write_joint_damping_to_sim", damping, env_ids)
        self._set_joint_effort_target(torch.zeros(num_reset_envs, self.robot.data.joint_pos.shape[1], device=self.device), env_ids)

    def _resolve_cat_entities(self):
        probe_cfg = self.cfg.probes
        self._probe_specs = [
            ("head", (probe_cfg.head_body_name,), probe_cfg.head_offsets),
            ("pelvis", (probe_cfg.pelvis_body_name,), probe_cfg.pelvis_offsets),
            ("torso", (probe_cfg.torso_body_name,), probe_cfg.torso_offsets),
            ("feet", probe_cfg.feet_body_names, probe_cfg.feet_offsets),
            ("hands", probe_cfg.hand_body_names, probe_cfg.hand_offsets),
            ("knees", probe_cfg.knee_body_names, probe_cfg.knee_offsets),
            ("shoulders", probe_cfg.shoulder_body_names, probe_cfg.shoulder_offsets),
        ]

        self._probe_body_ids = []
        all_offsets: list[tuple[float, float, float]] = []
        cursor = 0
        for group_name, body_names, offsets in self._probe_specs:
            if len(offsets) != len(body_names):
                raise RuntimeError(
                    f"CAT traversal probe group '{group_name}' has {len(body_names)} bodies but {len(offsets)} offsets."
                )
            ids, _ = self.robot.find_bodies(name_keys=list(body_names), preserve_order=True)
            if len(ids) != len(body_names):
                profile_hint = ""
                if probe_cfg.profile_name == "g1_cat_sites":
                    profile_hint = (
                        " Ensure cfg.scene.robot is a G1 articulation and the cfg was built with "
                        "make_unitree_g1_cat_env_cfg(), cat_traverse_g1, apply_g1_cat_site_profile(...), "
                        "or make_g1_cat_site_env_cfg(...)."
                    )
                raise RuntimeError(
                    f"CAT traversal probe group '{group_name}' could not resolve all bodies for profile "
                    f"'{probe_cfg.profile_name}': {body_names}.{profile_hint}"
                )
            self._probe_slices[group_name] = slice(cursor, cursor + len(body_names))
            for idx in ids:
                self._probe_body_ids.append(int(idx.item()) if hasattr(idx, "item") else int(idx))
            all_offsets.extend(offsets)
            cursor += len(body_names)

        if not self._probe_body_ids:
            raise RuntimeError("CAT traversal probe resolution failed: no body ids were resolved.")
        self._probe_offsets_local = torch.tensor(all_offsets, dtype=torch.float32, device=self.device)
        self._torso_body_id = self._probe_body_ids[self._probe_slices["torso"].start]

    def _reset_gait_and_delay(self, env_ids: torch.Tensor):
        if len(env_ids) == 0:
            return

        freq_lo, freq_hi = self.cfg.gait.freq_range
        gait_freq = torch.empty(len(env_ids), 1, device=self.device).uniform_(freq_lo, freq_hi)
        self._gait_phase_dt[env_ids] = 2.0 * math.pi * self.step_dt * gait_freq

        cond = torch.rand(len(env_ids), device=self.device) > 0.5
        init_phase = PHASE_INIT_LR.to(self.device).unsqueeze(0).repeat(len(env_ids), 1)
        alt_phase = PHASE_INIT_RL.to(self.device).unsqueeze(0).repeat(len(env_ids), 1)
        self._gait_phase[env_ids] = torch.where(cond.unsqueeze(-1), init_phase, alt_phase)

        foot_lo, foot_hi = self.cfg.gait.foot_height_range
        self._foot_height_target[env_ids] = torch.empty(len(env_ids), 1, device=self.device).uniform_(foot_lo, foot_hi)
        self._update_gait_mask(env_ids)

        self._motor_targets[env_ids] = self.robot.data.default_joint_pos[env_ids]
        self._delayed_root_pos_w[env_ids] = self.robot.data.root_pos_w[env_ids]
        self._delayed_root_quat_w[env_ids] = self.robot.data.root_quat_w[env_ids]
        self._current_command_world[env_ids] = 0.0
        self._last_command_world[env_ids] = 0.0
        self._stop_timestep[env_ids] = int(self.cfg.command.stop_timestep_reset)
        self._last_joint_vel[env_ids] = 0.0

    def _update_gait_mask(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            phase = self._gait_phase
            mask = self._gait_mask
        else:
            phase = self._gait_phase[env_ids]
            mask = self._gait_mask[env_ids]
        gait_cycle = torch.cos(phase)
        gait_bound = self.cfg.gait.gait_bound
        updated = torch.where(
            gait_cycle > gait_bound,
            torch.ones_like(gait_cycle),
            torch.where(gait_cycle < -gait_bound, -torch.ones_like(gait_cycle), torch.zeros_like(gait_cycle)),
        )
        if env_ids is None:
            mask.copy_(updated)
        else:
            self._gait_mask[env_ids] = updated

    def _update_gait_phase(self):
        self._gait_phase.add_(self._gait_phase_dt)
        self._gait_phase[:] = torch.remainder(self._gait_phase + math.pi, 2.0 * math.pi) - math.pi
        self._update_gait_mask()

    def _apply_stop_command_state_machine(
        self, command_current_raw_world: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._current_command_world is None or self._last_command_world is None or self._stop_timestep is None:
            raise RuntimeError("CAT traversal command state buffers are not initialized.")

        previous_command = self._current_command_world.clone()
        self._last_command_world.copy_(previous_command)

        last_task_mask = previous_command[:, 0] > 0.5
        task_mask = command_current_raw_world[:, 0] > 0.5

        stop_timestep = self._stop_timestep
        before_stop = stop_timestep > self.cfg.command.stop_hold_steps
        during_stop = (~before_stop) & (stop_timestep > 0)
        after_stop = (~before_stop) & (~during_stop)
        move_to_stop = last_task_mask & (~task_mask) & before_stop

        updated_stop_timestep = torch.where(
            move_to_stop,
            torch.full_like(stop_timestep, int(self.cfg.command.stop_hold_steps)),
            stop_timestep,
        )
        updated_stop_timestep = torch.where(during_stop, updated_stop_timestep - 1, updated_stop_timestep)
        self._stop_timestep.copy_(updated_stop_timestep)

        effective_command = torch.where(before_stop.unsqueeze(-1), command_current_raw_world, torch.zeros_like(command_current_raw_world))
        effective_command[:, 0] = torch.where(after_stop, 0.0, 1.0)
        self._current_command_world.copy_(effective_command)
        return effective_command, after_stop

    def _rotate_local_offsets(self, body_quat_w: torch.Tensor) -> torch.Tensor:
        if self._probe_offsets_local is None:
            raise RuntimeError("CAT traversal probe offsets are not initialized.")
        offsets_local = self._probe_offsets_local.unsqueeze(0).expand(self.num_envs, -1, -1)
        rot_w = quat_to_matrix(body_quat_w)
        return torch.einsum("enij,enj->eni", rot_w, offsets_local)

    def _probe_positions_world(self) -> torch.Tensor:
        body_pos_w = self.robot.data.body_pos_w[:, self._probe_body_ids, :3]
        body_quat_w = self.robot.data.body_quat_w[:, self._probe_body_ids, :]
        return body_pos_w + self._rotate_local_offsets(body_quat_w)

    def _probe_velocities_world(self) -> torch.Tensor:
        body_lin_vel_w = self.robot.data.body_lin_vel_w[:, self._probe_body_ids, :3]
        body_ang_vel_w = self.robot.data.body_ang_vel_w[:, self._probe_body_ids, :3]
        body_quat_w = self.robot.data.body_quat_w[:, self._probe_body_ids, :]
        offset_world = self._rotate_local_offsets(body_quat_w)
        return body_lin_vel_w + torch.cross(body_ang_vel_w, offset_world, dim=-1)

    def _compute_nav_rotation(self) -> torch.Tensor:
        root_rot_w = quat_to_matrix(self.robot.data.root_quat_w)
        x_axis = root_rot_w[:, :, 0]
        x_proj = x_axis.clone()
        x_proj[:, 2] = 0.0
        x_proj_norm = torch.linalg.norm(x_proj, dim=-1, keepdim=True).clamp_min(EPS)
        x_proj = x_proj / x_proj_norm

        z_axis = torch.zeros_like(x_proj)
        z_axis[:, 2] = 1.0
        y_axis = torch.cross(z_axis, x_proj, dim=-1)
        y_axis = y_axis / torch.linalg.norm(y_axis, dim=-1, keepdim=True).clamp_min(EPS)
        x_axis = torch.cross(y_axis, z_axis, dim=-1)
        return torch.stack((x_axis, y_axis, z_axis), dim=-1)

    def _world_to_nav(self, vectors_w: torch.Tensor, nav_to_world_rot: torch.Tensor) -> torch.Tensor:
        world_to_nav = nav_to_world_rot.transpose(1, 2)
        if vectors_w.ndim == 2:
            return torch.einsum("eij,ej->ei", world_to_nav, vectors_w)
        return torch.einsum("eij,enj->eni", world_to_nav, vectors_w)

    def _delay_points_with_root(self, points_w: torch.Tensor) -> torch.Tensor:
        root_pos_w = self.robot.data.root_pos_w
        root_rot_w = quat_to_matrix(self.robot.data.root_quat_w)
        delayed_rot_w = quat_to_matrix(self._delayed_root_quat_w)
        local_points = torch.einsum("eji,eni->enj", root_rot_w, points_w - root_pos_w.unsqueeze(1))
        return self._delayed_root_pos_w.unsqueeze(1) + torch.einsum("eij,enj->eni", delayed_rot_w, local_points)

    def _group(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        return tensor[:, self._probe_slices[name], ...]

    def _compute_pf_command(self, gf_world: torch.Tensor, bf_world: torch.Tensor) -> torch.Tensor:
        pelvis_gf = self._group(gf_world, "pelvis").squeeze(1)
        control_gf = torch.cat(
            (self._group(gf_world, "head"), self._group(gf_world, "feet"), self._group(gf_world, "hands")),
            dim=1,
        )
        control_bf = torch.cat(
            (self._group(bf_world, "head"), self._group(bf_world, "feet"), self._group(bf_world, "hands")),
            dim=1,
        )

        velocity_xy = pelvis_gf[:, :2] * self.cfg.command.root_gain
        boundary_norm = torch.linalg.norm(control_bf[..., :2], dim=-1, keepdim=True).clamp_min(EPS)
        boundary_hat = control_bf[..., :2] / boundary_norm
        ls = torch.sum(boundary_hat * control_gf[..., :2], dim=-1)
        bv = torch.sum(boundary_hat * velocity_xy.unsqueeze(1), dim=-1)
        denom = torch.sum(boundary_hat * boundary_hat, dim=-1, keepdim=True).clamp_min(EPS)
        delta = ((ls - bv).unsqueeze(-1) / denom) * boundary_hat
        delta = torch.where((ls > bv).unsqueeze(-1), delta, torch.zeros_like(delta))
        velocity_xy = velocity_xy + delta.mean(dim=1)

        command = torch.zeros(self.num_envs, 4, device=self.device, dtype=torch.float32)
        command[:, 0] = 1.0
        command[:, 1:3] = velocity_xy
        command *= self.cfg.command.output_scale
        stop_mask = torch.linalg.norm(command[:, 1:4], dim=-1) < self.cfg.command.stop_speed_threshold
        command[stop_mask] = 0.0
        return command

    def _normalize_pf_vectors(
        self,
        gf_world: torch.Tensor,
        bf_world: torch.Tensor,
        move_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        gf_world = gf_world / torch.linalg.norm(gf_world, dim=-1, keepdim=True).clamp_min(EPS)
        gf_world = gf_world * move_mask[:, None, :].float()
        bf_world = bf_world / torch.linalg.norm(bf_world, dim=-1, keepdim=True).clamp_min(EPS)
        return gf_world, bf_world

    def _compute_torso_nav_rpy(self, nav_to_world_rot: torch.Tensor) -> torch.Tensor:
        torso_rot_w = quat_to_matrix(self.robot.data.body_quat_w[:, self._torso_body_id, :])
        torso_rot_nav = torch.bmm(nav_to_world_rot.transpose(1, 2), torso_rot_w)
        return self._matrix_to_roll_pitch(torso_rot_nav)

    def _compute_pelvis_nav_rpy(self, nav_to_world_rot: torch.Tensor) -> torch.Tensor:
        pelvis_rot_w = quat_to_matrix(self.robot.data.root_quat_w)
        pelvis_rot_nav = torch.bmm(nav_to_world_rot.transpose(1, 2), pelvis_rot_w)
        return self._matrix_to_roll_pitch(pelvis_rot_nav)

    def _matrix_to_roll_pitch(self, rot_nav: torch.Tensor) -> torch.Tensor:
        roll = torch.atan2(rot_nav[:, 2, 1], rot_nav[:, 2, 2])
        pitch = torch.atan2(
            -rot_nav[:, 2, 0],
            torch.sqrt(rot_nav[:, 2, 1] ** 2 + rot_nav[:, 2, 2] ** 2).clamp_min(EPS),
        )
        return torch.stack((roll, pitch), dim=-1)

    def _sensor_pair_contact(self, sensor_name: str) -> torch.Tensor:
        sensor = self.scene.sensors[sensor_name]
        force_matrix_history = sensor.data.force_matrix_w_history
        if force_matrix_history is None:
            force_matrix = sensor.data.force_matrix_w
            if force_matrix is None:
                return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
            contact_force = torch.linalg.norm(force_matrix, dim=-1)
        else:
            contact_force = torch.linalg.norm(force_matrix_history, dim=-1)
        threshold = float(self.cfg.termination.pair_contact_force_threshold)
        return torch.any(contact_force.reshape(self.num_envs, -1) > threshold, dim=1)

    def _pair_contact_termination(self) -> torch.Tensor:
        return self._sensor_pair_contact("cat_left_pair_contact") | self._sensor_pair_contact("cat_right_pair_contact")

    def _nan_reset_mask(self) -> torch.Tensor:
        nan_reset = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        for tensor in (
            self.robot.data.root_pos_w,
            self.robot.data.root_quat_w,
            self.robot.data.joint_pos,
            self.robot.data.joint_vel,
        ):
            nan_reset |= torch.isnan(tensor.reshape(self.num_envs, -1)).any(dim=1)
        return nan_reset

    def _compute_cat_state(self) -> dict[str, torch.Tensor]:
        positions_w = self._probe_positions_world()
        velocities_w = self._probe_velocities_world()
        env_origins = self.scene.env_origins[:, None, :3]
        positions_local = positions_w - env_origins
        nav_to_world_rot = self._compute_nav_rotation()

        current_sample = self.field_loader.sample_all(positions_local)
        current_gf_raw = current_sample["gf"]
        current_bf_raw = current_sample["bf"]
        current_sdf = current_sample["sdf"]

        update_mask = torch.remainder(self.episode_length_buf - 1, self.cfg.delay.update_interval_steps) == 0
        update_mask |= self.episode_length_buf <= 1
        if torch.any(update_mask):
            self._delayed_root_pos_w[update_mask] = self.robot.data.root_pos_w[update_mask]
            self._delayed_root_quat_w[update_mask] = self.robot.data.root_quat_w[update_mask]

        delayed_positions_w = self._delay_points_with_root(positions_w)
        delayed_positions_local = delayed_positions_w - env_origins
        delayed_sample = self.field_loader.sample_all(delayed_positions_local)
        delayed_gf_raw = delayed_sample["gf"]
        delayed_bf_raw = delayed_sample["bf"]
        delayed_sdf = delayed_sample["sdf"]

        command_current_raw_world = self._compute_pf_command(current_gf_raw, current_bf_raw)
        command_current_world, after_stop = self._apply_stop_command_state_machine(command_current_raw_world)

        if torch.any(after_stop):
            stopped_env_ids = after_stop.nonzero(as_tuple=False).flatten()
            self._gait_phase[stopped_env_ids] = STANCE_PHASE.to(self.device)
            self._update_gait_mask(stopped_env_ids)

        move_mask = command_current_world[:, 0:1] > 0.5
        current_gf_world, current_bf_world = self._normalize_pf_vectors(current_gf_raw, current_bf_raw, move_mask)
        delayed_gf_world, delayed_bf_world = self._normalize_pf_vectors(delayed_gf_raw, delayed_bf_raw, move_mask)

        command_delay_world = self._compute_pf_command(delayed_gf_world, delayed_bf_world)

        current_gf_nav = self._world_to_nav(current_gf_world, nav_to_world_rot)
        current_bf_nav = self._world_to_nav(current_bf_world, nav_to_world_rot)
        delayed_gf_nav = self._world_to_nav(delayed_gf_world, nav_to_world_rot)
        delayed_bf_nav = self._world_to_nav(delayed_bf_world, nav_to_world_rot)

        command_current_nav = command_current_world.clone()
        command_current_nav[:, 1:4] = self._world_to_nav(command_current_world[:, 1:4], nav_to_world_rot)
        command_current_nav[:, 3] = 0.0

        command_delay_nav = command_delay_world.clone()
        command_delay_nav[:, 1:4] = self._world_to_nav(command_delay_world[:, 1:4], nav_to_world_rot)
        command_delay_nav[:, 3] = 0.0

        actor_command_nav = command_current_nav.clone()
        actor_command_nav[:, 1:4] = command_delay_nav[:, 1:4]
        actor_command_nav[:, 3] = 0.0

        actor_sdf = torch.clamp(delayed_sdf, min=-self.cfg.field.sdf_clip, max=self.cfg.field.obs_sdf_max)
        actor_bf_nav = delayed_bf_nav * (delayed_sdf.unsqueeze(-1) < self.cfg.field.bf_clear_threshold).float()
        pelvis_nav_rpy = self._compute_pelvis_nav_rpy(nav_to_world_rot)
        torso_nav_rpy = self._compute_torso_nav_rpy(nav_to_world_rot)
        torso_ang_vel_nav = self._world_to_nav(self.robot.data.body_ang_vel_w[:, self._torso_body_id, :], nav_to_world_rot)

        self.command_generator.command[:, :] = command_current_nav[:, 1:4]

        state = {
            "positions_w": positions_w,
            "positions_local": positions_local,
            "velocities_w": velocities_w,
            "delayed_positions_w": delayed_positions_w,
            "delayed_positions_local": delayed_positions_local,
            "current_sdf": current_sdf,
            "current_obs": current_sample["obs"],
            "current_gf_world": current_gf_world,
            "current_bf_world": current_bf_world,
            "current_gf_nav": current_gf_nav,
            "current_bf_nav": current_bf_nav,
            "delay_sdf": delayed_sdf,
            "delay_obs": delayed_sample["obs"],
            "delay_gf_world": delayed_gf_world,
            "delay_bf_world": delayed_bf_world,
            "delay_gf_nav": delayed_gf_nav,
            "delay_bf_nav": delayed_bf_nav,
            "actor_sdf": actor_sdf,
            "actor_bf_nav": actor_bf_nav,
            "command_current_raw_world": command_current_raw_world,
            "command_current_world": command_current_world,
            "command_current_nav": command_current_nav,
            "command_delay_world": command_delay_world,
            "command_delay_nav": command_delay_nav,
            "command_actor_nav": actor_command_nav,
            "last_command_world": self._last_command_world.clone(),
            "stop_timestep": self._stop_timestep.clone(),
            "nav_to_world_rot": nav_to_world_rot,
            "pelvis_nav_rpy": pelvis_nav_rpy,
            "torso_nav_rpy": torso_nav_rpy,
            "torso_ang_vel_nav": torso_ang_vel_nav,
            "field_dir": str(self.field_loader.field_dir),
            "obstacle_asset": str(resolve_field_dir(self.cfg.scene.mesh_obstacle.source_path))
            if self.cfg.scene.mesh_obstacle is not None and self.cfg.scene.mesh_obstacle.source_path
            else "",
        }
        self.latest_field_cache = state
        self._cat_state_step = int(self.sim_step_counter)
        return state

    def _ensure_cat_state(self) -> dict[str, torch.Tensor]:
        if self._cat_state_step != int(self.sim_step_counter) or not self.latest_field_cache:
            return self._compute_cat_state()
        return self.latest_field_cache

    def _flatten_pf_groups(self, gf: torch.Tensor, bf: torch.Tensor, sdf: torch.Tensor) -> torch.Tensor:
        parts = []
        for name in GROUP_ORDER:
            parts.append(self._group(gf, name).reshape(self.num_envs, -1))
            parts.append(self._group(bf, name).reshape(self.num_envs, -1))
            parts.append(self._group(sdf, name).reshape(self.num_envs, -1))
        return torch.cat(parts, dim=-1)

    def init_obs_buffer(self):
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            obs_joint_dim = len(self._obs_joint_ids)
            action_dim = self.num_actions
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales

            cursor = 0
            noise_vec[cursor : cursor + 3] = noise_scales.ang_vel * self.obs_scales.ang_vel
            cursor += 3
            noise_vec[cursor : cursor + 3] = noise_scales.projected_gravity * self.obs_scales.projected_gravity
            cursor += 3
            noise_vec[cursor : cursor + obs_joint_dim] = noise_scales.joint_pos * self.obs_scales.joint_pos
            cursor += obs_joint_dim
            noise_vec[cursor : cursor + obs_joint_dim] = noise_scales.joint_vel * self.obs_scales.joint_vel
            cursor += obs_joint_dim
            cursor += action_dim
            cursor += action_dim
            cursor += 4
            cursor += 1
            cursor += 4
            self.noise_scale_vec = noise_vec

            if self.cfg.scene.height_scanner.enable_height_scan:
                height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                    - self.cfg.normalization.height_scan_offset
                )
                height_scan_noise_vec = torch.zeros_like(height_scan[0])
                height_scan_noise_vec[:] = noise_scales.height_scan * self.obs_scales.height_scan
                self.height_scan_noise_vec = height_scan_noise_vec

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

    def compute_current_observations(self):
        field = self._ensure_cat_state()
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        ang_vel = self.robot.data.root_ang_vel_b * self.obs_scales.ang_vel
        projected_gravity = self.robot.data.projected_gravity_b * self.obs_scales.projected_gravity
        joint_pos = (
            (self.robot.data.joint_pos - self.robot.data.default_joint_pos)[:, self._obs_joint_ids]
            * self.obs_scales.joint_pos
        )
        joint_vel = (self.robot.data.joint_vel - self.robot.data.default_joint_vel)[:, self._obs_joint_ids] * self.obs_scales.joint_vel
        last_action = self.action_buffer._circular_buffer.buffer[:, -1, :] * self.obs_scales.actions
        motor_targets = self._motor_targets[:, self.action_joint_ids]
        gait_phase = torch.cat((torch.cos(self._gait_phase), torch.sin(self._gait_phase)), dim=-1)
        feet_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1), dim=1)[0] > 0.5
        ).float()

        actor_pf_obs = self._flatten_pf_groups(field["delay_gf_nav"], field["actor_bf_nav"], field["actor_sdf"])
        critic_pf_obs = self._flatten_pf_groups(field["current_gf_world"], field["current_bf_world"], field["current_sdf"])
        head_pos_w = self._group(field["positions_w"], "head").reshape(self.num_envs, -1)
        head_vel_w = self._group(field["velocities_w"], "head").reshape(self.num_envs, -1)
        pelvis_pos_w = self._group(field["positions_w"], "pelvis").reshape(self.num_envs, -1)
        torso_pos_w = self._group(field["positions_w"], "torso").reshape(self.num_envs, -1)
        feet_pos_w = self._group(field["positions_w"], "feet").reshape(self.num_envs, -1)
        feet_vel_w = self._group(field["velocities_w"], "feet").reshape(self.num_envs, -1)
        hands_pos_w = self._group(field["positions_w"], "hands").reshape(self.num_envs, -1)
        hands_vel_w = self._group(field["velocities_w"], "hands").reshape(self.num_envs, -1)
        rfi_action_scale = self._rfi_lim_scale[:, self.action_joint_ids]

        actor_obs = torch.cat(
            (
                ang_vel,
                projected_gravity,
                joint_pos,
                joint_vel,
                last_action,
                motor_targets,
                field["command_actor_nav"] * self.obs_scales.commands,
                self._foot_height_target,
                gait_phase,
                actor_pf_obs,
            ),
            dim=-1,
        )

        critic_obs = torch.cat(
            (
                ang_vel,
                projected_gravity,
                joint_pos,
                joint_vel,
                last_action,
                motor_targets,
                field["command_current_world"] * self.obs_scales.commands,
                self._foot_height_target,
                gait_phase,
                critic_pf_obs,
                self.robot.data.root_lin_vel_b * self.obs_scales.lin_vel,
                head_pos_w,
                head_vel_w,
                pelvis_pos_w,
                torso_pos_w,
                feet_pos_w,
                feet_vel_w,
                hands_pos_w,
                hands_vel_w,
                field["torso_nav_rpy"],
                self._gait_mask,
                feet_contact,
                self._kp_scale,
                self._kd_scale,
                rfi_action_scale,
                field["torso_ang_vel_nav"],
            ),
            dim=-1,
        )

        return actor_obs, critic_obs

    def compute_observations(self):
        actor_obs, critic_obs = super().compute_observations()
        if self.latest_field_cache:
            self.extras["cat_traverse"] = {
                "field_dir": self.latest_field_cache["field_dir"],
                "obstacle_asset": self.latest_field_cache["obstacle_asset"],
                "min_sdf": self.latest_field_cache["current_sdf"].amin(dim=1),
                "min_sdf_delay": self.latest_field_cache["delay_sdf"].amin(dim=1),
                "command_nav": self.latest_field_cache["command_current_nav"],
                "policy_command_nav": self.latest_field_cache["command_actor_nav"],
                "stop_timestep": self.latest_field_cache["stop_timestep"],
            }
        return actor_obs, critic_obs

    def reset(self, env_ids):
        super().reset(env_ids)
        if len(env_ids) == 0:
            return
        self._reset_gait_and_delay(env_ids)
        self._reset_domain_randomization(env_ids)
        self.command_generator.command[env_ids] = 0.0
        self._cat_state_step = -1
        self.latest_field_cache = {}
        self.scene.write_data_to_sim()
        self.sim.forward()

    def step(self, actions: torch.Tensor):
        delayed_actions = self.action_buffer.compute(actions)
        clipped_actions = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)
        previous_action_targets = self._motor_targets[:, self.action_joint_ids]
        updated_action_targets = previous_action_targets + clipped_actions * self.action_scale
        soft_lower = self.robot.data.soft_joint_pos_limits[:, self.action_joint_ids, 0]
        soft_upper = self.robot.data.soft_joint_pos_limits[:, self.action_joint_ids, 1]
        updated_action_targets = torch.maximum(updated_action_targets, soft_lower)
        updated_action_targets = torch.minimum(updated_action_targets, soft_upper)
        self._motor_targets.copy_(self.robot.data.default_joint_pos)
        self._motor_targets[:, self.action_joint_ids] = updated_action_targets

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

    def check_reset(self):
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        reset_buf = time_out_buf.clone()
        field = self._ensure_cat_state()

        projected_gravity = self.robot.data.projected_gravity_b
        fall_threshold = float(self.cfg.termination.fall_projected_gravity_z_threshold)
        fall_termination = projected_gravity[:, 2] > fall_threshold

        head_height = self._group(field["positions_w"], "head")[..., 2].amax(dim=1)
        head_low = head_height < self.cfg.probes.head_height_threshold

        pair_contact_termination = self._pair_contact_termination()
        pf_collision_termination = torch.any(field["current_sdf"] < -self.cfg.field.collision_threshold, dim=1)
        contact_grace_over = self.episode_length_buf >= self.cfg.field.grace_steps
        contact_termination = (pair_contact_termination | pf_collision_termination) & contact_grace_over

        nan_reset = self._nan_reset_mask()

        reset_buf |= fall_termination | head_low | contact_termination | nan_reset
        self.extras["cat_traverse_termination"] = {
            "time_out": time_out_buf,
            "fall": fall_termination,
            "head_low": head_low,
            "pair_contact": pair_contact_termination,
            "pf_collision": pf_collision_termination,
            "nan": nan_reset,
        }
        return reset_buf, time_out_buf
