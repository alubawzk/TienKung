# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env import BaseEnv
    from legged_lab.envs.tienkung.tienkung_env import TienKungEnv


_EPS = 1.0e-6


def _cat_field(env):
    return env._ensure_cat_state()


def _cat_group(env, tensor: torch.Tensor, name: str) -> torch.Tensor:
    return env._group(tensor, name)


def _cat_probe_body_ids(env, name: str) -> list[int]:
    return env._probe_body_ids[env._probe_slices[name]]


def _quat_to_matrix(quat_wxyz: torch.Tensor) -> torch.Tensor:
    quat_wxyz = quat_wxyz / torch.linalg.norm(quat_wxyz, dim=-1, keepdim=True).clamp_min(_EPS)
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
        (
            torch.stack((ww + xx - yy - zz, 2.0 * (xy - wz), 2.0 * (xz + wy)), dim=-1),
            torch.stack((2.0 * (xy + wz), ww - xx + yy - zz, 2.0 * (yz - wx)), dim=-1),
            torch.stack((2.0 * (xz - wy), 2.0 * (yz + wx), ww - xx - yy + zz), dim=-1),
        ),
        dim=-2,
    )


def cat_tracking_root_field(env) -> torch.Tensor:
    field = _cat_field(env)
    cmd_vel = field["command_current_world"][:, 1:4]
    lin_vel = env.robot.data.root_lin_vel_w[:, :3]
    lin_vel_error = torch.sum(torch.square(cmd_vel[:, :2] - lin_vel[:, :2]), dim=1)
    return torch.exp(-4.0 * lin_vel_error)


def cat_body_motion(env) -> torch.Tensor:
    field = _cat_field(env)
    cmd_xy = field["command_current_world"][:, 1:3]
    cmd_norm = torch.linalg.norm(cmd_xy, dim=1, keepdim=True)
    is_zero_cmd = cmd_norm.squeeze(1) < _EPS
    cmd_dir = torch.where(is_zero_cmd.unsqueeze(-1), torch.zeros_like(cmd_xy), cmd_xy / cmd_norm.clamp_min(_EPS))

    lin_xy = env.robot.data.root_lin_vel_w[:, :2]
    lin_xy_orth = lin_xy - torch.sum(lin_xy * cmd_dir, dim=1, keepdim=True) * cmd_dir
    cost_lin_xy_orth = torch.where(is_zero_cmd, torch.zeros_like(cmd_norm.squeeze(1)), torch.sum(torch.square(lin_xy_orth), dim=1))

    torso_ang_vel_nav = field["torso_ang_vel_nav"]
    cost = 1.2 * cost_lin_xy_orth + 0.4 * torch.abs(torso_ang_vel_nav[:, 0]) + 0.4 * torch.abs(torso_ang_vel_nav[:, 1])
    return torch.nan_to_num(cost)


def cat_tracking_orientation(env, torso_height_upper: float = 1.0) -> torch.Tensor:
    field = _cat_field(env)
    pelvis_rpy = field["pelvis_nav_rpy"]
    torso_rpy = field["torso_nav_rpy"]
    head_height = _cat_group(env, field["positions_local"], "head")[..., 2].amax(dim=1)
    idle_mask = head_height > (torso_height_upper + 0.1)

    err_roll = torch.abs(pelvis_rpy[:, 0]) + torch.abs(torso_rpy[:, 0])
    err_pitch_dire = torch.abs(torch.clamp(torso_rpy[:, 1], min=-math.pi, max=0.0))
    err_pitch_idle = idle_mask.float() * torch.abs(torso_rpy[:, 1])
    err_ori = err_roll + err_pitch_dire + err_pitch_idle
    rew = torch.exp(-0.5 * err_ori) - err_pitch_dire
    return torch.nan_to_num(rew)


def cat_foot_contact(env, threshold: float = 0.5) -> torch.Tensor:
    field = _cat_field(env)
    net_contact_forces = env.contact_sensor.data.net_forces_w_history
    feet_contact = torch.max(torch.norm(net_contact_forces[:, :, env.feet_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    stance = feet_contact.float()
    swing = (~feet_contact).float()
    gait_flag = env._gait_mask
    stance_des = (gait_flag == 1).float()
    swing_des = (gait_flag == -1).float()
    is_constrained = (gait_flag != 0).float()
    cost = torch.sum(torch.abs(stance - stance_des) * is_constrained, dim=1)
    cost += torch.sum(torch.abs(swing - swing_des) * is_constrained, dim=1)
    cost *= field["command_current_world"][:, 0]
    return torch.nan_to_num(cost)


def cat_foot_clearance(env, foot_height_stance: float = 0.0) -> torch.Tensor:
    field = _cat_field(env)
    foot_z = _cat_group(env, field["positions_local"], "feet")[..., 2]
    swing_des = (env._gait_mask == -1).float()
    foot_z_target = foot_height_stance + env._foot_height_target
    cost = torch.sum(swing_des * torch.square(foot_z - foot_z_target), dim=1)
    cost *= field["command_current_world"][:, 0]
    return torch.nan_to_num(cost)


def cat_foot_slip(env) -> torch.Tensor:
    field = _cat_field(env)
    stance_des = (env._gait_mask == 1).float()
    feet_vel = torch.linalg.norm(_cat_group(env, field["velocities_w"], "feet"), dim=-1)
    cost = torch.sum(torch.square(feet_vel) * stance_des, dim=1)
    return torch.nan_to_num(cost)


def cat_foot_balance(env) -> torch.Tensor:
    field = _cat_field(env)
    feet_pos_w = _cat_group(env, field["positions_w"], "feet")
    world_to_nav = field["nav_to_world_rot"].transpose(1, 2)
    root_pos_w = env.robot.data.root_pos_w[:, :3]

    support_world = torch.stack((root_pos_w, feet_pos_w[:, 0, :], feet_pos_w[:, 1, :]), dim=1)
    support_nav = torch.einsum("eij,ekj->eki", world_to_nav, support_world - root_pos_w.unsqueeze(1))
    foot_to_com_err = support_nav[:, 1:, :] - support_nav[:, 0:1, :]
    foot_center = foot_to_com_err[:, 0, :2] + foot_to_com_err[:, 1, :2]
    cost_support = torch.sum(torch.square(foot_center), dim=1)

    foot_distance = torch.linalg.norm(feet_pos_w[:, 0, :] - feet_pos_w[:, 1, :], dim=1)
    foot_spread_penalty = torch.where(foot_distance < 0.35, 0.35 - foot_distance, torch.zeros_like(foot_distance)) * 10.0
    return torch.nan_to_num(cost_support * (1.0 + foot_spread_penalty))


def cat_straight_knee(
    env,
    joint_patterns: list[str],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    import re

    asset: Articulation = env.scene[asset_cfg.name]
    joint_names = asset.data.joint_names
    joint_ids = [i for i, name in enumerate(joint_names) if any(re.fullmatch(pat, name) for pat in joint_patterns)]
    if not joint_ids:
        return torch.zeros(env.num_envs, device=env.device)
    joint_ids_t = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
    knee_pos = asset.data.joint_pos[:, joint_ids_t]
    penalty = torch.clamp(0.1 - knee_pos, min=0.0)
    return torch.nan_to_num(torch.sum(penalty, dim=1))


def cat_foot_far(env) -> torch.Tensor:
    field = _cat_field(env)
    feet_pos_w = _cat_group(env, field["positions_w"], "feet")
    foot_distance = torch.linalg.norm(feet_pos_w[:, 0, :] - feet_pos_w[:, 1, :], dim=1)
    return torch.where(foot_distance < 0.35, 0.35 - foot_distance, torch.zeros_like(foot_distance))


def cat_joint_pos_limits(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = -(asset.data.joint_pos - asset.data.soft_joint_pos_limits[..., 0]).clip(max=0.0)
    out_of_limits += (asset.data.joint_pos - asset.data.soft_joint_pos_limits[..., 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def cat_joint_torque(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque), dim=1)


def cat_smoothness_joint(
    env,
    joint_patterns: list[str] | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    import re

    asset: Articulation = env.scene[asset_cfg.name]
    if joint_patterns:
        joint_ids = [i for i, name in enumerate(asset.data.joint_names) if any(re.fullmatch(pat, name) for pat in joint_patterns)]
        if not joint_ids:
            return torch.zeros(env.num_envs, device=env.device)
        joint_ids_t = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
        qvel = asset.data.joint_vel[:, joint_ids_t]
        last_joint_vel = env._last_joint_vel[:, joint_ids_t]
    else:
        qvel = asset.data.joint_vel
        last_joint_vel = env._last_joint_vel
    qacc = (last_joint_vel - qvel) / env.step_dt
    cost = torch.sum(0.01 * torch.square(qvel) + torch.square(qacc), dim=1)
    return torch.nan_to_num(cost)


def cat_smoothness_action(env) -> torch.Tensor:
    action_hist = env.action_buffer._circular_buffer.buffer
    if action_hist.shape[1] < 3:
        act = action_hist[:, -1, :]
        return torch.nan_to_num(torch.sum(torch.square(act), dim=1))
    act = action_hist[:, -1, :]
    last_act = action_hist[:, -2, :]
    last_last_act = action_hist[:, -3, :]
    smooth_0th = torch.square(act)
    smooth_1st = torch.square(act - last_act)
    smooth_2nd = torch.square(act - 2.0 * last_act + last_last_act)
    return torch.nan_to_num(torch.sum(smooth_0th + smooth_1st + smooth_2nd, dim=1))


def cat_body_rotation(env, yaw_cmd_max: float = 0.5) -> torch.Tensor:
    field = _cat_field(env)
    leg_body_ids = _cat_probe_body_ids(env, "knees") + _cat_probe_body_ids(env, "feet")
    leg_quat_w = env.robot.data.body_quat_w[:, leg_body_ids, :]
    leg_rot_w = _quat_to_matrix(leg_quat_w)
    world_to_nav = field["nav_to_world_rot"].transpose(1, 2).unsqueeze(1)
    leg_rot_nav = torch.matmul(world_to_nav, leg_rot_w)

    cmd_vel = field["command_current_world"][:, 3]
    cmd_decay = torch.clamp((yaw_cmd_max - torch.abs(cmd_vel)) / max(yaw_cmd_max, _EPS), min=0.0, max=1.0) ** 2
    axis_roll_err = torch.mean(torch.abs(leg_rot_nav[:, :, 2, 1]), dim=1)
    axis_yaw_err = torch.mean(cmd_decay.unsqueeze(1) * torch.abs(leg_rot_nav[:, :, 0, 1]), dim=1)
    return torch.nan_to_num(torch.exp(-5.0 * (axis_roll_err + axis_yaw_err)))


def cat_feet_rotation(env) -> torch.Tensor:
    field = _cat_field(env)
    knee_body_ids = _cat_probe_body_ids(env, "knees")
    foot_body_ids = _cat_probe_body_ids(env, "feet")
    knee_rot_w = _quat_to_matrix(env.robot.data.body_quat_w[:, knee_body_ids, :])
    foot_rot_w = _quat_to_matrix(env.robot.data.body_quat_w[:, foot_body_ids, :])
    world_to_nav = field["nav_to_world_rot"].transpose(1, 2).unsqueeze(1)
    knee_rot_nav = torch.matmul(world_to_nav, knee_rot_w)
    foot_rot_nav = torch.matmul(world_to_nav, foot_rot_w)

    knees_roll_err = torch.sum(torch.abs(knee_rot_nav[:, :, 2, 1]), dim=1)
    knees_yaw_err = torch.sum(torch.abs(knee_rot_nav[:, :, 0, 1]), dim=1)
    ankles_roll_err = torch.sum(torch.abs(foot_rot_nav[:, :, 1, 2]), dim=1)
    ankles_pitch_err = torch.sum(torch.abs(foot_rot_nav[:, :, 0, 2]), dim=1)
    ankles_yaw_err = torch.sum(torch.square(foot_rot_nav[:, :, 0, 1]), dim=1)
    return torch.nan_to_num(
        torch.exp(-(knees_roll_err + knees_yaw_err + ankles_roll_err + ankles_pitch_err + ankles_yaw_err))
    )


def cat_pf_alignment_reward(
    env,
    group_name: str,
    tau: float,
    crossed_x: float = 1.5,
    block_stance_feet: bool = False,
) -> torch.Tensor:
    field = _cat_field(env)
    gf_vel = _cat_group(env, field["current_gf_world"], group_name)
    lin_vel = _cat_group(env, field["velocities_w"], group_name)
    sdf = _cat_group(env, field["current_sdf"], group_name)
    pos_x = _cat_group(env, field["positions_local"], group_name)[..., 0]

    g_norm = gf_vel / torch.linalg.norm(gf_vel, dim=-1, keepdim=True).clamp_min(_EPS)
    v_norm = lin_vel / torch.linalg.norm(lin_vel, dim=-1, keepdim=True).clamp_min(_EPS)
    cos_align = torch.sum(g_norm * v_norm, dim=-1)

    window = torch.sigmoid(40.0 * (tau - sdf))
    reward_near = window * (5.0 * cos_align)

    crossed = (field["command_current_world"][:, 0:1] < 0.5) | (pos_x > crossed_x)
    if block_stance_feet:
        crossed = crossed | (env._gait_mask == 1)
    reward_near = torch.where(crossed, torch.full_like(reward_near, 4.0), reward_near)
    return torch.nan_to_num(torch.mean(reward_near, dim=1))


def cat_pf_sdf_penalty(
    env,
    group_name: str,
    sdf_safe: float = 0.05,
    beta_inside: float = 0.02,
    pen_inside_scale: float = 20.0,
) -> torch.Tensor:
    field = _cat_field(env)
    sdf = _cat_group(env, field["current_sdf"], group_name)
    pen_inside = torch.nn.functional.softplus((sdf_safe - sdf) / beta_inside)
    penalty = pen_inside_scale * pen_inside
    return torch.nan_to_num(-torch.mean(penalty, dim=1))


def track_lin_vel_xy_yaw_frame_exp(
    env: BaseEnv | TienKungEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_apply_inverse(
        math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3]
    )
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: BaseEnv | TienKungEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def lin_vel_z_l2(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def energy(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_acc_l2(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: BaseEnv | TienKungEnv) -> torch.Tensor:
    return torch.sum(
        torch.square(
            env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]
        ),
        dim=1,
    )


def undesired_contacts(env: BaseEnv | TienKungEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def fly(env: BaseEnv | TienKungEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5


def flat_orientation_l2(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)


def is_terminated(env: BaseEnv | TienKungEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.reset_buf * ~env.time_out_buf


def feet_air_time_positive_biped(
    env: BaseEnv | TienKungEnv, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) > 0.1
    return reward


def feet_slide(
    env: BaseEnv | TienKungEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def body_force(
    env: BaseEnv | TienKungEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def joint_deviation_l1(env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    zero_flag = (
        torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])
    ) < 0.1
    return torch.sum(torch.abs(angle), dim=1) * zero_flag


def body_orientation_l2(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_apply_inverse(
        asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W
    )
    return torch.sum(torch.square(body_orientation[:, :2]), dim=1)


def feet_stumble(env: BaseEnv | TienKungEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(
        torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
        > 5 * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]),
        dim=1,
    )


def feet_too_near_humanoid(
    env: BaseEnv | TienKungEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


# Regularization Reward
def ankle_torque(env: TienKungEnv) -> torch.Tensor:
    """Penalize large torques on the ankle joints."""
    return torch.sum(torch.square(env.robot.data.applied_torque[:, env.ankle_joint_ids]), dim=1)


def ankle_action(env: TienKungEnv) -> torch.Tensor:
    """Penalize ankle joint actions."""
    return torch.sum(torch.abs(env.action[:, env.ankle_joint_ids]), dim=1)


def hip_roll_action(env: TienKungEnv) -> torch.Tensor:
    """Penalize hip roll joint actions."""
    return torch.sum(torch.abs(env.action[:, [env.left_leg_ids[0], env.right_leg_ids[0]]]), dim=1)


def hip_yaw_action(env: TienKungEnv) -> torch.Tensor:
    """Penalize hip yaw joint actions."""
    return torch.sum(torch.abs(env.action[:, [env.left_leg_ids[2], env.right_leg_ids[2]]]), dim=1)


def feet_y_distance(env: TienKungEnv) -> torch.Tensor:
    """Penalize foot y-distance when the commanded y-velocity is low, to maintain a reasonable spacing."""
    leftfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[0], :] - env.robot.data.root_link_pos_w[:, :]
    rightfoot = env.robot.data.body_pos_w[:, env.feet_body_ids[1], :] - env.robot.data.root_link_pos_w[:, :]
    leftfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), leftfoot)
    rightfoot_b = math_utils.quat_apply(math_utils.quat_conjugate(env.robot.data.root_link_quat_w[:, :]), rightfoot)
    y_distance_b = torch.abs(leftfoot_b[:, 1] - rightfoot_b[:, 1] - 0.299)
    y_vel_flag = torch.abs(env.command_generator.command[:, 1]) < 0.1
    return y_distance_b * y_vel_flag


# Periodic gait-based reward function
def gait_clock(phase, air_ratio, delta_t):
    """
    Generate periodic gait clock signals for foot swing and stance phases.

    This function constructs two phase-dependent signals:
    - `I_frc`: active during swing phase (used for penalizing ground force)
    - `I_spd`: active during stance phase (used for penalizing foot speed)

    Transitions between swing and stance are smoothed within a margin of `delta_t`
    to create differentiable transitions.

    Parameters
    ----------
    phase : torch.Tensor
        Normalized gait phase in [0, 1], shape: [num_envs].
    air_ratio : torch.Tensor
        Proportion of the gait cycle spent in swing phase, shape: [num_envs].
    delta_t : float
        Transition width around phase boundaries for smooth interpolation.

    Returns
    -------
    I_frc : torch.Tensor
        Gait-based swing-phase clock signal, range [0, 1], shape: [num_envs].
    I_spd : torch.Tensor
        Gait-based stance-phase clock signal, range [0, 1], shape: [num_envs].

    Notes
    -----
    - The transitions at the boundaries (e.g., swing→stance) are linear interpolations.
    - Used in reward shaping to associate expected behavior with gait phases.
    """
    swing_flag = (phase >= delta_t) & (phase <= (air_ratio - delta_t))
    stand_flag = (phase >= (air_ratio + delta_t)) & (phase <= (1 - delta_t))

    trans_flag1 = phase < delta_t
    trans_flag2 = (phase > (air_ratio - delta_t)) & (phase < (air_ratio + delta_t))
    trans_flag3 = phase > (1 - delta_t)

    I_frc = (
        1.0 * swing_flag
        + (0.5 + phase / (2 * delta_t)) * trans_flag1
        - (phase - air_ratio - delta_t) / (2.0 * delta_t) * trans_flag2
        + 0.0 * stand_flag
        + (phase - 1 + delta_t) / (2 * delta_t) * trans_flag3
    )
    I_spd = 1.0 - I_frc
    return I_frc, I_spd


def gait_feet_frc_perio(env: TienKungEnv, delta_t: float = 0.02) -> torch.Tensor:
    """Penalize foot force during the swing phase of the gait."""
    left_frc_swing_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[0]
    right_frc_swing_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[0]
    left_frc_score = left_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_swing_mask * (torch.exp(-200 * torch.square(env.avg_feet_force_per_step[:, 1])))
    return left_frc_score + right_frc_score


def gait_feet_spd_perio(env: TienKungEnv, delta_t: float = 0.02) -> torch.Tensor:
    """Penalize foot speed during the support phase of the gait."""
    left_spd_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_spd_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_spd_score = left_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 0])))
    right_spd_score = right_spd_support_mask * (torch.exp(-100 * torch.square(env.avg_feet_speed_per_step[:, 1])))
    return left_spd_score + right_spd_score


def gait_feet_frc_support_perio(env: TienKungEnv, delta_t: float = 0.02) -> torch.Tensor:
    """Reward that promotes proper support force during stance (support) phase."""
    left_frc_support_mask = gait_clock(env.gait_phase[:, 0], env.phase_ratio[:, 0], delta_t)[1]
    right_frc_support_mask = gait_clock(env.gait_phase[:, 1], env.phase_ratio[:, 1], delta_t)[1]
    left_frc_score = left_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 0])))
    right_frc_score = right_frc_support_mask * (1 - torch.exp(-10 * torch.square(env.avg_feet_force_per_step[:, 1])))
    return left_frc_score + right_frc_score


def joint_pos_limits_exclude(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    exclude_joint_names: list[str] | None = None,
) -> torch.Tensor:
    """Penalize joint positions that exceed soft limits, excluding specified joints."""
    import re
    asset: Articulation = env.scene[asset_cfg.name]
    joint_names = asset.data.joint_names

    if exclude_joint_names:
        include_ids = [
            i for i, name in enumerate(joint_names)
            if not any(re.fullmatch(pat, name) for pat in exclude_joint_names)
        ]
    else:
        include_ids = list(range(len(joint_names)))

    include_ids = torch.tensor(include_ids, device=env.device)
    out_of_limits = -(
        asset.data.joint_pos[:, include_ids]
        - asset.data.soft_joint_pos_limits[:, include_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos[:, include_ids]
        - asset.data.soft_joint_pos_limits[:, include_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)
