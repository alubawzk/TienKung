# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# Original code is licensed under BSD-3-Clause.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
# Modifications are licensed under BSD-3-Clause.
#
# This file contains code derived from Isaac Lab Project (BSD-3-Clause license)
# with modifications by Legged Lab Project (BSD-3-Clause license).

import isaaclab.sim as sim_utils
import isaacsim.core.utils.torch as torch_utils  # type: ignore
import numpy as np
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp.commands import (  # noqa: F401
    UniformVelocityCommand,
    UniformVelocityCommandCfg,
)
from isaaclab.managers import (
    CurriculumManager,
    EventManager,
    RewardManager,
    TerminationManager,
)
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
from isaaclab.utils.math import (  # noqa: F401
    quat_apply,
    quat_apply_inverse,
    quat_conjugate,
)
from rsl_rl.env import VecEnv
from rsl_rl.utils.general_motion_loader import AMPLoaderGeneral

from legged_lab.envs.amp.amp_base_config import AmpBaseEnvCfg
from legged_lab.mdp.commands import AnyBaseUniformVelocityCommand
from legged_lab.utils.env_utils.scene import SceneCfg


class AmpBaseEnv(VecEnv):
    def __init__(
        self,
        cfg: AmpBaseEnvCfg,
        headless,
    ):

        self.cfg: AmpBaseEnvCfg = cfg
        self.headless = headless
        self.device = self.cfg.device
        self.physics_dt = self.cfg.sim.dt
        self.step_dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.num_envs = self.cfg.scene.num_envs
        self.seed(cfg.scene.seed)

        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=cfg.sim.dt,
            render_interval=cfg.sim.decimation,
            gravity=cfg.sim.gravity,
            physx=PhysxCfg(gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        self.sim = SimulationContext(sim_cfg)

        scene_cfg = SceneCfg(config=cfg.scene, physics_dt=self.physics_dt, step_dt=self.step_dt)
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]

        if self.cfg.scene.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]

        self.init_command()

        self.reward_manager = RewardManager(self.cfg.reward, self)

        # Visualization AMP loader
        self.amp_loader_display = AMPLoaderGeneral(
            config_files=self.cfg.amp_config_files_display,
            motion_files=self.cfg.amp_motion_files_display,
            control_time_interval=self.physics_dt,
            device=self.device,
        )

        self.init_buffers()
        self.init_obs_buffers()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(self.cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

        self.curriculum_manager = CurriculumManager(self.cfg.curriculum, self)
        self.termination_manager = TerminationManager(self.cfg.termination, self)

        self.reset(env_ids)

    def init_command(self):
        command_cfg = UniformVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.commands.resampling_time_range,
            rel_standing_envs=self.cfg.commands.rel_standing_envs,
            rel_heading_envs=self.cfg.commands.rel_heading_envs,
            heading_command=self.cfg.commands.heading_command,
            heading_control_stiffness=self.cfg.commands.heading_control_stiffness,
            debug_vis=self.cfg.commands.debug_vis,
            ranges=self.cfg.commands.ranges,
        )
        self.command_generator = AnyBaseUniformVelocityCommand(
            cfg=command_cfg,
            env=self,
            asset_cfg=SceneEntityCfg("robot", body_names=[".*torso.*"]),
        )

    def init_buffers(self):
        self.extras = {}

        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.num_actions = self.robot.data.default_joint_pos.shape[1]
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations

        self.action_scale = self.cfg.robot.action_scale
        self.action_buffer = DelayBuffer(
            self.cfg.domain_rand.action_delay.params["max_delay"],
            self.num_envs,
            device=self.device,
        )
        self.action_buffer.compute(
            torch.zeros(
                self.num_envs,
                self.num_actions,
                dtype=torch.float,
                device=self.device,
                requires_grad=False,
            )
        )
        if self.cfg.domain_rand.action_delay.enable:
            time_lags = torch.randint(
                low=self.cfg.domain_rand.action_delay.params["min_delay"],
                high=self.cfg.domain_rand.action_delay.params["max_delay"] + 1,
                size=(self.num_envs,),
                dtype=torch.int,
                device=self.device,
            )
            self.action_buffer.set_time_lag(time_lags, torch.arange(self.num_envs, device=self.device))

        self.robot_cfg = SceneEntityCfg(name="robot")
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(
            name="contact_sensor",
            body_names=self.cfg.robot.terminate_contacts_body_names,
        )
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(name="contact_sensor", body_names=self.cfg.robot.feet_body_names)
        self.feet_cfg.resolve(self.scene)

        # 打印关节名称
        print(f"[INFO] Robot Joint Names: {self.robot.joint_names}")
        # import ipdb; ipdb.set_trace()  # noqa: E402

        self.feet_body_ids, _ = self.robot.find_bodies(name_keys=self.cfg.robot.feet_body_names, preserve_order=True)
        self.elbow_body_ids, _ = self.robot.find_bodies(name_keys=self.cfg.robot.hands_body_names, preserve_order=True)
        self.left_leg_ids, _ = self.robot.find_joints(
            name_keys=self.cfg.robot.left_leg_joint_names, preserve_order=True
        )
        self.right_leg_ids, _ = self.robot.find_joints(
            name_keys=self.cfg.robot.right_leg_joint_names, preserve_order=True
        )
        self.left_arm_ids, _ = self.robot.find_joints(
            name_keys=self.cfg.robot.left_arm_joint_names, preserve_order=True
        )
        self.right_arm_ids, _ = self.robot.find_joints(
            name_keys=self.cfg.robot.right_arm_joint_names, preserve_order=True
        )
        self.ankle_joint_ids, _ = self.robot.find_joints(
            name_keys=self.cfg.robot.ankle_joint_names, preserve_order=True
        )
        self.joint_ids, _ = self.robot.find_joints(
            name_keys=self.amp_loader_display.urdf_link_names, preserve_order=True
        )

        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise

        self.common_step_counter = 0  # 计数器common_step_counter ：记录总步数用于curriculum计算
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.phase_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.sim_step_counter = 0
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.action = torch.zeros(
            self.num_envs,
            self.num_actions,
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.avg_feet_force_per_step = torch.zeros(
            self.num_envs,
            len(self.feet_cfg.body_ids),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )
        self.avg_feet_speed_per_step = torch.zeros(
            self.num_envs,
            len(self.feet_cfg.body_ids),
            dtype=torch.float,
            device=self.device,
            requires_grad=False,
        )

    def init_obs_buffers(self):
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            # noise_vec[:3] = noise_scales.lin_vel * self.obs_scales.lin_vel
            noise_vec[0:3] = noise_scales.ang_vel * self.obs_scales.ang_vel
            noise_vec[3:6] = noise_scales.projected_gravity * self.obs_scales.projected_gravity
            noise_vec[6:9] = 0
            noise_vec[9 : 9 + self.num_actions] = noise_scales.joint_pos * self.obs_scales.joint_pos
            noise_vec[9 + self.num_actions : 9 + self.num_actions * 2] = (
                noise_scales.joint_vel * self.obs_scales.joint_vel
            )
            noise_vec[9 + self.num_actions * 2 : 9 + self.num_actions * 3] = 0.0
            noise_vec[9 + self.num_actions * 3 : 15 + self.num_actions * 3] = 0.0
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
        robot = self.robot
        net_contact_forces = self.contact_sensor.data.net_forces_w_history

        ang_vel = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        command = self.command_generator.command
        joint_pos = robot.data.joint_pos - robot.data.default_joint_pos
        joint_vel = robot.data.joint_vel - robot.data.default_joint_vel
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]
        root_lin_vel = robot.data.root_lin_vel_b
        feet_contact = (
            torch.max(
                torch.norm(net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1),
                dim=1,
            )[0]
            > 0.5
        )

        current_actor_obs = torch.cat(
            [
                # root_lin_vel * self.obs_scales.lin_vel,  # 3
                ang_vel * self.obs_scales.ang_vel,  # 3
                projected_gravity * self.obs_scales.projected_gravity,  # 3
                command * self.obs_scales.commands,  # 3
                joint_pos * self.obs_scales.joint_pos,  # 20
                joint_vel * self.obs_scales.joint_vel,  # 20
                action * self.obs_scales.actions,  # 20
            ],
            dim=-1,
        )
        current_critic_obs = torch.cat(
            [root_lin_vel * self.obs_scales.lin_vel, current_actor_obs, feet_contact], dim=-1
        )

        return current_actor_obs, current_critic_obs

    def compute_observations(self):
        current_actor_obs, current_critic_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec

        self.actor_obs_buffer.append(current_actor_obs)
        self.critic_obs_buffer.append(current_critic_obs)

        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)
        critic_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)
        if self.cfg.scene.height_scanner.enable_height_scan:
            height_scan = (
                self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self.height_scanner.data.ray_hits_w[..., 2]
                - self.cfg.normalization.height_scan_offset
            ) * self.obs_scales.height_scan
            critic_obs = torch.cat([critic_obs, height_scan], dim=-1)
            if self.add_noise:
                height_scan += (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
            actor_obs = torch.cat([actor_obs, height_scan], dim=-1)

        actor_obs = torch.clip(actor_obs, -self.clip_obs, self.clip_obs)
        critic_obs = torch.clip(critic_obs, -self.clip_obs, self.clip_obs)

        return actor_obs, critic_obs

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return

        self.extras["log"] = dict()
        if self.cfg.scene.terrain_generator is not None:
            if self.cfg.scene.terrain_generator.curriculum:
                terrain_levels = self.update_terrain_levels(env_ids)
                self.extras["log"].update(terrain_levels)

        # Update curriculum
        self.curriculum_manager.compute(env_ids=env_ids)

        self.scene.reset(env_ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=env_ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )

        reward_extras = self.reward_manager.reset(env_ids)
        self.extras["log"].update(reward_extras)
        self.extras["time_outs"] = self.time_out_buf

        # 重置Manager和buffer
        self.termination_manager.reset(env_ids)
        self.curriculum_manager.reset(env_ids)
        self.reward_manager.reset(env_ids)
        self.event_manager.reset(env_ids)
        self.command_generator.reset(env_ids)
        self.actor_obs_buffer.reset(env_ids)
        self.critic_obs_buffer.reset(env_ids)
        self.action_buffer.reset(env_ids)
        self.episode_length_buf[env_ids] = 0

        self.scene.write_data_to_sim()
        self.sim.forward()

    def step(self, actions: torch.Tensor):

        delayed_actions = self.action_buffer.compute(actions)

        cliped_actions = torch.clip(delayed_actions, -self.clip_actions, self.clip_actions).to(self.device)
        processed_actions = cliped_actions * self.action_scale + self.robot.data.default_joint_pos

        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(processed_actions)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)
            self.avg_feet_force_per_step += torch.norm(
                self.contact_sensor.data.net_forces_w[:, self.feet_cfg.body_ids, :3],
                dim=-1,
            )
            self.avg_feet_speed_per_step += torch.norm(self.robot.data.body_lin_vel_w[:, self.feet_body_ids, :], dim=-1)

        self.avg_feet_force_per_step /= self.cfg.sim.decimation
        self.avg_feet_speed_per_step /= self.cfg.sim.decimation

        if not self.headless:
            self.sim.render()

        self.common_step_counter += 1
        self.episode_length_buf += 1

        self.command_generator.compute(self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.reset_buf, self.time_out_buf, self.terminated_buf = self.check_reset()
        reward_buf = self.reward_manager.compute(self.step_dt)
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_env_ids = env_ids
        terminal_amp_states = self.get_amp_obs()[env_ids]
        self.reset(env_ids)

        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}

        return actor_obs, reward_buf, self.reset_buf, self.extras, terminal_amp_states

    def check_reset(self):
        reset_buf = self.termination_manager.compute()
        time_out_buf = self.termination_manager.time_outs
        terminated_buf = self.termination_manager.terminated

        return reset_buf, time_out_buf, terminated_buf

    def update_terrain_levels(self, env_ids):
        distance = torch.norm(self.robot.data.root_pos_w[env_ids, :2] - self.scene.env_origins[env_ids, :2], dim=1)
        move_up = distance > self.scene.terrain.cfg.terrain_generator.size[0] / 2
        move_down = (
            distance < torch.norm(self.command_generator.command[env_ids, :2], dim=1) * self.max_episode_length_s * 0.5
        )
        move_down *= ~move_up
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        extras = {"Curriculum/terrain_levels": torch.mean(self.scene.terrain.terrain_levels.float())}
        return extras

    def get_observations(self):
        actor_obs, critic_obs = self.compute_observations()
        self.extras["observations"] = {"critic": critic_obs}
        return actor_obs, self.extras

    def get_amp_obs(self):
        # left_foot_pos = (
        #     self.robot.data.body_state_w[:, self.feet_body_ids[0], :3] - self.robot.data.root_state_w[:, 0:3]
        # )
        # right_foot_pos = (
        #     self.robot.data.body_state_w[:, self.feet_body_ids[1], :3] - self.robot.data.root_state_w[:, 0:3]
        # )
        # left_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), left_foot_pos)
        # right_foot_pos = quat_apply(quat_conjugate(self.robot.data.root_state_w[:, 3:7]), right_foot_pos)
        # left_leg_dof_pos = self.robot.data.joint_pos[:, self.left_leg_ids]
        # right_leg_dof_pos = self.robot.data.joint_pos[:, self.right_leg_ids]
        # left_leg_dof_vel = self.robot.data.joint_vel[:, self.left_leg_ids]
        # right_leg_dof_vel = self.robot.data.joint_vel[:, self.right_leg_ids]
        # left_arm_dof_pos = self.robot.data.joint_pos[:, self.left_arm_ids]
        # right_arm_dof_pos = self.robot.data.joint_pos[:, self.right_arm_ids]
        # left_arm_dof_vel = self.robot.data.joint_vel[:, self.left_arm_ids]
        # right_arm_dof_vel = self.robot.data.joint_vel[:, self.right_arm_ids]
        dof_pos = self.robot.data.joint_pos
        dof_vel = self.robot.data.joint_vel

        # return torch.cat(
        #     (
        #         left_arm_dof_pos,
        #         right_arm_dof_pos,
        #         left_leg_dof_pos,
        #         right_leg_dof_pos,
        #         left_arm_dof_vel,
        #         right_arm_dof_vel,
        #         left_leg_dof_vel,
        #         right_leg_dof_vel,
        #         left_foot_pos,
        #         right_foot_pos,
        #     ),
        #     dim=-1,
        # )
        return torch.cat(
            (
                dof_pos,
                dof_vel,
            ),
            dim=-1,
        )

    def visualize_motion(self, time, clip_id: int = 0):
        """
        Update the robot simulation state based on the AMP motion capture data at a given time.

        This function sets the joint positions and velocities, root position and orientation,
        and linear/angular velocities according to the AMP motion frame at the specified time,
        then steps the simulation and updates the scene.

        Args:
            time (float): The time (in seconds) at which to fetch the AMP motion frame.

        Returns:
            None
        """
        _root_pos, _root_rot, _joint_pos, _joint_vel, _keypoint_pos, _obs_state = (
            self.amp_loader_display.sample_states_on_time(clip_id, time)
        )
        device = self.device
        _root_pos = _root_pos.squeeze(0).to(device)
        _root_rot = _root_rot.squeeze(0).to(device)
        _joint_pos = _joint_pos.squeeze(0).to(device)
        _joint_vel = _joint_vel.squeeze(0).to(device)
        _obs_state = _obs_state.squeeze(0).to(device)

        # dof_pos = torch.zeros((self.num_envs, self.robot.num_joints), device=device)
        # dof_vel = torch.zeros((self.num_envs, self.robot.num_joints), device=device)
        # dof_pos[:, self.left_leg_ids] = _joint_pos[8:14]
        # dof_pos[:, self.right_leg_ids] = _joint_pos[14:20]
        # dof_pos[:, self.left_arm_ids] = _joint_pos[0:4]
        # dof_pos[:, self.right_arm_ids] = _joint_pos[4:8]

        # dof_vel[:, self.left_leg_ids] = _joint_vel[8:14]
        # dof_vel[:, self.right_leg_ids] = _joint_vel[14:20]
        # dof_vel[:, self.left_arm_ids] = _joint_vel[0:4]
        # dof_vel[:, self.right_arm_ids] = _joint_vel[4:8]
        # dof_pos[:, self.joint_ids] = _joint_pos
        # dof_vel[:, self.joint_ids] = _joint_vel

        # dof_pos = _joint_pos
        # dof_vel = _joint_vel
        dof_pos = _obs_state[0 : self.robot.num_joints]
        dof_vel = _obs_state[self.robot.num_joints : self.robot.num_joints * 2]

        root_pos = _root_pos.clone()
        root_pos[2] += 0.3

        root_rot = _root_rot.clone()
        quat_wxyz = torch.tensor(
            [root_rot[3], root_rot[0], root_rot[1], root_rot[2]],
            dtype=torch.float32,
            device=device,
        )

        env_ids = torch.arange(self.num_envs, device=device)

        # root state: [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        root_state = torch.zeros((self.num_envs, 13), device=device)
        root_state[:, 0:3] = torch.tile(root_pos.unsqueeze(0), (self.num_envs, 1))
        root_state[:, 3:7] = torch.tile(quat_wxyz.unsqueeze(0), (self.num_envs, 1))
        # root_state[:, 7:10] = torch.tile(lin_vel.unsqueeze(0), (self.num_envs, 1))
        # root_state[:, 10:13] = torch.tile(ang_vel.unsqueeze(0), (self.num_envs, 1))

        self.robot.write_root_state_to_sim(root_state, env_ids)
        self.robot.write_joint_position_to_sim(dof_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim(dof_vel, env_ids=env_ids)
        self.robot.write_joint_damping_to_sim(torch.zeros_like(dof_pos), env_ids=env_ids)
        self.robot.write_joint_stiffness_to_sim(torch.zeros_like(dof_pos), env_ids=env_ids)

        self.sim.render()
        self.sim.step()
        self.scene.update(dt=self.step_dt)
        # self.scene.update(dt=self.physics_dt)

    @staticmethod
    def seed(seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep  # type: ignore

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)
