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

import argparse
import glob
import os
import struct
import sys
import threading

import mujoco
import mujoco_viewer
import numpy as np
import torch
from pynput import keyboard
import time

JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS = 0x02
JS_EVENT_INIT = 0x80
JS_EVENT_SIZE = 8


class JoystickReader:
    """Read /dev/input/js* device directly without ROS2 dependency."""

    def __init__(self, device_path=None):
        self.device_path = device_path
        self.device_file = None
        self.axes = {}
        self.buttons = {}
        self.running = False
        self.read_thread = None

    def find_joystick_device(self, target_name=None):
        devices = sorted(glob.glob("/dev/input/js*"))
        if not devices:
            return None
        if self.device_path and os.path.exists(self.device_path):
            return self.device_path
        if target_name:
            for device in devices:
                try:
                    name_path = device.replace("/dev/input/js", "/sys/class/input/js") + "/device/name"
                    if os.path.exists(name_path):
                        with open(name_path, "r") as f:
                            if target_name.upper() in f.read().strip().upper():
                                return device
                except Exception:
                    pass
        return devices[0]

    def open_device(self, device_path=None, target_name="DF39"):
        if device_path:
            self.device_path = device_path
        else:
            self.device_path = self.find_joystick_device(target_name)
        if not self.device_path:
            all_devices = sorted(glob.glob("/dev/input/js*"))
            if all_devices:
                print(f"[JOYSTICK] Devices found but name '{target_name}' not matched: {all_devices}")
                print(f"[JOYSTICK] Trying first device: {all_devices[0]}")
                self.device_path = all_devices[0]
            else:
                print("[JOYSTICK] No /dev/input/js* devices found. Is the joystick connected?")
                print("[JOYSTICK] Hint: run 'ls /dev/input/js*' to check.")
                return False
        try:
            self.device_file = open(self.device_path, "rb")
            return True
        except PermissionError:
            print(f"[JOYSTICK] Permission denied: {self.device_path}")
            print(f"[JOYSTICK] Fix: sudo chmod 666 {self.device_path}")
            return False
        except Exception as e:
            print(f"[JOYSTICK] Failed to open {self.device_path}: {e}")
            return False

    def _read_loop(self, callback):
        while self.running:
            try:
                data = self.device_file.read(JS_EVENT_SIZE)
                if len(data) != JS_EVENT_SIZE:
                    time.sleep(0.001)
                    continue
                _time_ms, value, event_type, number = struct.unpack("IhBB", data)
                if event_type & JS_EVENT_AXIS:
                    self.axes[number] = value / 32767.0
                elif event_type & JS_EVENT_BUTTON:
                    self.buttons[number] = value
                if callback:
                    callback({"type": event_type, "value": value, "number": number})
            except Exception:
                time.sleep(0.001)

    def start_reading(self, callback=None):
        if self.running or not self.device_file:
            return False
        self.running = True
        self.read_thread = threading.Thread(target=self._read_loop, args=(callback,), daemon=True)
        self.read_thread.start()
        return True

    def get_axis(self, number, default=0.0):
        return self.axes.get(number, default)

    def get_button(self, number, default=0):
        return self.buttons.get(number, default)

    def stop(self):
        self.running = False
        if self.read_thread and self.read_thread.is_alive():
            self.read_thread.join(timeout=1.0)
        if self.device_file:
            self.device_file.close()
            self.device_file = None


class SimToSimCfg:
    """Configuration class for sim2sim parameters.

    Must be kept consistent with the training configuration.
    """

    class sim:
        sim_duration = 100.0
        num_action = 21
        num_obs_per_step = 78
        actor_obs_history_length = 10
        dt = 0.002
        decimation = 10
        clip_observations = 100.0
        clip_actions = 100.0
        action_scale = 0.25

    class robot:
        gait_air_ratio_l: float = 0.38
        gait_air_ratio_r: float = 0.38
        gait_phase_offset_l: float = 0.38
        gait_phase_offset_r: float = 0.88
        gait_cycle: float = 0.55


class MujocoRunner:
    """
    Sim2Sim runner that loads a policy and a MuJoCo model
    to run real-time humanoid control simulation.

    Args:
        cfg (SimToSimCfg): Configuration object for simulation.
        policy_path (str): Path to the TorchScript exported policy.
        model_path (str): Path to the MuJoCo XML model.
    """

    def __init__(self, cfg: SimToSimCfg, policy_path, model_path):
        self.cfg = cfg
        network_path = policy_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.model.opt.timestep = self.cfg.sim.dt

        self.policy = torch.jit.load(network_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer._render_every_frame = False
        self.left_foot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
        self.right_foot_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
        self.init_variables()

    def init_variables(self) -> None:
        """Initialize simulation variables and joint index mappings."""
        self.dt = self.cfg.sim.decimation * self.cfg.sim.dt
        self.dof_pos = np.zeros(self.cfg.sim.num_action)
        self.dof_vel = np.zeros(self.cfg.sim.num_action)
        self.action = np.zeros(self.cfg.sim.num_action)
        self.default_dof_pos = np.array(
            [
                -0.4,  # right_hip_pitch_joint
                -0.1,   # right_hip_roll_joint
                0.0,    # right_hip_yaw_joint
                0.8,    # right_knee_pitch_joint
                -0.45,  # right_ankle_pitch_joint
                0.1,    # right_ankle_roll_joint
                -0.4,  # left_hip_pitch_joint
                0.1,    # left_hip_roll_joint
                0.0,    # left_hip_yaw_joint
                0.8,    # left_knee_pitch_joint
                -0.45,  # left_ankle_pitch_joint
                -0.1,   # left_ankle_roll_joint
                0.0,    # waist_yaw_joint
                0.0,    # right_shoulder_pitch_joint
                -0.25,  # right_shoulder_roll_joint
                0.0,    # right_shoulder_yaw_joint
                1.0,    # right_elbow_pitch_joint
                0.0,    # left_shoulder_pitch_joint
                0.25,   # left_shoulder_roll_joint
                0.0,    # left_shoulder_yaw_joint
                1.0,    # left_elbow_pitch_joint
            ]
        )
        self.episode_length_buf = 0
        self.gait_phase = np.zeros(2)
        self.gait_phase_accum_time = 0.0
        self._was_moving = False
        self.gait_cycle = self.cfg.robot.gait_cycle
        self.phase_ratio = np.array([self.cfg.robot.gait_air_ratio_l, self.cfg.robot.gait_air_ratio_r])
        self.phase_offset = np.array([self.cfg.robot.gait_phase_offset_l, self.cfg.robot.gait_phase_offset_r])
        # PD gains and torque limits in MuJoCo joint order.
        self.kp = np.array(
            [
                35,  # right_hip_pitch_joint
                20,  # right_hip_roll_joint
                20,  # right_hip_yaw_joint
                35,  # right_knee_pitch_joint
                35,  # right_ankle_pitch_joint
                20,  # right_ankle_roll_joint
                35,  # left_hip_pitch_joint
                20,  # left_hip_roll_joint
                20,  # left_hip_yaw_joint
                35,  # left_knee_pitch_joint
                35,  # left_ankle_pitch_joint
                20,  # left_ankle_roll_joint
                20,  # waist_yaw_joint
                20,  # right_shoulder_pitch_joint
                15,  # right_shoulder_roll_joint
                10,  # right_shoulder_yaw_joint
                10,  # right_elbow_pitch_joint
                20,  # left_shoulder_pitch_joint
                15,  # left_shoulder_roll_joint
                10,  # left_shoulder_yaw_joint
                10,  # left_elbow_pitch_joint
            ],
            dtype=np.float64,
        )
        self.kd = np.array(
            [
                2.0,  # right_hip_pitch_joint
                2.0,  # right_hip_roll_joint
                2.0,  # right_hip_yaw_joint
                2.0,  # right_knee_pitch_joint
                1.5,  # right_ankle_pitch_joint
                1.5,  # right_ankle_roll_joint
                2.0,  # left_hip_pitch_joint
                2.0,  # left_hip_roll_joint
                2.0,  # left_hip_yaw_joint
                2.0,  # left_knee_pitch_joint
                1.5,  # left_ankle_pitch_joint
                1.5,  # left_ankle_roll_joint
                1.0,  # waist_yaw_joint
                1.0,  # right_shoulder_pitch_joint
                1.0,  # right_shoulder_roll_joint
                1.0,  # right_shoulder_yaw_joint
                1.0,  # right_elbow_pitch_joint
                1.0,  # left_shoulder_pitch_joint
                1.0,  # left_shoulder_roll_joint
                1.0,  # left_shoulder_yaw_joint
                1.0,  # left_elbow_pitch_joint
            ],
            dtype=np.float64,
        )
        self.torque_limit = np.array(
            [
                97,  # right_hip_pitch_joint
                28,  # right_hip_roll_joint
                28,  # right_hip_yaw_joint
                97,  # right_knee_pitch_joint
                20,  # right_ankle_pitch_joint
                20,  # right_ankle_roll_joint
                97,  # left_hip_pitch_joint
                28,  # left_hip_roll_joint
                28,  # left_hip_yaw_joint
                97,  # left_knee_pitch_joint
                20,  # left_ankle_pitch_joint
                20,  # left_ankle_roll_joint
                28,  # waist_yaw_joint
                10,  # right_shoulder_pitch_joint
                10,  # right_shoulder_roll_joint
                10,  # right_shoulder_yaw_joint
                10,  # right_elbow_pitch_joint
                10,  # left_shoulder_pitch_joint
                10,  # left_shoulder_roll_joint
                10,  # left_shoulder_yaw_joint
                10,  # left_elbow_pitch_joint
            ],
            dtype=np.float64,
        )
        self.mujoco_joint_names = [
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_pitch_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_pitch_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "waist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_pitch_joint",
        ]
        self.qpos_adr = np.array(
            [
                self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
                for name in self.mujoco_joint_names
            ],
            dtype=np.int32,
        )
        self.dof_adr = np.array(
            [
                self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
                for name in self.mujoco_joint_names
            ],
            dtype=np.int32,
        )

        self.mujoco_to_isaac_idx = [
            6,   # left_hip_pitch_joint
            0,   # right_hip_pitch_joint
            12,  # waist_yaw_joint
            7,   # left_hip_roll_joint
            1,   # right_hip_roll_joint
            17,  # left_shoulder_pitch_joint
            13,  # right_shoulder_pitch_joint
            8,   # left_hip_yaw_joint
            2,   # right_hip_yaw_joint
            18,  # left_shoulder_roll_joint
            14,  # right_shoulder_roll_joint
            9,   # left_knee_pitch_joint
            3,   # right_knee_pitch_joint
            19,  # left_shoulder_yaw_joint
            15,  # right_shoulder_yaw_joint
            10,  # left_ankle_pitch_joint
            4,   # right_ankle_pitch_joint
            20,  # left_elbow_pitch_joint
            16,  # right_elbow_pitch_joint
            11,  # left_ankle_roll_joint
            5,   # right_ankle_roll_joint
        ]
        self.isaac_to_mujoco_idx = [
            1,   # right_hip_pitch_joint
            4,   # right_hip_roll_joint
            8,   # right_hip_yaw_joint
            12,  # right_knee_pitch_joint
            16,  # right_ankle_pitch_joint
            20,  # right_ankle_roll_joint
            0,   # left_hip_pitch_joint
            3,   # left_hip_roll_joint
            7,   # left_hip_yaw_joint
            11,  # left_knee_pitch_joint
            15,  # left_ankle_pitch_joint
            19,  # left_ankle_roll_joint
            2,   # waist_yaw_joint
            6,   # right_shoulder_pitch_joint
            10,  # right_shoulder_roll_joint
            14,  # right_shoulder_yaw_joint
            18,  # right_elbow_pitch_joint
            5,   # left_shoulder_pitch_joint
            9,   # left_shoulder_roll_joint
            13,  # left_shoulder_yaw_joint
            17,  # left_elbow_pitch_joint
        ]
        # Initial command vel
        self.command_vel = np.array([0.0, 0.0, 0.0])
        self.ang_vel_sensor_name = self._resolve_sensor_name(["angular-velocity", "base_link_site_angvel"])
        self.orientation_sensor_name = self._resolve_sensor_name(["orientation", "base_link_site_quat"])
        self.obs_history = np.zeros(
            (self.cfg.sim.num_obs_per_step * self.cfg.sim.actor_obs_history_length,), dtype=np.float32
        )
        self.joystick_reader = None
        self.joystick_lock = threading.Lock()
        self.joystick_vx_scale = 1.0
        self.joystick_vy_scale = 0.5
        self.joystick_dyaw_scale = 1.57
        self.joystick_deadzone = 0.1
        # Contact force tracking
        self.left_foot_forces: list[float] = []
        self.right_foot_forces: list[float] = []

    def _resolve_sensor_name(self, candidates: list[str]) -> str:
        """Resolve the first available MuJoCo sensor name from candidates."""
        for name in candidates:
            try:
                self.data.sensor(name)
                return name
            except KeyError:
                continue
        valid_names = [self.model.sensor(i).name for i in range(self.model.nsensor)]
        raise RuntimeError(f"Missing sensors {candidates}. Available sensors: {valid_names}")

    def get_obs(self) -> np.ndarray:
        """
        Compute current observation vector from MuJoCo sensors and internal state.

        Returns:
            np.ndarray: Normalized and clipped observation history.
        """
        self.dof_pos = self.data.qpos[self.qpos_adr].copy()
        self.dof_vel = self.data.qvel[self.dof_adr].copy()

        obs = np.concatenate(
            [
                self.data.sensor(self.ang_vel_sensor_name).data.astype(np.double),  # 3
                self.quat_rotate_inverse(
                    self.data.sensor(self.orientation_sensor_name).data[[1, 2, 3, 0]].astype(np.double),
                    np.array([0, 0, -1]),
                ),  # 3
                self.command_vel,  # 3
                (self.dof_pos - self.default_dof_pos)[self.mujoco_to_isaac_idx],  # 21
                self.dof_vel[self.mujoco_to_isaac_idx],  # 21
                np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions),  # 21
                np.sin(2 * np.pi * self.gait_phase),  # 2
                np.cos(2 * np.pi * self.gait_phase),  # 2
                self.phase_ratio,  # 2
            ],
            axis=0,
        ).astype(np.float32)
        # if self.episode_length_buf % round(0.5 / self.dt) == 0:
        #     print(f"[PHASE] {self.gait_phase[0]:.4f}  {self.gait_phase[1]:.4f}")

        # Update observation history
        self.obs_history = np.roll(self.obs_history, shift=-self.cfg.sim.num_obs_per_step)
        self.obs_history[-self.cfg.sim.num_obs_per_step :] = obs.copy()

        return np.clip(self.obs_history, -self.cfg.sim.clip_observations, self.cfg.sim.clip_observations)

    def torque_control(self) -> np.ndarray:
        """
        Apply PD control in joint space and output torques.

        Returns:
            np.ndarray: Target torques in MuJoCo order.
        """
        actions_scaled = self.action * self.cfg.sim.action_scale
        target_pos = actions_scaled[self.isaac_to_mujoco_idx] + self.default_dof_pos
        dof_pos = self.data.qpos[self.qpos_adr]
        dof_vel = self.data.qvel[self.dof_adr]
        torques = self.kp * (target_pos - dof_pos) - self.kd * dof_vel
        return np.clip(torques, -self.torque_limit, self.torque_limit)

    def _collect_contact_forces(self) -> None:
        """Read MuJoCo contact forces and append magnitudes for each foot."""
        force_buf = np.zeros(6)
        left_f = 0.0
        right_f = 0.0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1_body = self.model.geom_bodyid[contact.geom1]
            g2_body = self.model.geom_bodyid[contact.geom2]
            involved = {g1_body, g2_body}
            if self.left_foot_body_id in involved or self.right_foot_body_id in involved:
                mujoco.mj_contactForce(self.model, self.data, i, force_buf)
                fmag = float(np.linalg.norm(force_buf[:3]))
                if self.left_foot_body_id in involved:
                    left_f += fmag
                if self.right_foot_body_id in involved:
                    right_f += fmag
        self.left_foot_forces.append(left_f)
        self.right_foot_forces.append(right_f)

    def _print_contact_stats(self) -> None:
        """Print top-10 force bins and peak force events for each foot."""
        for name, forces in [("Left foot ", self.left_foot_forces), ("Right foot", self.right_foot_forces)]:
            arr = np.array(forces)
            nonzero = arr[arr > 0.0]
            if nonzero.size == 0:
                print(f"\n[{name}] No contact forces recorded.")
                continue

            # Histogram: 10 equal-width bins over the nonzero range
            counts, edges = np.histogram(nonzero, bins=10)
            order = np.argsort(counts)[::-1]
            print(f"\n[{name}] Contact force distribution (top-10 bins by frequency):")
            print(f"  {'Force range (N)':>24s}  {'Count':>8s}  {'Freq %':>8s}")
            for rank, idx in enumerate(order[:10]):
                lo, hi = edges[idx], edges[idx + 1]
                pct = counts[idx] / nonzero.size * 100.0
                print(f"  {rank+1:2d}.  [{lo:8.1f}, {hi:8.1f})   {counts[idx]:8d}  {pct:7.2f}%")

            # Top-10 peak force events
            peak_idx = np.argsort(arr)[::-1][:10]
            print(f"\n[{name}] Top-10 peak contact forces:")
            print(f"  {'Rank':>4s}  {'Step':>8s}  {'Force (N)':>12s}")
            for rank, idx in enumerate(peak_idx):
                print(f"  {rank+1:4d}  {idx:8d}  {arr[idx]:12.2f}")

    def run(self) -> None:
        """
        Run the simulation loop with keyboard-controlled commands.
        """
        self.setup_keyboard_listener()
        self.listener.start()

        if self.setup_joystick():
            print(f"[INFO] Joystick connected: {self.joystick_reader.device_path}")
            print("[INFO]   Left stick Y  -> vx  (forward/back)")
            print("[INFO]   Right stick X -> vy  (left/right)")
            print("[INFO]   L/R triggers  -> dyaw (turn)")
        else:
            print("[INFO] No joystick found, using keyboard control.")

        while self.data.time < self.cfg.sim.sim_duration:
            self.obs_history = self.get_obs()
            self.action[:] = (
                self.policy(torch.tensor(self.obs_history, dtype=torch.float32)).detach().numpy()[: self.cfg.sim.num_action]
            )
            self.action = np.clip(self.action, -self.cfg.sim.clip_actions, self.cfg.sim.clip_actions)
            # self.action = np.zeros_like(self.action)

            for sim_update in range(self.cfg.sim.decimation):
                step_start_time = time.time()

                self.data.ctrl = self.torque_control()
                mujoco.mj_step(self.model, self.data)
                self._collect_contact_forces()
                self.viewer.render()

                elapsed = time.time() - step_start_time
                sleep_time = self.cfg.sim.dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            self.episode_length_buf += 1
            if self.episode_length_buf % round(1.0 / self.dt) == 0:
                lf = self.left_foot_forces[-1] if self.left_foot_forces else 0.0
                rf = self.right_foot_forces[-1] if self.right_foot_forces else 0.0
                lmax = max(self.left_foot_forces) if self.left_foot_forces else 0.0
                rmax = max(self.right_foot_forces) if self.right_foot_forces else 0.0
                print(
                    f"[FORCE] t={self.data.time:6.1f}s | "
                    f"L now={lf:7.1f}N  max={lmax:7.1f}N | "
                    f"R now={rf:7.1f}N  max={rmax:7.1f}N"
                )
            self.calculate_gait_para()

        self.listener.stop()
        if self.joystick_reader:
            self.joystick_reader.stop()
        self.viewer.close()
        self._print_contact_stats()

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Rotate a vector by the inverse of a quaternion.

        Args:
            q (np.ndarray): Quaternion (x, y, z, w) format.
            v (np.ndarray): Vector to rotate.

        Returns:
            np.ndarray: Rotated vector.
        """
        q_w = q[-1]
        q_vec = q[:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * q_w * 2.0
        c = q_vec * np.dot(q_vec, v) * 2.0

        return a - b + c

    def calculate_gait_para(self) -> None:
        """
        Update gait phase parameters based on command state and offset.
        Aligned with mini3_env._calculate_gait_para.
        """
        command = self.command_vel
        moving = np.linalg.norm(command[:3]) > 0.01
        just_started = moving and (not self._was_moving)
        stopped = not moving

        # if stopped:
        #     self.gait_phase_accum_time = 0.0
        #     phase_stopped = self.gait_phase.copy()
        #     delta = ((phase_stopped + 0.5) % 1.0) - 0.5
        #     step = self.dt / max(self.gait_cycle, 1e-6)
        #     delta_step = np.clip(delta, -step, step)
        #     phase_next = (phase_stopped - delta_step) % 1.0
        #     phase_next = np.where(np.abs(delta - delta_step) < 1e-6, 0.0, phase_next)
        #     self.gait_phase = phase_next

        advance = moving and (not just_started)
        if True: #advance:
            self.gait_phase_accum_time += self.dt

        if True: #moving:
            t = self.gait_phase_accum_time / self.gait_cycle
            self.gait_phase[0] = (t + self.phase_offset[0]) % 1.0
            self.gait_phase[1] = (t + self.phase_offset[1]) % 1.0

        # if just_started:
        #     self.gait_phase[:] = 0.0

        self._was_moving = moving

    def setup_joystick(self, device_path=None, target_name="DF39") -> bool:
        """Try to open a joystick device and start reading in a background thread.

        Returns True if a joystick was found and opened successfully.
        """
        reader = JoystickReader(device_path)
        if not reader.open_device(device_path, target_name):
            return False
        reader.start_reading(callback=self._on_joystick_event)
        self.joystick_reader = reader
        return True

    def _on_joystick_event(self, event) -> None:
        """Update command_vel from joystick axes (called from background thread)."""
        # Print raw axis events on first contact to help identify axis mapping.
        # if not hasattr(self, "_joystick_debug_count"):
        #     self._joystick_debug_count = 0
        # if self._joystick_debug_count < 30 and (event["type"] & JS_EVENT_AXIS) and not (event["type"] & JS_EVENT_INIT):
        #     print(f"[JOYSTICK] axis {event['number']:2d} = {event['value'] / 32767.0:+.3f}")
        #     self._joystick_debug_count += 1

        left_y = self.joystick_reader.get_axis(1)
        right_x = self.joystick_reader.get_axis(2)
        left_trigger_raw = self.joystick_reader.get_axis(4)
        right_trigger_raw = self.joystick_reader.get_axis(5)

        left_trigger = max(0.0, min(1.0, (left_trigger_raw + 1.0) / 2.0))
        right_trigger = max(0.0, min(1.0, (right_trigger_raw + 1.0) / 2.0))

        if abs(left_y) < self.joystick_deadzone:
            left_y = 0.0
        if abs(right_x) < self.joystick_deadzone:
            right_x = 0.0

        vx = -left_y * self.joystick_vx_scale
        vy = -right_x * self.joystick_vy_scale
        dyaw = (right_trigger - left_trigger) * self.joystick_dyaw_scale

        with self.joystick_lock:
            self.command_vel[0] = float(np.clip(vx, -0.3, 0.6))
            self.command_vel[1] = float(np.clip(vy, -0.3, 0.3))
            self.command_vel[2] = float(np.clip(dyaw, -0.5, 0.5))
            print(f"[CMD] vel = [{self.command_vel[0]:+.3f}, {self.command_vel[1]:+.3f}, {self.command_vel[2]:+.3f}]")

    def adjust_command_vel(self, idx: int, increment: float) -> None:
        """
        Adjust command velocity vector.

        Args:
            idx (int): Index of velocity component (0=x, 1=y, 2=yaw).
            increment (float): Value to increment.
        """
        self.command_vel[idx] += increment
        self.command_vel[idx] = np.clip(self.command_vel[idx], -1.0, 1.0)  # vel clip

    def setup_keyboard_listener(self) -> None:
        """
        Set up keyboard event listener for user control input.
        """

        def on_press(key):
            try:
                if key.char == "8":  # NumPad 8      x += 0.2
                    self.adjust_command_vel(0, 0.2)
                elif key.char == "2":  # NumPad 2      x -= 0.2
                    self.adjust_command_vel(0, -0.2)
                elif key.char == "4":  # NumPad 4      y -= 0.2
                    self.adjust_command_vel(1, -0.2)
                elif key.char == "6":  # NumPad 6      y += 0.2
                    self.adjust_command_vel(1, 0.2)
                elif key.char == "7":  # NumPad 7      yaw += 0.2
                    self.adjust_command_vel(2, -0.2)
                elif key.char == "9":  # NumPad 9      yaw -= 0.2
                    self.adjust_command_vel(2, 0.2)
            except AttributeError:
                pass

        self.listener = keyboard.Listener(on_press=on_press)


if __name__ == "__main__":
    LEGGED_LAB_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
    parser = argparse.ArgumentParser(description="Run sim2sim Mujoco controller.")
    parser.add_argument(
        "--task",
        type=str,
        default="walk",
        choices=["walk", "run"],
        help="Task type: 'walk' or 'run' to set gait parameters",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Path to policy.pt. If not specified, it will be set automatically based on --task",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(LEGGED_LAB_ROOT_DIR, "legged_lab/assets/mini3/mjcf/scene.xml"),
        help="Path to model.xml",
    )
    parser.add_argument("--duration", type=float, default=100.0, help="Simulation duration in seconds")
    args = parser.parse_args()

    if args.policy is None:
        args.policy = os.path.join(LEGGED_LAB_ROOT_DIR, "Exported_policy", f"{args.task}.pt")

    if not os.path.isfile(args.policy):
        print(f"[ERROR] Policy file not found: {args.policy}")
        sys.exit(1)
    if not os.path.isfile(args.model):
        print(f"[ERROR] MuJoCo model file not found: {args.model}")
        sys.exit(1)

    print(f"[INFO] Loaded task preset: {args.task.upper()}")
    print(f"[INFO] Loaded policy: {args.policy}")
    print(f"[INFO] Loaded model: {args.model}")

    sim_cfg = SimToSimCfg()
    sim_cfg.sim.sim_duration = args.duration

    # Set gait parameters according to task
    if args.task == "walk":
        sim_cfg.robot.gait_air_ratio_l = 0.38
        sim_cfg.robot.gait_air_ratio_r = 0.38
        sim_cfg.robot.gait_phase_offset_l = 0.38
        sim_cfg.robot.gait_phase_offset_r = 0.88
        sim_cfg.robot.gait_cycle = 0.55
    elif args.task == "run":
        sim_cfg.robot.gait_air_ratio_l = 0.6
        sim_cfg.robot.gait_air_ratio_r = 0.6
        sim_cfg.robot.gait_phase_offset_l = 0.6
        sim_cfg.robot.gait_phase_offset_r = 0.1
        sim_cfg.robot.gait_cycle = 0.5

    runner = MujocoRunner(
        cfg=sim_cfg,
        policy_path=args.policy,
        model_path=args.model,
    )
    runner.run()
