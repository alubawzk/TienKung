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
import os
from pathlib import Path

import torch
from isaaclab.app import AppLauncher

from legged_lab.utils import task_registry
from rsl_rl.runners import AmpOnPolicyRunner, OnPolicyRunner

# local imports
import legged_lab.utils.cli_args as cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--vx", type=float, default=1.0, help="Forward velocity command (m/s).")
parser.add_argument("--vy", type=float, default=0.0, help="Lateral velocity command (m/s).")
parser.add_argument("--vz", type=float, default=0.0, help="Yaw angular velocity command (rad/s).")
parser.add_argument(
    "--field_path",
    type=str,
    default=None,
    help="Override CAT field asset directory, e.g. data/assets/TypiObs/narrow0.",
)

# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
# Start camera rendering
if "sensor" in args_cli.task:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg


def _freeze_cat_reset_for_play(env_cfg):
    events = getattr(getattr(env_cfg, "domain_rand", None), "events", None)
    if events is None:
        return

    reset_base = getattr(events, "reset_base", None)
    if reset_base is not None and getattr(reset_base, "params", None) is not None:
        pose_range = reset_base.params.get("pose_range")
        if pose_range is not None:
            reset_base.params["pose_range"] = {key: (0.0, 0.0) for key in pose_range.keys()}
        velocity_range = reset_base.params.get("velocity_range")
        if velocity_range is not None:
            reset_base.params["velocity_range"] = {key: (0.0, 0.0) for key in velocity_range.keys()}

    reset_robot_joints = getattr(events, "reset_robot_joints", None)
    if reset_robot_joints is not None and getattr(reset_robot_joints, "params", None) is not None:
        if "position_range" in reset_robot_joints.params:
            reset_robot_joints.params["position_range"] = (1.0, 1.0)
        if "velocity_range" in reset_robot_joints.params:
            reset_robot_joints.params["velocity_range"] = (0.0, 0.0)


def play():
    runner: OnPolicyRunner
    env_cfg: BaseEnvCfg  # noqa:F405

    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)
    is_cat_task = env_class_name.startswith("cat_traverse")

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.max_episode_length_s = 40.0
    env_cfg.scene.num_envs = 1 if is_cat_task else 50
    env_cfg.scene.env_spacing = 6.0 if is_cat_task else 2.5
    env_cfg.commands.rel_standing_envs = 0.0
    env_cfg.commands.ranges.lin_vel_x = (args_cli.vx, args_cli.vx)
    env_cfg.commands.ranges.lin_vel_y = (args_cli.vy, args_cli.vy)
    env_cfg.commands.ranges.ang_vel_z = (args_cli.vz, args_cli.vz)
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)

    env_cfg.scene.terrain_generator = None
    env_cfg.scene.terrain_type = "plane"

    if env_cfg.scene.terrain_generator is not None:
        env_cfg.scene.terrain_generator.num_rows = 5
        env_cfg.scene.terrain_generator.num_cols = 5
        env_cfg.scene.terrain_generator.curriculum = False
        env_cfg.scene.terrain_generator.difficulty_range = (0.4, 0.4)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    if args_cli.field_path is not None:
        if not hasattr(env_cfg, "field") or not hasattr(env_cfg.field, "path"):
            raise ValueError(f"Task '{env_class_name}' does not support '--field_path'.")
        env_cfg.field.path = args_cli.field_path
        if hasattr(env_cfg, "scene") and getattr(env_cfg.scene, "mesh_obstacle", None) is not None:
            env_cfg.scene.mesh_obstacle.source_path = str(Path(args_cli.field_path) / "obs.obj")

    if is_cat_task:
        _freeze_cat_reset_for_play(env_cfg)

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    runner_class: OnPolicyRunner | AmpOnPolicyRunner = eval(agent_cfg.runner_class_name)
    runner = runner_class(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)

    policy = runner.get_inference_policy(device=env.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    os.makedirs(export_model_dir, exist_ok=True)
    export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(
        runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
    )
    if is_cat_task:
        print(f"[INFO] CAT field path: {env_cfg.field.path}")
        print("[INFO] CAT play resets use a fixed centered spawn with default joint posture.")
    print(f"[INFO] Exported policy directory: {export_model_dir}")

    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard

        keyboard = Keyboard(env)  # noqa:F841

    obs, _ = env.get_observations()

    while simulation_app.is_running():

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)


if __name__ == "__main__":
    play()
    simulation_app.close()
