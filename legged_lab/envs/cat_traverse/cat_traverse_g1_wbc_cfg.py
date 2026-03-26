# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
#
# cat_traverse_g1_wbc_cfg.py
#
# Configuration for the G1 WBC task: policy outputs 8 planner control signals
# which are fed into planner_sonic.onnx to generate full-body joint targets.
#
# Policy output layout (num_actions = 8):
#   [0]   target_vel          (float, <= 0 means use mode default speed)
#   [1]   mode                (float, rounded+clamped to int in [0, 26])
#   [2:5] movement_direction  (3D vector, L2-normalized before planner)
#   [5:8] facing_direction    (3D vector, L2-normalized before planner)

import dataclasses

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from legged_lab.assets.unitree_description import UNITREE_G1_CFG

from .cat_traverse_cfg import (
    CatTraverseAgentCfg,
    CatTraverseEnvCfg,
    make_g1_cat_probe_cfg,
    make_g1_cat_reward_cfg,
    make_g1_cat_robot_cfg,
)

# 29 joint names in the order used by planner_sonic.onnx (matches
# g1_mjx_feetonly_torque.xml joint declaration order).
PLANNER_JOINT_NAMES: list = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


@configclass
class PlannerCfg:
    """Configuration for the planner_sonic.onnx inference step."""

    # Path to the ONNX model, relative to the TienKung project root.
    onnx_path: str = "Exported_policy/planner/target_vel/V2/planner_sonic.onnx"

    # 29 joint names in planner internal order.
    planner_joint_names: list = dataclasses.field(
        default_factory=lambda: list(PLANNER_JOINT_NAMES)
    )

    # Root height passed to planner. < 0 disables height control.
    default_height: float = -1.0

    # When True, only the first predicted frame (t=0) is used as joint target.
    use_first_frame_only: bool = True


@configclass
class CatTraverseG1WbcEnvCfg(CatTraverseEnvCfg):
    """CAT traversal env cfg for the G1 + WBC (planner-in-the-loop) variant."""

    planner: PlannerCfg = dataclasses.field(default_factory=PlannerCfg)


def make_g1_wbc_env_cfg() -> CatTraverseG1WbcEnvCfg:
    """Build a CatTraverseG1WbcEnvCfg pre-wired for the G1 robot and WBC mode."""
    cfg = CatTraverseG1WbcEnvCfg()
    cfg.probes = make_g1_cat_probe_cfg()
    cfg.robot = make_g1_cat_robot_cfg()
    cfg.reward = make_g1_cat_reward_cfg()
    cfg.scene.robot = UNITREE_G1_CFG
    cfg.variant = "cat_wbc"
    return cfg


def make_g1_flow_mimic_env_cfg(
    pt_path: str = "Exported_policy/flow_mimic.pt",
) -> CatTraverseG1WbcEnvCfg:
    """Build a CatTraverseG1WbcEnvCfg for the flow_mimic.pt controller.

    The planner.onnx_path field is repurposed to hold the flow_mimic.pt path.
    """
    cfg = make_g1_wbc_env_cfg()
    cfg.planner.onnx_path = pt_path
    return cfg


@configclass
class CatTraverseG1WbcAgentCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL runner config for the G1 WBC task."""

    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 5000
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        class_name="rsl_rl.modules.ActorCriticTanh",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        symmetry_cfg=None,
        rnd_cfg=None,
    )
    clip_actions = None
    save_interval = 100
    runner_class_name = "OnPolicyRunner"
    experiment_name = "cat_traverse_g1_wbc"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "cat_traverse"
    wandb_project = "cat_traverse"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
