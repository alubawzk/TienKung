# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
#
# This file is part of the CAT IsaacLab task scaffold for Click-and-Traverse.

import math
from copy import deepcopy

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

import legged_lab.mdp as mdp
from legged_lab.assets.tienkung2_lite import TIENKUNG2LITE_CFG
from legged_lab.assets.unitree_description import UNITREE_G1_CFG
from legged_lab.envs.base.base_config import (
    ActionDelayCfg,
    BaseSceneCfg,
    CommandRangesCfg,
    CommandsCfg,
    DomainRandCfg,
    EventCfg,
    HeightScannerCfg,
    MeshObstacleCfg,
    NoiseCfg,
    NoiseScalesCfg,
    NormalizationCfg,
    ObsScalesCfg,
    PhysxCfg,
    RobotCfg,
    SimCfg,
)

PROXY_ACTION_JOINT_NAMES = [
    "hip_pitch_l_joint",
    "hip_roll_l_joint",
    "hip_yaw_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "ankle_roll_l_joint",
    "hip_pitch_r_joint",
    "hip_roll_r_joint",
    "hip_yaw_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_r_joint",
]

PROXY_OBS_JOINT_NAMES = [
    "hip_pitch_l_joint",
    "hip_roll_l_joint",
    "hip_yaw_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "ankle_roll_l_joint",
    "hip_pitch_r_joint",
    "hip_roll_r_joint",
    "hip_yaw_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_r_joint",
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
]

G1_ACTION_JOINT_NAMES = [
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
]

G1_OBS_JOINT_NAMES = [
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
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]


@configclass
class CatTraverseFieldCfg:
    path: str = "data/assets/TypiObs/empty"
    dx: float = 0.04
    origin: tuple[float, float, float] = (-0.5, -1.0, 0.0)
    sdf_clip: float = 1.0
    obs_sdf_max: float = 0.5
    collision_threshold: float = 0.04
    grace_steps: int = 50
    bf_clear_threshold: float = 0.5


@configclass
class CatTraverseProbeCfg:
    profile_name: str = "tienkung_proxy"
    head_body_name: str = "pelvis"
    head_offsets: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.55),)
    pelvis_body_name: str = "pelvis"
    pelvis_offsets: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0),)
    torso_body_name: str = "pelvis"
    torso_offsets: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.25),)
    feet_body_names: tuple[str, str] = ("ankle_roll_l_link", "ankle_roll_r_link")
    feet_offsets: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    hand_body_names: tuple[str, str] = ("elbow_pitch_l_link", "elbow_pitch_r_link")
    hand_offsets: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    knee_body_names: tuple[str, str] = ("knee_pitch_l_link", "knee_pitch_r_link")
    knee_offsets: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    # The original MuJoCo CAT termination uses left/right shin collision geoms.
    # In IsaacLab/URDF these usually live on dedicated lower-leg links or on the
    # knee links that carry the imported shin collision capsules.
    shin_body_names: tuple[str, str] = ("knee_pitch_l_link", "knee_pitch_r_link")
    shoulder_body_names: tuple[str, str] = ("shoulder_roll_l_link", "shoulder_roll_r_link")
    shoulder_offsets: tuple[tuple[float, float, float], ...] = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    head_height_threshold: float = 0.7


def make_tienkung_proxy_probe_cfg() -> CatTraverseProbeCfg:
    """Fallback CAT probe mapping for the current TienKung2Lite asset."""
    return CatTraverseProbeCfg()


def make_g1_cat_probe_cfg() -> CatTraverseProbeCfg:
    """Exact CAT site layout from data/assets/unitree_g1/g1_mjx_feetonly_torque.xml.

    The IsaacLab articulation still needs to expose the corresponding G1 body names.
    This preset only adapts the CAT probe geometry; it does not switch the robot asset.
    """

    return CatTraverseProbeCfg(
        profile_name="g1_cat_sites",
        # site "head" on body "torso_link"
        head_body_name="torso_link",
        head_offsets=((0.0, 0.0, 0.4),),
        # site "imu_in_pelvis" on body "pelvis"
        pelvis_body_name="pelvis",
        pelvis_offsets=((0.04525, 0.0, -0.08339),),
        # site "imu_in_torso" on body "torso_link"
        torso_body_name="torso_link",
        torso_offsets=((-0.03959, -0.00224, 0.14792),),
        # sites "left_foot" / "right_foot" on ankle-roll links
        feet_body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
        feet_offsets=((0.04, 0.0, -0.037), (0.04, 0.0, -0.037)),
        # sites "left_palm" / "right_palm" on wrist-yaw links
        hand_body_names=("left_wrist_yaw_link", "right_wrist_yaw_link"),
        hand_offsets=((0.08, 0.0, 0.0), (0.08, 0.0, 0.0)),
        # sites "left_knee" / "right_knee" on knee links
        knee_body_names=("left_knee_link", "right_knee_link"),
        knee_offsets=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        # original MuJoCo shin geoms are attached to the knee links
        shin_body_names=("left_knee_link", "right_knee_link"),
        # sites "left_shoulder" / "right_shoulder" on shoulder-pitch links
        shoulder_body_names=("left_shoulder_pitch_link", "right_shoulder_pitch_link"),
        shoulder_offsets=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        head_height_threshold=0.7,
    )


def _make_undesired_contacts_term(body_names: list[str]) -> RewTerm:
    return RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=body_names), "threshold": 1.0},
    )


@configclass
class CatTraverseGaitCfg:
    gait_bound: float = 0.6
    freq_range: tuple[float, float] = (1.3, 1.5)
    foot_height_range: tuple[float, float] = (0.07, 0.07)


@configclass
class CatTraverseDelayCfg:
    update_interval_steps: int = 5


@configclass
class CatTraverseCommandCfg:
    root_gain: float = 0.7
    output_scale: float = 0.75
    stop_speed_threshold: float = 0.2
    stop_hold_steps: int = 50
    stop_timestep_reset: int = 100


@configclass
class CatTraverseTerminationCfg:
    pair_contact_force_threshold: float = 1.0
    fall_projected_gravity_z_threshold: float = 0.0


@configclass
class CatTraverseDmRandCfg:
    enable_pd: bool = True
    kp_range: tuple[float, float] = (0.8, 1.2)
    kd_range: tuple[float, float] = (0.8, 1.2)
    enable_rfi: bool = True
    rfi_lim: float = 0.1
    rfi_lim_range: tuple[float, float] = (0.8, 1.2)


@configclass
class CatTraverseRewardCfg:
    tracking_orientation = RewTerm(func=mdp.cat_tracking_orientation, weight=2.0, params={"torso_height_upper": 1.0})
    tracking_root_field = RewTerm(func=mdp.cat_tracking_root_field, weight=1.0)
    body_motion = RewTerm(func=mdp.cat_body_motion, weight=-0.5)
    body_rotation = RewTerm(func=mdp.cat_body_rotation, weight=1.0, params={"yaw_cmd_max": 0.5})
    feet_rotation = RewTerm(func=mdp.cat_feet_rotation, weight=0.0)
    foot_contact = RewTerm(func=mdp.cat_foot_contact, weight=-1.0)
    foot_clearance = RewTerm(func=mdp.cat_foot_clearance, weight=-15.0, params={"foot_height_stance": 0.0})
    foot_slip = RewTerm(func=mdp.cat_foot_slip, weight=-0.5)
    foot_balance = RewTerm(func=mdp.cat_foot_balance, weight=-30.0)
    foot_far = RewTerm(func=mdp.cat_foot_far, weight=-0.0)
    straight_knee = RewTerm(func=mdp.cat_straight_knee, weight=-30.0, params={"joint_patterns": [".*_knee_joint"]})
    joint_limits = RewTerm(func=mdp.cat_joint_pos_limits, weight=-1.0)
    joint_torque = RewTerm(func=mdp.cat_joint_torque, weight=-1.0e-4)
    smoothness_joint = RewTerm(
        func=mdp.cat_smoothness_joint,
        weight=-1.0e-6,
        params={
            "joint_patterns": [
                ".*hip.*_joint",
                ".*knee.*_joint",
                ".*ankle.*_joint",
                ".*waist.*_joint",
                ".*shoulder.*_joint",
                ".*elbow.*_joint",
            ]
        },
    )
    smoothness_action = RewTerm(func=mdp.cat_smoothness_action, weight=-1.0e-3)
    headgf = RewTerm(func=mdp.cat_pf_alignment_reward, weight=1.0, params={"group_name": "head", "tau": 0.5})
    feetgf = RewTerm(
        func=mdp.cat_pf_alignment_reward,
        weight=1.0,
        params={"group_name": "feet", "tau": 0.3, "block_stance_feet": True},
    )
    handsgf = RewTerm(func=mdp.cat_pf_alignment_reward, weight=1.0, params={"group_name": "hands", "tau": 0.5})
    headdf = RewTerm(func=mdp.cat_pf_sdf_penalty, weight=1.0, params={"group_name": "head"})
    feetdf = RewTerm(func=mdp.cat_pf_sdf_penalty, weight=1.0, params={"group_name": "feet"})
    handsdf = RewTerm(func=mdp.cat_pf_sdf_penalty, weight=1.0, params={"group_name": "hands"})
    kneesdf = RewTerm(func=mdp.cat_pf_sdf_penalty, weight=1.0, params={"group_name": "knees"})
    shldsdf = RewTerm(func=mdp.cat_pf_sdf_penalty, weight=1.0, params={"group_name": "shoulders"})


def make_tienkung_proxy_robot_cfg() -> RobotCfg:
    return RobotCfg(
        actor_obs_history_length=1,
        critic_obs_history_length=1,
        action_scale=0.5,
        action_joint_names=PROXY_ACTION_JOINT_NAMES,
        obs_joint_names=PROXY_OBS_JOINT_NAMES,
        terminate_contacts_body_names=["knee_pitch.*", "shoulder_roll.*", "elbow_pitch.*", "pelvis"],
        feet_body_names=["ankle_roll_l_link", "ankle_roll_r_link"],
    )


def make_g1_cat_robot_cfg() -> RobotCfg:
    """Robot contact/body-name profile for the G1 CAT articulation."""

    return RobotCfg(
        actor_obs_history_length=1,
        critic_obs_history_length=1,
        action_scale=0.5,
        action_joint_names=G1_ACTION_JOINT_NAMES,
        obs_joint_names=G1_OBS_JOINT_NAMES,
        terminate_contacts_body_names=[".*_knee_link", ".*_shoulder_roll_link", ".*_elbow_link", "pelvis"],
        feet_body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
    )


def make_tienkung_proxy_reward_cfg() -> CatTraverseRewardCfg:
    return CatTraverseRewardCfg(
        straight_knee=RewTerm(
            func=mdp.cat_straight_knee,
            weight=-30.0,
            params={"joint_patterns": ["knee_pitch_l_joint", "knee_pitch_r_joint"]},
        )
    )


def make_g1_cat_reward_cfg() -> CatTraverseRewardCfg:
    """Reward profile matching the G1 CAT articulation naming."""

    return CatTraverseRewardCfg(
        feet_rotation=RewTerm(func=mdp.cat_feet_rotation, weight=0.0),
        straight_knee=RewTerm(
            func=mdp.cat_straight_knee,
            weight=-30.0,
            params={"joint_patterns": [".*_knee_joint"]},
        )
    )


def make_g1_cat_pri_reward_cfg() -> CatTraverseRewardCfg:
    return CatTraverseRewardCfg(
        feet_rotation=RewTerm(func=mdp.cat_feet_rotation, weight=1.0),
        foot_balance=RewTerm(func=mdp.cat_foot_balance, weight=-10.0),
        straight_knee=RewTerm(
            func=mdp.cat_straight_knee,
            weight=-30.0,
            params={"joint_patterns": [".*_knee_joint"]},
        ),
    )


@configclass
class CatTraverseEnvCfg:
    device: str = "cuda:0"
    variant: str = "cat"
    field: CatTraverseFieldCfg = CatTraverseFieldCfg()
    # Default stays on the current TienKung2Lite proxy mapping so the task keeps
    # resolving with the existing robot asset. Use make_g1_cat_site_env_cfg(...)
    # or apply_g1_cat_site_profile(...) once the scene robot is changed to G1.
    probes: CatTraverseProbeCfg = make_tienkung_proxy_probe_cfg()
    gait: CatTraverseGaitCfg = CatTraverseGaitCfg()
    delay: CatTraverseDelayCfg = CatTraverseDelayCfg()
    command: CatTraverseCommandCfg = CatTraverseCommandCfg()
    termination: CatTraverseTerminationCfg = CatTraverseTerminationCfg()
    dm_rand: CatTraverseDmRandCfg = CatTraverseDmRandCfg()

    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        num_envs=1024,
        env_spacing=3.0,
        robot=TIENKUNG2LITE_CFG,
        terrain_type="plane",
        terrain_generator=None,
        max_init_terrain_level=1,
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,
            prim_body_name="pelvis",
            resolution=0.1,
            size=(1.6, 1.0),
            debug_vis=False,
            drift_range=(0.0, 0.0),
        ),
        mesh_obstacle=MeshObstacleCfg(
            enable=True,
            source_path="",
            prim_path="{ENV_REGEX_NS}/Obstacle",
            collision_group=0,
        ),
    )
    robot: RobotCfg = make_tienkung_proxy_robot_cfg()
    reward = make_tienkung_proxy_reward_cfg()
    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(
            lin_vel=1.0,
            ang_vel=1.0,
            projected_gravity=1.0,
            commands=1.0,
            joint_pos=1.0,
            joint_vel=1.0,
            actions=1.0,
            height_scan=1.0,
        ),
        clip_observations=100.0,
        clip_actions=1.0,
        height_scan_offset=0.5,
    )
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=CommandRangesCfg(
            lin_vel_x=(-0.4, 0.8),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )
    noise: NoiseCfg = NoiseCfg(
        add_noise=True,
        noise_scales=NoiseScalesCfg(
            lin_vel=0.2,
            ang_vel=0.2,
            projected_gravity=0.05,
            joint_pos=0.01,
            joint_vel=1.5,
            height_scan=0.1,
        ),
    )
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=EventCfg(
            physics_material=EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (0.6, 1.0),
                    "dynamic_friction_range": (0.4, 0.8),
                    "restitution_range": (0.0, 0.005),
                    "num_buckets": 64,
                },
            ),
            add_base_mass=EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
                    "mass_distribution_params": (-5.0, 5.0),
                    "operation": "add",
                },
            ),
            reset_base=EventTerm(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    # Spawn the robot in the clear open area at the start of
                    # the field. The field origin is (-0.5, -1.0) in env-local
                    # frame; a ±1 m range sends robots outside the field or
                    # into obstacles. ±0.3 m keeps the robot well within the
                    # obstacle-free starting zone of all standard CAT fields.
                    "pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-math.pi / 2.0, math.pi / 2.0)},
                    "velocity_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (-0.5, 0.5),
                        "roll": (-0.5, 0.5),
                        "pitch": (-0.5, 0.5),
                        "yaw": (-0.5, 0.5),
                    },
                },
            ),
            reset_robot_joints=EventTerm(
                func=mdp.reset_joints_by_scale,
                mode="reset",
                params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
            ),
            push_robot=EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(10.0, 15.0),
                params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
            ),
        ),
        action_delay=ActionDelayCfg(enable=False, params={"max_delay": 5, "min_delay": 0}),
    )
    sim: SimCfg = SimCfg(dt=0.005, decimation=4, physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))


def apply_g1_cat_site_profile(cfg: CatTraverseEnvCfg) -> CatTraverseEnvCfg:
    """Return a cfg copy with G1 CAT site/body-name mappings applied.

    This helper only switches the CAT probe layout and the contact/body-name
    patterns. The caller still needs to replace ``cfg.scene.robot`` with a G1
    articulation that exposes the corresponding body names, such as
    ``UNITREE_G1_CFG``.
    """

    updated_cfg = deepcopy(cfg)
    updated_cfg.probes = make_g1_cat_probe_cfg()
    updated_cfg.robot = make_g1_cat_robot_cfg()
    updated_cfg.reward = make_g1_cat_reward_cfg()
    return updated_cfg


def apply_g1_cat_pri_profile(cfg: CatTraverseEnvCfg) -> CatTraverseEnvCfg:
    updated_cfg = apply_g1_cat_site_profile(cfg)
    updated_cfg.variant = "cat_pri"
    updated_cfg.reward = make_g1_cat_pri_reward_cfg()
    updated_cfg.gait = CatTraverseGaitCfg(gait_bound=0.6, freq_range=(1.3, 1.5), foot_height_range=(0.05, 0.05))
    updated_cfg.command = CatTraverseCommandCfg(
        root_gain=0.6,
        output_scale=1.0,
        stop_speed_threshold=0.2,
        stop_hold_steps=50,
        stop_timestep_reset=100,
    )
    return updated_cfg


def make_g1_cat_site_env_cfg(scene_robot=None) -> CatTraverseEnvCfg:
    """Build a CAT env cfg pre-wired for the G1 CAT site layout.

    Args:
        scene_robot: Optional G1 articulation cfg. When omitted, only the probe
            and contact naming profiles are switched.
    """

    cfg = apply_g1_cat_site_profile(CatTraverseEnvCfg())
    if scene_robot is not None:
        cfg.scene.robot = scene_robot
    return cfg


def make_unitree_g1_cat_env_cfg() -> CatTraverseEnvCfg:
    """Build a CAT env cfg bound to the vendored Unitree G1 IsaacLab asset."""

    return make_g1_cat_site_env_cfg(UNITREE_G1_CFG)


def make_unitree_g1_cat_pri_env_cfg() -> CatTraverseEnvCfg:
    cfg = apply_g1_cat_pri_profile(CatTraverseEnvCfg())
    cfg.scene.robot = UNITREE_G1_CFG
    return cfg


@configclass
class CatTraverseAgentCfg(RslRlOnPolicyRunnerCfg):
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
    experiment_name = "cat_traverse"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "cat_traverse"
    wandb_project = "cat_traverse"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
