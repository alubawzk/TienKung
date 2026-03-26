# cat_traverse_flow_mimic_cfg.py
#
# Standalone env cfg + agent cfg for training CatTraverseFlowMimicEnv with G1.
# Does NOT inherit from CatTraverseEnvCfg or any WBC variant — all fields are
# defined directly, matching the style of CatTraverseRewardCfg.

import dataclasses
import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

import legged_lab.mdp as mdp
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

from .cat_traverse_cfg import (
    CatTraverseFieldCfg,
    CatTraverseProbeCfg,
    CatTraverseGaitCfg,
    CatTraverseDelayCfg,
    CatTraverseCommandCfg,
    CatTraverseTerminationCfg,
    CatTraverseDmRandCfg,
    make_g1_cat_probe_cfg,
    make_g1_cat_robot_cfg,
)


# ──────────────────────────────────────────────────────────────────────────────
# Reward cfg  (standalone, not inherited)
# ──────────────────────────────────────────────────────────────────────────────

@configclass
class FlowMimicRewardCfg:
    tracking_orientation = RewTerm(func=mdp.cat_tracking_orientation, weight=2.0, params={"torso_height_upper": 1.0})
    tracking_root_field  = RewTerm(func=mdp.cat_tracking_root_field, weight=1.0)
    body_motion          = RewTerm(func=mdp.cat_body_motion, weight=-0.5)
    body_rotation        = RewTerm(func=mdp.cat_body_rotation, weight=1.0, params={"yaw_cmd_max": 0.5})
    feet_rotation        = RewTerm(func=mdp.cat_feet_rotation, weight=0.0)
    foot_contact         = RewTerm(func=mdp.cat_foot_contact, weight=-1.0)
    foot_clearance       = RewTerm(func=mdp.cat_foot_clearance, weight=-15.0, params={"foot_height_stance": 0.0})
    foot_slip            = RewTerm(func=mdp.cat_foot_slip, weight=-0.5)
    foot_balance         = RewTerm(func=mdp.cat_foot_balance, weight=-30.0)
    foot_far             = RewTerm(func=mdp.cat_foot_far, weight=-0.0)
    straight_knee        = RewTerm(func=mdp.cat_straight_knee, weight=-30.0, params={"joint_patterns": [".*_knee_joint"]})
    joint_limits         = RewTerm(func=mdp.cat_joint_pos_limits, weight=-1.0)
    joint_torque         = RewTerm(func=mdp.cat_joint_torque, weight=-1.0e-4)
    smoothness_joint     = RewTerm(
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
    headgf  = RewTerm(func=mdp.cat_pf_alignment_reward, weight=1.0, params={"group_name": "head",      "tau": 0.5})
    feetgf  = RewTerm(func=mdp.cat_pf_alignment_reward, weight=1.0, params={"group_name": "feet",      "tau": 0.3, "block_stance_feet": True})
    handsgf = RewTerm(func=mdp.cat_pf_alignment_reward, weight=1.0, params={"group_name": "hands",     "tau": 0.5})
    headdf  = RewTerm(func=mdp.cat_pf_sdf_penalty,      weight=1.0, params={"group_name": "head"})
    feetdf  = RewTerm(func=mdp.cat_pf_sdf_penalty,      weight=1.0, params={"group_name": "feet"})
    handsdf = RewTerm(func=mdp.cat_pf_sdf_penalty,      weight=1.0, params={"group_name": "hands"})
    kneesdf = RewTerm(func=mdp.cat_pf_sdf_penalty,      weight=1.0, params={"group_name": "knees"})
    shldsdf = RewTerm(func=mdp.cat_pf_sdf_penalty,      weight=1.0, params={"group_name": "shoulders"})


# ──────────────────────────────────────────────────────────────────────────────
# Env cfg  (standalone, not inherited from CatTraverseEnvCfg)
# ──────────────────────────────────────────────────────────────────────────────

@configclass
class CatTraverseFlowMimicEnvCfg:
    """Standalone CAT traversal env cfg for flow_mimic training with G1.

    All fields are defined here directly — no inheritance from CatTraverseEnvCfg
    or any WBC variant.  The extra field `flow_mimic_pt_path` points to the
    TransformerTeacherPolicy checkpoint used to initialise the policy weights.
    """

    # ── flow_mimic specific ──────────────────────────────────────────────────
    flow_mimic_pt_path: str = "Exported_policy/flow_mimic.pt"

    # ── identity ─────────────────────────────────────────────────────────────
    device:  str = "cuda:0"
    variant: str = "cat_flow_mimic"

    # ── CAT sub-configs ───────────────────────────────────────────────────────
    field:       CatTraverseFieldCfg       = CatTraverseFieldCfg()
    probes:      CatTraverseProbeCfg       = make_g1_cat_probe_cfg()
    gait:        CatTraverseGaitCfg        = CatTraverseGaitCfg()
    delay:       CatTraverseDelayCfg       = CatTraverseDelayCfg()
    command:     CatTraverseCommandCfg     = CatTraverseCommandCfg()
    termination: CatTraverseTerminationCfg = CatTraverseTerminationCfg()
    dm_rand:     CatTraverseDmRandCfg      = CatTraverseDmRandCfg()

    # ── scene ─────────────────────────────────────────────────────────────────
    scene: BaseSceneCfg = BaseSceneCfg(
        max_episode_length_s=20.0,
        num_envs=4096,
        env_spacing=3.0,
        robot=UNITREE_G1_CFG,
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

    # ── robot / reward ────────────────────────────────────────────────────────
    robot:  RobotCfg        = make_g1_cat_robot_cfg()
    reward: FlowMimicRewardCfg = FlowMimicRewardCfg()

    # ── normalization ─────────────────────────────────────────────────────────
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

    # ── commands ──────────────────────────────────────────────────────────────
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

    # ── noise ─────────────────────────────────────────────────────────────────
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

    # ── domain randomisation ──────────────────────────────────────────────────
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=EventCfg(
            physics_material=EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range":  (0.6, 1.0),
                    "dynamic_friction_range": (0.4, 0.8),
                    "restitution_range":      (0.0, 0.005),
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
                    "pose_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "yaw": (-math.pi / 2.0, math.pi / 2.0)},
                    "velocity_range": {
                        "x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.5, 0.5),
                        "roll": (-0.5, 0.5), "pitch": (-0.5, 0.5), "yaw": (-0.5, 0.5),
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

    # ── sim ───────────────────────────────────────────────────────────────────
    sim: SimCfg = SimCfg(dt=0.005, decimation=4, physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))


# ──────────────────────────────────────────────────────────────────────────────
# Policy cfg  (standard MLP — RL policy outputs 14*9=126 dim body poses)
# ──────────────────────────────────────────────────────────────────────────────

@configclass
class FlowMimicPolicyCfg(RslRlPpoActorCriticCfg):
    """Standard MLP actor-critic for the high-level RL policy.

    The policy takes cat_traverse-style obs (~150 dims) and outputs
    14-body poses (126 dims) in anchor frame.  flow_mimic.pt is loaded
    and frozen inside CatTraverseFlowMimicEnv — the policy class here
    is a plain RSL-RL ActorCritic.
    """
    class_name: str = "rsl_rl.modules.ActorCriticTanh"

    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"
    actor_hidden_dims: list = dataclasses.field(default_factory=lambda: [1024, 512, 256])
    critic_hidden_dims: list = dataclasses.field(default_factory=lambda: [1024, 512, 256])
    activation: str = "elu"


# ──────────────────────────────────────────────────────────────────────────────
# Agent (runner) cfg
# ──────────────────────────────────────────────────────────────────────────────

@configclass
class CatTraverseFlowMimicAgentCfg(RslRlOnPolicyRunnerCfg):
    """RSL-RL runner cfg for training CatTraverseFlowMimicEnv."""

    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 5000
    empirical_normalization = False

    policy = FlowMimicPolicyCfg()

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
    experiment_name = "cat_traverse_g1_flow_mimic"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "cat_traverse"
    wandb_project = "cat_traverse"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"
