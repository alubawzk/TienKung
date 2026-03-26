# cat_traverse_flow_mimic_cfg.py
#
# Agent (runner) configuration for training with FlowMimicActorCritic.

import dataclasses

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from .cat_traverse_g1_wbc_cfg import CatTraverseG1WbcEnvCfg, make_g1_wbc_env_cfg


def make_g1_flow_mimic_train_env_cfg(
    pt_path: str = "Exported_policy/flow_mimic.pt",
) -> CatTraverseG1WbcEnvCfg:
    """Env cfg for flow_mimic TRAINING (policy outputs 29 joint targets)."""
    cfg = make_g1_wbc_env_cfg()
    cfg.planner.onnx_path = pt_path
    return cfg


@configclass
class FlowMimicPolicyCfg(RslRlPpoActorCriticCfg):
    """Policy cfg that adds flow_mimic_pt_path on top of the base PPO actor-critic cfg.

    All MISSING fields from the parent are given dummy defaults here because
    FlowMimicActorCritic does not use them (they are filtered out by
    _filter_init_kwargs and silently ignored via **kwargs).
    """

    class_name: str = (
        "legged_lab.envs.cat_traverse.flow_mimic_policy.FlowMimicActorCritic"
    )
    # Path to flow_mimic.pt for weight initialisation.
    # Relative paths are resolved from the TienKung project root.
    flow_mimic_pt_path: str = "Exported_policy/flow_mimic.pt"

    # Required by parent — not used by FlowMimicActorCritic.
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False
    actor_hidden_dims: list = dataclasses.field(default_factory=lambda: [256, 128, 64])
    critic_hidden_dims: list = dataclasses.field(default_factory=lambda: [512, 256, 128])
    activation: str = "elu"


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
