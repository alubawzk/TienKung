# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TienKung-Lab is an RL-based locomotion control framework for humanoid robots, built on **IsaacLab 2.1.0** and **IsaacSim 4.5.0**. It integrates Adversarial Motion Priors (AMP) with periodic gait rewards. The framework supports TienKung (full-sized humanoid), Mini3, and Mini2 robots, and was validated in the first Humanoid Robot Half Marathon.

## Common Commands

### Installation
```bash
# From within the IsaacLab conda environment
pip install -e .                  # Install legged_lab
cd rsl_rl && pip install -e .     # Install RSL-RL fork
pre-commit install                # Install code quality hooks
```

### Training
```bash
python legged_lab/scripts/train.py --task=tienkung_walk --headless --logger=tensorboard --num_envs=4096

# Multi-GPU background training
CUDA_VISIBLE_DEVICES=0 nohup python legged_lab/scripts/train.py \
    --task walk --logger tensorboard --max_iterations 5000 \
    --num_envs 8192 --headless --run_name <name> > output.log 2>&1 &
```

### Playing / Inference
```bash
python legged_lab/scripts/play.py --task walk --num_envs 1 \
    --load_run <timestamp_run_name> --vx 0.5 --vy 0.0 --vz 0.5
```

### Visualization
```bash
# Visualize AMP motion data
python legged_lab/scripts/play_amp_animation.py --task=walk --num_envs=1

# Save expert motion dataset
python legged_lab/scripts/play_amp_animation.py --task=walk --num_envs=1 \
    --save_path legged_lab/envs/tienkung/datasets/motion_amp_expert/motion.txt --fps 30.0
```

### Sim2Sim (MuJoCo validation)
```bash
python legged_lab/scripts/sim2sim.py --task walk \
    --policy logs/mini3_walk/<timestamp>/exported/policy.pt --duration 100
```

### Motion Retargeting Pipeline
```bash
# 1. Retarget SMPLX motion using GMR
python scripts/smplx_to_robot.py --smplx_file <path> --robot tienkung --save_path <output.pkl>

# 2. Convert for visualization
python legged_lab/scripts/gmr_data_conversion.py \
    --input_pkl <output.pkl> \
    --output_txt legged_lab/envs/tienkung/datasets/motion_visualization/motion.txt

# 3. Generate expert AMP data (step after visualization)
python legged_lab/scripts/play_amp_animation.py --task=walk --num_envs=1 \
    --save_path legged_lab/envs/tienkung/datasets/motion_amp_expert/motion.txt --fps 30.0
```

### Code Quality
```bash
pre-commit run --all-files   # Run black, flake8, isort
```
Line length limit: **120** (configured in `.flake8`).

## Architecture

### Directory Layout
```
legged_lab/
├── assets/          # Robot USD/URDF + configs (mini2_ToC, mini3, tienkung2_lite)
├── envs/
│   ├── base/        # BaseEnv, BaseEnvCfg — shared logic for all robots
│   ├── tienkung/    # TienKung tasks: walk, run, with_sensor variants
│   ├── mini3/       # Mini3 quadruped tasks
│   └── mini2_ToC/   # Mini2 quadruped tasks
├── mdp/             # MDP components (rewards.py)
├── scripts/         # Entry points: train, play, sim2sim, play_amp_animation
├── sensors/         # Custom camera and LiDAR sensor wrappers
├── symmetry/        # Left-right symmetry augmentation module
├── terrains/        # Procedural terrain generators
└── utils/           # task_registry, scene helpers, keyboard input

rsl_rl/              # Fork of RSL-RL with AMP support
├── algorithms/      # PPO, AMP discriminator
├── runners/         # OnPolicyRunner, AmpOnPolicyRunner
├── modules/         # Actor-Critic network modules
└── storage/         # Rollout and AMP replay buffers
```

### Task Registration
All tasks are registered in [legged_lab/envs/__init__.py](legged_lab/envs/__init__.py) via:
```python
task_registry.register("task_name", EnvClass, EnvCfg(), AgentCfg())
```
`task_registry` (in `legged_lab/utils/`) maps task names to environment classes and configs. This is the primary entry point for adding new tasks.

### Configuration Hierarchy
Configs use dataclass inheritance:
```
BaseEnvCfg
  └── <Robot><Task>FlatEnvCfg   (e.g. TienKungWalkFlatEnvCfg)
        ├── scene: BaseSceneCfg
        │     ├── robot: ArticulationCfg (USD path, joint names, foot bodies)
        │     ├── terrain: TerrainGeneratorCfg
        │     ├── height_scanner: RayCasterCfg
        │     └── sensors (optional): lidar, depth_camera
        ├── reward: RewardCfg      (all reward term weights)
        ├── normalization: obs/action scaling factors
        ├── commands: velocity command ranges
        ├── noise: sensor noise levels
        ├── domain_rand: DR event configs
        └── amp_motion_files: path list for expert motion data
```

### Observation Pipeline
Each step, `compute_current_observations()` stacks:
- Root angular velocity + projected gravity
- Velocity commands
- Joint positions (relative to default), velocities, accelerations
- Foot contact forces
- Height scan (optional)
- 10-frame history buffer
- All scaled by `obs_scales`

### Action Pipeline
```
policy output → action_buffer (delay sim) → smoothing filter → clip & scale (×0.25) → PD joint targets
```

### Reward System
Rewards are a weighted sum of terms defined in task-specific `*_cfg.py` files. Key terms include velocity tracking (exponential), energy penalties, joint smoothness, contact penalties, orientation penalties, and AMP motion-matching. The termination penalty is −200.

### AMP (Adversarial Motion Priors)
- Expert motion stored in two formats:
  - `motion_visualization/`: full frames `[root_pos, root_rot, dof_pos, root_lin_vel, root_ang_vel, dof_vel]`
  - `motion_amp_expert/`: compact AMP data `[dof_pos, dof_vel, end_effector_pos]`
- `AmpOnPolicyRunner` (in `rsl_rl/runners/`) extends PPO with a discriminator that scores how "natural" the policy's motion is.

### Logging Structure
```
logs/<robot>/<task>/<timestamp>_<run_name>/
├── checkpoints/          # Periodic policy checkpoints
├── exported/policy.pt    # JIT-compiled policy for deployment
├── params/env.yaml       # Environment config snapshot
└── runs/tensorboard/     # TensorBoard event files
```

## Key Conventions

- **Adding a new robot/task**: create env class + cfg in `legged_lab/envs/<robot>/`, then register in `legged_lab/envs/__init__.py`.
- **Domain randomization** is configured via `EventCfg` inside the task cfg; uses IsaacLab's `EventManager`.
- **Sensor variants** (`walk_with_sensor`) extend the base task cfg by adding `height_scanner`, `lidar`, or `depth_camera` configs.
- **Symmetry augmentation**: apply via `legged_lab/symmetry/` when the robot has bilateral symmetry.
- The `rsl_rl/` subdirectory is a local editable package — changes there take effect without reinstalling.
