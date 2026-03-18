from __future__ import annotations

import torch


def _num_actions(env) -> int:
    # Prefer VecEnv attribute if present.
    na = getattr(env, "num_actions", None)
    if isinstance(na, int) and na > 0:
        return na
    # Fallback to robot joint count.
    return int(env.robot.data.default_joint_pos.shape[1])


def _actor_frame_dim(env) -> int:
    # Mini3_Env.compute_current_observations layout:
    # 12 (vel/gravity/command) + 3*num_actions (jpos/jvel/last_action) + 6 (gait)
    na = _num_actions(env)
    return 18 + 3 * na


def _critic_frame_dim(env) -> int:
    # actor frame + feet_contact(2)
    return _actor_frame_dim(env) + 2


def _build_joint_mirror_maps(env) -> tuple[torch.Tensor, torch.Tensor]:
    """Build joint index permutation + sign flips for mirroring.

    We mirror across the sagittal plane (x-z plane): swap left/right joints and flip sign
    for joints whose axis changes sign under reflection (roll and yaw).
    """
    joint_names = list(env.robot.data.joint_names)
    n = len(joint_names)

    perm = torch.arange(n, device=env.device, dtype=torch.long)
    sign = torch.ones(n, device=env.device, dtype=torch.float32)

    name_to_idx = {name: i for i, name in enumerate(joint_names)}

    def axis_flips(name: str) -> float:
        # Under reflection across x-z plane:
        # - pitch (rotation about y) keeps sign
        # - roll (rotation about x) flips sign
        # - yaw  (rotation about z) flips sign
        if "roll" in name or "yaw" in name:
            return -1.0
        return 1.0

    for i, name in enumerate(joint_names):
        if name.startswith("left_"):
            mirror_name = "right_" + name[len("left_") :]
            j = name_to_idx.get(mirror_name)
            if j is not None:
                perm[i] = j
                sign[i] = axis_flips(name)
        elif name.startswith("right_"):
            mirror_name = "left_" + name[len("right_") :]
            j = name_to_idx.get(mirror_name)
            if j is not None:
                perm[i] = j
                sign[i] = axis_flips(name)
        else:
            # midline joints: keep index, but flip sign for yaw/roll
            sign[i] = axis_flips(name)

    return perm, sign


def _maybe_get_cached_maps(env) -> tuple[torch.Tensor, torch.Tensor]:
    cache = getattr(env, "_symmetry_cache_mini3", None)
    if cache is None or cache.get("device") != str(env.device):
        perm, sign = _build_joint_mirror_maps(env)
        cache = {"device": str(env.device), "perm": perm, "sign": sign}
        setattr(env, "_symmetry_cache_mini3", cache)
    return cache["perm"], cache["sign"]


def _mirror_actor_frame(env, frame: torch.Tensor) -> torch.Tensor:
    """Mirror a single actor frame (last dim = 18 + 3*num_actions)."""
    expected_dim = _actor_frame_dim(env)
    if frame.shape[-1] != expected_dim:
        raise ValueError(f"Expected actor frame dim {expected_dim}, got {frame.shape[-1]}")

    perm_j, sign_j = _maybe_get_cached_maps(env)
    na = _num_actions(env)

    # layout (Mini3_Env.compute_current_observations):
    # 0:3   root_lin_vel (vx, vy, vz)
    # 3:6   ang_vel      (wx, wy, wz)
    # 6:9   projected_gravity
    # 9:12  command      (vx, vy, yaw_rate)
    # 12:12+na          joint_pos
    # 12+na:12+2na      joint_vel
    # 12+2na:12+3na     last_action
    # 12+3na:...        gait sin/cos/phase_ratio (2,2,2)

    out = frame.clone()

    # vectors mirrored across x-z plane: y flips sign
    out[..., 1] *= -1.0  # lin_vel_y
    out[..., 7] *= -1.0  # gravity_y
    out[..., 10] *= -1.0  # command_vy

    # angular velocity is a pseudovector: w' = -R w, with R = diag(1,-1,1) -> (-wx, wy, -wz)
    out[..., 3] *= -1.0  # wx
    out[..., 5] *= -1.0  # wz

    # command yaw rate flips sign
    out[..., 11] *= -1.0

    # swap (and sign-flip) joint-related terms
    jpos = out[..., 12 : 12 + na]
    jvel = out[..., 12 + na : 12 + 2 * na]
    lact = out[..., 12 + 2 * na : 12 + 3 * na]

    jpos_m = jpos[..., perm_j] * sign_j
    jvel_m = jvel[..., perm_j] * sign_j
    lact_m = lact[..., perm_j] * sign_j

    out[..., 12 : 12 + na] = jpos_m
    out[..., 12 + na : 12 + 2 * na] = jvel_m
    out[..., 12 + 2 * na : 12 + 3 * na] = lact_m

    # gait features: swap left/right (2 dims each)
    gait_start = 12 + 3 * na
    for k in range(3):  # sin, cos, phase_ratio
        seg = out[..., gait_start + 2 * k : gait_start + 2 * (k + 1)]
        out[..., gait_start + 2 * k : gait_start + 2 * (k + 1)] = seg.flip(-1)

    return out


def _mirror_critic_frame(env, frame: torch.Tensor) -> torch.Tensor:
    """Mirror a single critic frame (last dim = actor_frame_dim + 2)."""
    actor_dim = _actor_frame_dim(env)
    expected_dim = actor_dim + 2
    if frame.shape[-1] != expected_dim:
        raise ValueError(f"Expected critic frame dim {expected_dim}, got {frame.shape[-1]}")

    actor = frame[..., :actor_dim]
    feet = frame[..., actor_dim:]
    actor_m = _mirror_actor_frame(env, actor)
    feet_m = feet.flip(-1)  # swap left/right foot contact
    return torch.cat([actor_m, feet_m], dim=-1)


@torch.no_grad()
def get_symmetric_states(
    *,
    obs: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    env=None,
    obs_type: str = "policy",
):
    """Return (augmented_obs, augmented_actions) for symmetry.

    The returned tensors are concatenated as: [original, mirrored].

    This matches rsl_rl's symmetry augmentation and mirror-loss code paths.
    """
    if env is None:
        raise ValueError("env must be provided for symmetry augmentation")

    obs_out = None
    act_out = None

    if obs is not None:
        frame_dim = _actor_frame_dim(env) if obs_type == "policy" else _critic_frame_dim(env)
        if obs.shape[-1] < frame_dim:
            raise ValueError(f"obs last dim {obs.shape[-1]} smaller than expected frame dim {frame_dim}")

        n_frames = obs.shape[-1] // frame_dim
        tail = obs[..., n_frames * frame_dim :]

        frames = obs[..., : n_frames * frame_dim].view(-1, n_frames, frame_dim)
        if obs_type == "policy":
            mirrored = torch.stack([_mirror_actor_frame(env, frames[:, i, :]) for i in range(n_frames)], dim=1)
        else:
            mirrored = torch.stack([_mirror_critic_frame(env, frames[:, i, :]) for i in range(n_frames)], dim=1)

        mirrored = mirrored.reshape(obs.shape[0], -1)
        mirrored = torch.cat([mirrored, tail], dim=-1) if tail.numel() else mirrored
        obs_out = torch.cat([obs, mirrored], dim=0)

    if actions is not None:
        perm_j, sign_j = _maybe_get_cached_maps(env)
        mirrored_actions = actions[..., perm_j] * sign_j
        act_out = torch.cat([actions, mirrored_actions], dim=0)

    return obs_out, act_out

