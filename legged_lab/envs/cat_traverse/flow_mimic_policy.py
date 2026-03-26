# flow_mimic_policy.py
#
# TransformerTeacherPolicy + FlowMimicActorCritic (RSL-RL wrapper).
# TransformerTeacherPolicy — reconstructed from flow_mimic.pt weight shapes.
#
# Architecture (inferred from model_state_dict):
#   actor_history_proj   Linear(D_AH=160,  d_model=256)
#   actor_future_proj    Linear(D_AF=630,  d_model=256)
#   actor_pos_encoding   PositionalEncoding(d_model=256, max_len=20)
#   actor_transformer    TransformerEncoder(4 layers, d_model=256,
#                                           nhead=8, dim_feedforward=512)
#   actor_mean_head      Linear(256, 29)
#   std                  learnable [29]
#   actor_history_normalizer  RunningMeanStd([1, 160])
#   actor_future_normalizer   RunningMeanStd([1, 630])
#   critic_history_normalizer RunningMeanStd([1, 286])
#   critic_future_normalizer  RunningMeanStd([1, 630])
#   critic.mlp           MLP(10*286 + 630=3490 → 512 → 256 → 128 → 1)

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
from torch.distributions import Normal

# ──────────────────────────────────────────────────────────────────────────────
# Architecture constants (must match flow_mimic.pt)
# ──────────────────────────────────────────────────────────────────────────────
D_ACTOR_HISTORY: int = 160   # per-frame actor obs dimension
D_ACTOR_FUTURE: int = 630    # actor "future/goal" obs dimension
D_CRITIC_HISTORY: int = 286  # per-frame critic obs dimension (includes privileged)
D_CRITIC_FUTURE: int = 630   # critic future obs (same as actor)
SEQ_LEN: int = 10            # history length (H frames)
NUM_ACTIONS: int = 29        # number of joint targets output
D_MODEL: int = 256
N_HEAD: int = 8
N_LAYERS: int = 4
DIM_FFN: int = 512


# ──────────────────────────────────────────────────────────────────────────────
# Sub-modules
# ──────────────────────────────────────────────────────────────────────────────

class RunningMeanStd(nn.Module):
    """Online running normalizer (matches training impl)."""

    def __init__(self, shape: tuple[int, ...]):
        super().__init__()
        self.register_buffer("_mean", torch.zeros(shape))
        self.register_buffer("_var",  torch.ones(shape))
        self.register_buffer("_std",  torch.ones(shape))
        self.register_buffer("count", torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self._mean) / (self._std + 1e-8)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding, pre-computed up to max_len tokens."""

    def __init__(self, d_model: int, max_len: int = 20):
        super().__init__()
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, seq, d_model)
        return x + self.pe[:, : x.size(1), :]


class CriticMLP(nn.Module):
    """MLP critic: Linear(3490→512→256→128→1)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


# ──────────────────────────────────────────────────────────────────────────────
# Main policy
# ──────────────────────────────────────────────────────────────────────────────

class TransformerTeacherPolicy(nn.Module):
    """TransformerTeacherPolicy as saved in flow_mimic.pt.

    Actor forward pass:
        actor_history  (B, H, D_AH)  →  actor_history_proj  →  (B, H, d_model)
        actor_future   (B, D_AF)      →  actor_future_proj   →  (B, 1, d_model)
        concat → (B, H+1, d_model)   →  pos_encoding + transformer
        last token [:, -1, :]         →  actor_mean_head     →  mean (B, 29)
        std is a learnable parameter
    """

    def __init__(self):
        super().__init__()
        # ── Normalizers ──────────────────────────────────────────────────────
        self.actor_history_normalizer  = RunningMeanStd((1, D_ACTOR_HISTORY))
        self.actor_future_normalizer   = RunningMeanStd((1, D_ACTOR_FUTURE))
        self.critic_history_normalizer = RunningMeanStd((1, D_CRITIC_HISTORY))
        self.critic_future_normalizer  = RunningMeanStd((1, D_CRITIC_FUTURE))

        # ── Actor ────────────────────────────────────────────────────────────
        self.actor_history_proj = nn.Linear(D_ACTOR_HISTORY, D_MODEL)
        self.actor_future_proj  = nn.Linear(D_ACTOR_FUTURE,  D_MODEL)
        self.actor_pos_encoding = PositionalEncoding(D_MODEL, max_len=20)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEAD,
            dim_feedforward=DIM_FFN,
            batch_first=True,
        )
        self.actor_transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.actor_mean_head   = nn.Linear(D_MODEL, NUM_ACTIONS)
        self.std               = nn.Parameter(torch.ones(NUM_ACTIONS))

        # ── Critic ───────────────────────────────────────────────────────────
        critic_input_dim = SEQ_LEN * D_CRITIC_HISTORY + D_CRITIC_FUTURE  # 3490
        self.critic = CriticMLP(critic_input_dim)

    # ──────────────────────────────────────────────────────────────────────────

    def act(
        self,
        actor_history: torch.Tensor,   # (B, H, 160)
        actor_future:  torch.Tensor,   # (B, 630)
    ) -> torch.Tensor:
        """Return mean action (B, 29) — no sampling, suitable for inference."""
        return self._actor_forward(actor_history, actor_future)

    def forward(
        self,
        actor_history:  torch.Tensor,   # (B, H, 160)
        actor_future:   torch.Tensor,   # (B, 630)
        critic_history: torch.Tensor | None = None,  # (B, H, 286)
        critic_future:  torch.Tensor | None = None,  # (B, 630)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Returns (mean, std, value).  value is None if critic inputs are absent."""
        mean = self._actor_forward(actor_history, actor_future)
        value = None
        if critic_history is not None and critic_future is not None:
            value = self._critic_forward(critic_history, critic_future)
        return mean, self.std.exp(), value

    # ──────────────────────────────────────────────────────────────────────────

    def _actor_forward(
        self,
        actor_history: torch.Tensor,   # (B, H, D_AH)
        actor_future:  torch.Tensor,   # (B, D_AF)
    ) -> torch.Tensor:
        B, H, _ = actor_history.shape

        # Normalise
        ah = self.actor_history_normalizer(actor_history)   # (B, H, 160)
        af = self.actor_future_normalizer(actor_future)     # (B, 630)

        # Project to d_model
        h_tokens = self.actor_history_proj(ah)              # (B, H, 256)
        f_token  = self.actor_future_proj(af).unsqueeze(1)  # (B, 1, 256)

        # Concatenate history + future token → (B, H+1, 256)
        tokens = torch.cat([h_tokens, f_token], dim=1)

        # Positional encoding + transformer
        tokens = self.actor_pos_encoding(tokens)
        tokens = self.actor_transformer(tokens)             # (B, H+1, 256)

        # Take the last token (future token aggregates full context)
        out = tokens[:, -1, :]                              # (B, 256)
        mean = self.actor_mean_head(out)                    # (B, 29)
        return mean

    def _critic_forward(
        self,
        critic_history: torch.Tensor,   # (B, H, 286)
        critic_future:  torch.Tensor,   # (B, 630)
    ) -> torch.Tensor:
        ch = self.critic_history_normalizer(critic_history)
        cf = self.critic_future_normalizer(critic_future)
        flat = torch.cat([ch.reshape(ch.size(0), -1), cf], dim=-1)  # (B, 3490)
        return self.critic(flat)  # (B, 1)


# ──────────────────────────────────────────────────────────────────────────────
# Loading helper
# ──────────────────────────────────────────────────────────────────────────────

def load_flow_mimic(pt_path: str, device: str = "cpu") -> TransformerTeacherPolicy:
    """Load TransformerTeacherPolicy from a flow_mimic.pt checkpoint.

    The checkpoint is expected to be a dict with key 'model_state_dict' whose
    keys are prefixed with '_orig_mod.' (from torch.compile).

    Returns the model in eval mode on the specified device.
    """
    ckpt = torch.load(pt_path, map_location=device, weights_only=False)
    raw_sd = ckpt["model_state_dict"]

    # Strip '_orig_mod.' prefix added by torch.compile
    sd = {k.replace("_orig_mod.", "", 1): v for k, v in raw_sd.items()}

    model = TransformerTeacherPolicy()
    missing, unexpected = model.load_state_dict(sd, strict=True)
    if missing:
        raise RuntimeError(f"Missing keys in flow_mimic.pt: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys in flow_mimic.pt: {unexpected}")

    model.to(device)
    model.eval()
    return model


# ──────────────────────────────────────────────────────────────────────────────
# RSL-RL-compatible actor-critic wrapper
# ──────────────────────────────────────────────────────────────────────────────

class FlowMimicActorCritic(nn.Module):
    """RSL-RL-compatible actor-critic using TransformerTeacherPolicy.

    Expected flat obs layout (assembled by CatTraverseFlowMimicEnv):
        actor  obs: (N, SEQ_LEN * D_ACTOR_HISTORY  + D_ACTOR_FUTURE)  = (N, 2230)
        critic obs: (N, SEQ_LEN * D_CRITIC_HISTORY + D_CRITIC_FUTURE) = (N, 3490)

    These are split back into (history, future) tensors and forwarded through
    TransformerTeacherPolicy.
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        flow_mimic_pt_path: str = "",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            print(
                f"FlowMimicActorCritic: ignoring unexpected kwargs: {list(kwargs.keys())}"
            )
        super().__init__()

        self.policy = TransformerTeacherPolicy()

        if flow_mimic_pt_path:
            if not os.path.isabs(flow_mimic_pt_path):
                import legged_lab as _ll
                root = os.path.dirname(os.path.dirname(os.path.abspath(_ll.__file__)))
                flow_mimic_pt_path = os.path.join(root, flow_mimic_pt_path)
            if os.path.isfile(flow_mimic_pt_path):
                ckpt = torch.load(flow_mimic_pt_path, map_location="cpu", weights_only=False)
                raw_sd = ckpt["model_state_dict"]
                sd = {k.replace("_orig_mod.", "", 1): v for k, v in raw_sd.items()}
                self.policy.load_state_dict(sd, strict=True)
                print(f"[FlowMimicActorCritic] Loaded weights from {flow_mimic_pt_path}")
            else:
                print(
                    f"[FlowMimicActorCritic] flow_mimic_pt_path not found: "
                    f"'{flow_mimic_pt_path}'. Training from scratch."
                )

        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    # ── Interface ─────────────────────────────────────────────────────────────

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    # ── Obs splitting ─────────────────────────────────────────────────────────

    def _split_actor_obs(self, obs: torch.Tensor):
        B = obs.shape[0]
        h_flat = obs[:, : SEQ_LEN * D_ACTOR_HISTORY]        # (B, 1600)
        future  = obs[:, SEQ_LEN * D_ACTOR_HISTORY :]       # (B, 630)
        history = h_flat.view(B, SEQ_LEN, D_ACTOR_HISTORY)  # (B, 10, 160)
        return history, future

    def _split_critic_obs(self, obs: torch.Tensor):
        B = obs.shape[0]
        h_flat  = obs[:, : SEQ_LEN * D_CRITIC_HISTORY]       # (B, 2860)
        future   = obs[:, SEQ_LEN * D_CRITIC_HISTORY :]       # (B, 630)
        history  = h_flat.view(B, SEQ_LEN, D_CRITIC_HISTORY)  # (B, 10, 286)
        return history, future

    # ── RSL-RL interface ──────────────────────────────────────────────────────

    def update_distribution(self, observations: torch.Tensor):
        history, future = self._split_actor_obs(observations)
        mean = self.policy._actor_forward(history, future)
        # Guard against NaN/inf in mean (from gradient explosion or bad obs).
        mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
        # policy.std is a log-std parameter (consistent with TransformerTeacherPolicy
        # which returns self.std.exp() in forward()).  Clamp log-std before exp to
        # prevent overflow and NaN after gradient updates.
        log_std = torch.nan_to_num(self.policy.std, nan=0.0, posinf=2.0, neginf=-4.0)
        log_std = log_std.clamp(-4.0, 2.0)
        std = log_std.exp().expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        history, future = self._split_actor_obs(observations)
        return self.policy._actor_forward(history, future)

    def evaluate(self, critic_observations: torch.Tensor, **kwargs) -> torch.Tensor:
        history, future = self._split_critic_obs(critic_observations)
        return self.policy._critic_forward(history, future)

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
