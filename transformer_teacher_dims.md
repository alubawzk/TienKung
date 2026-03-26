# Transformer Teacher Policy - 输入输出维度说明

## 整体架构

`TransformerTeacherPolicy` 是主策略类，内部包含：
- **Actor**：Transformer 编码器，负责生成动作分布
- **Critic**：MLP，负责估计状态价值

---

## 核心类：`TransformerTeacherPolicy`

### 观测输入（TensorDict）

策略通过 `obs_groups` 配置将观测分为两类：

| 类型 | 形状 | 说明 |
|------|------|------|
| History 组（3D） | `(B, H, D_state)` | 历史帧观测序列，`H` 为历史长度，`D_state` 为单帧特征维度 |
| Future 组（2D） | `(B, D_future)` | 当前目标/未来轨迹信息，已展平 |

其中：
- `B`：batch size
- `H`：历史序列长度（`seq_len`，通常为 10）
- `D_state`：单帧观测特征维度（actor 和 critic 分别统计）
- `D_future`：目标/未来观测特征维度

Actor 和 Critic 各自有独立的 history/future 观测组，由 `obs_groups["policy"]` 和 `obs_groups["critic"]` 配置。

---

## Actor 网络（Transformer）

### 数据流

```
History 观测:  (B, H, D_actor_history)
                    ↓ actor_history_proj (Linear)
               (B, H, d_model)
                    ↓ concat (dim=1)
Future 观测:   (B, D_actor_future)
                    ↓ actor_future_proj (Linear) + unsqueeze(1)
               (B, 1, d_model)
                    ↓
完整 tokens:   (B, H+1, d_model)
                    ↓ PositionalEncoding
                    ↓ TransformerEncoder (多层自注意力)
               (B, H+1, d_model)
                    ↓ 取最后一个 token [:, -1, :]
               (B, d_model)
                    ↓ actor_mean_head (Linear)
动作均值 mean: (B, num_actions)
动作标准差 std: (B, num_actions)  ← 可学习参数展开
```

### 各层参数说明

| 模块 | 输入形状 | 输出形状 | 说明 |
|------|---------|---------|------|
| `actor_history_proj` | `(B, H, D_actor_history)` | `(B, H, d_model)` | 将每帧历史观测投影到 d_model 维度 |
| `actor_future_proj` | `(B, D_actor_future)` | `(B, 1, d_model)` | 将未来目标投影为单个 token |
| `actor_pos_encoding` | `(B, H+1, d_model)` | `(B, H+1, d_model)` | 正弦位置编码，给序列注入时序信息 |
| `actor_transformer` | `(B, H+1, d_model)` | `(B, H+1, d_model)` | 多头自注意力（Flash Attention / SDPA） |
| `actor_mean_head` | `(B, d_model)` | `(B, num_actions)` | 输出动作均值 |

> **为何取最后一个 token（future token）作输出**：future token 在自注意力中可以同时关注所有 history token，因此它聚合了完整的历史上下文 + 目标信息，是最丰富的表示。

### Actor 输出

| 输出 | 形状 | 说明 |
|------|------|------|
| `mean` | `(B, num_actions)` | 动作高斯分布的均值 |
| `std` | `(B, num_actions)` | 动作高斯分布的标准差（可学习参数） |
| `actions`（采样后） | `(B, num_actions)` | 从 Normal(mean, std) 采样得到的动作 |

---

## Critic 网络（MLP）

Critic 使用 MLP 而非 Transformer，速度更快且对价值估计通常已足够。

### 数据流

```
History 观测:  (B, H, D_critic_history)
                    ↓ reshape
               (B, H * D_critic_history)
                    ↓ concat (dim=-1)
Future 观测:   (B, D_critic_future)
                    ↓
拼接输入:      (B, H * D_critic_history + D_critic_future)
                    ↓ MLP [512 → 256 → 128 → 1]
价值估计:      (B, 1)
```

### 各层参数说明

| 模块 | 输入形状 | 输出形状 | 说明 |
|------|---------|---------|------|
| Flatten | `(B, H, D_critic_history)` | `(B, H * D_critic_history)` | 展平历史帧 |
| Concat | `(B, H*D + D_future)` | — | 拼接历史和未来 |
| MLP Layer 1 | `(B, H*D + D_future)` | `(B, 512)` | ELU 激活 |
| MLP Layer 2 | `(B, 512)` | `(B, 256)` | ELU 激活 |
| MLP Layer 3 | `(B, 256)` | `(B, 128)` | ELU 激活 |
| 输出层 | `(B, 128)` | `(B, 1)` | 价值估计 |

### Critic 输出

| 输出 | 形状 | 说明 |
|------|------|------|
| `values` | `(B, 1)` | 当前状态的估计价值 |

---

## 辅助类维度说明

### `PositionalEncoding`

| 参数 | 说明 |
|------|------|
| `d_model` | 模型维度，与 token 维度一致 |
| `max_len` | 支持的最大序列长度（默认 5000，实际用 `seq_len + 10`） |
| 输入/输出 | `(B, seq_len, d_model)` → `(B, seq_len, d_model)` |

### `TransformerTeacherActor`（独立 Actor，非主策略内部使用）

| 参数 | 说明 |
|------|------|
| 输入 | `(B, seq_len * num_obs_per_step)` 展平历史，或 `(B, seq_len, num_obs_per_step)` |
| 特权观测 | 与普通观测同结构，拼接后共同输入 |
| 输出 mean | `(B, num_actions)` |
| 输出 std | `(B, num_actions)` |

### `TransformerTeacherCritic`（独立 Critic）

| 参数 | 说明 |
|------|------|
| 输入 | 同 `TransformerTeacherActor` |
| 输出 | `(B,)` 价值估计（squeeze 掉最后一维） |

---

## 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `d_model` | 256 | Transformer 隐层维度 |
| `nhead` | 8 | 多头注意力头数 |
| `num_layers` | 3 | Transformer 编码器层数 |
| `dim_feedforward` | 512 | Transformer FFN 中间层维度 |
| `seq_len` | 10 | 历史序列长度 H |
| `num_actions` | — | 动作空间维度（由环境决定） |
| `init_noise_std` | 1.0 | 动作噪声初始标准差 |

---

## 信息流总结

```
观测 TensorDict
├── actor history groups  → (B, H, D_ah) ─┐
├── actor future groups   → (B, D_af)    ─┤→ Actor Transformer → mean(B,A), std(B,A)
├── critic history groups → (B, H, D_ch) ─┐
└── critic future groups  → (B, D_cf)    ─┤→ Critic MLP       → value(B,1)
```

- **A** = `num_actions`（关节动作数）
- **H** = `seq_len`（历史帧数，默认 10）
- **D_ah / D_af** = actor 历史/未来观测总特征维度
- **D_ch / D_cf** = critic 历史/未来观测总特征维度
