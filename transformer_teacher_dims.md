# Transformer Teacher Policy - 输入输出维度说明

> 以下维度以 **Unitree G1** 机器人为例（29 个关节，14 个追踪 body）。
>
> **关节空间 vs 三维坐标**：`command`（参考运动指令）是**标量关节角度**（rad），不是笛卡尔三维坐标。`command = [joint_pos(29) || joint_vel(29)] = 58 维`。如需三维关节位置，使用的是 body 级别的观测（`motion_body_pos_b` 等）。

## 整体架构

`TransformerTeacherPolicy` 是主策略类，内部包含：
- **Actor**：Transformer 编码器，负责生成动作分布
- **Critic**：MLP，负责估计状态价值

obs_groups 配置（见 `TrackingTeacherTransformerPPORunnerCfg`）：
```python
obs_groups = {
    "policy": ["history", "future"],           # Actor 用
    "critic": ["critic_history", "critic_future"],  # Critic 用
}
```

---

## 核心类：`TransformerTeacherPolicy`

### 观测输入（TensorDict）

策略通过 `obs_groups` 配置将观测分为两类：

| 类型 | 形状（G1） | 说明 |
|------|-----------|------|
| History 组（3D） | `(B, 10, D_state)` | 历史帧观测序列，`H=10` 帧，`D_state` 为单帧特征维度 |
| Future 组（2D） | `(B, D_future)` | 未来轨迹目标信息，已展平 |

其中：
- `B`：batch size
- `H=10`：历史序列长度（`history_length=10`，`flatten_history_dim=False`）
- Actor history `D_state = 131`，Critic history `D_state = 257`（含特权信息）
- Future `D_future = 630`（actor 和 critic 相同）

Actor 和 Critic 各自有独立的 history/future 观测组，由 `obs_groups["policy"]` 和 `obs_groups["critic"]` 配置。

| 观测组 | 形状（G1） | 说明 |
|--------|-----------|------|
| Actor History (`history`) | `(B, 10, 160)` | 含噪声的本体感知+目标历史 |
| Actor Future (`future`) | `(B, 630)` | 未来 5 帧 × 14 body × 9 维轨迹 |
| Critic History (`critic_history`) | `(B, 10, 286)` | 无噪声 + 特权 body 状态 |
| Critic Future (`critic_future`) | `(B, 630)` | 同 Actor Future |

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

### Actor History 观测组：`TransformerHistoryObsCfg`

形状：`(B, 10, 160)`，每帧 160 维，各字段按拼接顺序：

| 字段 | 函数 | 维度 | 物理含义 | 参考系 | 噪声 |
|------|------|------|---------|-------|------|
| `command` | `generated_commands` | **58** | 参考运动的关节状态：`[joint_pos(29) \|\| joint_vel(29)]`，均为**标量关节角度/角速度（rad/rad·s⁻¹）** | 关节空间 | 无 |
| `motion_anchor_pos_b` | `motion_anchor_pos_b` | 3 | 参考运动 anchor（骨盆）位置相对机器人当前 anchor 的偏移 (x,y,z) | 机器人 anchor 系 | ±0.25 m |
| `motion_anchor_ori_b` | `motion_anchor_ori_b` | 6 | 参考运动 anchor 姿态的 6D 旋转表示（旋转矩阵前两列展平） | 机器人 anchor 系 | ±0.05 |
| `base_lin_vel` | `base_lin_vel` | 3 | 机器人基座线速度 (vx, vy, vz) | 基座系 | ±0.5 m/s |
| `base_ang_vel` | `base_ang_vel` | 3 | 机器人基座角速度 (ωx, ωy, ωz) | 基座系 | ±0.2 rad/s |
| `joint_pos` | `joint_pos_rel` | 29 | 机器人当前关节位置相对于默认站立姿态的偏差（29 个关节，rad） | 关节空间 | ±0.01 rad |
| `joint_vel` | `joint_vel_rel` | 29 | 机器人当前关节速度（29 个关节，rad/s） | 关节空间 | ±0.5 rad/s |
| `actions` | `last_action` | 29 | 上一时刻的动作输出（29 个关节位置偏移量） | 关节空间 | 无 |
| **合计** | | **160** | | | |

> **注意**：`command` 中的 `joint_pos/joint_vel` 是参考运动数据（目标值），而下方的 `joint_pos/joint_vel` 字段是机器人当前实测值。两者共同提供 **"目标 - 当前"** 的对比信息，是 Transformer 学习误差修正的核心依据。
>
> 29 个关节顺序（G1）：左腿 6 + 右腿 6 + 腰部 3 + 左臂 7 + 右臂 7

### Actor Future 观测组：`TransformerFutureObsCfg`

形状：`(B, 630)`，展平的未来轨迹目标 token：

| 字段 | 函数 | 维度 | 物理含义 |
|------|------|------|---------|
| `future_trajectory` | `future_body_trajectory_b` | 5 × 14 × 9 = **630** | 未来 5 帧（每 2 仿真步采样一次）× 14 个追踪 body，每 body = 位置 3D + 6D 旋转 |

每个 body 的 9 维布局：

| 子字段 | 维度 | 含义 |
|--------|------|------|
| `pos_x, pos_y, pos_z` | 3 | body 位置在机器人 anchor 系下的偏移 |
| `ori_6d` | 6 | body 姿态的 6D 旋转（旋转矩阵前两列），比四元数更平滑，无奇点 |

14 个追踪 body（G1）：

| 编号 | body 名称 | 部位 |
|------|----------|------|
| 0 | `pelvis` | 骨盆（anchor） |
| 1 | `left_hip_roll_link` | 左髋 |
| 2 | `left_knee_link` | 左膝 |
| 3 | `left_ankle_roll_link` | 左踝 |
| 4 | `right_hip_roll_link` | 右髋 |
| 5 | `right_knee_link` | 右膝 |
| 6 | `right_ankle_roll_link` | 右踝 |
| 7 | `torso_link` | 躯干 |
| 8 | `left_shoulder_roll_link` | 左肩 |
| 9 | `left_elbow_link` | 左肘 |
| 10 | `left_wrist_yaw_link` | 左腕 |
| 11 | `right_shoulder_roll_link` | 右肩 |
| 12 | `right_elbow_link` | 右肘 |
| 13 | `right_wrist_yaw_link` | 右腕 |

> **变长未来视野（Variable-horizon masking）**：每个 episode 随机激活 1~5 帧中的前 N 帧，超出范围的 future 步骤置零，训练策略应对不同预测时域。

### 各层参数说明（G1 具体尺寸）

| 模块 | 输入形状 | 输出形状 | 说明 |
|------|---------|---------|------|
| `actor_history_proj` | `(B, 10, 160)` | `(B, 10, 256)` | 将每帧历史观测投影到 d_model=256 |
| `actor_future_proj` | `(B, 630)` | `(B, 1, 256)` | 将未来轨迹目标投影为单个 token |
| `actor_pos_encoding` | `(B, 11, 256)` | `(B, 11, 256)` | 正弦位置编码，给序列注入时序信息 |
| `actor_transformer` | `(B, 11, 256)` | `(B, 11, 256)` | 4 层多头自注意力（nhead=8，Flash Attention） |
| `actor_mean_head` | `(B, 256)` | `(B, 29)` | 输出动作均值 |

> **为何取最后一个 token（future token）作输出**：future token 在自注意力中可以同时关注所有 history token，因此它聚合了完整的历史上下文 + 目标信息，是最丰富的表示。

### Actor 输出

| 输出 | 形状（G1） | 说明 |
|------|-----------|------|
| `mean` | `(B, 29)` | 动作高斯分布的均值（29 个关节） |
| `std` | `(B, 29)` | 动作高斯分布的标准差（可学习参数） |
| `actions`（采样后） | `(B, 29)` | 从 Normal(mean, std) 采样得到的关节位置偏移量 |

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

### Critic History 观测组：`TransformerCriticHistoryObsCfg`

形状：`(B, 10, 286)`，与 Actor History 相同字段但**无噪声**，额外增加两项特权信息：

| 字段 | 维度 | 物理含义 | 备注 |
|------|------|---------|------|
| `command` | **58** | 参考运动关节状态 `[joint_pos(29) \|\| joint_vel(29)]` | 同 Actor History，无噪声 |
| `motion_anchor_pos_b` | 3 | anchor 位置偏移 | 无噪声 |
| `motion_anchor_ori_b` | 6 | anchor 6D 姿态 | 无噪声 |
| `base_lin_vel` | 3 | 基座线速度 | 无噪声 |
| `base_ang_vel` | 3 | 基座角速度 | 无噪声 |
| `joint_pos` | 29 | 关节位置偏差 | 无噪声 |
| `joint_vel` | 29 | 关节速度 | 无噪声 |
| `actions` | 29 | 上一动作 | 无噪声 |
| `body_pos` ⭐ | 14×3 = **42** | 机器人实际 14 个 body 的位置（在 anchor 系下）| **特权信息**，部署时不可用 |
| `body_ori` ⭐ | 14×6 = **84** | 机器人实际 14 个 body 的 6D 姿态（在 anchor 系下）| **特权信息**，部署时不可用 |
| **合计** | **286** | | |

### Critic Future 观测组：`TransformerCriticFutureObsCfg`

形状：`(B, 630)`，与 Actor Future 完全相同（同一未来轨迹，无噪声）。

### 各层参数说明（G1 具体尺寸）

| 模块 | 输入形状 | 输出形状 | 说明 |
|------|---------|---------|------|
| Flatten history | `(B, 10, 286)` | `(B, 2860)` | 展平历史帧 |
| Concat | `(B, 2860 + 630)` = `(B, 3490)` | — | 拼接历史和未来 |
| MLP Layer 1 | `(B, 3200)` | `(B, 512)` | ELU 激活 |
| MLP Layer 2 | `(B, 512)` | `(B, 256)` | ELU 激活 |
| MLP Layer 3 | `(B, 256)` | `(B, 128)` | ELU 激活 |
| 输出层 | `(B, 128)` | `(B, 1)` | 价值估计 |

### Critic 输出

| 输出 | 形状（G1） | 说明 |
|------|-----------|------|
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

| 参数 | 值（G1 配置） | 说明 |
|------|-------------|------|
| `d_model` | 256 | Transformer 隐层维度 |
| `nhead` | 8 | 多头注意力头数（每头 32 维） |
| `num_layers` | 4 | Transformer 编码器层数 |
| `dim_feedforward` | 512 | Transformer FFN 中间层维度 |
| `seq_len` | 10 | 历史序列长度 H |
| `num_actions` | 29 | 动作空间维度（G1 关节数） |
| `init_noise_std` | 1.0 | 动作噪声初始标准差 |
| `actor_obs_normalization` | True | 对 history/future 分别做 EmpiricalNormalization |
| `critic_obs_normalization` | True | 同上 |

---

## 信息流总结（G1 完整尺寸）

```
观测 TensorDict
├── "history"        (B, 10, 160) ─┐
│                                  ├→ Actor Transformer → mean(B,29), std(B,29)
├── "future"         (B, 630)     ─┘
│
├── "critic_history" (B, 10, 286) ─┐
│                                  ├→ Critic MLP       → value(B,1)
└── "critic_future"  (B, 630)     ─┘
```

### Actor History 字段汇总

| 偏移 | 字段 | 维度 | 含义 |
|------|------|------|------|
| 0 | command：参考关节位置（joint_pos） | 29 | 目标关节角度（rad） |
| 29 | command：参考关节速度（joint_vel） | 29 | 目标关节角速度（rad/s） |
| 58 | motion_anchor_pos_b | 3 | 骨盆目标位置偏移 |
| 61 | motion_anchor_ori_b | 6 | 骨盆目标姿态 6D |
| 67 | base_lin_vel | 3 | 机器人基座线速度 |
| 70 | base_ang_vel | 3 | 机器人基座角速度 |
| 73 | joint_pos（相对默认姿态） | 29 | 当前关节位置偏差 |
| 102 | joint_vel | 29 | 当前关节速度 |
| 131 | actions（上一动作） | 29 | 控制连续性信息 |
| **合计** | | **160** | |

### Future 字段汇总

| 偏移 | 内容 | 维度 |
|------|------|------|
| 0 | 第 1 未来帧，14 body × 9 = 126 维 | 126 |
| 126 | 第 2 未来帧 | 126 |
| 252 | 第 3 未来帧 | 126 |
| 378 | 第 4 未来帧 | 126 |
| 504 | 第 5 未来帧 | 126 |
| **合计** | | **630** |

每帧每 body 9 维 = `[pos_x, pos_y, pos_z, ori_col0_x, ori_col0_y, ori_col0_z, ori_col1_x, ori_col1_y, ori_col1_z]`，均在机器人 anchor（骨盆）坐标系下。
