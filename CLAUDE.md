# collis_wbc + planner_sonic.onnx 集成设计

## 目标

将 `TienKung/collis_wbc` 分支中训练的 policy 输出不再直接作用于机器人关节，而是先输入到 `planner_sonic.onnx`，再以 planner 的输出轨迹控制机器人全身关节。

整体数据流如下：

```
IsaacSim 状态
    │
    ▼
Policy (RL, collis_wbc)
    │  policy_action: [num_envs, N]
    ▼
构造 planner_sonic.onnx 输入
    │  11 个输入 tensor（见下）
    ▼
planner_sonic.onnx 推理
    │  mujoco_qpos:     [1, 64, 36]
    │  num_pred_frames: [1]
    ▼
提取有效帧 → 关节目标
    │  joint_pos_target: [num_envs, 29]
    ▼
robot.set_joint_position_target(...)
```

---

## 1. Policy 输出的重新定义

### 现状（collis_wbc 当前行为）

`cat_traverse_env.py` 中的 `step()` 方法：

```python
# legged_lab/envs/cat_traverse/cat_traverse_env.py : step()
updated_action_targets = previous_action_targets + clipped_actions * self.action_scale
...
self._motor_targets[:, self.action_joint_ids] = updated_action_targets
self.robot.set_joint_position_target(self._motor_targets)   # 直接写入关节
```

policy 的输出是关节位置增量（delta），直接累加到 `_motor_targets` 后写入仿真。

### 修改后

policy 输出改为 **planner 的高级控制量**，不再是关节增量。具体输出维度（`num_actions`）为 **8**：

| 索引 | 含义 | 对应 planner 输入 | 备注 |
|---|---|---|---|
| `0` | `target_vel` | `target_vel [1]` | m/s，`<= 0` 表示使用 mode 默认速度 |
| `1` | `mode` (归一化) | `mode [1]` | 实际使用时 round + clamp 到 `[0, 26]` |
| `2:5` | `movement_direction` | `movement_direction [1, 3]` | 世界系方向向量，推理前 L2 归一化 |
| `5:8` | `facing_direction` | `facing_direction [1, 3]` | 世界系方向向量，推理前 L2 归一化 |

`height` 固定由 cfg 给出（不作为 policy 的输出），默认设为 `< 0`（禁用高度控制）。

如需 policy 也输出 `height`，可将 `num_actions` 扩展为 9，追加一维。

---

## 2. planner_sonic.onnx 接口回顾

> 详细说明见 [planner_sonic_io.md](./planner_sonic_io.md)。

### 输入（11 个）

| Tensor | Shape | 来源 |
|---|---|---|
| `context_mujoco_qpos` | `[1, 4, 36]` | 维护 4 帧历史 qpos buffer（见第 3 节） |
| `target_vel` | `[1]` | policy 输出索引 0 |
| `mode` | `[1]` int64 | policy 输出索引 1，round + clamp |
| `movement_direction` | `[1, 3]` | policy 输出索引 2:5，归一化 |
| `facing_direction` | `[1, 3]` | policy 输出索引 5:8，归一化 |
| `random_seed` | `[1]` int64 | 固定管理（见第 5 节） |
| `has_specific_target` | `[1, 1]` int64 | 固定为 `0`（不启用 waypoint） |
| `specific_target_positions` | `[1, 4, 3]` | 固定为全零（has_specific_target=0 时不生效） |
| `specific_target_headings` | `[1, 4]` | 固定为全零（has_specific_target=0 时不生效） |
| `allowed_pred_num_tokens` | `[1, 11]` int64 | 初始全为 `1`，可按需约束 |
| `height` | `[1]` | cfg 中固定配置，默认 `-1.0`（禁用） |

### 输出（2 个）

| Tensor | Shape | 含义 |
|---|---|---|
| `mujoco_qpos` | `[1, 64, 36]` | 预测轨迹，padding 到 64 帧 |
| `num_pred_frames` | `[1]` int32 | 有效帧数 |

单帧 `qpos[36]` 布局：

```
[0:3]  root position  (x, y, z)
[3:7]  root quaternion (w, x, y, z)
[7:36] 29 个关节角 (rad)
```

---

## 3. context_mujoco_qpos 的构造与维护

planner 需要最近 4 帧的 `qpos[36]`。IsaacSim 中需要自行维护一个循环 buffer。

### 3.1 数据来源

每个仿真 step 结束后，从 IsaacSim 读取当前帧状态：

```python
# root position: [num_envs, 3]
root_pos = self.robot.data.root_pos_w              # (x, y, z)

# root quaternion (IsaacSim 约定: w, x, y, z)
root_quat = self.robot.data.root_quat_w            # (w, x, y, z)

# joint positions (29 个关节，需按 planner 期望顺序排列)
joint_pos = self.robot.data.joint_pos[:, self.planner_joint_ids]  # [num_envs, 29]
```

拼接成单帧 qpos：

```python
qpos_frame = torch.cat([root_pos, root_quat, joint_pos], dim=-1)  # [num_envs, 36]
```

### 3.2 关节顺序对齐

IsaacSim 中的关节顺序与 planner 内部 29 个关节顺序**不一定相同**，必须显式维护映射：

```python
# cat_traverse_cfg.py 中新增
planner_joint_names: list[str] = [
    # 按 planner_sonic.onnx 训练时的关节顺序列出全部 29 个关节名称
    # 必须与 planner 训练数据中的 MuJoCo model 关节顺序完全一致
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    ...  # 共 29 个
]
```

`planner_joint_ids` 在 `__init__` 中通过 `SceneEntityCfg` 解析得到。

### 3.3 坐标系对齐

planner 使用 **MuJoCo Z-up** 坐标系（x 前、y 左、z 上）。

IsaacSim 同样使用 Z-up，但需确认：

- `root_pos_w` 的世界原点与 planner 期望的参考原点是否一致
- `root_quat_w` 的四元数约定是否为 `(w, x, y, z)`（IsaacSim 默认是，planner 也是）

如有偏差，需在填入 buffer 前做坐标变换。

### 3.4 Buffer 维护

```python
# __init__ 中初始化
self._planner_context_buf = torch.zeros(
    self.num_envs, 4, 36, dtype=torch.float32, device=self.device
)

# 每个 step 更新（在调用 planner 推理之前）
def _update_planner_context(self, qpos_frame: torch.Tensor):
    # 滚动：丢弃最旧帧，追加最新帧
    self._planner_context_buf = torch.roll(self._planner_context_buf, shifts=-1, dims=1)
    self._planner_context_buf[:, -1, :] = qpos_frame

# reset 时对指定 env 用当前静止 qpos 填满全部 4 帧
def _reset_planner_context(self, env_ids: torch.Tensor):
    qpos_frame = self._get_current_qpos_frame(env_ids)   # [len(env_ids), 36]
    self._planner_context_buf[env_ids] = qpos_frame.unsqueeze(1).expand(-1, 4, -1)
```

---

## 4. planner 推理与关节目标提取

### 4.1 构造输入并推理

由于 IsaacSim 通常为多 env 并行，而 planner 每次推理 batch_size=1，需要**逐 env** 或做适配。

**方案 A（逐 env，适用于 num_envs 较少或 play 场景）**：

```python
for i in range(self.num_envs):
    ort_inputs = {
        "context_mujoco_qpos": context_buf[i:i+1].cpu().numpy(),   # [1, 4, 36]
        "target_vel":          target_vel[i:i+1].cpu().numpy(),    # [1]
        "mode":                mode_i[i:i+1].cpu().numpy(),        # [1] int64
        "movement_direction":  move_dir[i:i+1].cpu().numpy(),      # [1, 3]
        "facing_direction":    face_dir[i:i+1].cpu().numpy(),      # [1, 3]
        "random_seed":         np.array([self._planner_seeds[i]], dtype=np.int64),
        "has_specific_target": np.zeros((1, 1), dtype=np.int64),
        "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
        "specific_target_headings":  np.zeros((1, 4), dtype=np.float32),
        "allowed_pred_num_tokens":   np.ones((1, 11), dtype=np.int64),
        "height":              np.array([-1.0], dtype=np.float32),
    }
    mujoco_qpos, num_pred_frames = self._planner_session.run(None, ort_inputs)
    # mujoco_qpos: [1, 64, 36], num_pred_frames: [1]
```

**方案 B（批量，需要 planner 支持动态 batch）**：若 planner_sonic.onnx 支持 batch > 1，可一次传入所有 env。实际 onnx 中 batch 维度固定为 1，因此默认使用方案 A。

### 4.2 从 planner 输出提取关节目标

planner 输出轨迹 `mujoco_qpos[1, 64, 36]`，有效帧数为 `num_pred_frames[0]`。

对于单步控制，取**第 0 帧**（最近的预测帧）的关节角作为目标：

```python
n_valid = int(num_pred_frames[0])
# 取第 0 帧，关节角在 [7:36]
joint_target_i = mujoco_qpos[0, 0, 7:36]   # [29]
```

或取多帧做简单插值（可选，视控制频率与 planner 频率比例而定）。

### 4.3 写入关节目标

将 29 个 planner 关节角按 `planner_joint_ids` 写回 `_motor_targets`：

```python
self._motor_targets[:, self.planner_joint_ids] = joint_targets   # [num_envs, 29]
self.robot.set_joint_position_target(self._motor_targets)
```

---

## 5. 固定管理量的处理

| 量 | 处理方式 |
|---|---|
| `random_seed` | 每 env 维护独立整数计数器 `_planner_seeds[num_envs]`，每次推理后 +1 |
| `has_specific_target` | 固定为 `0`，不启用 waypoint |
| `specific_target_*` | 固定为全零 |为什么
| `allowed_pred_num_tokens` | 默认全 `1`；如需固定预测帧数，只打开对应索引位 |
| `height` | cfg 中配置，默认 `-1.0` 禁用；如需站立高度控制，设为正值（单位 m） |

---

## 6. 需要修改的代码位置

### 6.1 `cat_traverse_cfg.py`

```python
@dataclass
class PlannerCfg:
    onnx_path: str = "path/to/planner_sonic.onnx"   # 模型路径
    planner_joint_names: list[str] = field(default_factory=list)  # 29 个关节名（planner 顺序）
    default_height: float = -1.0    # < 0 禁用高度控制
    use_first_frame_only: bool = True   # 只取第 0 帧作为关节目标

# CatTraverseEnvCfg 中添加
planner: PlannerCfg = field(default_factory=PlannerCfg)

# 修改 num_actions
num_actions: int = 8   # [target_vel(1), mode(1), move_dir(3), face_dir(3)]
```

### 6.2 `cat_traverse_env.py`

需要修改的方法：

| 方法 | 修改内容 |
|---|---|
| `__init__` | 加载 ONNX session；初始化 `_planner_context_buf`；解析 `planner_joint_ids` |
| `_post_reset_idx` | 调用 `_reset_planner_context(env_ids)` |
| `_get_obs` | 将 `motor_targets` 替换为 planner 上次输出的关节目标（可选） |
| `step` | 解析 policy 输出 → 构造 planner 输入 → 推理 → 提取关节目标 → 写入仿真 |

### 6.3 `step()` 主体改写

```python
def step(self, actions: torch.Tensor):
    # 1. 解析 policy 输出
    target_vel   = actions[:, 0:1]                          # [N, 1]
    mode_raw     = actions[:, 1:2]                          # [N, 1]
    move_dir     = F.normalize(actions[:, 2:5], dim=-1)     # [N, 3]
    face_dir     = F.normalize(actions[:, 5:8], dim=-1)     # [N, 3]
    mode_int     = mode_raw.round().clamp(0, 26).long()     # [N, 1] int64

    # 2. 更新 context buffer（用上一步末的 qpos）
    self._update_planner_context(self._get_current_qpos_frame())

    # 3. 逐 env 推理 planner
    joint_targets = torch.zeros(self.num_envs, 29, device=self.device)
    for i in range(self.num_envs):
        ort_inputs = self._build_planner_inputs(i, target_vel, mode_int, move_dir, face_dir)
        mujoco_qpos, num_pred_frames = self._planner_session.run(None, ort_inputs)
        joint_targets[i] = torch.from_numpy(mujoco_qpos[0, 0, 7:36]).to(self.device)
        self._planner_seeds[i] += 1

    # 4. 写入关节目标
    self._motor_targets.copy_(self.robot.data.default_joint_pos)
    self._motor_targets[:, self.planner_joint_ids] = joint_targets
    for _ in range(self.cfg.decimation):
        self.robot.set_joint_position_target(self._motor_targets)
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        self.scene.update(self.step_dt)

    # 5. 后续观测、奖励、终止计算（不变）
    ...
```

---

## 7. 奖励与终止逻辑的兼容

原有 CAT 奖励（势场、障碍距离）和终止逻辑均基于 `root_pos`、关节速度等量，不依赖 policy action 的含义，因此**无需修改**。

唯一需要注意：

- 原来的 `_motor_targets`（用于 obs 中的 `motor_targets`）现在来自 planner 输出，而非 policy 直接输出。
- 如果观测中保留了 `last_action`，其含义变为"上一步的 planner 控制量"，需要在 `_get_obs` 中做对应调整。

---

## 8. 训练时的注意事项

### 8.1 planner 推理不参与梯度

planner_sonic.onnx 通过 `onnxruntime` 推理，不在 PyTorch 计算图中，梯度不会反传穿过 planner。
Policy 的梯度来源是：**planner 输出关节目标 → 仿真物理 → 状态 → 奖励**，通过 PPO 的值函数估计实现。

### 8.2 推理频率

planner 内部以 **30 fps** 生成轨迹（每 token 4 帧 × 30 fps = 约 0.133 s/token）。
IsaacSim 通常以 500 Hz 控制，decimation=4 时策略频率为 125 Hz。

建议：

- **每隔 N 个 policy step 调用一次 planner**，在两次调用之间复用上一次 planner 输出的关节轨迹。
- 例如：planner 每 4 个 policy step 调用一次，期间按帧索引顺序消费 `mujoco_qpos` 的前几帧。

### 8.3 训练性能

逐 env 循环调用 ONNX 在 `num_envs` 较大时会成为瓶颈。训练阶段建议：

- 将 `num_envs` 限制在合理范围（如 64 以内）
- 或使用 TensorRT 推理替代 onnxruntime，并尝试支持批量推理

---

## 9. 验收检查清单

- [ ] `cat_traverse_cfg.py` 中 `num_actions = 8`，`PlannerCfg` 配置完整
- [ ] `planner_joint_names` 列表与 planner_sonic.onnx 训练用 MuJoCo model 关节顺序一致
- [ ] `_planner_context_buf` 在 reset 时正确初始化（4 帧重复当前 qpos）
- [ ] `step()` 中 policy 输出不再写入关节，而是传给 planner
- [ ] planner 输出 `mujoco_qpos[0, 0, 7:36]` 正确映射回 IsaacSim 关节目标
- [ ] 坐标系（Z-up，四元数 w 在前）与 planner 约定对齐
- [ ] 奖励和终止逻辑不依赖 policy action 语义，验证不受影响
- [ ] 在 `num_envs=1` 下 `play.py` 可运行且关节运动符合预期
