# `cat_traverse_g1_flow_mimic` 改为"输出位姿增量"方案待办清单

## 目标

将当前 `cat_traverse_g1_flow_mimic` 中高层 policy 的输出语义从：

- `14 x (pos[3] + rot6d[6])`  ← 绝对位姿

改为：

- `14 x (delta_pos[3] + delta_rot6d[6])`  ← 相对当前帧的位姿增量

然后在环境内部：

1. 取当前时刻 14 个 tracked bodies 在 pelvis anchor frame 下的实际 pose
2. 将 policy 输出的 `delta_pos` 增量叠加到当前位置上
3. 将 policy 输出的 `delta_rot6d` 经 Gram-Schmidt 正交化后得到 `R_delta`，与当前旋转矩阵组合得到目标旋转 `R_target = R_current @ R_delta`
4. 取 `R_target` 前两列作为 flow mimic 所需的 `rot6d[6]`（列向量拼接）
5. 将组合后的 `14 x (pos[3] + rot6d[6])` 输入给 frozen `flow_mimic.pt`

---

## Anchor Frame 精确定义

**Anchor frame = 当前步的 pelvis 局部坐标系，每个仿真 step 随 pelvis 实时更新。**

具体定义：
- 原点：当前 pelvis 的世界坐标位置 `p_pelvis_w`
- 朝向：由当前 pelvis 的旋转矩阵 `R_pelvis_w` 定义，与 pelvis body frame 完全对齐

相对变换公式：

```text
R_aw = R_pelvis_w              # world 系中 pelvis 的旋转矩阵（anchor 轴在 world 中的表示）
R_wa = R_aw^T                  # world → anchor 变换

p_body_a = R_wa @ (p_body_w - p_pelvis_w)   # body 位置在 anchor frame 下
R_body_a = R_wa @ R_body_w                  # body 旋转在 anchor frame 下
```

由于 pelvis 自身即为 anchor 原点，`pelvis_pos_a = [0,0,0]`，`R_pelvis_a = I`。
其他 13 个 body 的 pose 是非零的，但代码仍统一处理全部 14 个（pelvis 作为 body 0）。

---

## 动作尺度约定

| 量 | 单位 | 说明 |
|---|---|---|
| `delta_pos[3]` | 米 (m) | 在 anchor (pelvis) frame 下的位置增量，经 `delta_pos_scale` 缩放后叠加 |
| `delta_rot6d[6]` | 无量纲 | delta 旋转矩阵 `R_delta` 的前两列（列向量按顺序拼接），Gram-Schmidt 正交化后与当前旋转组合 |

Policy 输出经 `clip_actions=1.0` 裁剪到 `[-1, 1]`。

- `delta_pos` 在叠加前乘以 `cfg.delta_pos_scale`（建议默认 `0.15` m），使 ±1 对应最大 ±0.15 m 的单步位移
- `delta_rot6d` 作为旋转矩阵列向量，数值天然在 `[-1, 1]` 内；零输出方向 `[1,0,0, 0,1,0]` 对应 R_delta ≈ I（无旋转变化）

---

## 先明确的一件事

从 **绝对位姿** 改为 **delta 位姿（rot6d 表示）** 后，每个 body 仍然输出 9 维：

- `delta_pos[3]` + `delta_rot6d[6]` = **9 dims per body**
- 总动作维度 = `14 × 9 = 126`，**与原方案相同，无需修改 RL_ACTION_DIM**

因此：
- `RL_ACTION_DIM`、`N_TRACKED_BODIES`、`BODY_POSE_DIM`、`DelayBuffer` 维度均**不需要修改**
- 旧 PPO checkpoint 在网络结构上维度兼容，但语义已完全改变，**建议从新实验开始训练**
- Actor obs 因新增 body_pos 而维度变化，旧 checkpoint 加载会直接报错，无法 resume

---

## 当前实现里会被影响的核心链路

当前代码中：

- 高层 policy 直接输出 `126` 维 absolute future pose
- `actor_future[:, :126] = clipped_actions`（直接写 policy 原始输出）
- `_build_actor_history_frame()` 把 `actions[:, 0:3]` 和 `actions[:, 3:9]` 直接当 pelvis absolute anchor pose

改完后应变成：

- 高层 policy 输出 `126` 维 delta pose（`delta_pos + delta_rot6d`，语义变化）
- 环境先读取当前 14-body pose，合成 absolute target pose
- `actor_future[:, :126]` 写入**合成后的 absolute pose**，而不是原始 policy 输出
- `_build_actor_history_frame()` 使用**合成后的 pelvis absolute pose**（composed_pose[:, 0, :]）

---

## 涉及文件

- `legged_lab/envs/cat_traverse/cat_traverse_flow_mimic_env.py`
- `legged_lab/envs/cat_traverse/cat_traverse_flow_mimic_cfg.py`
- `legged_lab/envs/cat_traverse/rot_utils.py`（新建）
- `transformer_teacher_dims.md`

---

## 必做事项

### 1. 明确新的动作定义

高层 policy 的每个 body 输出重新定义为：

- `delta_pos = [dx, dy, dz]`：在 anchor (pelvis) frame 下，单位 m，乘以 `delta_pos_scale` 后叠加
- `delta_rot6d = [c1_x, c1_y, c1_z, c2_x, c2_y, c2_z]`：delta 旋转矩阵 `R_delta` 的前两列，列向量按顺序拼接

旋转增量组合规则（全程旋转矩阵，无 RPY，无万向锁）：

```text
R_delta    = gram_schmidt(delta_rot6d)               # (3, 3) 正交旋转矩阵
R_target   = R_current @ R_delta                     # 当前旋转右乘 delta
rot6d_out  = concat(R_target[:, 0], R_target[:, 1]) # 前两列列向量拼接（非 reshape）
```

`delta_rot6d` 零方向输出 `[1,0,0, 0,1,0]` → R_delta = I → R_target = R_current（无旋转变化）。

参考系：所有量均在 **anchor (pelvis) frame** 下定义，anchor frame 每步随 pelvis 实时更新。

---

### 2. `RL_ACTION_DIM` 维持 126（无需修改）

`delta_pos[3] + delta_rot6d[6] = 9` dims per body，`14 × 9 = 126`，因此以下量**不需要修改**：

- `N_TRACKED_BODIES = 14`
- `BODY_POSE_DIM = 9`
- `RL_ACTION_DIM = 126`
- `self.num_actions = 126`
- `DelayBuffer` 初始化维度
- PPO actor 输出维度

**只需修改**：

- 顶部注释（"absolute body pose" → "delta_pos(3) + delta_rot6d(6) per body"）
- `step()` 和 `_build_actor_history_frame()` 的内部逻辑（见后续事项）
- `cfg.py` 新增 `delta_pos_scale` 超参数

---

### 3. 新增 14 个 tracked bodies 的 body-id 解析

在 `CatTraverseFlowMimicEnv` 中新增 `_resolve_tracked_body_ids()`，解析顺序必须严格对齐 `transformer_teacher_dims.md`：

| 索引 | body 名称 |
|------|-----------|
| 0 | `pelvis` |
| 1 | `left_hip_roll_link` |
| 2 | `left_knee_link` |
| 3 | `left_ankle_roll_link` |
| 4 | `right_hip_roll_link` |
| 5 | `right_knee_link` |
| 6 | `right_ankle_roll_link` |
| 7 | `torso_link` |
| 8 | `left_shoulder_roll_link` |
| 9 | `left_elbow_link` |
| 10 | `left_wrist_yaw_link` |
| 11 | `right_shoulder_roll_link` |
| 12 | `right_elbow_link` |
| 13 | `right_wrist_yaw_link` |

---

### 4. 提取当前 14 个 tracked bodies 在 anchor frame 下的实际 pose

新增 helper `_get_current_tracked_body_pose_anchor()`：

```python
def _get_current_tracked_body_pose_anchor(self):
    """返回 14 个 tracked bodies 在 pelvis anchor frame 下的位置和旋转矩阵。

    Returns:
        current_body_pos_a: (N, 14, 3)
        current_body_rot_a: (N, 14, 3, 3)
    """
    body_pos_w  = self.robot.data.body_pos_w[:, self._tracked_body_ids, :]   # (N, 14, 3)
    body_quat_w = self.robot.data.body_quat_w[:, self._tracked_body_ids, :]  # (N, 14, 4)

    pelvis_pos_w  = self.robot.data.root_pos_w    # (N, 3)
    pelvis_quat_w = self.robot.data.root_quat_w   # (N, 4)  (w, x, y, z)

    R_pelvis_w = quat_to_matrix(pelvis_quat_w)          # (N, 3, 3)
    R_wa = R_pelvis_w.transpose(-1, -2)                 # world → anchor

    diff = body_pos_w - pelvis_pos_w.unsqueeze(1)        # (N, 14, 3)
    current_body_pos_a = torch.einsum("nij,nbj->nbi", R_wa, diff)     # (N, 14, 3)

    R_body_w = quat_to_matrix(body_quat_w)               # (N, 14, 3, 3)
    current_body_rot_a = torch.einsum("nij,nbjk->nbik", R_wa, R_body_w)  # (N, 14, 3, 3)

    return current_body_pos_a, current_body_rot_a
```

可在 play 模式下验证：`current_body_pos_a[:, 0, :]`（pelvis 自身）应接近全零。

---

### 5. 新增旋转工具文件 `rot_utils.py`

新建 `legged_lab/envs/cat_traverse/rot_utils.py`，包含以下函数：

```python
def quat_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """(N, 4) [w,x,y,z] → (N, 3, 3) 旋转矩阵"""

def gram_schmidt(rot6d: torch.Tensor) -> torch.Tensor:
    """(..., 6) rot6d → (..., 3, 3) 正交旋转矩阵。

    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = normalize(a1)                         # clamp norm >= 1e-6
    b2 = normalize(a2 - dot(b1, a2) * b1)     # clamp norm >= 1e-6
    b3 = cross(b1, b2)
    return stack([b1, b2, b3], dim=-1)         # 列向量构成矩阵
    """

def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """(..., 3, 3) → (..., 6)，取前两列列向量显式拼接。

    return concat(R[..., :, 0], R[..., :, 1], dim=-1)   # 不用 reshape
    """
```

**注意**：不需要 `matrix_to_rpy` 和 `rpy_to_matrix`，旋转全程在矩阵空间进行，彻底避免 RPY 万向锁。

---

### 6. 定义增量叠加规则

```python
# policy 输出 reshape
actions_r  = clipped.view(N, 14, 9)           # (N, 14, 9)
delta_pos  = actions_r[..., 0:3]              # (N, 14, 3)  单位 m
delta_rot6d = actions_r[..., 3:9]             # (N, 14, 6)

# 读取当前 body pose（anchor frame）
current_pos_a, current_rot_a = self._get_current_tracked_body_pose_anchor()
# current_pos_a:  (N, 14, 3)
# current_rot_a:  (N, 14, 3, 3)

# 位置增量叠加
target_pos_a = current_pos_a + delta_pos * self.cfg.delta_pos_scale  # (N, 14, 3)

# 旋转增量叠加（无奇异性）
R_delta    = gram_schmidt(delta_rot6d)         # (N, 14, 3, 3)
R_target   = current_rot_a @ R_delta          # (N, 14, 3, 3)

# 转为 flow mimic 所需 rot6d（前两列列向量拼接）
rot6d_target = matrix_to_rot6d(R_target)      # (N, 14, 6)

# 合成最终 pose
composed_pose = torch.cat([target_pos_a, rot6d_target], dim=-1)  # (N, 14, 9)
```

---

### 7. 确认 `rot6d` 的展平顺序

已确认：flow_mimic 使用**列向量优先拼接**：

```python
rot6d = torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)
```

**不要**使用 `R[..., :2].reshape(-1, 6)`（这是行优先，与训练约定不符）。
`matrix_to_rot6d()` 函数内部必须使用显式列拼接。

---

### 8. 用"合成后的 absolute future pose"重建 `actor_future`

```python
actor_future = torch.zeros(self.num_envs, D_ACTOR_FUTURE, device=self.device)
actor_future[:, :RL_ACTION_DIM] = composed_pose.view(N, -1)  # (N, 126) absolute pose
```

`actor_future[:, :126]` 写入的是**合成后的 absolute pose**，而不是原始 delta policy 输出。
后续 `D_ACTOR_FUTURE = 630` 不变，后 504 维继续保持为零。

---

### 9. 修改 `_build_actor_history_frame()` 接口

原签名使用 `actions`（原始 policy 输出），改为接收 `composed_pose`（合成后的 absolute 14-body pose）：

```python
def _build_actor_history_frame(
    self,
    composed_pose: torch.Tensor   # (N, 14, 9)：合成后 absolute pos+rot6d，anchor frame
) -> torch.Tensor:
    # pelvis absolute pose（body 0）
    pelvis_pos_a = composed_pose[:, 0, 0:3]   # (N, 3)  ← 合成后，非原始 delta
    pelvis_ori_a = composed_pose[:, 0, 3:9]   # (N, 6)
    ...
```

其余字段（cmd_pos, cmd_vel, 关节状态等）与原实现相同。

同步修改 `step()` 中的调用：

```python
# 原来：new_frame = self._build_actor_history_frame(clipped)
# 改为：
new_frame = self._build_actor_history_frame(composed_pose)
```

---

### 10. 增强 RL Policy 的 Observation

新增 14 个 tracked bodies 在 anchor frame 下的当前位置（**42 dims**）作为 actor obs：

```python
# compute_current_observations() 中新增
current_body_pos_a, _ = self._get_current_tracked_body_pose_anchor()
body_pos_flat = current_body_pos_a.view(self.num_envs, -1)   # (N, 42)
```

更新 actor obs 拼接（在 `actor_pf` 之后追加）：

```python
actor_obs = torch.cat([
    ang_vel,           # 3
    proj_grav,         # 3
    joint_pos,         # 23
    joint_vel,         # 23
    last_fm,           # 12
    field["command_actor_nav"] * self.obs_scales.commands,  # 4
    self._foot_height_target,   # 1
    gait_phase,        # 4
    actor_pf,          # 77
    body_pos_flat,     # 42  ← 新增：14 bodies 在 pelvis anchor frame 下的当前位置
], dim=-1)             # ≈ 192
```

**同步更新 `init_obs_buffer()` 里的 `noise_vec` 构建**：对 `body_pos_flat` 段补充零噪声（初始阶段不建议加噪）。

可选后续：如果训练中旋转信息明显不足，可进一步追加 14 个 bodies 的 `rot6d`（84 dims），但建议先用 42 dims 跑通训练。

---

### 11. 更新注释、文档和命名

至少需要同步更新：

- `cat_traverse_flow_mimic_env.py` 顶部数据流注释
- `BODY_POSE_DIM` 注释（改为 `delta_pos(3) + delta_rot6d(6) per body`）
- `RL_ACTION_DIM` 注释
- `_build_actor_history_frame()` docstring
- `transformer_teacher_dims.md` 中关于当前 task 封装方式的说明
- 所有写着 "absolute body pose" 或 "absolute future pose" 的注释

---

### 12. 处理 Checkpoint 兼容性

因为：

- `RL_ACTION_DIM` 未变（仍为 126），网络输出头兼容
- Actor obs 维度已变（新增 42 dims body_pos），网络输入头**不兼容**

所以：

- 旧 PPO checkpoint 直接加载会因 obs 维度不符而报错
- **使用新的 experiment name**，不要 resume 旧实验

```python
experiment_name = "cat_traverse_g1_flow_mimic_delta"
```

---

### 13. 调试验证步骤（新增）

在 `play.py` 模式（`num_envs=1`）下，在 `step()` 末尾新增调试日志方法：

```python
def _debug_log_step(self, step_idx, delta_pos, delta_rot6d, R_delta,
                    composed_pose, joint_targets):
    """每 50 step 打印一次关键中间量，用于验证 delta→absolute→flow_mimic 链路。"""
    if step_idx % 50 != 0:
        return

    # 1. policy delta 输出统计
    print(f"[DBG] delta_pos    mean={delta_pos[0].mean():.4f}  std={delta_pos[0].std():.4f}")
    print(f"[DBG] delta_rot6d  mean={delta_rot6d[0].mean():.4f}  std={delta_rot6d[0].std():.4f}")

    # 2. Gram-Schmidt 后 R_delta 的行列式（应全部接近 1.0）
    det = torch.det(R_delta[0])   # (14,)
    print(f"[DBG] R_delta det  min={det.min():.4f}  max={det.max():.4f}  (expect ≈ 1.0)")

    # 3. pelvis 在 anchor frame 下的合成后 absolute pose
    pelv = composed_pose[0, 0]
    print(f"[DBG] composed pelvis  pos_a={pelv[:3].tolist()}  rot6d={pelv[3:].tolist()}")

    # 4. flow_mimic 输出的关节目标范围
    print(f"[DBG] joint_targets  min={joint_targets[0].min():.4f}  max={joint_targets[0].max():.4f}")

    # 5. NaN / Inf 检测
    for name, t in [
        ("delta_pos",     delta_pos),
        ("delta_rot6d",   delta_rot6d),
        ("composed_pose", composed_pose),
        ("joint_targets", joint_targets),
    ]:
        if torch.isnan(t).any() or torch.isinf(t).any():
            print(f"[WARN] NaN/Inf detected in {name}!")
```

**验证重点**：

- `R_delta` 行列式全部接近 `1.0`（Gram-Schmidt 正确）
- `composed_pose[:, 0, :3]`（pelvis pos_a）接近 `[0,0,0]`（anchor frame 原点）
- `joint_targets` 在 soft joint limits 内，无 NaN
- `delta_pos` 的实际幅度合理（单步不超过 `delta_pos_scale`）

---

## 建议新增的中间接口

| 函数 | 位置 | 说明 |
|------|------|------|
| `quat_to_matrix(quat)` | `rot_utils.py` | `(N,4)→(N,3,3)` |
| `gram_schmidt(rot6d)` | `rot_utils.py` | `(...,6)→(...,3,3)`，含 norm clamp |
| `matrix_to_rot6d(R)` | `rot_utils.py` | `(...,3,3)→(...,6)`，显式列拼接 |
| `_resolve_tracked_body_ids()` | env | 解析 14-body ids，启动时调用 |
| `_get_current_tracked_body_pose_anchor()` | env | `→(N,14,3),(N,14,3,3)` |
| `_compose_delta_pose(...)` | env | delta+current → `(N,14,9)` absolute |
| `_build_actor_future(composed_pose)` | env | `(N,14,9)→(N,630)` |
| `_debug_log_step(...)` | env | play 模式调试，生产时可 no-op |

---

## 需要提前确认的设计点

### A. Anchor frame 随 pelvis 每步更新（已确认）

Policy 的 delta 输出始终是相对于**当前步 pelvis 坐标系**的，学习信号稳定，不会因机器人远距离移动而积累偏差。

### B. "当前帧"的时间点约定

建议统一：**当前帧 = 调用 flow_mimic 推理之前、本步仿真 step 之前**的机器人状态。

每步执行顺序：

1. 读当前机器人状态 → `current_body_pose_a`
2. 接收 policy delta 输出 → `composed_pose`
3. 写入 `actor_future` → 推理 flow_mimic → `joint_targets`
4. `robot.set_joint_position_target(...)` → 推进物理仿真
5. 更新 `actor_history`（写入 `composed_pose[:, 0, :]` 作为 pelvis anchor）

### C. Gram-Schmidt 数值稳定性

当 policy 初期输出接近零向量时，两列可能接近平行，导致 b2 接近零向量。

在 `gram_schmidt()` 中对每个列向量的范数做 clamp（最小值 `1e-6`）：

```python
b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp(min=1e-6)
proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
b2_raw = a2 - proj
b2 = b2_raw / b2_raw.norm(dim=-1, keepdim=True).clamp(min=1e-6)
b3 = torch.cross(b1, b2, dim=-1)
```

### D. `rot6d` 列展开顺序（已确认）

flow_mimic 训练约定：**列向量优先拼接**，代码中显式使用 `cat([R[..., :, 0], R[..., :, 1]], dim=-1)`，不使用 `reshape`。

### E. `delta_pos_scale` 超参数

在 `CatTraverseFlowMimicEnvCfg` 中新增：

```python
delta_pos_scale: float = 0.15   # 单步最大位置变化 ±0.15 m
```

---

## 验证清单

- [ ] `RL_ACTION_DIM` 仍为 126，顶部注释已更新为 `delta_pos(3) + delta_rot6d(6) per body`
- [ ] `actor_future[:, :126]` 写入的是合成后的 absolute pose，而不是原始 delta
- [ ] 14 tracked body ids 解析顺序严格对齐 `transformer_teacher_dims.md`
- [ ] `_get_current_tracked_body_pose_anchor()` 验证：pelvis `pos_a ≈ [0,0,0]`
- [ ] `gram_schmidt()` 输出的旋转矩阵行列式接近 `1.0`（单元测试覆盖）
- [ ] `matrix_to_rot6d()` 使用显式列拼接，非 reshape
- [ ] `composed_pose[:, 0, :]` 正确进入 `_build_actor_history_frame()` 作为 pelvis anchor
- [ ] actor obs 新增了 `body_pos_flat`（42 dims）
- [ ] `init_obs_buffer()` 中 `noise_vec` 同步扩展（body_pos 段为零噪声）
- [ ] 调试日志在 play 模式下可正常输出，无 NaN / Inf
- [ ] 旧 checkpoint 未被错误 resume（新 experiment name：`cat_traverse_g1_flow_mimic_delta`）

---

## 建议的实现顺序

1. 实现 `rot_utils.py`（`quat_to_matrix`, `gram_schmidt`, `matrix_to_rot6d`），写简单单元测试验证正确性（特别是 det ≈ 1.0 和列拼接顺序）
2. 实现 `_resolve_tracked_body_ids()`，启动后打印 body 名称顺序确认
3. 实现 `_get_current_tracked_body_pose_anchor()`，play 模式下验证 pelvis pos_a ≈ 零
4. 实现 `_compose_delta_pose()`，验证 R_delta 行列式和 rot6d 展平顺序
5. 修改 `step()` 主逻辑：接入 delta→absolute 合成，更新 actor_future 和 actor_history
6. 修改 `_build_actor_history_frame()` 接口，改用 `composed_pose`
7. 修改 `compute_current_observations()`，新增 `body_pos_flat`（42 dims）
8. 更新 `init_obs_buffer()` 中的 `noise_vec` 构建
9. 新增 `_debug_log_step()`，在 play 模式下走一遍验收清单
10. 统一更新注释、`experiment_name` 和文档

---

## 最简版改造摘要

这次改造在高层 policy 与 frozen flow mimic 之间插入一层位姿合成逻辑：

```text
policy delta 输出（126 dims，语义变化）
    ↓  reshape → (N, 14, 9) → delta_pos (N,14,3) + delta_rot6d (N,14,6)
读取当前 14-body pose（anchor frame = pelvis 局部坐标，每步更新）
    ↓  叠加
    target_pos   = current_pos + delta_pos × delta_pos_scale
    R_delta      = gram_schmidt(delta_rot6d)
    R_target     = current_rot @ R_delta
    rot6d_target = concat(R_target[:,:,0], R_target[:,:,1])  ← 列向量显式拼接
    ↓  拼接为 (N, 126) → 写入 actor_future[:, :126]
冻结的 flow_mimic.pt
    ↓
29 维 joint targets
    ↓
robot.set_joint_position_target(...)
```

核心变化三点：

1. **Policy 输出语义**：absolute pose → delta pose（`delta_pos + delta_rot6d`），dim 不变
2. **环境中新增**：读取当前 body pose + delta 合成 absolute pose 的步骤（全程旋转矩阵，无 RPY）
3. **Actor obs 扩展**：新增 14-body 当前位置（42 dims），帮助 policy 感知当前状态以输出合理 delta
