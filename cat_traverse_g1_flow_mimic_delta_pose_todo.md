# `cat_traverse_g1_flow_mimic` 改为“输出位姿增量”方案待办清单

## 目标

将当前 `cat_traverse_g1_flow_mimic` 中高层 policy 的输出语义从：

- `14 x (pos[3] + rot6d[6])`

改为：

- `14 x (delta_pos[3] + delta_rpy[3])`

然后在环境内部：

1. 取当前时刻 14 个 tracked bodies 的实际 pose
2. 将 policy 输出的增量叠加到当前 pose 上
3. 把叠加后的姿态角 `roll/pitch/yaw` 转成 `3x3` 旋转矩阵
4. 取旋转矩阵前两列作为 flow mimic 所需的 `rot6d[6]`
5. 再将组合后的 `14 x (pos[3] + rot6d[6])` 输入给 frozen `flow_mimic.pt`

## 先明确的一件事

如果每个 body 的输出从 `pos[3] + rot6d[6]` 改为 `delta_pos[3] + delta_rpy[3]`，那么高层 policy 的动作维度会从：

- `14 x 9 = 126`

变成：

- `14 x 6 = 84`

所以这次改动默认意味着：

- `RL_ACTION_DIM` 需要从 `126` 改成 `84`
- 训练好的旧 checkpoint 与新结构不兼容，不能直接 resume

如果你希望“仍然保持 policy 输出 126 维”，那就需要另外定义剩余 `42` 维的含义；按当前设想，动作维度应当改为 `84`。

## 当前实现里会被影响的核心链路

当前代码中：

- 高层 policy 直接输出 `126` 维 absolute future pose
- `actor_future[:, :126] = clipped_actions`
- `_build_actor_history_frame()` 直接把 `actions[:, 0:3]` 和 `actions[:, 3:9]` 当作 pelvis 的 absolute anchor pose

改完后应变成：

- 高层 policy 输出 `84` 维 delta pose
- 环境先根据当前实际 body pose 合成 absolute target pose
- `actor_future[:, :126]` 写入的是合成后的 absolute pose，而不是原始 policy 输出
- `_build_actor_history_frame()` 使用的 pelvis anchor 也必须是“合成后的 absolute pelvis pose”，而不是原始 delta

## 涉及文件

- `legged_lab/envs/cat_traverse/cat_traverse_flow_mimic_env.py`
- `legged_lab/envs/cat_traverse/cat_traverse_flow_mimic_cfg.py`
- `transformer_teacher_dims.md`
- 可能还会涉及：
  - `legged_lab/envs/cat_traverse/cat_traverse_env.py`
  - 新增一个小的旋转/姿态工具文件，避免把转换逻辑堆进 env

## 必做事项

## 1. 明确新的动作定义

需要把高层 policy 的每个 body 输出重新定义为：

- `delta_pos = [dx, dy, dz]`
- `delta_rpy = [droll, dpitch, dyaw]`

并明确这些量的参考系：

- 推荐：都在 flow mimic 使用的同一个 anchor frame 下定义
- 不建议混用 world frame / nav frame / body local frame

需要确认的实现约定：

- `delta_pos` 是在 anchor frame 下相对当前实际 body 位置的增量
- `delta_rpy` 是在 anchor frame 下相对当前实际 body 姿态角的增量

## 2. 将 `RL_ACTION_DIM` 从 126 改为 84

需要同步修改：

- `N_TRACKED_BODIES`
- `BODY_POSE_DIM`
- `RL_ACTION_DIM`
- `self.num_actions`
- `DelayBuffer` 初始化维度
- PPO actor 输出维度
- 相关注释和文档

注意：

- `flow mimic` 的输入维度 `D_ACTOR_FUTURE = 630` 不变
- 变化的是高层 policy 输出维度，不是 flow mimic future token 的维度

## 3. 新增 14 个 tracked bodies 的 body-id 解析

当前 `CatTraverseEnv` 维护的是 probe bodies，不是 flow mimic future 需要的完整 14 个 tracked bodies。

因此需要在 `CatTraverseFlowMimicEnv` 中单独维护一套 tracked body ids，顺序必须严格对齐 `transformer_teacher_dims.md`：

1. `pelvis`
2. `left_hip_roll_link`
3. `left_knee_link`
4. `left_ankle_roll_link`
5. `right_hip_roll_link`
6. `right_knee_link`
7. `right_ankle_roll_link`
8. `torso_link`
9. `left_shoulder_roll_link`
10. `left_elbow_link`
11. `left_wrist_yaw_link`
12. `right_shoulder_roll_link`
13. `right_elbow_link`
14. `right_wrist_yaw_link`

这一步是硬需求，否则无法从仿真状态中正确恢复当前 14-body pose。

## 4. 提取当前 14 个 tracked bodies 的实际 pose

需要新增一个 helper，获取当前时刻 14 个 tracked bodies 的：

- 位置
- 姿态

建议统一转到 flow mimic 所要求的 anchor frame 下。

建议输出中间量：

- `current_body_pos_a`: `(N, 14, 3)`
- `current_body_rot_a`: `(N, 14, 3, 3)`

推荐公式：

```text
R_aw = root rotation matrix (anchor -> world)
R_wa = R_aw^T

p_body_a = R_wa * (p_body_w - p_root_w)
R_body_a = R_wa * R_body_w
```

这样得到的是“当前实际 body pose 在当前 anchor frame 下的表示”。

## 5. 增加旋转表示转换工具

至少需要以下工具函数：

- `quat_to_matrix`
- `matrix_to_rpy`
- `rpy_to_matrix`
- `matrix_to_rot6d`

其中：

- `matrix_to_rpy` 用于从当前实际姿态提取 `roll/pitch/yaw`
- `rpy_to_matrix` 用于将叠加后的姿态角转回旋转矩阵
- `matrix_to_rot6d` 用于生成 flow mimic 所需的 `6D rotation`

建议把这些工具集中管理，避免散落在 env 里。

## 6. 定义增量叠加规则

当前设想下，合成后的目标 pose 应为：

```text
target_pos_a = current_pos_a + delta_pos
target_rpy_a = wrap_to_pi(current_rpy_a + delta_rpy)
target_rot_a = rpy_to_matrix(target_rpy_a)
target_rot6d = first_two_columns(target_rot_a)
```

这里有几个要点必须落实：

- 对 `rpy` 累加后要做角度 wrap，避免数值爆掉
- 需要明确欧拉角顺序，推荐写死为 `roll -> pitch -> yaw`
- 需要明确 `rpy_to_matrix` 的乘法顺序，例如 `Rz(yaw) @ Ry(pitch) @ Rx(roll)` 或等价定义，并全程保持一致

## 7. 明确 `rot6d` 的展平顺序

flow mimic 文档里写的是“旋转矩阵前两列”，但代码实现时还要确认 6 维向量的排列顺序。

需要明确是：

- 列优先拼接：`[c1; c2]`

还是：

- 简单 reshape 导致的行优先顺序

建议不要直接 `reshape(6)`，而是显式拼接两列，避免顺序出错。

示意：

```text
rot6d = concat(R[..., :, 0], R[..., :, 1])
```

这一步必须和原始 flow mimic 训练时的预处理保持一致。

## 8. 用“合成后的 absolute future pose”重建 `actor_future`

当前代码是：

```text
actor_future[:, :RL_ACTION_DIM] = clipped
```

改完后应变成：

1. policy 输出 `84` 维 delta
2. env 根据当前实际 pose 合成 `14 x 9 = 126` 维 absolute pose
3. 将这 `126` 维 absolute pose 写入 `actor_future[:, :126]`

也就是说：

- policy action dim 变小
- flow mimic future dim 不变
- 中间多出一层“delta -> absolute pose”的合成逻辑

## 9. 修改 `_build_actor_history_frame()`

当前 `_build_actor_history_frame()` 默认：

- `actions[:, 0:3]` 是 pelvis absolute position
- `actions[:, 3:9]` 是 pelvis absolute 6D rotation

改完后这个假设不成立。

因此需要改成：

- 从“合成后的 absolute 14-body target pose”里取第 0 个 body，也就是 pelvis
- 使用 pelvis 的 absolute `pos[3] + rot6d[6]` 写入 history frame 的：
  - `motion_anchor_pos_b`
  - `motion_anchor_ori_b`

也就是说，history 里写入的是“合成后的 pelvis absolute pose”，而不是原始 delta。

## 10. 检查 observation 是否需要增强

严格来说，当前 actor 只看：

- joint positions / velocities
- gait / command / PF
- last flow-mimic leg targets

它并没有显式看到 14 个 tracked bodies 的当前 absolute pose。

从可实现性看：

- 可以不改 actor observation，直接依赖关节状态隐式表达当前 body pose

但从可学性看，需要评估是否应加入：

- pelvis 当前 rpy
- torso 当前 rpy
- 或完整 14-body 当前 pose 的压缩表示

这项不是绝对必改，但建议在实现前先判断是否要顺手一起做。

## 11. 更新注释、文档和命名

至少需要同步更新：

- `cat_traverse_flow_mimic_env.py` 顶部数据流注释
- `BODY_POSE_DIM` 的注释
- `RL_ACTION_DIM` 的注释
- `transformer_teacher_dims.md` 中关于当前 task 封装方式的说明
- 任何写着“policy 输出 126 维 absolute body pose”的地方

否则后续容易在代码和文档之间出现两套语义。

## 12. 处理 checkpoint 兼容性

因为高层 policy 的输出维度会从 `126` 改成 `84`，所以：

- 旧的 PPO checkpoint 不能直接加载到新模型上
- 旧日志和新日志需要分开
- `resume=True` 时要特别小心

建议：

- 单独起一个新的 experiment name
- 或至少不要复用旧 run

## 建议新增的中间接口

为了避免 `step()` 里逻辑过于拥挤，建议新增这些 helper：

- `_resolve_tracked_body_ids()`
- `_get_current_tracked_body_pose_anchor()`
- `_matrix_to_rpy()`
- `_rpy_to_matrix()`
- `_matrix_to_rot6d()`
- `_compose_delta_future_pose()`
- `_build_actor_future_from_composed_pose()`

其中：

- `_compose_delta_future_pose()` 输出建议为 `(N, 14, 9)`
- `_build_actor_future_from_composed_pose()` 再把它展平并写到 `(N, 630)` 的 future token 中

## 需要提前确认的设计点

下面这些点建议在正式改代码前先拍板：

### A. 动作维度是否接受改为 84

这是最关键的一条。如果不能接受动作维度变化，就需要重新定义动作表示。

### B. “上一帧”到底指什么

建议默认解释为：

- 当前 step 的实际机器人 body pose

而不是：

- 上一时刻 policy 预测的目标 pose
- 上一时刻 flow mimic 的 future pose

如果这里定义不一致，训练行为会明显不同。

### C. 姿态增量是否真的是“欧拉角逐分量相加”

你当前要求是“roll/pitch/yaw 增量再叠加到当前状态”，这在实现上是直接的，但它不等价于严格的旋转群乘法。

如果你就是要这个语义，那实现上没有问题；只是需要明确这是有意设计，不是几何上最严谨的旋转增量定义。

### D. anchor frame 的精确定义

必须统一：

- 当前 tracked body pose 在哪个 frame 下求出
- delta 在哪个 frame 下定义
- 合成后的 target pose 在哪个 frame 下送入 flow mimic

建议全都使用同一个 anchor frame，避免混乱。

### E. `rot6d` 列展开顺序

这一项很容易被忽略，但会直接影响 flow mimic 输入语义是否正确。

## 验证清单

代码改完后，建议至少检查以下内容：

- policy 输出维度是否已从 `126` 正确变成 `84`
- `actor_future` 仍然是 `630` 维
- `actor_future[:, :126]` 是否写入了合成后的 absolute pose，而不是原始 delta
- pelvis 当前 pose 在 anchor frame 下是否符合预期
- pelvis 合成后 pose 是否正确进入 `actor_history`
- `rot6d` 是否由旋转矩阵前两列显式拼接而成
- `rpy` 累加后是否做了 wrap
- 是否出现 NaN 或奇异姿态
- tracked body 顺序是否严格匹配文档
- 旧 checkpoint 是否被错误地 resume

## 建议的实现顺序

1. 先把 tracked body ids 和当前 body pose 提取打通
2. 再实现 `matrix <-> rpy` 和 `matrix -> rot6d`
3. 再实现 `delta -> absolute pose` 合成
4. 再接入 `actor_future`
5. 最后修改 `_build_actor_history_frame()` 使用合成后的 pelvis pose
6. 完成后再统一改注释、日志名和文档

## 最简版改造摘要

这次改造本质上是在高层 policy 与 frozen flow mimic 之间插入一层：

```text
delta pose action
    -> 结合当前 tracked-body 实际 pose
    -> 合成为 absolute body pose
    -> rpy 转旋转矩阵
    -> 取前两列生成 rot6d
    -> 写入 actor_future
    -> flow mimic 推理 joint targets
```

如果只看真正需要动的主线，可以压缩成三件事：

1. 把 policy 输出从 absolute pose 改成 delta pose
2. 在 env 中恢复当前实际 pose 并合成 absolute pose
3. 用合成后的 absolute pose 替换当前直接送给 flow mimic 的 `126` 维 future pose
