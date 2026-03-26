# planner_sonic.onnx 输入输出说明

本文整理的是这份模型的实际接口：

- `gear_sonic_deploy/planner/target_vel/V2/planner_sonic.onnx`

内容基于三部分信息交叉确认：

- 直接解析 ONNX graph 得到的真实输入输出名、dtype 和 shape
- `docs/source/references/planner_onnx.md`
- `gear_sonic_deploy/src/g1/g1_deploy_onnx_ref/include/localmotion_kplanner_tensorrt.hpp`

这份 `planner_sonic.onnx` 属于 **V2 planner**，实际有 **11 个输入**、**2 个输出**。

## 坐标系约定

- 使用 MuJoCo 的 **Z-up** 世界坐标系
- `x`：前方
- `y`：左侧
- `z`：上方

方向向量、位置、heading 都按这个坐标系解释。

## 总览

### 输入

| Tensor | Dtype | Shape | 说明 |
|---|---|---|---|
| `context_mujoco_qpos` | `float32` | `[1, 4, 36]` | 4 帧历史上下文 |
| `target_vel` | `float32` | `[1]` | 目标速度 |
| `mode` | `int64` | `[1]` | 动作模式索引 |
| `movement_direction` | `float32` | `[1, 3]` | 运动方向 |
| `facing_direction` | `float32` | `[1, 3]` | 朝向方向 |
| `random_seed` | `int64` | `[1]` | 随机种子 |
| `has_specific_target` | `int64` | `[1, 1]` | 是否启用 waypoint 目标 |
| `specific_target_positions` | `float32` | `[1, 4, 3]` | 4 个 waypoint 位置 |
| `specific_target_headings` | `float32` | `[1, 4]` | 4 个 waypoint heading |
| `allowed_pred_num_tokens` | `int64` | `[1, 11]` | 允许的预测 token 数掩码 |
| `height` | `float32` | `[1]` | 根部目标高度 |

### 输出

| Tensor | Dtype | Shape | 说明 |
|---|---|---|---|
| `mujoco_qpos` | `float32` | `[1, 64, 36]` | padding 后的预测轨迹 |
| `num_pred_frames` | `int32` | `[1]` | 有效预测帧数 |

## 输入张量逐项说明

### 1. `context_mujoco_qpos`

- Shape：`[1, 4, 36]`
- 各维含义：
  - 第 0 维 `1`：batch size，固定为 1
  - 第 1 维 `4`：上下文帧数，固定使用最近 4 帧
  - 第 2 维 `36`：每帧 MuJoCo `qpos`

单帧 `qpos[36]` 的展开方式：

| 索引 | 含义 | 说明 |
|---|---|---|
| `0:3` | root position | 根位置 `(x, y, z)`，单位米 |
| `3:7` | root quaternion | 根姿态四元数 `(w, x, y, z)` |
| `7:36` | joint positions | 29 个关节角，单位弧度 |

也就是说：

- `context_mujoco_qpos[0, t, 0:3]`：第 `t` 帧根位置
- `context_mujoco_qpos[0, t, 3:7]`：第 `t` 帧根四元数
- `context_mujoco_qpos[0, t, 7:36]`：第 `t` 帧 29 个关节角

### 2. `target_vel`

- Shape：`[1]`
- 各维含义：
  - 第 0 维 `1`：batch size / 单标量

语义：

- `<= 0`：使用当前 `mode` 的默认速度
- `> 0`：使用该值作为目标速度，单位 m/s

### 3. `mode`

- Shape：`[1]`
- 各维含义：
  - 第 0 维 `1`：batch size / 单标量

语义：

- 表示动作模式索引
- 对这份 V2 planner，部署代码按 `0..26` 使用

### 4. `movement_direction`

- Shape：`[1, 3]`
- 各维含义：
  - 第 0 维 `1`：batch size
  - 第 1 维 `3`：方向向量 `(x, y, z)`

语义：

- 表示机器人希望移动的方向
- 通常主要使用水平分量 `(x, y)`，`z` 一般为 `0`

### 5. `facing_direction`

- Shape：`[1, 3]`
- 各维含义：
  - 第 0 维 `1`：batch size
  - 第 1 维 `3`：方向向量 `(x, y, z)`

语义：

- 表示机器人希望朝向的方向
- heading 通常由 `(x, y)` 计算得到

### 6. `random_seed`

- Shape：`[1]`
- 各维含义：
  - 第 0 维 `1`：batch size / 单标量

语义：

- 控制模型内部随机性

### 7. `has_specific_target`

- Shape：`[1, 1]`
- 各维含义：
  - 第 0 维 `1`：batch size
  - 第 1 维 `1`：布尔标量位置，取值 `0` 或 `1`

语义：

- `0`：忽略 `specific_target_positions` 和 `specific_target_headings`
- `1`：启用 waypoint 约束

### 8. `specific_target_positions`

- Shape：`[1, 4, 3]`
- 各维含义：
  - 第 0 维 `1`：batch size
  - 第 1 维 `4`：4 个 waypoint 帧
  - 第 2 维 `3`：每个 waypoint 的 `(x, y, z)` 位置

语义：

- 仅当 `has_specific_target = 1` 时生效

### 9. `specific_target_headings`

- Shape：`[1, 4]`
- 各维含义：
  - 第 0 维 `1`：batch size
  - 第 1 维 `4`：4 个 waypoint 对应的 heading

语义：

- 仅当 `has_specific_target = 1` 时生效
- 单位为弧度

### 10. `allowed_pred_num_tokens`

- Shape：`[1, 11]`
- 各维含义：
  - 第 0 维 `1`：batch size
  - 第 1 维 `11`：允许的 token 数掩码

这 11 个位置对应 token 数 `6..16`：

| 索引 | token 数 | 帧数 |
|---|---:|---:|
| `0` | `6` | `24` |
| `1` | `7` | `28` |
| `2` | `8` | `32` |
| `3` | `9` | `36` |
| `4` | `10` | `40` |
| `5` | `11` | `44` |
| `6` | `12` | `48` |
| `7` | `13` | `52` |
| `8` | `14` | `56` |
| `9` | `15` | `60` |
| `10` | `16` | `64` |

其中：

- `1`：允许该长度
- `0`：禁止该长度
- 每个 token 对应 4 帧

### 11. `height`

- Shape：`[1]`
- 各维含义：
  - 第 0 维 `1`：batch size / 单标量

语义：

- `< 0`：禁用高度控制
- `>= 0`：将其作为目标根部高度，单位米

## 输出张量逐项说明

### 1. `mujoco_qpos`

- Shape：`[1, 64, 36]`
- 各维含义：
  - 第 0 维 `1`：batch size
  - 第 1 维 `64`：最大输出帧数，固定 padding 到 64
  - 第 2 维 `36`：单帧 `qpos`

单帧 `qpos[36]` 的含义与输入 `context_mujoco_qpos` 完全一致：

| 索引 | 含义 |
|---|---|
| `0:3` | 根位置 `(x, y, z)` |
| `3:7` | 根四元数 `(w, x, y, z)` |
| `7:36` | 29 个关节角 |

注意：

- 输出虽然固定为 64 帧，但并不是 64 帧都有效
- 只有前 `num_pred_frames[0]` 帧有效

常见读取方式：

```python
valid_qpos = mujoco_qpos[0, :num_pred_frames[0], :]
```

### 2. `num_pred_frames`

- Shape：`[1]`
- 各维含义：
  - 第 0 维 `1`：batch size / 单标量

语义：

- 表示 `mujoco_qpos` 中实际有效的预测帧数
- 其值等于 `num_pred_tokens * 4`

## 实现注意点

### 1. 这份模型的真实输出类型与参考文档不完全一致

仓库里的英文参考文档写的是：

- `num_pred_frames`：`scalar int64`

但这份实际模型导出的结果是：

- `num_pred_frames`：`[1] int32`

部署代码里也按 `int32[1]` 申请输出 buffer。

### 2. 输出最大帧数固定为 64

这与 `allowed_pred_num_tokens` 的最大 token 数 `16` 一致：

- `16 tokens * 4 frames/token = 64 frames`

### 3. 运行时一般不会直接手改高级输入

通常外部只需要关心：

- `context_mujoco_qpos`
- `target_vel`
- `mode`
- `movement_direction`
- `facing_direction`
- `height`

而下面这些通常由 C++ 部署层管理：

- `random_seed`
- `has_specific_target`
- `specific_target_positions`
- `specific_target_headings`
- `allowed_pred_num_tokens`

## 一页总结

如果只看最关键的信息，可以记成下面这组：

- 输入主上下文：`context_mujoco_qpos = [1, 4, 36]`
- 单帧状态：`qpos = [36] = [root_pos(3), root_quat(4), joints(29)]`
- 主控制量：
  - `target_vel = [1]`
  - `mode = [1]`
  - `movement_direction = [1, 3]`
  - `facing_direction = [1, 3]`
  - `height = [1]`
- 输出轨迹：`mujoco_qpos = [1, 64, 36]`
- 有效长度：`num_pred_frames = [1]`

