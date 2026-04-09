# `cat_traverse_g1_flow_mimic` 训练逻辑梳理

## 结论

当前工程中的 `cat_traverse_g1_flow_mimic` 采用的是一套分层控制训练逻辑：

- 上层训练一个 PPO policy。
- 这个 policy 输出的不是最终关节命令，而是 `126` 维高层 body pose / motion anchor。
- 环境内部把这份高层输出与一段固定格式的历史状态拼装后，送入已经训练好的 `flow_mimic.pt`。
- `flow_mimic.pt` 输出 `29` 维关节目标。
- 最终通过 `robot.set_joint_position_target(...)` 作用到机器人关节上。

换句话说，这个 task 训练的是高层策略，不是直接训练 `flow mimic` 网络本身。

## 一句话数据流

```text
CAT actor observation
    -> PPO 高层策略
    -> 126 维 body pose / anchor
    -> 与 actor_history 组合成 flow mimic 输入
    -> 冻结的 flow_mimic.pt
    -> 29 维 joint targets
    -> robot.set_joint_position_target(...)
```

## 相关代码位置

- Task 注册: `legged_lab/envs/__init__.py`
- 训练入口: `legged_lab/scripts/train.py`
- Task 配置: `legged_lab/envs/cat_traverse/cat_traverse_flow_mimic_cfg.py`
- 环境主逻辑: `legged_lab/envs/cat_traverse/cat_traverse_flow_mimic_env.py`
- Flow mimic 网络定义和备用 wrapper: `legged_lab/envs/cat_traverse/flow_mimic_policy.py`

## 训练时到底训练了什么

`CatTraverseFlowMimicAgentCfg` 里配置的 policy 类是：

```python
class_name: str = "rsl_rl.modules.ActorCriticTanh"
```

这说明当前 task 实际训练的是一个标准的 RSL-RL actor-critic MLP，而不是 `FlowMimicActorCritic`。

同时算法配置为：

```python
class_name="PPO"
```

因此这个 task 的训练范式是标准 PPO on-policy 训练。

## RL policy 的输入和输出

### 1. Policy 输入

高层 RL actor 使用的是 CAT 风格 observation，主要包含：

- `ang_vel(3)`
- `projected_gravity(3)`
- `joint_pos_rel(23)`
- `joint_vel(23)`
- `last_flow_mimic_leg_targets(12)`
- `command(4)`
- `foot_height_target(1)`
- `gait_phase(4)`
- `actor_pf_obs(77)`

总维度约为 `150`。

这里有一个容易混淆的点：

- RL policy 自己看到的 observation history 长度并不是 `10`。
- G1 的 robot 配置里 `actor_obs_history_length=1`、`critic_obs_history_length=1`。
- 那个 `10` 帧 history 是专门给下游 `flow mimic` 使用的内部缓存，不是 PPO actor 直接看到的输入。

### 2. Policy 输出

环境在初始化时会把 `num_actions` 改成 `126`：

```python
self.num_actions = RL_ACTION_DIM
RL_ACTION_DIM = 14 * 9 = 126
```

因此 PPO actor 输出的是 `126` 维高层动作，对应 `14` 个 tracked bodies 的 pose 表示，每个 body `9` 维：

- 位置 `3`
- 旋转 `6D`

这些输出不是关节角，而是给 `flow mimic` 的高层未来条件。

## `flow_mimic.pt` 是怎么接入的

环境初始化时会做两件关键事情：

1. 加载 `flow_mimic.pt`
2. 冻结其参数

代码逻辑上是：

- `load_flow_mimic(...)`
- `eval()`
- 对所有参数执行 `requires_grad_(False)`

这说明 `flow_mimic.pt` 在当前 task 中是一个固定下游控制器，不参与训练更新。

## Flow mimic 的输入是什么

下游 `flow mimic` 并不是只吃高层 policy 输出，而是吃两部分输入：

### 1. `actor_future`，形状 `(N, 630)`

构造方式是：

- 先创建全零张量
- 再把高层 policy 当前 step 的 `126` 维输出放到前 `126` 维
- 其余维度保持为 `0`

也就是：

```text
actor_future[:, :126] = RL policy output
actor_future[:, 126:] = 0
```

注释里把它解释为：

- `frame_0[:126] = RL policy output`
- `frames 1-4 = 0`

### 2. `actor_history`，形状 `(N, 10, 160)`

这是环境内部维护的 10 帧滚动缓存，每一帧 `160` 维，主要包含：

- 上一时刻 flow mimic 输出的 `29` 维关节 target
- target 的速度估计
- RL 输出中的 pelvis position / orientation anchor
- base 线速度 / 角速度
- 当前 `29` 个关节相对默认位姿的位置
- 当前 `29` 个关节速度
- 上一时刻 action

这个 history 格式是刻意对齐 `flow_mimic.pt` 原始训练格式的。

## 每个环境 step 里发生了什么

在 `CatTraverseFlowMimicEnv.step()` 中，执行顺序可以概括为：

1. 接收 PPO policy 输出的 `126` 维动作。
2. 对这 `126` 维动作做 delay 和 clip。
3. 构造 `actor_future`。
4. 取出内部维护的 `actor_history`。
5. 在 `torch.no_grad()` 下调用冻结的 `flow_mimic.pt` 前向，得到 `29` 维 `joint_targets`。
6. 将 `joint_targets` clamp 到 soft joint limits。
7. 写入 `_motor_targets`，并调用 `robot.set_joint_position_target(...)` 推动物理仿真。
8. 仿真后更新：
   - gait phase
   - command
   - 奖励
   - reset 状态
   - `last_flow_mimic_targets`
   - `actor_history`

这说明 `flow mimic` 是在环境内部执行的下游控制环节，而不是 PPO policy 本体的一部分。

## 奖励是如何作用到训练上的

PPO 并不会直接对 `flow_mimic.pt` 反向传播。

实际训练机制是：

- 高层 policy 输出 `126` 维高层动作
- 这些动作经过冻结的 `flow mimic` 转成 `29` 维关节目标
- 机器人执行后获得任务奖励
- PPO 根据回报更新高层 policy

因此从优化角度看：

- 被训练的是高层策略
- `flow_mimic.pt` 只是固定的环境内下游模块
- 高层策略通过奖励间接学会输出更合适的 body pose / anchor

## 和 `FlowMimicActorCritic` 的关系

`legged_lab/envs/cat_traverse/flow_mimic_policy.py` 中确实定义了一个 `FlowMimicActorCritic`，它会把 observation 拆成 history/future 后直接走 `TransformerTeacherPolicy`。

但当前 `cat_traverse_g1_flow_mimic` task 并没有使用这个类，因为配置里明确指定的是：

```python
class_name: str = "rsl_rl.modules.ActorCriticTanh"
```

所以当前生效的是：

- 高层 PPO MLP policy
- 环境里 frozen flow mimic

而不是：

- 直接训练 `FlowMimicActorCritic`

## 最终训练产物意味着什么

当前 task 训练完成后，得到的主要是高层 PPO policy 权重。

这意味着部署时如果想复现同样逻辑，仍然需要：

1. 加载训练好的高层 policy
2. 同时保留并加载 `flow_mimic.pt`
3. 按环境中的同样方式构造 `actor_future` 和 `actor_history`
4. 再由 `flow mimic` 输出最终关节目标

也就是说，最终控制链路依赖两个部分：

- 训练得到的高层 policy
- 预训练且冻结的 `flow_mimic.pt`

## 最简总结

`cat_traverse_g1_flow_mimic` 的当前实现可以概括为：

> 使用 PPO 训练一个高层 body-pose policy；该 policy 的输出与环境内部维护的历史状态一起输入到冻结的 `flow_mimic.pt`；`flow_mimic.pt` 输出 29 维关节目标，再将这些关节目标作用到机器人上。

这个理解是准确的。
