# Copyright (c) 2021-2024, The RSL-RL Project Developers.
# All rights reserved.
# Original code is licensed under the BSD-3-Clause license.
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The Legged Lab Project Developers.
# All rights reserved.
#
# Copyright (c) 2025-2026, The TienKung-Lab Project Developers.
# All rights reserved.
# Modifications are licensed under the BSD-3-Clause license.
#
# This file contains code derived from the RSL-RL, Isaac Lab, and Legged Lab Projects,
# with additional modifications by the TienKung-Lab Project,
# and is distributed under the BSD-3-Clause license.

from legged_lab.envs.base.base_env import BaseEnv
from legged_lab.envs.base.base_env_config import BaseAgentCfg, BaseEnvCfg
from legged_lab.envs.cat_traverse import (
    CatTraverseAgentCfg,
    CatTraverseEnv,
    CatTraverseEnvCfg,
    make_unitree_g1_cat_env_cfg,
)
from legged_lab.envs.tienkung.run_cfg import TienKungRunAgentCfg, TienKungRunFlatEnvCfg
from legged_lab.envs.tienkung.run_with_sensor_cfg import (
    TienKungRunWithSensorAgentCfg,
    TienKungRunWithSensorFlatEnvCfg,
)
from legged_lab.envs.tienkung.tienkung_env import TienKungEnv
from legged_lab.envs.tienkung.walk_cfg import (
    TienKungWalkAgentCfg,
    TienKungWalkFlatEnvCfg,
)
from legged_lab.envs.tienkung.walk_with_sensor_cfg import (
    TienKungWalkWithSensorAgentCfg,
    TienKungWalkWithSensorFlatEnvCfg,
)
from legged_lab.utils.task_registry import task_registry

## Add mini3 register
from legged_lab.envs.mini3.mini3_env import Mini3_Env
from legged_lab.envs.mini3.walk_cfg import (
    Mini3_WalkAgentCfg,
    Mini3_WalkFlatEnvCfg,
)

## Add mini3 register
task_registry.register("walk", Mini3_Env, Mini3_WalkFlatEnvCfg(), Mini3_WalkAgentCfg())

task_registry.register("cat_traverse", CatTraverseEnv, CatTraverseEnvCfg(), CatTraverseAgentCfg())
task_registry.register("cat_traverse_g1", CatTraverseEnv, make_unitree_g1_cat_env_cfg(), CatTraverseAgentCfg())
task_registry.register("tienkun_walk", TienKungEnv, TienKungWalkFlatEnvCfg(), TienKungWalkAgentCfg())
task_registry.register("tienkun_run", TienKungEnv, TienKungRunFlatEnvCfg(), TienKungRunAgentCfg())
task_registry.register(
    "tienkun_walk_with_sensor", TienKungEnv, TienKungWalkWithSensorFlatEnvCfg(), TienKungWalkWithSensorAgentCfg()
)
task_registry.register(
    "tienkun_run_with_sensor", TienKungEnv, TienKungRunWithSensorFlatEnvCfg(), TienKungRunWithSensorAgentCfg()
)
