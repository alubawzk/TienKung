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

from pathlib import Path
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, patterns
from isaaclab.terrains.terrain_importer_cfg import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

from legged_lab.sensors.camera import TiledCameraCfg
from legged_lab.terrains.ray_caster_cfg import RayCasterCfg

if TYPE_CHECKING:
    from legged_lab.envs.base.base_env_config import BaseSceneCfg


REPO_ROOT = Path(__file__).resolve().parents[4]


def _resolve_scene_asset_path(raw_path: str | Path) -> Path:
    raw_path = Path(raw_path)
    if raw_path.is_absolute():
        return raw_path.resolve()

    candidates = [
        Path.cwd() / raw_path,
        REPO_ROOT / raw_path,
        REPO_ROOT / "TienKung" / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (REPO_ROOT / raw_path).resolve()


def _build_mesh_obstacle_asset(config: "BaseSceneCfg") -> AssetBaseCfg | None:
    obstacle_cfg = getattr(config, "mesh_obstacle", None)
    if obstacle_cfg is None or not obstacle_cfg.enable:
        return None
    if not obstacle_cfg.source_path:
        raise ValueError("Mesh obstacle is enabled, but 'source_path' is empty.")

    source_path = _resolve_scene_asset_path(obstacle_cfg.source_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Mesh obstacle source does not exist: {source_path}")

    obstacle_usd_path = source_path
    obstacle_scale = obstacle_cfg.scale
    if source_path.suffix.lower() not in {".usd", ".usda"}:
        usd_dir = (
            Path(obstacle_cfg.usd_dir)
            if obstacle_cfg.usd_dir is not None
            else source_path.parent / "_isaaclab_usd"
        )
        if not usd_dir.is_absolute():
            usd_dir = _resolve_scene_asset_path(usd_dir)
        converter = sim_utils.MeshConverter(
            sim_utils.MeshConverterCfg(
                asset_path=str(source_path),
                usd_dir=str(usd_dir),
                usd_file_name=obstacle_cfg.usd_file_name or source_path.stem,
                force_usd_conversion=False,
                make_instanceable=True,
                collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=obstacle_cfg.collision_enabled),
                mesh_collision_props=sim_utils.TriangleMeshPropertiesCfg(),
                scale=obstacle_cfg.scale,
            )
        )
        obstacle_usd_path = Path(converter.usd_path)
        obstacle_scale = None

    return AssetBaseCfg(
        prim_path=obstacle_cfg.prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(obstacle_usd_path),
            scale=obstacle_scale,
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=obstacle_cfg.collision_enabled),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=obstacle_cfg.pos,
            rot=obstacle_cfg.rot,
        ),
        collision_group=obstacle_cfg.collision_group,
    )


@configclass
class SceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    def __init__(self, config: "BaseSceneCfg", physics_dt, step_dt):
        super().__init__(num_envs=config.num_envs, env_spacing=config.env_spacing)

        self.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type=config.terrain_type,
            terrain_generator=config.terrain_generator,
            max_init_terrain_level=config.max_init_terrain_level,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
                project_uvw=True,
                texture_scale=(0.25, 0.25),
            ),
            debug_vis=False,
        )

        obstacle_asset = _build_mesh_obstacle_asset(config)
        if obstacle_asset is not None:
            self.obstacle = obstacle_asset

        self.robot: ArticulationCfg = config.robot.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.contact_sensor = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, update_period=physics_dt
        )
        filtered_contact_sensors = getattr(config, "filtered_contact_sensors", None)
        if filtered_contact_sensors:
            for sensor_name, sensor_cfg in filtered_contact_sensors.items():
                setattr(self, sensor_name, sensor_cfg)

        self.light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
        )
        self.sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                intensity=750.0,
                texture_file=(
                    f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
                ),
            ),
        )

        if config.height_scanner.enable_height_scan:
            self.height_scanner = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + config.height_scanner.prim_body_name,
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
                attach_yaw_only=True,
                pattern_cfg=patterns.GridPatternCfg(
                    resolution=config.height_scanner.resolution, size=config.height_scanner.size
                ),
                debug_vis=config.height_scanner.debug_vis,
                mesh_prim_paths=["/World/ground"],
                update_period=step_dt,
                drift_range=config.height_scanner.drift_range,
            )

        if config.lidar.enable_lidar:
            self.lidar = RayCasterCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + config.lidar.prim_body_name,
                offset=RayCasterCfg.OffsetCfg(pos=config.lidar.offset, rot=config.lidar.rotation),
                attach_yaw_only=True,
                pattern_cfg=config.lidar.pattern_cfg,
                debug_vis=config.lidar.debug_vis,
                mesh_prim_paths=config.lidar.mesh_prim_paths,
                max_distance=config.lidar.max_distance,
            )

        if config.depth_camera.enable_depth_camera:
            self.depth_camera = TiledCameraCfg(
                prim_path="{ENV_REGEX_NS}/Robot/" + config.depth_camera.prim_body_name,
                offset=config.depth_camera.offset,
                height=config.depth_camera.height,
                width=config.depth_camera.width,
                data_types=config.depth_camera.data_types,
                spawn=config.depth_camera.spawn,
                debug_vis=config.depth_camera.debug_vis,
                visualizer_cfg=config.depth_camera.visualizer_cfg,
            )
