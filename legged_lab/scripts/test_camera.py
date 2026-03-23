"""Test script for attaching an IsaacLab camera to the mini3 robot's base_link.

Camera parameters:
  - update_period: 0.033s (30 Hz)
  - resolution: 640x480
  - data_types: ["rgb", "distance_to_image_plane"]

Run with:
  python legged_lab/scripts/test_camera.py
"""

import argparse

from isaaclab.app import AppLauncher

# ── CLI ────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Test IsaacLab camera on mini3 base_link.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Camera rendering must be enabled before the app launches
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Imports (must come after AppLauncher) ──────────────────────────────────────
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from legged_lab.assets.mini3.mini3 import MINI3_CFG  # noqa: E402
from legged_lab.sensors.camera import TiledCameraCfg  # noqa: E402
from legged_lab.sensors.camera.tiled_camera import TiledCamera  # noqa: E402

# ── Scene configuration ────────────────────────────────────────────────────────

# Focal length derived from 87-deg HFOV, same as D455
import math  # noqa: E402
_HFOV_DEG = 87.0
_H_APERTURE_CM = 2.4
_FOCAL_LENGTH_CM = (_H_APERTURE_CM / 2.0) / math.tan(math.radians(_HFOV_DEG) / 2.0)


@configclass
class Mini3CameraSceneCfg(InteractiveSceneCfg):
    """Scene with mini3 robot and an RGB-D camera attached to base_link."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Distant light
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    # Mini3 robot — must be defined before camera so the body prim exists at spawn time
    robot: ArticulationCfg = MINI3_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Camera attached to base_link.
    # prim_path = {ENV_REGEX_NS}/Robot/<body>/<camera_prim_name>
    # For mini3 the articulation root body is "base_link".
    camera = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base_link/test_camera",
        update_period=0.033,   # 30 Hz
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_FOCAL_LENGTH_CM,
            horizontal_aperture=_H_APERTURE_CM,
            clipping_range=(0.1, 20.0),
        ),
        # Offset: slightly forward and up from base_link, facing forward (ROS convention)
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.15, 0.0, 0.05),
            rot=(0.5, -0.5, 0.5, -0.5),
            convention="ros",
        ),
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def run_simulation(sim: SimulationContext, scene: InteractiveScene):
    """Step the simulation and print camera data info once per second."""
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    camera: TiledCamera = scene["camera"]

    print("\n[test_camera] Starting simulation loop. Press Ctrl+C to exit.\n")

    while simulation_app.is_running():
        # Periodically reset so the robot doesn't fall indefinitely
        if count % 500 == 0:
            scene.reset()
            print(f"[test_camera] Scene reset at step {count}")

        sim.step()
        scene.update(sim_dt)
        sim_time += sim_dt
        count += 1

        # Visualize every frame (camera updates at 30 Hz, physics at 200 Hz)
        output = camera.data.output

        if "rgb" in output:
            rgb = output["rgb"]  # (N, H, W, 3) uint8
            rgb_np = rgb[0].cpu().numpy()
            cv2.imshow("RGB", cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR))

        if "distance_to_image_plane" in output:
            depth = output["distance_to_image_plane"]  # (N, H, W, 1) float32
            depth_np = depth[0, :, :, 0].cpu().numpy()
            # Normalize finite values to [0, 255]; set inf/nan to 0 (black)
            finite_mask = np.isfinite(depth_np)
            depth_vis = np.zeros_like(depth_np, dtype=np.uint8)
            if finite_mask.any():
                d_min = depth_np[finite_mask].min()
                d_max = depth_np[finite_mask].max()
                if d_max > d_min:
                    depth_vis[finite_mask] = (
                        (depth_np[finite_mask] - d_min) / (d_max - d_min) * 255
                    ).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_TURBO)
            cv2.imshow("Depth", depth_color)

        # Print stats once per simulated second
        if count % max(1, int(1.0 / sim_dt)) == 0:
            if "rgb" in output:
                print(
                    f"[t={sim_time:.1f}s] RGB   shape={tuple(rgb.shape)}"
                    f"  min={rgb[0].float().min():.0f}  max={rgb[0].float().max():.0f}"
                )
            if "distance_to_image_plane" in output:
                valid = depth[depth.isfinite()]
                if valid.numel() > 0:
                    print(
                        f"[t={sim_time:.1f}s] Depth min={valid.min().item():.3f}m"
                        f"  max={valid.max().item():.3f}m"
                    )
                else:
                    print(f"[t={sim_time:.1f}s] Depth (no valid pixels)")

        # Process GUI events; press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.002,           # 200 Hz physics
        render_interval=1,
        device=args_cli.device,
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.5, 2.5, 2.0], target=[0.0, 0.0, 0.5])

    scene_cfg = Mini3CameraSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.5)
    scene = InteractiveScene(scene_cfg)

    sim.reset()
    print("[test_camera] Setup complete. Running simulation …")
    run_simulation(sim, scene)


if __name__ == "__main__":
    main()
    cv2.destroyAllWindows()
    simulation_app.close()
