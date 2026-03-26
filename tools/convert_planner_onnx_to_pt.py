#!/usr/bin/env python3
"""Convert planner_sonic.onnx to planner_sonic.pt (PyTorch format).

Two strategies are attempted in order:
  1. onnx2torch  — produces a true nn.Module with no onnxruntime dependency.
  2. OnnxWrapper — wraps the ONNX session in an nn.Module; requires onnxruntime
     at inference time but allows the model to be loaded via torch.load().

Usage:
    python tools/convert_planner_onnx_to_pt.py
    python tools/convert_planner_onnx_to_pt.py --onnx path/to/planner_sonic.onnx --out path/to/output.pt
    python tools/convert_planner_onnx_to_pt.py --force-wrapper   # skip onnx2torch attempt
"""

import argparse
import os
import sys


# ---------------------------------------------------------------------------
# OnnxWrapper fallback
# ---------------------------------------------------------------------------

class OnnxWrapper:
    """Thin wrapper that calls onnxruntime and presents a dict-in / tuple-out API.

    Saved with torch.save; loaded with torch.load.  Requires onnxruntime at
    inference time.

    Inputs (keyword arguments, all numpy or torch tensors):
        context_mujoco_qpos       float32  [1, 4, 36]
        target_vel                float32  [1]
        mode                      int64    [1]
        movement_direction        float32  [1, 3]
        facing_direction          float32  [1, 3]
        random_seed               int64    [1]
        has_specific_target       int64    [1, 1]
        specific_target_positions float32  [1, 4, 3]
        specific_target_headings  float32  [1, 4]
        allowed_pred_num_tokens   int64    [1, 11]
        height                    float32  [1]

    Returns:
        (mujoco_qpos: torch.Tensor [1, 64, 36],
         num_pred_frames: torch.Tensor [1] int32)
    """

    def __init__(self, onnx_path: str):
        if not os.path.isabs(onnx_path):
            onnx_path = os.path.abspath(onnx_path)
        self.onnx_path = onnx_path
        self._session = None

    # ------------------------------------------------------------------
    # Lazy session creation (so the object can be unpickled / torch.load'd)
    # ------------------------------------------------------------------

    def _get_session(self):
        if self._session is None:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            self._session = ort.InferenceSession(
                self.onnx_path,
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
        return self._session

    # ------------------------------------------------------------------
    # Callable interface
    # ------------------------------------------------------------------

    def __call__(self, **kwargs):
        import numpy as np
        import torch

        def _to_np(v):
            if isinstance(v, torch.Tensor):
                return v.cpu().numpy()
            return np.asarray(v)

        feeds = {k: _to_np(v) for k, v in kwargs.items()}
        mujoco_qpos, num_pred_frames = self._get_session().run(None, feeds)
        return torch.from_numpy(mujoco_qpos.copy()), torch.from_numpy(num_pred_frames.copy())

    def __repr__(self):
        return f"OnnxWrapper(onnx_path={self.onnx_path!r})"


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def try_onnx2torch(onnx_path: str, pt_path: str) -> bool:
    """Attempt conversion via onnx2torch. Returns True on success."""
    try:
        import onnx
        import onnx2torch
        import torch
    except ImportError as exc:
        print(f"[onnx2torch] Not available: {exc}")
        return False

    try:
        print("[onnx2torch] Loading ONNX model …")
        onnx_model = onnx.load(onnx_path)
        print("[onnx2torch] Converting to PyTorch nn.Module …")
        torch_model = onnx2torch.convert(onnx_model)
        torch_model.eval()
        print(f"[onnx2torch] Saving to {pt_path} …")
        torch.save(torch_model, pt_path)
        print(f"[onnx2torch] Success → {pt_path}")
        return True
    except Exception as exc:
        print(f"[onnx2torch] Conversion failed: {exc}")
        return False


def save_ort_wrapper(onnx_path: str, pt_path: str):
    """Save an OnnxWrapper to pt_path using torch.save."""
    import torch

    model = OnnxWrapper(onnx_path)
    torch.save(model, pt_path)
    print(f"[OnnxWrapper] Saved → {pt_path}")
    print("[OnnxWrapper] Note: onnxruntime must be installed when loading this model.")


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------

def smoke_test(pt_path: str):
    """Run a quick forward pass to confirm the saved model works."""
    import numpy as np
    import torch

    print("[smoke_test] Loading saved model …")
    model = torch.load(pt_path, weights_only=False)

    dummy = {
        "context_mujoco_qpos": np.zeros((1, 4, 36), dtype=np.float32),
        "target_vel":          np.array([0.5], dtype=np.float32),
        "mode":                np.array([0], dtype=np.int64),
        "movement_direction":  np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
        "facing_direction":    np.array([[0.0, 1.0, 0.0]], dtype=np.float32),
        "random_seed":         np.array([42], dtype=np.int64),
        "has_specific_target": np.zeros((1, 1), dtype=np.int64),
        "specific_target_positions": np.zeros((1, 4, 3), dtype=np.float32),
        "specific_target_headings":  np.zeros((1, 4), dtype=np.float32),
        "allowed_pred_num_tokens":   np.ones((1, 11), dtype=np.int64),
        "height":              np.array([-1.0], dtype=np.float32),
    }

    if callable(model):
        # OnnxWrapper or onnx2torch module
        if isinstance(model, OnnxWrapper):
            qpos, nframes = model(**dummy)
        else:
            # onnx2torch: positional args in ONNX graph order
            import torch
            args = [torch.from_numpy(v) for v in dummy.values()]
            result = model(*args)
            qpos = result[0] if isinstance(result, (list, tuple)) else result
    else:
        raise TypeError(f"Unexpected model type: {type(model)}")

    print(f"[smoke_test] mujoco_qpos shape: {tuple(qpos.shape)}  (expected (1, 64, 36))")
    assert tuple(qpos.shape) == (1, 64, 36), f"Unexpected shape {qpos.shape}"
    print("[smoke_test] PASSED")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert planner_sonic.onnx to .pt")
    parser.add_argument(
        "--onnx",
        default="Exported_policy/planner/target_vel/V2/planner_sonic.onnx",
        help="Path to planner_sonic.onnx (relative to TienKung root or absolute)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output .pt path (default: same directory as --onnx, with .pt suffix)",
    )
    parser.add_argument(
        "--force-wrapper",
        action="store_true",
        help="Skip onnx2torch attempt and go straight to OnnxWrapper",
    )
    parser.add_argument(
        "--no-smoke-test",
        action="store_true",
        help="Skip the smoke-test forward pass",
    )
    args = parser.parse_args()

    # Resolve paths relative to this script's location (TienKung/tools/)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)  # TienKung/

    onnx_path = args.onnx
    if not os.path.isabs(onnx_path):
        onnx_path = os.path.join(project_dir, onnx_path)

    if not os.path.isfile(onnx_path):
        print(f"ERROR: ONNX file not found: {onnx_path}", file=sys.stderr)
        sys.exit(1)

    pt_path = args.out
    if pt_path is None:
        pt_path = os.path.splitext(onnx_path)[0] + ".pt"
    elif not os.path.isabs(pt_path):
        pt_path = os.path.join(project_dir, pt_path)

    print(f"Source : {onnx_path}")
    print(f"Target : {pt_path}")
    print()

    success = False
    if not args.force_wrapper:
        success = try_onnx2torch(onnx_path, pt_path)

    if not success:
        print("[OnnxWrapper] Falling back to OnnxWrapper strategy …")
        save_ort_wrapper(onnx_path, pt_path)
        success = True

    if success and not args.no_smoke_test:
        print()
        smoke_test(pt_path)

    print()
    print(f"Output saved to: {pt_path}")


if __name__ == "__main__":
    main()
