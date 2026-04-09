# rot_utils.py
#
# Rotation utility functions for delta-pose composition in anchor frame.
# All operations stay in rotation-matrix space to avoid gimbal lock.
#
# Conventions:
#   quaternion order  : (w, x, y, z)  — matches IsaacLab / IsaacSim convention
#   rot6d layout      : concat(R[..., :, 0], R[..., :, 1], dim=-1)
#                       i.e. first two **columns** concatenated as column vectors
#   gram_schmidt zero : [1,0,0, 0,1,0] → R_delta = I  (identity, no rotation change)

import torch


def quat_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion(s) to rotation matrix/matrices.

    Args:
        quat: (..., 4) in (w, x, y, z) order.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    w, x, y, z = quat.unbind(dim=-1)

    R = torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),      2*(x*z + w*y),
        2*(x*y + w*z),      1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),      2*(y*z + w*x),       1 - 2*(x*x + y*y),
    ], dim=-1)  # (..., 9)

    shape = quat.shape[:-1] + (3, 3)
    return R.view(shape)


def gram_schmidt(rot6d: torch.Tensor) -> torch.Tensor:
    """Orthogonalize 6D rotation representation to SO(3) via Gram-Schmidt.

    Args:
        rot6d: (..., 6) — first two columns of a rotation matrix concatenated
               as column vectors: [c0_x, c0_y, c0_z, c1_x, c1_y, c1_z].
               Zero input [1,0,0, 0,1,0] maps to identity matrix.

    Returns:
        (..., 3, 3) proper rotation matrices with det ≈ +1.
    """
    a1 = rot6d[..., 0:3]   # (..., 3)  first column candidate
    a2 = rot6d[..., 3:6]   # (..., 3)  second column candidate

    # First column: normalise a1 (clamp norm to avoid division by zero)
    b1 = a1 / a1.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    # Second column: remove projection onto b1, then normalise
    proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2_raw = a2 - proj
    b2 = b2_raw / b2_raw.norm(dim=-1, keepdim=True).clamp(min=1e-6)

    # Third column: right-hand cross product
    b3 = torch.cross(b1, b2, dim=-1)

    # Stack as columns → (..., 3, 3)
    return torch.stack([b1, b2, b3], dim=-1)


def matrix_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    """Extract 6D rotation representation from rotation matrix.

    Takes the first two **columns** and concatenates them as column vectors.
    This is the inverse of gram_schmidt (up to numerical precision).

    Args:
        R: (..., 3, 3) rotation matrices.

    Returns:
        (..., 6) — concat(R[..., :, 0], R[..., :, 1], dim=-1).

    Note:
        Uses explicit column extraction, NOT reshape, to preserve the
        column-vector convention expected by flow_mimic.pt training.
    """
    col0 = R[..., :, 0]   # (..., 3)  first column
    col1 = R[..., :, 1]   # (..., 3)  second column
    return torch.cat([col0, col1], dim=-1)  # (..., 6)
