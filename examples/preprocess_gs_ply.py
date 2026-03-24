"""Pre-rotate Gaussian Splatting PLY files for use with identity-orientation bodies.

When a PLY was captured in a scene where the GS reference body has a non-identity
euler rotation (e.g. euler="0.959931089 0.052359878 0" in the Franka table30 scene),
the PLY positions and SH coefficients are expressed in that body's local frame.

To use the PLY with an identity-orientation body in MuJoCo (which avoids SH distortion
at render time), the PLY must be pre-rotated by the same euler rotation so that:
  - Gaussian xyz positions are expressed in the world-aligned (identity) body frame
  - SH coefficients are rotated via e3nn to match the new frame
  - Gaussian orientations (rot) are updated accordingly

See docs/gs_rendering_alignment.md for the full explanation.

Usage examples::

    # Rotate a single PLY by a quaternion (xyzw)
    python examples/preprocess_gs_ply.py \\
        third_party/.../button_blue.ply \\
        -o assets/gs/scenes/press_three_buttons/button_blue.ply \\
        -r 0.46159 0.02322 0.01209 0.88671

    # Rotate all PLYs in a directory using MuJoCo euler angles (intrinsic XYZ, radians)
    python examples/preprocess_gs_ply.py \\
        third_party/.../3dgs/ \\
        -o assets/gs/scenes/my_task/ \\
        --euler 0.959931089 0.052359878 0.0

    # Dry run: print what would be done without writing files
    python examples/preprocess_gs_ply.py \\
        third_party/.../3dgs/ \\
        -o assets/gs/scenes/my_task/ \\
        --euler 0.959931089 0.052359878 0.0 \\
        --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Quaternion helpers
# ---------------------------------------------------------------------------


def euler_intrinsic_xyz_to_quat_xyzw(ax: float, ay: float, az: float) -> np.ndarray:
    """Convert MuJoCo intrinsic XYZ euler angles (radians) to xyzw quaternion.

    MuJoCo ``euler="ax ay az"`` with default sequence xyz uses intrinsic rotations:
    q = qx(ax) ⊗ qy(ay) ⊗ qz(az), where each qi is a rotation about that axis.

    Args:
        ax: Rotation around X axis in radians.
        ay: Rotation around Y axis in radians.
        az: Rotation around Z axis in radians.

    Returns:
        np.ndarray: Quaternion [x, y, z, w].
    """
    cx, sx = np.cos(ax / 2), np.sin(ax / 2)
    cy, sy = np.cos(ay / 2), np.sin(ay / 2)
    cz, sz = np.cos(az / 2), np.sin(az / 2)

    def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        ax_, ay_, az_, aw = a
        bx, by, bz, bw = b
        return np.array(
            [
                aw * bx + ax_ * bw + ay_ * bz - az_ * by,
                aw * by - ax_ * bz + ay_ * bw + az_ * bx,
                aw * bz + ax_ * by - ay_ * bx + az_ * bw,
                aw * bw - ax_ * bx - ay_ * by - az_ * bz,
            ]
        )

    qx = np.array([sx, 0.0, 0.0, cx])
    qy = np.array([0.0, sy, 0.0, cy])
    qz = np.array([0.0, 0.0, sz, cz])
    return qmul(qmul(qx, qy), qz)


# ---------------------------------------------------------------------------
# Core transform
# ---------------------------------------------------------------------------


def preprocess_ply(
    src: Path, dst: Path, quat_xyzw: np.ndarray, dry_run: bool = False
) -> None:
    """Pre-rotate a single PLY file and write the result.

    Args:
        src: Input PLY path.
        dst: Output PLY path.
        quat_xyzw: Rotation quaternion [x, y, z, w] to bake into the PLY.
        dry_run: If True, print what would be done but do not write.
    """
    print(f"  {src.name}  →  {dst}")
    if dry_run:
        return

    # Import here so the script can be imported without GPU available
    from gaussian_renderer.core.util_gau import load_ply, save_ply
    from gaussian_renderer.transform_gs_model import transform_gaussian
    from scipy.spatial.transform import Rotation

    dst.parent.mkdir(parents=True, exist_ok=True)

    gaussian_data = load_ply(str(src))

    R = Rotation.from_quat(quat_xyzw).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R

    transform_gaussian(gaussian_data, T, scale_factor=1.0, silent=True)
    save_ply(gaussian_data, str(dst))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "src",
        help="Source: a single .ply file, or a directory containing .ply files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Destination: output .ply file (when src is a file) or output directory (when src is a dir).",
    )

    rot_group = parser.add_mutually_exclusive_group(required=True)
    rot_group.add_argument(
        "-r",
        "--rotation",
        nargs=4,
        type=float,
        metavar=("X", "Y", "Z", "W"),
        help="Rotation quaternion in xyzw order.",
    )
    rot_group.add_argument(
        "--euler",
        nargs=3,
        type=float,
        metavar=("AX", "AY", "AZ"),
        help=(
            "MuJoCo intrinsic XYZ euler angles in radians "
            "(matches the euler= attribute in scene XML). "
            "Converted to xyzw quaternion internally."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing any files.",
    )
    parser.add_argument(
        "--glob",
        default="*.ply",
        help="Glob pattern used when src is a directory (default: '*.ply').",
    )
    args = parser.parse_args()

    # Resolve quaternion
    if args.rotation is not None:
        quat_xyzw = np.array(args.rotation, dtype=np.float64)
    else:
        ax, ay, az = args.euler
        quat_xyzw = euler_intrinsic_xyz_to_quat_xyzw(ax, ay, az)
        print(
            f"Euler ({ax:.6f}, {ay:.6f}, {az:.6f}) rad  →  xyzw quaternion: {quat_xyzw}"
        )

    norm = np.linalg.norm(quat_xyzw)
    if abs(norm - 1.0) > 1e-4:
        print(f"Warning: quaternion norm = {norm:.6f}, normalizing.", file=sys.stderr)
        quat_xyzw = quat_xyzw / norm

    src = Path(args.src)
    dst = Path(args.output)

    if src.is_file():
        # Single-file mode
        if not src.suffix.lower() == ".ply":
            parser.error(f"src file must be a .ply file, got: {src}")
        out_path = dst if dst.suffix.lower() == ".ply" else dst / src.name
        print(f"{'[DRY RUN] ' if args.dry_run else ''}Pre-rotating 1 file:")
        preprocess_ply(src, out_path, quat_xyzw, dry_run=args.dry_run)

    elif src.is_dir():
        # Directory mode
        ply_files = sorted(src.glob(args.glob))
        if not ply_files:
            print(f"No files matching '{args.glob}' found in {src}", file=sys.stderr)
            sys.exit(1)
        print(
            f"{'[DRY RUN] ' if args.dry_run else ''}Pre-rotating {len(ply_files)} file(s):"
        )
        for ply in ply_files:
            preprocess_ply(ply, dst / ply.name, quat_xyzw, dry_run=args.dry_run)

    else:
        parser.error(f"src does not exist: {src}")

    if not args.dry_run:
        print("Done.")


if __name__ == "__main__":
    main()
