"""Start an rpyc server that exposes PolicyEvaluator remotely.

Clients connect and call ``from_config`` / ``from_yaml`` to initialize,
then drive evaluation via ``reset`` / ``update`` / ``get_observation``.

Usage::

    python examples/policy_eval_server.py
    python examples/policy_eval_server.py --host 0.0.0.0 --port 9999
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


_NVIDIA_EGL_VENDOR_JSON = Path("/usr/share/glvnd/egl_vendor.d/10_nvidia.json")
_NVIDIA_EGL_LIB = Path("/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0")


def _configure_headless_egl(
    gpu: int | None,
    *,
    force_vendor_json: bool,
    preload_egl: bool,
) -> dict[str, str]:
    # Verified locally on 2026-04-04:
    # - MUJOCO_GL=egl is the only strict requirement for GS env init on this host.
    # - PYOPENGL_PLATFORM=egl and EGL_PLATFORM=device are kept as safe headless defaults.
    # - CUDA/EGL/MuJoCo device vars are useful to pin all GPU consumers to one card.
    updates = {
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "EGL_PLATFORM": "device",
    }
    if gpu is not None:
        gpu_str = str(gpu)
        updates["CUDA_VISIBLE_DEVICES"] = gpu_str
        updates["EGL_VISIBLE_DEVICES"] = gpu_str
        updates["MUJOCO_EGL_DEVICE_ID"] = gpu_str
    if force_vendor_json and _NVIDIA_EGL_VENDOR_JSON.exists():
        updates["__EGL_VENDOR_LIBRARY_FILENAMES"] = str(_NVIDIA_EGL_VENDOR_JSON)
    if preload_egl and _NVIDIA_EGL_LIB.exists():
        updates["LD_PRELOAD"] = str(_NVIDIA_EGL_LIB)
    for key, value in updates.items():
        os.environ[key] = value
    return updates


def _prepend_python_bin_to_path() -> str:
    python_bin = str(Path(sys.executable).parent)
    path_parts = os.environ.get("PATH", "").split(os.pathsep)
    if python_bin not in path_parts:
        os.environ["PATH"] = os.pathsep.join([python_bin, *path_parts])
    return python_bin


def main() -> None:
    parser = argparse.ArgumentParser(description="PolicyEvaluator rpyc server")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=18861)
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="Optional physical GPU index for CUDA + EGL + MuJoCo.",
    )
    parser.add_argument(
        "--headless-egl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply verified headless EGL environment variables before importing auto_atom.",
    )
    parser.add_argument(
        "--force-vendor-json",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also set __EGL_VENDOR_LIBRARY_FILENAMES to the NVIDIA GLVND JSON.",
    )
    parser.add_argument(
        "--preload-nvidia-egl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also set LD_PRELOAD to libEGL_nvidia.so.0.",
    )
    args = parser.parse_args()

    python_bin = _prepend_python_bin_to_path()
    print(f"Prepended python bin to PATH: {python_bin}")

    if args.headless_egl:
        applied = _configure_headless_egl(
            args.gpu,
            force_vendor_json=args.force_vendor_json,
            preload_egl=args.preload_nvidia_egl,
        )
        print(f"Applied headless EGL env: {applied}")

    from auto_atom.ipc import serve_policy_evaluator

    serve_policy_evaluator(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
