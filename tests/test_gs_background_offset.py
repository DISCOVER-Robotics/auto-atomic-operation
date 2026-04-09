import numpy as np

from gaussian_renderer.core.gaussiandata import GaussianData
from gaussian_renderer.core.util_gau import load_ply, save_ply

from auto_atom.basis.mjc.gs_mujoco_env import (
    GaussianRenderConfig,
    _materialize_shifted_background_ply,
)


def _write_dummy_ply(path) -> None:
    save_ply(
        GaussianData(
            xyz=np.array([[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]], dtype=np.float32),
            rot=np.array(
                [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=np.float32
            ),
            scale=np.ones((2, 3), dtype=np.float32),
            opacity=np.full((2,), 0.5, dtype=np.float32),
            sh=np.zeros((2, 3), dtype=np.float32),
        ),
        path,
    )


def test_gaussian_render_config_resolves_background_offset_by_stem():
    cfg = GaussianRenderConfig(
        background_ply="assets/gs/backgrounds/discover-lab2.ply",
        background_offsets={
            "discover-lab2": (0.01, -0.02, 0.03),
        },
    )
    assert cfg.resolved_background_offset() == (0.01, -0.02, 0.03)


def test_explicit_background_offset_takes_precedence():
    cfg = GaussianRenderConfig(
        background_ply="assets/gs/backgrounds/discover-lab2.ply",
        background_offset=(0.4, 0.5, 0.6),
        background_offsets={
            "discover-lab2": (0.01, -0.02, 0.03),
        },
    )
    assert cfg.resolved_background_offset() == (0.4, 0.5, 0.6)


def test_materialize_shifted_background_ply_applies_xyz_offset(tmp_path):
    src = tmp_path / "background_0.ply"
    _write_dummy_ply(src)

    shifted_path = _materialize_shifted_background_ply(str(src), (0.25, -0.5, 1.0))
    shifted = load_ply(shifted_path)

    np.testing.assert_allclose(
        shifted.xyz,
        np.array([[0.25, -0.4, 1.2], [1.25, 0.6, 2.2]], dtype=np.float32),
    )
