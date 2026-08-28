import pytest

from deepks.deephf.workflow import _normalize_gpu_dft_args


def test_default_gpu_dft_controls_preserve_strict_lda_contract():
    assert _normalize_gpu_dft_args(None) == {
        "xc": "LDA_X + LDA_C_VWN",
        "grid_mode": "strict",
        "grid_level": 3,
        "small_rho_cutoff": 0.0,
    }


def test_published_b3lyp_gpu_dft_controls_are_serializable():
    assert _normalize_gpu_dft_args(
        {
            "xc": "B3LYP5",
            "grid_mode": "default",
            "grid_level": 3,
            "small_rho_cutoff": 0,
        }
    ) == {
        "xc": "B3LYP5",
        "grid_mode": "default",
        "grid_level": 3,
        "small_rho_cutoff": 0.0,
    }


@pytest.mark.parametrize(
    "controls",
    (
        {"grid_mode": "unknown"},
        {"grid_level": -1},
        {"small_rho_cutoff": -1.0},
        {"unknown": 1},
    ),
)
def test_invalid_gpu_dft_controls_are_rejected(controls):
    with pytest.raises(ValueError):
        _normalize_gpu_dft_args(controls)
