from pathlib import Path
import tomllib

import pytest
import torch

from deepks.gpu import (
    GPUConfigurationError,
    GPU_DIRECT_SCF_TOL,
    _preload_bundled_cuda_libraries,
    require_cuda_device,
)
from deepks.iterate.template import (
    DEFAULT_SCF_RES,
    DEFAULT_SCF_SUB_RES,
    DEFAULT_TRN_RES,
)
from deepks.task.job.shell import Shell
from deepks.task.job.slurm import Slurm


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_locked_dependencies_select_gpu4pyscf_and_cuda_pytorch():
    configuration = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    dependencies = configuration["project"]["dependencies"]
    assert any(value.startswith("gpu4pyscf-cuda12x") for value in dependencies)
    assert "cupy-cuda12x==13.4.1" in dependencies
    assert "cutensor-cu12==2.2.0" in dependencies
    assert configuration["tool"]["uv"]["sources"]["torch"] == {
        "index": "pytorch-cu128"
    }
    indexes = configuration["tool"]["uv"]["index"]
    assert indexes == [
        {
            "name": "pytorch-cu128",
            "url": "https://download.pytorch.org/whl/cu128",
            "explicit": True,
        }
    ]


def test_runtime_rejects_cpu_device():
    with pytest.raises(GPUConfigurationError, match="require a CUDA device"):
        require_cuda_device("cpu")


def test_runtime_rejects_cuda_outside_slurm(monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(GPUConfigurationError, match="inside a Slurm allocation"):
        require_cuda_device("cuda")


def test_runtime_rejects_missing_cuda_inside_slurm(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(GPUConfigurationError, match="CUDA is unavailable"):
        require_cuda_device("cuda")


def test_all_calculation_templates_request_one_gpu():
    assert DEFAULT_SCF_RES["numb_gpu"] == 1
    assert DEFAULT_SCF_SUB_RES["numb_gpu"] == 1
    assert DEFAULT_TRN_RES["numb_gpu"] == 1


def test_gpu_scf_screening_tolerance_matches_gradient_accuracy_contract():
    assert GPU_DIRECT_SCF_TOL == 1.0e-14


def test_locked_cuda_runtime_preloads_blackwell_capable_nvrtc():
    _preload_bundled_cuda_libraries()
    from cupy_backends.cuda.libs import nvrtc

    assert nvrtc.getVersion() >= (12, 8)


def test_slurm_defaults_and_scripts_require_gpu_resources():
    slurm = object.__new__(Slurm)
    resources = slurm.default_resources(None)
    assert resources["numb_gpu"] == 1
    assert "#SBATCH --gres=gpu:1\n" in slurm.sub_script_head(resources)
    assert "--gres=gpu:1" in slurm.sub_step_head({"numb_gpu": 1})
    with pytest.raises(ValueError, match="positive integer"):
        slurm.default_resources({"numb_gpu": 0})


def test_shell_dispatch_rejects_gpu_requests():
    shell = object.__new__(Shell)
    with pytest.raises(ValueError, match="submitted through Slurm"):
        shell.default_resources({"numb_gpu": 1})


def test_self_consistent_methods_use_gpu4pyscf_bases():
    source = (PROJECT_ROOT / "deepks/deepks/method.py").read_text()
    assert "class RDeePKS(ModelCorrectionMixin, PenaltyMixin, gpu_rks.RKS)" in source
    assert "class UDeePKS(ModelCorrectionMixin, PenaltyMixin, gpu_uks.UKS)" in source
    assert "from pyscf import dft" not in source


def test_deephf_reference_scf_uses_gpu4pyscf_constructors():
    source = (PROJECT_ROOT / "deepks/deephf/workflow.py").read_text()
    gpu_constructors = (
        "gpu_scf.RHF",
        "gpu_scf.UHF",
        "gpu_dft.RKS",
        "gpu_dft.UKS",
    )
    for constructor in gpu_constructors:
        assert constructor in source
    for constructor in ("scf.RHF", "scf.UHF", "dft.RKS", "dft.UKS"):
        assert f"reference = {constructor}" not in source
