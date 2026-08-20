import numpy as np
import pytest
from pyscf import dft, gto, scf

from deepks.deephf import DeePHF, RKSDeePHF, UHFDeePHF, UKSDeePHF, build_reference, evaluate_molecule, make_deephf
from deepks.deephf.workflow import main


def make_h2():
    return gto.M(atom="H 0 0 0; H 0 0 1.4", basis="sto-3g", unit="Bohr", spin=0, verbose=0)


def test_public_factory_and_direct_zvector_workflow_agree():
    reference = build_reference(make_h2(), "rhf")
    method = make_deephf(reference, None, projector_basis=[[0, [0.8, 1.0]]])
    assert type(method) is DeePHF
    direct = method.gradient(backend="direct")
    zvector = method.gradient(backend="zvector")
    np.testing.assert_allclose(direct, zvector, rtol=0.0, atol=1.0e-10)
    result = evaluate_molecule(make_h2(), None, family="rhf", backend="zvector", projector_basis=[[0, [0.8, 1.0]]])
    assert set(result) == {"converged", "e_base", "e_corr", "e_tot", "descriptor", "gradient", "force"}
    np.testing.assert_allclose(result["gradient"], -result["force"], rtol=0.0, atol=0.0)
    assert result["e_corr"] == 0.0


def test_public_workflow_persists_canonical_outputs(tmp_path):
    xyz = tmp_path / "h2.xyz"
    xyz.write_text("2\nH2\nH 0 0 0\nH 0 0 0.7408480953\n", encoding="utf-8")
    outputs = main(
        [str(xyz)],
        reference="rhf",
        model_file=None,
        basis="sto-3g",
        projector_basis=[[0, [0.8, 1.0]]],
        backend="direct",
        dump_dir=str(tmp_path / "out"),
    )
    assert len(outputs) == 1
    output_directory, collected = outputs[0]
    for name, value in collected.items():
        np.testing.assert_array_equal(np.load(f"{output_directory}/{name}.npy"), value)


@pytest.mark.parametrize(
    ("family", "molecule", "reference_type", "method_type"),
    [
        ("rhf", make_h2, scf.hf.RHF, DeePHF),
        ("rks", make_h2, dft.rks.RKS, RKSDeePHF),
        ("uhf", lambda: gto.M(atom="Li 0 0 0", basis="sto-3g", spin=1, verbose=0), scf.uhf.UHF, UHFDeePHF),
        ("uks", lambda: gto.M(atom="Li 0 0 0", basis="sto-3g", spin=1, verbose=0), dft.uks.UKS, UKSDeePHF),
    ],
)
def test_public_factory_dispatches_every_accepted_reference_family(family, molecule, reference_type, method_type):
    reference = build_reference(molecule(), family)
    method = make_deephf(reference, None, projector_basis=[[0, [0.8, 1.0]]])
    assert type(reference) is reference_type
    assert type(method) is method_type
    assert np.isfinite(method.kernel())
