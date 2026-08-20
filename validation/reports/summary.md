# DeePHF Release Validation Report

Overall status: PASS.

## Scientific matrices

- RHF formaldehyde.xyz: PASS; direct/Z-vector max error 7.810e-15 Eh/Bohr; finite-difference gradient max error 4.143e-07 Eh/Bohr; relaxed-descriptor max error 1.991e-07 Bohr^-1; response signal 3.703e-01 Bohr^-1.
- RKS formaldehyde.xyz: PASS; direct/Z-vector max error 4.691e-15 Eh/Bohr; finite-difference gradient max error 3.493e-07 Eh/Bohr; relaxed-descriptor max error 1.050e-06 Bohr^-1; response signal 3.729e-01 Bohr^-1.
- UHF hydroxymethyl.xyz: PASS; direct/Z-vector max error 2.631e-14 Eh/Bohr; finite-difference gradient max error 2.655e-07 Eh/Bohr; relaxed-descriptor max error 6.795e-07 Bohr^-1; response signal 3.430e-01 Bohr^-1.
- UKS hydroxymethyl.xyz: PASS; direct/Z-vector max error 2.686e-14 Eh/Bohr; finite-difference gradient max error 2.946e-07 Eh/Bohr; relaxed-descriptor max error 3.175e-06 Bohr^-1; response signal 3.243e-01 Bohr^-1.

## Water-dimer training

- teacher water-dimer workflow: PASS; held-out RHF baseline energy/force RMSE 3.779e-01 Eh and 1.889e-02 Eh/Bohr; energy-only 2.196e-04 and 8.533e-03; energy-plus-force 1.377e-04 and 2.814e-03.
- mp2 water-dimer workflow: PASS; held-out RHF baseline energy/force RMSE 2.552e-01 Eh and 1.492e-02 Eh/Bohr; energy-only 1.903e-04 and 2.216e-02; energy-plus-force 1.887e-04 and 2.690e-03.

## Performance samples

- complete_direct_gradient: median 1.914776 s, minimum 1.913646 s, maximum 1.917411 s, MAD 0.001130 s.
- complete_zvector_gradient: median 1.129219 s, minimum 1.126977 s, maximum 1.129491 s, MAD 0.000272 s.
- descriptor_and_correction_energy: median 0.042302 s, minimum 0.042190 s, maximum 0.042468 s, MAD 0.000083 s.
- direct_response_construction_and_solve: median 0.957010 s, minimum 0.956428 s, maximum 0.958580 s, MAD 0.000582 s.
- force_data_generation_per_frame: median 0.914469 s, minimum 0.914096 s, maximum 0.916277 s, MAD 0.000120 s.
- native_rhf_reference: median 0.169223 s, minimum 0.169036 s, maximum 0.230984 s, MAD 0.000036 s.
- scanner_first_frame: median 1.262638 s, minimum 1.261404 s, maximum 1.343172 s, MAD 0.001234 s.
- scanner_subsequent_frame: median 1.244284 s, minimum 1.243881 s, maximum 1.245625 s, MAD 0.000403 s.
- zvector_operator_construction_and_solve: median 0.992038 s, minimum 0.991546 s, maximum 0.993111 s, MAD 0.000451 s.

## Verification

- PASS: `uv sync --locked --python 3.11`; log `validation/reports/verification_locked_sync.log`.
- PASS: `uv run pytest tests/baseline`; log `validation/reports/verification_baseline_tests.log`.
- PASS: `uv run pytest`; log `validation/reports/verification_complete_tests.log`.
- PASS: `uv build --out-dir validation/outputs/build`; log `validation/reports/verification_build.log`.

## Archive

The validation archive contains 218 hashed files listed in `validation/reports/artifact_manifest.json`; machine-readable stage status is in `validation/reports/master.json`.
