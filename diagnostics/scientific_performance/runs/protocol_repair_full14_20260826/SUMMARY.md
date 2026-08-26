# Scientific Correctness and Performance Validation Report

Run: `protocol_repair_full14_20260826`

Revision: `802a40c15bc5be7075028d66c9e0011a98dfc0ab`

Results: 14

## Outcome categories

| Category | Passed | Failed | Not applicable or pending |
| --- | ---: | ---: | ---: |
| scientific | 14 | 0 | 0 |
| integrity | 14 | 0 | 0 |
| performance | 0 | 0 | 14 |
| resource | 14 | 0 | 0 |

## Scientific accuracy

| Workload | Family | Direct/Z-vector max abs | FD component max abs | FD direction max abs | Descriptor FD max abs | Passed |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| L1-def2-SVP | rhf | 2.912e-14 | 3.534e-07 | 8.563e-08 | 5.267e-06 | True |
| L1-def2-SVP | rks | 1.845e-13 | 3.286e-07 | 8.193e-08 | 8.991e-07 | True |
| L1-def2-TZVP | rhf | 2.553e-14 | 1.526e-07 | 7.882e-08 | 7.753e-06 | True |
| L1-def2-TZVP | rks | 1.127e-14 | 2.602e-07 | 8.892e-08 | 1.699e-06 | True |
| L2-def2-SVP | rhf | 1.153e-12 | 2.152e-07 | 3.518e-08 | 1.925e-06 | True |
| L2-def2-SVP | rks | 6.242e-14 | 2.218e-07 | 3.525e-08 | 1.886e-06 | True |
| L3-def2-SVP | uhf | 5.517e-14 | 1.77e-06 | 2.865e-07 | 9.23e-06 | True |
| L3-def2-SVP | uks | 2.273e-12 | 2.815e-06 | 4.011e-07 | 7.131e-06 | True |
| S1-6-31G | rhf | 9.067e-15 | 4.145e-07 | 2.46e-07 | 1.993e-07 | True |
| S1-6-31G | rks | 1.854e-14 | 3.492e-07 | 2.52e-07 | 2.328e-07 | True |
| S2-def2-TZVP | rhf | 8.188e-15 | 3.885e-07 | 3.225e-07 | 1.616e-07 | True |
| S2-def2-TZVP | rks | 1.213e-13 | 3.299e-07 | 3.392e-07 | 2.803e-07 | True |
| S3-def2-SVP | uhf | 3.73e-14 | 2.707e-07 | 1.484e-07 | 3.918e-07 | True |
| S3-def2-SVP | uks | 4.417e-14 | 2.82e-07 | 1.681e-07 | 5.066e-06 | True |

## Finite-difference accuracy by predeclared step

| Workload | Family | Step (Bohr) | Force max abs | Worst coordinate | Descriptor max abs | Worst descriptor | Direction max abs | Worst direction | Passed |
| --- | --- | ---: | ---: | --- | ---: | --- | ---: | ---: | --- |
| L1-def2-SVP | rhf | 0.001 | 3.534e-07 | atom 0 y | 5.267e-06 | atom 0 x; q[0,2] | 8.563e-08 | 2 | True |
| L1-def2-SVP | rhf | 0.0003 | 2.978e-08 | atom 0 y | 4.741e-07 | atom 0 x; q[0,2] | 7.373e-09 | 2 | True |
| L1-def2-SVP | rks | 0.001 | 3.286e-07 | atom 0 x | 8.991e-07 | atom 0 y; q[0,2] | 8.193e-08 | 2 | True |
| L1-def2-SVP | rks | 0.0003 | 2.937e-08 | atom 0 x | 8.067e-08 | atom 0 y; q[0,2] | 6.456e-09 | 2 | True |
| L1-def2-TZVP | rhf | 0.0003 | 2.422e-08 | atom 0 y | 1.09e-06 | atom 0 x; q[0,2] | 9.776e-09 | 2 | True |
| L1-def2-TZVP | rhf | 0.0008 | 1.526e-07 | atom 6 x | 7.753e-06 | atom 0 x; q[0,2] | 7.882e-08 | 2 | True |
| L1-def2-TZVP | rks | 0.001 | 2.602e-07 | atom 0 x | 1.699e-06 | atom 0 x; q[0,2] | 8.892e-08 | 2 | True |
| L1-def2-TZVP | rks | 0.0003 | 2.503e-08 | atom 0 x | 1.529e-07 | atom 0 x; q[0,2] | 6.483e-09 | 2 | True |
| L2-def2-SVP | rhf | 0.001 | 2.152e-07 | atom 0 x | 1.925e-06 | atom 2 x; q[2,3] | 3.518e-08 | 3 | True |
| L2-def2-SVP | rhf | 0.0003 | 1.989e-08 | atom 0 x | 1.731e-07 | atom 2 x; q[2,3] | 2.845e-09 | 3 | True |
| L2-def2-SVP | rks | 0.001 | 2.218e-07 | atom 0 x | 1.886e-06 | atom 0 x; q[0,3] | 3.525e-08 | 3 | True |
| L2-def2-SVP | rks | 0.0003 | 2.048e-08 | atom 0 x | 2.381e-07 | atom 1 z; q[5,1] | 5.471e-09 | 3 | True |
| L3-def2-SVP | uhf | 0.002 | 1.122e-06 | atom 0 x | 7.957e-06 | atom 11 z; q[0,1] | 1.795e-07 | 1 | True |
| L3-def2-SVP | uhf | 0.0025 | 1.77e-06 | atom 0 x | 9.23e-06 | atom 7 y; q[7,2] | 2.865e-07 | 1 | True |
| L3-def2-SVP | uks | 0.002 | 1.268e-06 | atom 0 x | 5.026e-06 | atom 11 z; q[4,1] | 1.812e-07 | 1 | True |
| L3-def2-SVP | uks | 0.003 | 2.815e-06 | atom 0 x | 7.131e-06 | atom 6 x; q[6,2] | 4.011e-07 | 1 | True |
| S1-6-31G | rhf | 0.001 | 4.145e-07 | atom 1 x | 1.993e-07 | atom 1 x; q[1,3] | 2.46e-07 | 3 | True |
| S1-6-31G | rhf | 0.0001 | 4.412e-09 | atom 1 x | 2.447e-08 | atom 0 x; q[0,1] | 2.891e-09 | 3 | True |
| S1-6-31G | rhf | 0.0003 | 3.759e-08 | atom 1 x | 1.792e-08 | atom 1 x; q[1,3] | 2.236e-08 | 3 | True |
| S1-6-31G | rks | 0.001 | 3.492e-07 | atom 1 x | 1.737e-07 | atom 0 x; q[1,1] | 2.52e-07 | 3 | True |
| S1-6-31G | rks | 0.0001 | 3.605e-09 | atom 1 x | 8.096e-08 | atom 2 y; q[0,1] | 2.359e-09 | 3 | True |
| S1-6-31G | rks | 0.0003 | 3.004e-08 | atom 1 x | 2.328e-07 | atom 0 y; q[0,1] | 2.15e-08 | 3 | True |
| S2-def2-TZVP | rhf | 0.001 | 3.885e-07 | atom 1 x | 1.616e-07 | atom 1 x; q[1,1] | 3.225e-07 | 1 | True |
| S2-def2-TZVP | rhf | 0.0003 | 3.557e-08 | atom 1 x | 1.442e-08 | atom 1 x; q[1,3] | 2.885e-08 | 1 | True |
| S2-def2-TZVP | rks | 0.001 | 3.299e-07 | atom 1 x | 1.597e-07 | atom 3 y; q[3,3] | 3.392e-07 | 1 | True |
| S2-def2-TZVP | rks | 0.0003 | 2.943e-08 | atom 1 x | 2.803e-07 | atom 2 z; q[0,1] | 3.097e-08 | 1 | True |
| S3-def2-SVP | uhf | 0.001 | 2.707e-07 | atom 1 x | 3.918e-07 | atom 1 x; q[1,3] | 1.484e-07 | 3 | True |
| S3-def2-SVP | uhf | 0.0003 | 2.402e-08 | atom 1 x | 3.528e-08 | atom 1 x; q[1,3] | 1.336e-08 | 3 | True |
| S3-def2-SVP | uks | 0.001 | 2.82e-07 | atom 1 x | 5.066e-06 | atom 1 z; q[4,1] | 1.681e-07 | 3 | True |
| S3-def2-SVP | uks | 0.0003 | 2.527e-08 | atom 1 z | 4.508e-07 | atom 1 z; q[4,2] | 1.457e-08 | 3 | True |

## Dense replay

| Workload | Family | Dimension | Condition | Solution relative L2 | Solution max abs | Gradient max abs | Passed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |

## Backend ratios

| Profile | Workload | Family | Z compact/direct compact time | Z compact/direct compact RSS | Z compact/Z detailed time | Z compact/Z detailed RSS |
| --- | --- | --- | ---: | ---: | ---: | ---: |

## Unresolved limits

- `campaign-performance-acceptance` `None` `None`: campaign performance rule failed: all_large_compact_time
- `campaign-performance-acceptance` `None` `None`: campaign performance rule failed: all_large_compact_rss
- `campaign-performance-acceptance` `None` `None`: campaign performance rule failed: two_large_closed_shell_speedups
- `campaign-performance-acceptance` `None` `None`: campaign performance rule failed: zvector_scalar_gmres_structure
- `campaign-performance-acceptance` `None` `None`: campaign performance rule failed: largest_common_case_memory
- `campaign-selection-acceptance` `None` `None`: campaign selection rule failed: selected_rows
- `campaign-selection-acceptance` `None` `None`: campaign selection rule failed: coordinate_blocks
- `campaign-selection-acceptance` `None` `None`: campaign selection rule failed: x1_coordinate_memory_reduction

## Reproduction

```bash
uv sync --locked --python 3.11
uv run python validation/scientific_performance/scripts/run_campaign.py --run-root /home/mwding/WorkSpace/Projects/deepks-kit/diagnostics/scientific_performance/runs/protocol_repair_full14_20260826
uv run python validation/scientific_performance/scripts/aggregate.py /home/mwding/WorkSpace/Projects/deepks-kit/diagnostics/scientific_performance/runs/protocol_repair_full14_20260826
```
