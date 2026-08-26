# Scientific Correctness and Performance Validation Report

Run: `protocol_repair_l1_tzvp_rhf_20260825`

Revision: `802a40c15bc5be7075028d66c9e0011a98dfc0ab`

Results: 1

## Outcome categories

| Category | Passed | Failed | Not applicable or pending |
| --- | ---: | ---: | ---: |
| scientific | 1 | 0 | 0 |
| integrity | 1 | 0 | 0 |
| performance | 0 | 0 | 1 |
| resource | 1 | 0 | 0 |

## Scientific accuracy

| Workload | Family | Direct/Z-vector max abs | FD component max abs | FD direction max abs | Descriptor FD max abs | Passed |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| L1-def2-TZVP | rhf | 2.553e-14 | 1.526e-07 | 7.882e-08 | 7.753e-06 | True |

## Finite-difference accuracy by predeclared step

| Workload | Family | Step (Bohr) | Force max abs | Worst coordinate | Descriptor max abs | Worst descriptor | Direction max abs | Worst direction | Passed |
| --- | --- | ---: | ---: | --- | ---: | --- | ---: | ---: | --- |
| L1-def2-TZVP | rhf | 0.0003 | 2.422e-08 | atom 0 y | 1.09e-06 | atom 0 x; q[0,2] | 9.776e-09 | 2 | True |
| L1-def2-TZVP | rhf | 0.0008 | 1.526e-07 | atom 6 x | 7.753e-06 | atom 0 x; q[0,2] | 7.882e-08 | 2 | True |

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
uv run python validation/scientific_performance/scripts/run_campaign.py --run-root /home/mwding/WorkSpace/Projects/deepks-kit/diagnostics/scientific_performance/runs/protocol_repair_l1_tzvp_rhf_20260825
uv run python validation/scientific_performance/scripts/aggregate.py /home/mwding/WorkSpace/Projects/deepks-kit/diagnostics/scientific_performance/runs/protocol_repair_l1_tzvp_rhf_20260825
```
