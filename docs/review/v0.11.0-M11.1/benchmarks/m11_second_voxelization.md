# M11.1 Exact Second Voxelization Benchmark

Generated: 2026-08-29T03:24:06.150431+00:00

CPU percentage is the system-wide Windows load during the case. Peak memory is the parent-process working set and does not include simultaneous child working sets.
Save/reopen timings use a complete `.dfnproj` containing the synthetic M9 field, M10 geometry, and M11 result.

| Fractures | Workers | Seconds | Fractures/s | CPU % machine | Candidate pairs | Positive pairs | Result MiB | Parent peak MiB | Save s | Reopen s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 4 | 3.403 | 2,938 | 15.9 | 14,427 | 14,426 | 5.27 | 153.60 | 0.041 | 0.017 |
| 100,000 | 4 | 18.542 | 5,393 | 20.2 | 144,098 | 144,089 | 7.25 | 173.20 | 0.173 | 0.035 |
| 721,786 | 4 | 63.895 | 11,296 | 32.3 | 1,037,347 | 1,037,234 | 20.88 | 353.68 | 0.796 | 0.152 |
