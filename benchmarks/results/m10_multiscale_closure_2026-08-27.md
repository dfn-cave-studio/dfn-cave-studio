# M10 large explicit-DFN benchmark

Source: `C:\Users\Administrator\Desktop\untitled0812.dfnproj`

Generated: `2026-08-27T18:28:20+0800`
Python: `3.14.5 (tags/v3.14.5:5607950, May 10 2026, 10:43:50) [MSC v.1944 64 bit (AMD64)]` | NumPy: `2.5.1` | logical CPUs: 20
Generator: `m10-multiscale-1` | geometry: `columnar-multiscale-v1` | threshold: `area-weighted-cdf-1`
Threshold mode: `auto` | enabled: `MEDIUM, LARGE` | base seed: 42 | strategy: `base_plus_index`
Size parameters: `{"manual_medium_large_radius": 2.0, "manual_small_medium_radius": 0.5, "medium_large_cumulative_share": 0.7, "small_area_share": 0.1}`

| Actual | Workers | Time (s) | fractures/s | Peak RSS (MiB) | Arrays (MiB) | Estimate error | Save (s) | Reopen (s) | LOD (s) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 9,995 | 1 | 0.524 | 19,074 | 25.4 | 3.0 | 0.17% | 0.182 | 0.029 | 0.405 |
| 99,946 | 1 | 0.546 | 182,945 | 32.7 | 13.2 | 0.08% | 0.314 | 0.050 | 0.031 |
| 721,786 | 1 | 0.788 | 915,671 | 104.9 | 83.8 | 0.04% | 1.160 | 0.194 | 0.228 |
