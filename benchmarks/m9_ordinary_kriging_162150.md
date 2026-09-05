# M9 Ordinary Kriging - 162,150 voxel benchmark

This fixed-seed benchmark uses synthetic data only.

- Elapsed: 9.345 s
- Throughput: 17,351 voxels/s
- Persistent output arrays: 2.320 MiB
- Benchmark tracemalloc peak: 11.206 MiB
- Modelled cells: 162,150
- CPU: Intel64 Family 6 Model 183 Stepping 1, GenuineIntel (20 logical cores)

benchmark_tracemalloc_peak_mib starts before benchmark-owned input, target, and output arrays are created, but tracemalloc does not fully capture NumPy/SciPy native allocator peaks or VTK buffers. This machine-specific measurement is not a speed guarantee for other systems.
