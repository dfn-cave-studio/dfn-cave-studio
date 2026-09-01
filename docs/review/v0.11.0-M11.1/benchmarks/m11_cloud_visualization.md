# M11 Interactive P32 Cloud Visualization Benchmark

CPU-side VTK geometry benchmark for a `69 × 50 × 47 = 162,150` voxel field. A no-op plotter is used; these figures do not claim real OpenGL frame-rate performance.

| Mode | Time (s) | Cells | Points | Retained array/point memory |
|---|---:|---:|---:|---:|
| Outer Surface — Exact | 0.047226 | 18,086 | 18,088 | 578,792 B |
| Outer Surface — Smooth | 0.039777 | 18,086 | 18,088 | 578,800 B |
| X section | 0.027041 | 2,350 | 2,448 | 38,776 B |
| Y section | 0.030123 | 3,243 | 3,360 | 53,292 B |
| Z section | 0.030575 | 3,450 | 3,570 | 56,640 B |
| Arbitrary plane | 0.038875 | 4,308 | 2,400 | 46,032 B |
| Clipped outer surface | 0.054346 | 18,474 | 9,626 | 337,200 B |

- First interactive Z-section actor registration: `0.032135 s`.
- Ten stable-slot section updates: `0.189102 s` total (the GUI separately applies an 80 ms debounce).
- After ten updates: one M11 actor, one M11 scalar bar, and one active plane widget.
- Exact and smooth figures report retained VTK point/scalar-array payload, not Python/VTK allocator overhead or GPU memory.
- Clipped surface geometry is intentionally open at the clipping boundary; it is not presented as a closed internally coloured scientific cut face.
- Real OpenGL appearance and interaction remain subject to manual acceptance.
