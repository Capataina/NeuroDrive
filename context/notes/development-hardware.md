# Development Hardware

The primary development and testing machine for NeuroDrive.

## Specs

| Component | Detail |
|-----------|--------|
| **Machine** | MacBook Air (M2, 2022) |
| **CPU** | Apple M2 — 8 cores (4 performance + 4 efficiency) @ 3.50 GHz |
| **GPU** | Apple M2 — 8 cores @ 1.40 GHz (integrated, unified memory) |
| **Memory** | 8 GB unified (shared between CPU and GPU) |
| **Disk** | 228 GB APFS |
| **Display** | 14" 2940x1912 @ 60 Hz |
| **OS** | macOS Tahoe (Darwin 25.3.0, arm64) |
| **Architecture** | ARM64 (Apple Silicon) |

## Implications for NeuroDrive

- **Unified memory architecture**: CPU and GPU share the same 8 GB pool — no discrete VRAM. Memory-intensive work (large rollout buffers, trace captures) competes with rendering.
- **8 GB total**: profiling infrastructure must be memory-lean. No unbounded buffers, no large in-memory trace histories.
- **ARM64 / Apple Silicon**: SIMD via NEON, not SSE/AVX. Any future SIMD optimisation must target `std::arch::aarch64` or use portable abstractions.
- **Battery-powered**: frame budget discipline matters — the profiler itself must cost negligible overhead.
- **60 Hz display**: the 16.67ms frame budget is the natural target for smooth rendering.
