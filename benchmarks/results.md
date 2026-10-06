# Geometry Engine Benchmark

Command: `python benchmarks/bench_geometry.py`
Machine: Windows workstation, Python venv, NumPy 2.5. Numbers are indicative; re-run locally.

Scenario `leo-ku-europe`: 528 satellites x 239 terminals. One chunk = 40 time steps.

| Method | Links | Time [s] | ns / link | Speed-up | Peak mem [MB] |
|---|---:|---:|---:|---:|---:|
| loop (Python) | 252,384 | 0.498 | 1974 | 1x | - |
| broadcast (NumPy 4-D) | 5,047,680 | 0.316 | 63 | 32x | 364 |
| **engine float64** | 5,047,680 | **0.107** | **21** | **93x** | **127** |
| engine float32 out | 5,047,680 | 0.121 | 24 | 82x | 187 |

Full scenario (721 steps x 528 sats x 239 terminals = 91 M links, propagation + geometry, chunked):

| Method | Time |
|---|---:|
| engine | 2.0 s |
| Python loop (extrapolated) | ~3 min |

A 1-vCPU Linux container gave the same engine cost (~20 ns/link) with a slower Python loop (~3.7 us/link, 188x speed-up).

## Takeaways

1. **Vectorising alone (loop -> broadcast) gives ~30-50x**, but the `(T, S, G, 3)` intermediate makes memory the next bottleneck.
2. **Rewriting the maths as dot products (BLAS) + in-place updates gives another ~3x speed and ~3x less memory.**
3. **float32 outputs** cost a cast inside the call but halve the size of results kept across chunks (use when storing full time series).
4. **Chunking bounds memory**: peak memory depends on chunk size, not on simulation length.

## What this means at operational scale

Example: 4,400 satellites, 10,000 terminals, 24 h at 10 s steps = 3.8 x 10^11 links.

| Approach | Estimated time |
|---|---:|
| Python loop | ~9 days |
| Engine, 1 core | ~2.2 h |
| Engine, 32 workers (chunks in parallel, Dask/Ray) | ~4 min |

Next optimisations, in order of value:
- Cull satellites below the horizon per chunk before the full maths (coarse geocentric angle test)
- Run chunks in parallel (they are independent by design)
- Numba / GPU (CuPy) only if profiling still shows geometry as the hotspot
