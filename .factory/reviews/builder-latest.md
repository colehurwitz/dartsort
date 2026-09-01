# Builder Review — temporal_upsamples 4→2

**Date:** 2026-09-01
**Parameter:** `temporal_upsamples`
**File:** `benchmark_60s.yaml`
**Change:** `4 → 2` (default was 4; explicitly set to 2)

## What Was Done

Added `temporal_upsamples: 2` under `sorter.params` in `benchmark_60s.yaml`. This is the only file modified.

## Why

Per the strategy analysis, `temporal_upsamples` is the highest-ranked parameter for speed optimization (Rank 1). It targets the matching stage (228.9s, 28.4% of dartsort core time) with low-to-medium accuracy risk:

- **Mechanism:** Halves the number of upsampled template versions from 4 to 2, reducing SVD projections, pairwise overlap computation, and fine-pass matching candidates.
- **Expected speed savings:** 26–43s (3.2–5.3% of dartsort core time).
- **Expected accuracy impact:** −0.001 to −0.005 (projected 0.824–0.828, well above 0.8207 threshold).
- **Temporal precision:** Moves from 1/4 sample (±8.3μs) to 1/2 sample (±16.7μs) at 30kHz — still high resolution.

## Verification

- Only `benchmark_60s.yaml` was modified (config file, `.yaml` extension).
- No `.py`, `.cu`, `.cpp`, `.c`, `.h`, `.pyx`, or `.sh` files were touched.
- The resulting YAML is valid and matches the target structure from the strategy document.

## Resulting Config (sorter section)

```yaml
sorter:
  name: dartsort
  timeout_s: 1800
  params:
    preprocessing: ibllikecmr
    device: cuda
    n_jobs_cpu: 0
    matching_iterations: 1
    temporal_upsamples: 2
```

## Next Steps

Run the benchmark: `python benchmark_adapter.py benchmark_60s.yaml results/temporal_upsamples_2.json`
