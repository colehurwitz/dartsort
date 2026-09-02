# Builder Review — benchmark_adapter.py timing fix

## What was done

Fixed `benchmark_adapter.py` to read the actual sorting time from `dartsort_output/timing.json` instead of using `metrics.speed_s` from `results.json`, which only contains the comparison time (0.23s vs 706.65s actual).

## Changes

- **benchmark_adapter.py**: Modified `translate()` to accept an optional `results_dir` parameter. When provided, it reads `{results_dir}/dartsort_output/timing.json` and:
  - Uses `timing_data["total"]` (706.65s) as `speed_seconds` instead of `metrics["speed_s"]` (0.23s)
  - Populates `stage_timing` from all non-"total" entries, each prefixed with `dartsort_` (e.g., `dartsort_initial_detection`, `dartsort_cluster0`)
  - Falls back to old behavior (`metrics.speed_s`, empty stage_timing) when timing.json is absent
- **main()**: Passes `results_dir` to `translate()`

## Testing

Verified with real data files:
- Fallback mode (no results_dir): `speed_seconds = 0.23`, empty `stage_timing` ✓
- With timing.json: `speed_seconds = 706.65`, 9 stage timing entries with `dartsort_` prefix ✓

## Scope

Only `benchmark_adapter.py` was modified — single file, focused change.
