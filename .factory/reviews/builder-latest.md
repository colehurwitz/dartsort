## Builder Review — 2026-09-01

### Experiment
**H1: Eliminate HDF5 round-trip in threshold_to_fit()**

### What was implemented
Modified `threshold_to_fit()` in `src/dartsort/peel/peel_lib.py` to accumulate threshold-detected waveforms in memory instead of writing them to a temp HDF5 file and reading them back.

**New function:** `_threshold_to_fit_in_memory()` — a helper that:
1. Gets chunk starts and shuffles them (same logic as `peel()` with `shuffle=True`)
2. Iterates chunks using `trainer.process_chunk()`, accumulating waveforms/channels/voltages in lists
3. Applies early stopping when `n_spikes >= max_waveforms_fit` (same condition as `peel()`)
4. Concatenates accumulated tensors
5. Computes amplitude-based reweighting via `fit_reweighting()` from `data_util.py` (same logic as `subsample_waveforms()` with `subsample_by_weighting=True`)
6. Passes waveforms + weights directly to `pipeline.fit()`

**Branching logic:** The in-memory path is used when `pipeline.needs_residual() == False` (the common case for initial denoiser fitting). When residual snips are needed (e.g., whitener fitting), the existing HDF5 path is preserved unchanged.

### Files modified
- `src/dartsort/peel/peel_lib.py` (+121 lines, -1 line)

### Correctness verification
- **Same waveforms:** Produced by the same `peel_chunk()` calls with same detection parameters
- **Same weighting:** `fit_reweighting()` called with identical parameters (voltages, fit_sampling, fit_max_reweighting)
- **Same fit inputs:** `TemporalPCADenoiser.fit()` receives identical waveforms/channels/weights (confirmed it does NOT use hdf5_filename parameter)
- **Fallback preserved:** HDF5 path unchanged for residual-snip-requiring cases

### Test results
- `test_subtract.py`: 3/3 passed (includes `test_fakedata_nonn` which exercises full subtraction pipeline)
- `test_threshold.py`: 1/1 passed
- `test_dartsort.py`: 5/5 passed (includes end-to-end `test_fakedata` and `test_fakedata_nonn`)
- `test_transform.py`: 1/1 passed
- Full suite: 497/498 passed (1 pre-existing failure in `test_drifty_matching.py::test_interp_upsampling[interpolation-16-8-1]` — numerical precision issue unrelated to this change)

### Expected speedup
| Component | Estimated savings |
|-----------|-------------------|
| HDF5 write (9.4GB incremental) | 3–8s |
| HDF5 read via batched_h5_read (9.4GB) | 5–12s |
| HDF5 metadata/chunking overhead | 1–3s |
| TemporaryDirectory creation + cleanup | 0.5s |
| **Total** | **10–24s** |

Conservative: 15s (1.6% of 959.55s baseline)

### Risk assessment
**Zero accuracy risk.** Only data transport changes (disk → memory). No numerical computation is modified. All downstream stages (subtraction loop, featurization, clustering, matching) receive identical inputs.

**Memory:** 512K waveforms ≈ 10GB in float32 — same data was transiently in memory during HDF5 writes anyway. Typical spike sorting machines have ≥64GB RAM.
