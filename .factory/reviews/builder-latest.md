## Builder Review — Eliminate HDF5 round-trip in fit_featurization_pipeline

### Change Summary
Added `_fit_featurization_in_memory()` method to `BasePeeler` in `src/dartsort/peel/peel_base.py` that bypasses the HDF5 write→read cycle when fitting the featurization pipeline. Modified `fit_featurization_pipeline()` to use the in-memory path when guard conditions are met.

### Guard Conditions (fall back to HDF5 if any fail)
1. `featurization_pipeline.needs_residual() == False` — no whitener/decollider needing residual snips
2. `featurization_pipeline.needs_more_features() == False` — no MixtureClassifier needing 2M+ waveforms
3. No transformer in the pipeline has `fits_from_disk == True`

### What the in-memory path does
Instead of:
1. Swap pipeline → temp `[Voltage, Waveform]` pipeline
2. `run_subsampled_peeling()` → write detected waveforms to temp HDF5 file
3. `subsample_waveforms()` → read waveforms back from HDF5
4. Fit pipeline from read-back waveforms

It now:
1. Swap pipeline → temp `[Voltage, Waveform]` pipeline
2. Loop over shuffled chunks, call `process_chunk()` directly
3. Accumulate `peeled_waveforms_fit`, `channels`, `peeled_voltages_fit`, `times_seconds` in CPU memory
4. Early-stop when `max_waveforms_fit` spikes are gathered
5. Concatenate, apply `fit_reweighting()` in memory
6. Fit pipeline directly from in-memory tensors (`hdf5_filename=None`)

### Pattern
Follows the same approach as the existing `_threshold_to_fit_in_memory()` in `peel_lib.py` (commit 6f5ed692).

### Bug fix during implementation
The initial implementation placed `weights` on the CUDA device. This caused `RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'` in `AmortizedLocalization._fit()`, which uses `WeightedRandomSampler` with a CPU-based torch generator. Fixed by keeping `weights` on CPU — the transformer moves data to device internally as needed.

### Test Results
- All 8 `test_dartsort.py` tests pass (including `test_initial_detection_swap[threshold]` which exercises the new in-memory path)
- All other test failures are pre-existing and unrelated to this change

### Expected Impact
- Saves ~5-15s per call on `initial_detection` and `matching1` model fits
- Estimated total savings: ~10-30s (1.4-4.2% of 708.7s pipeline)
- Eliminates HDF5 dataset creation, per-chunk dataset resizing, writes, and read-back operations
