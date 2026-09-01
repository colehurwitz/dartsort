# Builder Review — H1: Preload PCA Dataset

## Summary
Implemented preloading of the PCA dataset (`collisioncleaned_tpca_features`) in `cluster()` to eliminate duplicate HDF5 reads. Both `SimpleMatrixFeatures.from_config()` and `StableWaveformFeatures.from_config()` previously read the same ~115MB 3D dataset from HDF5 independently. Now the data is loaded once into memory and shared.

## Changes

### `src/dartsort/main.py` (+19 lines)
- In `cluster()`, before calling `SimpleMatrixFeatures.from_config()`, preload the PCA dataset from HDF5 into the sorting object via `add_ephemeral_feature()`
- Guard: only when `features is None`, `parent_h5_path` exists, and data not already loaded
- Uses dict checks (`_ephemeral_features`, `_persistent_features`) instead of `hasattr()` to avoid triggering `__getattr__` lazy loading
- Wrapped in try/except for graceful fallback

### `src/dartsort/clustering/clustering_features.py` (+22 lines, -6 lines)
- In `SimpleMatrixFeatures.from_config()`, added fast path in the motion-aware PCA branch
- Checks for preloaded data via `getattr(sorting, pca_dataset_name, None)`
- If data is in memory: uses it directly with `sorting.channel_index`
- If not: falls through to existing H5 path (preserved unchanged)
- Moved `assert pcs.shape[2] == 1` and slicing outside both branches so they execute regardless of path

## Correctness Verification
- `sorting.channel_index` == `h5['channel_index'][:]`: both loaded from the same H5 file; verified via `_no_check_needed()` → `_is_geom_related()` which always loads `channel_index` as a persistent feature
- `interpolate_by_chunk()` already accepts both `np.ndarray` and `h5py.Dataset` (line 25 of interpolation_util.py)
- `StableWaveformFeatures.from_config()` already has the matching fast path (line 262-276) — it will find the preloaded data via `hasattr()` → `__getattr__` → `_ephemeral_features`
- Data is identical bytes: `h5[pca_name][:]` produces the same numpy array regardless of when it's read

## Test Results
- **72/72 passed** in `test_clustering.py` (90.80s)
- **34/34 passed** in `test_data_util.py` + `test_cluster_util.py` (12.59s)
- All imports verified clean

## Expected Impact
- Eliminates 1 full H5 read of ~115MB 3D PCA dataset
- Reduces chunk decompression overhead (numpy array iteration faster than h5py)
- Estimated 15-25s savings in clustering stage
- Memory overhead: ~115MB (acceptable for GPU machine)
