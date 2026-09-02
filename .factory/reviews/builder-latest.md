## Builder Review — GPU-accelerated nearest_higher_density_neighbor

### Changes Made

**`src/dartsort/clustering/density.py`**
- Added `sort_nhdn()` function (~120 lines) after `sort_density()`
  - Uses same sorted-first-dimension sliding window as `sort_density()`
  - Computes GPU-accelerated squared Euclidean distances via `sqeuc_cdist_known_norm()`
  - Tracks k-nearest neighbors per query point via incremental topk-merge across column batches
  - Applies density filtering (higher density, non-neginf) to k nearest, selects closest
  - Maps sorted indices back to original ordering
  - Matches `nearest_higher_density_neighbor()` behavior exactly (k-limited search)
- Modified `density_peaks()` to route to GPU NHDN when device is not CPU
  - Added `use_gpu_nhdn` flag based on device type
  - Conditionally skips expensive 8D KDTree rebuild (only for sort density + GPU + no borders)
  - Falls back to CPU `nearest_higher_density_neighbor()` when device is CPU

**`src/dartsort/clustering/clustering.py`**
- Modified `DensityPeaksClusterer._cluster_extra()` for lazy KDTree build
  - When `res["kdtree"]` is None (GPU path skipped it), builds KDTree on `X_fit` for `nearest_neighbor_assign()` in subsampling path

### Correctness Verification
- `sort_nhdn()` produces 100% exact match with `nearest_higher_density_neighbor()` on test data (1000 points, 3D)
- `density_peaks()` with GPU device produces identical labels to CPU path
- All 8 density tests pass
- All 72 clustering tests pass (including 8 accuracy subtests)

### Key Design Decisions
1. **k-nearest limit**: Initially implemented without k-limit (searching ALL neighbors within radius). This caused test failures because it bridged separate clusters. Added topk-merge approach to match KDTree's k=n_neighbors_search limit exactly.
2. **Incremental topk-merge**: For each column batch, find top-k candidates, merge with accumulated top-k buffer. Efficient on GPU (tiny tensor operations on batch_size × k matrices).
3. **Conditional KDTree skip**: Only skip KDTree rebuild when safe (GPU + sort density + no border removal). Preserves all other code paths.

### Expected Performance Impact
- Eliminates 8D KDTree build (~5s) and CPU NHDN queries (~30-60s) for ~200k points
- GPU topk-merge overhead is minimal (20-element buffers per query point)
- Estimated cluster0 savings: 30-60s (15-29% of cluster0 stage)
