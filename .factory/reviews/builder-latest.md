# Builder Review — torch.compile Inductor Backend Switch

## Change Summary

Switched `torch_compile()` in `src/dartsort/util/torch_util.py` from the deprecated `torch.jit.script` to `torch.compile` with Inductor backend for GPUs with compute capability ≥ 7.

## What Changed

- **File:** `src/dartsort/util/torch_util.py` (lines 133–147)
- **Before:** Early return with `torch.jit.script(fn)`, commented-out `torch.compile` path
- **After:** Uncommented the `torch.compile` path; uses `torch.compile(fn, dynamic=dynamic, fullgraph=fullgraph)` for CUDA GPUs with compute capability ≥ 7, falls back to `torch.jit.script` for older GPUs or CPU-only

## Correctness Verification

- `tests/test_spiketorch.py`: 7/7 passed ✅
- `tests/test_torch_optimization_util.py`: passed ✅
- `tests/test_config.py`: passed ✅
- `tests/test_data_util.py`: passed ✅
- Manual test: `torch_compile(dummy_fn)` returns `torch.compile` result on L40S (compute 8.9), outputs match exactly

## Behavioral Preservation

- Same function signature, same arguments, same return semantics
- `dynamic=True` handles variable input shapes (batched EM processing)
- `fullgraph` parameter passed through from each call site (7/15 functions use `fullgraph=False`)
- Fallback to `torch.jit.script` for compute capability < 7 preserves backward compatibility

## Expected Impact

- Affects all 15 `@torch_compiler`-decorated functions across clustering, matching, and utility modules
- Inductor backend enables operator fusion, memory planning, and reduced kernel launch overhead
- Estimated 2–4% total pipeline speedup (15–25 seconds) after warm-up amortization
- First invocation has warm-up cost (~1–5s per function); cached on subsequent runs

## Risk

- Low: mathematical equivalence verified (max_abs_diff = 0.0 in researcher benchmarks)
- Warm-up cost on first run may offset gains for single-recording benchmarks
- `dynamic=True` may limit some static-shape Inductor optimizations
