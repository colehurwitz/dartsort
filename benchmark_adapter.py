#!/usr/bin/env python3
"""Adapter: wraps spike-sort-bench and translates output to optimize-sorting format.

Usage: benchmark_adapter.py <config.yaml> [output.json]

If output.json is provided, translated results are written there (for workflow {output} substitution).
Results are always also printed to stdout.
"""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml


def translate(raw: dict, results_dir: Path | None = None) -> dict:
    """Translate spike-sort-bench output to optimize-sorting format.

    Parameters
    ----------
    raw : dict
        Parsed results.json from spike-sort-bench.
    results_dir : Path, optional
        Directory containing the benchmark output.  When provided the
        adapter reads ``{results_dir}/dartsort_output/timing.json`` for
        the actual sorting time and per-stage breakdown.  Falls back to
        ``metrics.speed_s`` (comparison-only time) when the file is
        absent.
    """
    metrics = raw["metrics"]
    per_unit = metrics.get("per_unit", {})

    per_unit_accuracy = {}
    if "unit_ids" in per_unit and "accuracy" in per_unit:
        for uid, acc in zip(per_unit["unit_ids"], per_unit["accuracy"]):
            per_unit_accuracy[str(uid)] = acc

    # --- timing -----------------------------------------------------------
    # Prefer dartsort_output/timing.json (actual sorting time) over
    # metrics.speed_s (which only measures the comparison step).
    speed_seconds = metrics["speed_s"]
    stage_timing: dict[str, float] = {}

    if results_dir is not None:
        timing_path = results_dir / "dartsort_output" / "timing.json"
        if timing_path.exists():
            with open(timing_path) as f:
                timing_data: dict[str, float] = json.load(f)
            speed_seconds = timing_data.get("total", speed_seconds)
            for key, val in timing_data.items():
                if key != "total":
                    stage_timing[f"dartsort_{key}"] = val

    return {
        "accuracy": metrics["accuracy"],
        "speed_seconds": speed_seconds,
        "per_unit_accuracy": per_unit_accuracy,
        "stage_timing": stage_timing,
    }


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: benchmark_adapter.py <config.yaml> [output.json]", file=sys.stderr)
        sys.exit(1)

    config_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    # Clear stale dartsort output so overwrite=False doesn't skip computation
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    results_dir = Path(cfg["output"]["output_dir"])
    dartsort_output = results_dir / "dartsort_output"
    if dartsort_output.exists():
        shutil.rmtree(dartsort_output)
    stale_results = results_dir / "results.json"
    if stale_results.exists():
        stale_results.unlink()

    result = subprocess.run(
        ["/usr/bin/python3", "-m", "spike_sort_bench", "--config", config_path],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"spike-sort-bench failed (exit {result.returncode})", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(result.returncode)

    results_file = cfg["output"].get("results_file", "results.json")
    results_path = results_dir / results_file

    with open(results_path) as f:
        raw = json.load(f)

    if "error" in raw:
        print(f"Benchmark error: {raw['error']}", file=sys.stderr)
        sys.exit(2)

    translated = translate(raw, results_dir=results_dir)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(translated, f, indent=2)

    json.dump(translated, sys.stdout, indent=2)
    print()


if __name__ == "__main__":
    main()
