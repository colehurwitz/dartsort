import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import spikeinterface.core as sc
import spikeinterface.preprocessing as spre

from ..templates import TemplateData
from ..util.data_util import DARTsortSorting, ensure_path
from ..util.job_util import ensure_computation_config
from ..util.motion import MotionInfo
from ..util.py_util import databag
from .sim_template_tools import get_template_simulator
from .simlib import InjectSpikesPreprocessor


@databag
class HybridDataset:
    gt_sorting: DARTsortSorting
    gt_templates: TemplateData
    gt_unit_info: pd.DataFrame
    recording: sc.BaseRecording
    motion: MotionInfo
    metadata: dict[str, Any]


def load_hybrid_recording(folder: str | Path) -> HybridDataset | None:
    folder = ensure_path(folder)
    if not folder.exists():
        return None
    recording_dir = folder / "recording"
    templates_npz = folder / "templates.npz"
    sorting_h5 = folder / "dartsort_sorting.h5"
    unit_info_csv = folder / "unit_information.csv"
    meta_json = folder / "metadata.json"
    if not meta_json.exists():
        return None

    recording = sc.read_binary_folder(recording_dir)
    templates = TemplateData.from_npz(templates_npz)
    sorting = DARTsortSorting.from_peeling_hdf5(sorting_h5)
    motion = MotionInfo.try_load(folder)
    assert motion is not None
    unit_info_df = pd.read_csv(unit_info_csv)
    with meta_json.open("r") as jsonf:
        metadata = json.load(jsonf)

    return HybridDataset(
        recording=recording,
        gt_templates=templates,
        gt_sorting=sorting,
        motion=motion,
        gt_unit_info=unit_info_df,
        metadata=metadata,
    )


def make_hybrid_recording(
    *,
    folder: str | Path,
    overwrite: bool = False,
    target_recording: sc.BaseRecording,
    injected_sorting: sc.BaseSorting,
    motion: MotionInfo,
    templates: sc.Templates,
    filter: bool = True,
    trim_t_start_rel: float = 0.0,
    trim_t_len: float | None = None,
    remove_bad_channels: bool = True,
    template_peak_range: tuple[float, float] | None = (25.0, 100.0),
    reset_times: bool = True,
    target_sampling_frequency: float = 30_000.0,
    amplitude_jitter: float = 0.05,
    amp_jitter_family: Literal["gamma", "uniform", "normal"] = "normal",
    temporal_jitter: int = 8,
    save_chunk_len_s: float = 0.5,
    template_simulator_kwargs: dict | None = None,
    metadata: dict[str, Any] | None = None,
    computation_cfg=None,
    extra_unit_information: pd.DataFrame | None = None,
    rg: np.random.Generator | int = 0,
) -> HybridDataset:
    assert templates.is_in_uV
    assert templates.num_units == injected_sorting.get_num_units()
    assert injected_sorting._recording is None
    assert target_recording.get_num_segments() == 1
    assert injected_sorting.get_num_segments() == 1
    assert injected_sorting.get_last_spike_frame() < target_recording.get_num_frames()
    assert abs(templates.sampling_frequency - target_sampling_frequency) < 10

    if folder is not None and not overwrite:
        if (ret := load_hybrid_recording(folder)) is not None:
            return ret
    computation_cfg = ensure_computation_config(computation_cfg)

    if trim_t_len:
        frame_start = int(trim_t_start_rel * target_recording.sampling_frequency)
        frame_end = frame_start + int(trim_t_len * target_recording.sampling_frequency)
        target_recording = target_recording.frame_slice(frame_start, frame_end)
        injected_sorting = injected_sorting.frame_slice(frame_start, frame_end)

    if reset_times:
        target_recording.reset_times()
        if abs(target_recording.sampling_frequency - target_sampling_frequency) > 10.0:
            raise ValueError("Sampling...")
        target_recording._sampling_frequency = target_sampling_frequency
        target_recording.reset_times()

    if template_peak_range is not None:
        templates = rescale_templates(templates, template_peak_range, rg)

    if filter:
        target_recording = spre.highpass_filter(target_recording, dtype="float32")
        if "inter_sample_shift" in target_recording.get_property_keys():
            target_recording = spre.phase_shift(target_recording)
    if remove_bad_channels:
        assert filter
        bcids = spre.detect_bad_channels(target_recording, seed=0)
        target_recording = target_recording.remove_channels(bcids[0])
    target_recording = spre.scale_to_uV(target_recording)  # ty: ignore[invalid-argument-type]

    template_simulator_kwargs = dict(template_simulator_kwargs or {})
    template_simulator_kwargs.setdefault("trough_offset_samples", templates.nbefore)
    if templates.probe is not None:
        # the library's own geometry is where its templates were sampled
        template_simulator_kwargs.setdefault(
            "source_geom", templates.get_channel_locations()
        )
    template_simulator = get_template_simulator(
        n_units=templates.num_units,
        templates_kind="library",
        template_library=templates.templates_array,
        geom=target_recording.get_channel_locations(),
        sampling_frequency=target_sampling_frequency,
        temporal_jitter=temporal_jitter,
        random_seed=rg,
        **template_simulator_kwargs,
    )
    hybrid_recording = InjectSpikesPreprocessor(
        recording=target_recording,
        sorting=injected_sorting,
        motion=motion,
        template_simulator=template_simulator,
        amplitude_jitter=amplitude_jitter,
        amp_jitter_family=amp_jitter_family,
        temporal_jitter=temporal_jitter,
        extract_radius=0.0,
        random_seed=rg,
        compute_collision_waveforms=False,
    )

    hybrid_recording.save_simulation(
        folder,
        overwrite=overwrite,
        n_jobs=computation_cfg.actual_n_jobs(),
        featurization_cfg=None,
        computation_cfg=computation_cfg,
        chunk_len_s=save_chunk_len_s,
        save_injected_waveforms=False,
        save_noise_waveforms=False,
        save_collision_waveforms=False,
        save_collisioncleaned_waveforms=False,
        save_collidedness=False,
        extra_unit_information=extra_unit_information,
        n_residual_snips=0,
    )
    metadata = dict(metadata or {})
    metadata.update(
        filter=filter,
        trim_t_start_rel=trim_t_start_rel,
        trim_t_len=trim_t_len,
        remove_bad_channels=remove_bad_channels,
        template_peak_range=template_peak_range,
        reset_times=reset_times,
        target_sampling_frequency=target_sampling_frequency,
        amplitude_jitter=amplitude_jitter,
        amp_jitter_family=amp_jitter_family,
        temporal_jitter=temporal_jitter,
    )
    with (ensure_path(folder) / "metadata.json").open("w") as jsonf:
        json.dump(metadata, jsonf)

    res = load_hybrid_recording(folder)
    assert res is not None
    return res


def rescale_templates(
    templates: sc.Templates,
    template_peak_range: tuple[float, float] = (25.0, 100.0),
    rg: np.random.Generator | int = 0,
):
    t = deepcopy(templates)
    del templates
    rg = np.random.default_rng(rg)
    for i in range(t.num_units):
        pk = np.abs(t.templates_array[i]).max()
        assert pk > 0
        targ = rg.uniform(*template_peak_range)
        t.templates_array[i] *= targ / pk
    return t
