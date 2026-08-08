import numpy as np
import pytest

from dartsort.clustering.cluster_util import (
    decrumb_labels,
    recluster,
    reorder_by_depth,
    violation_matrix,
)
from dartsort.util.data_util import DARTsortSorting


@pytest.mark.parametrize("n_units", [0, 1, 2, 8])
def test_reorder_by_depth(n_units):
    # generate a sorting with n_units units where each unit has a specific
    # depth coordinate
    n_spikes_per_unit = 10
    n_spikes = n_units * n_spikes_per_unit

    times_samples = np.arange(n_spikes)
    channels = np.zeros(n_spikes, dtype=np.int64)
    labels = np.repeat(np.arange(n_units), n_spikes_per_unit)

    # centroids backwards
    centroids = np.arange(n_units, dtype=np.float64)[::-1]

    sorting = DARTsortSorting(
        times_samples=times_samples, channels=channels, labels=labels
    )

    reordered_sorting, reorder = reorder_by_depth(sorting, centroids=centroids)
    assert reordered_sorting.labels is not None

    # test that reorder[j] is the new label of unit j
    for j in range(n_units):
        in_j = labels == j
        assert np.all(reordered_sorting.labels[in_j] == reorder[j])

    # test that centroids are increasing in the reordered sorting
    new_centroids = np.empty(n_units)
    new_centroids[reorder] = centroids
    assert (np.diff(new_centroids) >= 0).all()


def test_recluster():
    # basic check of recluster
    coords = np.array([0.0, 0.1, 1.0, 1.1, 2.0])
    n_units = len(coords)
    dists = np.abs(coords[:, None] - coords[None, :])

    sorting = DARTsortSorting(
        times_samples=np.arange(n_units),
        channels=np.zeros(n_units, dtype=np.int64),
        labels=np.arange(n_units),
    )

    new_sorting, new_ids = recluster(sorting=sorting, dists=dists, threshold=0.15)
    assert new_sorting.labels is not None

    # check that new_ids[j] is the new label of unit j
    for j in range(n_units):
        assert np.all(new_sorting.labels[sorting.labels == j] == new_ids[j])

    # merges correspond to the 1d coordinate:
    # close pairs share a new label; distant units do not
    assert new_ids[0] == new_ids[1]  # distance 0.1 < threshold
    assert new_ids[2] == new_ids[3]  # distance 0.1 < threshold
    assert new_ids[0] != new_ids[2]
    assert new_ids[0] != new_ids[4]
    assert new_ids[2] != new_ids[4]


@pytest.mark.parametrize("n_units", [0, 1, 8])
def test_decrumb(n_units):
    min_size = 5
    # even-indexed units get 10 spikes (big enough); odd-indexed get 2 (crumbs)
    counts = np.where(np.arange(n_units) % 2 == 0, 10, 2)
    labels = np.repeat(np.arange(n_units), counts)

    new_labels = decrumb_labels(labels, min_size=min_size)

    # big units are relabeled 0, 1, 2, ... in original order
    for new_id, old_id in enumerate(range(0, n_units, 2)):
        assert np.all(new_labels[labels == old_id] == new_id)

    # small units are set to -1
    for old_id in range(1, n_units, 2):
        assert np.all(new_labels[labels == old_id] == -1)


def _viol_sorting(times_samples, labels):
    return DARTsortSorting(
        times_samples=np.asarray(times_samples, dtype=np.int64),
        channels=np.zeros(len(times_samples), dtype=np.int64),
        labels=np.asarray(labels, dtype=np.int64),
        sampling_frequency=10_000.0,
    )


def _brute_force_viols(times, labels, n_units):
    viols = np.zeros((n_units, n_units), dtype=np.int64)
    viol_samples = 10  # using 10kHz above
    for i in range(len(times)):
        for j in range(i + 1, len(times)):
            if labels[i] < 0 or labels[j] < 0:
                continue
            if abs(times[i] - times[j]) > viol_samples:
                continue
            viols[labels[i], labels[j]] += 1
            if labels[i] != labels[j]:
                viols[labels[j], labels[i]] += 1
    return viols


def test_violation_matrix_explicit():
    times = [0, 10, 21, 31, 31, 60]
    labels = [0, 0, 1, 0, 1, -1]
    res = violation_matrix(_viol_sorting(times, labels), viol_ms=1.0)

    assert np.array_equal(res.unit_ids, [0, 1])
    assert np.array_equal(res.spike_counts, [3, 2])
    assert np.array_equal(res.viol_counts, [[1, 2], [2, 1]])


def test_violation_matrix_sametime():
    times = [0] * 7
    labels = [0, 0, 1, 1, 1, 2, -1]
    res = violation_matrix(_viol_sorting(times, labels), viol_ms=1.0)

    assert np.array_equal(res.unit_ids, [0, 1, 2])
    assert np.array_equal(res.spike_counts, [2, 3, 1])
    assert np.array_equal(res.viol_counts, [[1, 6, 2], [6, 3, 3], [2, 3, 0]])


@pytest.mark.parametrize(
    "n_units, n_spikes, seed",
    [(0, 0, 0), (1, 1, 0), (1, 50, 1), (3, 200, 2), (6, 500, 3)],
)
def test_violation_matrix(n_units, n_spikes, seed):
    rg = np.random.default_rng(seed)

    # mean spacing is below the violation window, so violations are common.
    # integer times drawn with replacement also give us plenty of ties.
    times = np.sort(rg.integers(0, 5 * n_spikes + 1, size=n_spikes))

    # every unit is present, so that the matrix is (n_units, n_units), and
    # some spikes are triaged
    labels = np.concatenate(
        (np.arange(n_units), rg.integers(-1, n_units, size=max(0, n_spikes - n_units)))
    )
    rg.shuffle(labels)

    sorting = _viol_sorting(times, labels)
    res = violation_matrix(sorting, viol_ms=1.0)

    assert res.viol_counts.shape == (n_units, n_units)
    assert np.array_equal(res.unit_ids, np.arange(n_units))
    assert np.array_equal(
        res.spike_counts, np.bincount(labels[labels >= 0], minlength=n_units)
    )
    assert np.array_equal(res.viol_counts, res.viol_counts.T)
    assert np.array_equal(res.viol_counts, _brute_force_viols(times, labels, n_units))

    # unsorted spike times land on the same answer
    order = rg.permutation(n_spikes)
    shuffled = violation_matrix(_viol_sorting(times[order], labels[order]), viol_ms=1.0)
    assert np.array_equal(shuffled.viol_counts, res.viol_counts)

    # a window of 0ms only counts simultaneous spikes
    zero = violation_matrix(sorting, viol_ms=0.0)
    same_time = np.zeros((n_units, n_units), dtype=np.int64)
    for t in np.unique(times):
        in_t = labels[times == t]
        in_t = in_t[in_t >= 0]
        for a in range(len(in_t)):
            for b in range(a + 1, len(in_t)):
                same_time[in_t[a], in_t[b]] += 1
                if in_t[a] != in_t[b]:
                    same_time[in_t[b], in_t[a]] += 1
    assert np.array_equal(zero.viol_counts, same_time)
