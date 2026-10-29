import numpy as np

from src.multimap.prepare_intra_map_csi import select_strongest_active_link
from src.multimap.compare_intra_map_csi import summarize


def test_select_strongest_active_link_ignores_unavailable_station():
    channels = np.zeros((2, 3, 2, 2), dtype=np.complex64)
    channels[:, 0] = 1.0
    channels[:, 1] = 100.0
    channels[:, 2] = 3.0 + 4.0j
    availability = np.asarray([[1, 0, 1], [1, 1, 0]], dtype=bool)

    selected, indices = select_strongest_active_link(channels, availability)

    assert indices.tolist() == [2, 1]
    assert np.allclose(selected[0], channels[0, 2])
    assert np.allclose(selected[1], channels[1, 1])


def test_normalized_summary_uses_coverage_diagonal():
    metrics = summarize(np.asarray([10.0, 20.0]), diagonal_m=100.0)
    assert np.isclose(metrics["mean_m"], 15.0)
    assert np.isclose(metrics["mean_normalized_pct"], 15.0)
