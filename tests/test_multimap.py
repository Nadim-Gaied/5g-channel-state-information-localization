import torch
import numpy as np

from src.models.multimap import MultiMapLocalizer
from src.multimap.dataset import MultiMapScenarioDataset
from src.multimap.normalization import (
    compute_global_feature_stats,
    load_normalization,
    save_normalization,
)


def test_multimap_model_shapes_and_station_permutation_invariance():
    torch.manual_seed(7)
    model = MultiMapLocalizer(embedding_dim=32, num_heads=4, set_depth=1).eval()
    csi = torch.randn(2, 3, 2, 32, 64)
    bs_positions = torch.randn(2, 3, 3)
    bs_mask = torch.tensor([[True, True, False], [True, True, True]])
    map_raster = torch.randn(2, 2, 64, 64)

    with torch.no_grad():
        output = model(csi, bs_positions, bs_mask, map_raster)
        permutation = torch.tensor([2, 0, 1])
        permuted = model(
            csi[:, permutation],
            bs_positions[:, permutation],
            bs_mask[:, permutation],
            map_raster,
        )
    assert output.shape == (2, 2)
    assert torch.isfinite(output).all()
    assert torch.allclose(output, permuted, atol=1e-5)


def test_multimap_dataset_variable_bts_mask(tmp_path):
    channels = np.ones((3, 2, 4, 8), dtype=np.complex64)
    path = tmp_path / "scenario.npz"
    np.savez_compressed(
        path,
        channels=channels,
        bs_positions=np.asarray([[0, 0, 5], [10, 0, 5]], dtype=np.float32),
        bs_available=np.asarray([[1, 1], [1, 0], [0, 1]], dtype=bool),
        positions=np.asarray([[1, 1, 2], [2, 2, 2], [3, 3, 2]], dtype=np.float32),
        map_raster=np.zeros((2, 16, 16), dtype=np.float32),
        map_bounds=np.asarray([-10, -10, 20, 20], dtype=np.float32),
        feature_mean=np.zeros(2, dtype=np.float32),
        feature_std=np.ones(2, dtype=np.float32),
    )
    dataset = MultiMapScenarioDataset(
        path, max_bts=3, random_bts_subsets=True, minimum_bts=1
    )
    sample = dataset[0]
    assert sample["csi"].shape == (3, 2, 4, 8)
    assert 1 <= int(sample["bs_mask"].sum()) <= 2
    assert not sample["bs_mask"][2]
    assert torch.count_nonzero(sample["csi"][~sample["bs_mask"]]) == 0


def test_global_normalization_uses_only_active_links(tmp_path):
    channels = np.asarray(
        [
            [[[1 + 3j, 2 + 4j]], [[100 + 100j, 100 + 100j]]],
            [[[100 + 100j, 100 + 100j]], [[5 + 7j, 6 + 8j]]],
        ],
        dtype=np.complex64,
    )
    scenario = tmp_path / "scenario.npz"
    np.savez_compressed(
        scenario,
        channels=channels,
        bs_positions=np.asarray([[0, 0, 5], [10, 0, 5]], dtype=np.float32),
        bs_available=np.asarray([[1, 0], [0, 1]], dtype=bool),
        positions=np.asarray([[1, 1, 2], [2, 2, 2]], dtype=np.float32),
        map_raster=np.zeros((2, 8, 8), dtype=np.float32),
        map_bounds=np.asarray([-10, -10, 20, 20], dtype=np.float32),
        feature_mean=np.asarray([50, 50], dtype=np.float32),
        feature_std=np.ones(2, dtype=np.float32),
    )

    report = compute_global_feature_stats([scenario])
    assert report["active_links"] == 2
    assert report["scalar_values_per_component"] == 4
    assert np.allclose(report["feature_mean"], [3.5, 5.5])
    assert np.allclose(report["feature_std"], [np.sqrt(4.25), np.sqrt(4.25)])

    normalization = tmp_path / "normalization.npz"
    save_normalization(report, normalization)
    mean, std = load_normalization(normalization)
    dataset = MultiMapScenarioDataset(scenario, normalization_path=normalization)
    assert dataset.normalization_source == str(normalization.resolve())
    assert np.allclose(mean, dataset.feature_mean.numpy())
    assert np.allclose(std, dataset.feature_std.numpy())
    assert torch.allclose(
        dataset[0]["csi"][0, 0, 0, 0],
        torch.tensor((1.0 - 3.5) / np.sqrt(4.25), dtype=torch.float32),
    )

