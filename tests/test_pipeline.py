from pathlib import Path

import numpy as np
import torch

from src.models import build_model
from src.data.dataset import load_csi_splits
from src.experiment_config import experiment_paths, load_experiment_config
from src.train import add_random_complex_awgn


DATA_PATH = Path("data/processed/city_0_newyork_3p5_max_32x64.npz")


def test_processed_dataset_integrity():
    with np.load(DATA_PATH) as data:
        features = data["features"]
        assert features.shape == (17337, 2, 32, 64)
        assert np.isfinite(features).all()

        train = set(data["train_idx"].tolist())
        validation = set(data["val_idx"].tolist())
        test = set(data["test_idx"].tolist())
        assert not (train & validation)
        assert not (train & test)
        assert not (validation & test)
        assert len(train | validation | test) == len(features)

        train_features = features[data["train_idx"]]
        means = train_features.mean(axis=(0, 2, 3))
        stds = train_features.std(axis=(0, 2, 3))
        assert np.allclose(means, 0.0, atol=1e-5)
        assert np.allclose(stds, 1.0, atol=1e-3)


def test_model_output_shapes():
    inputs = torch.randn(4, 2, 16, 32)
    for name in ("cnn", "vit", "cnn_residual", "vit_deep"):
        output = build_model(name)(inputs)
        assert output.shape == (4, 2)
        assert torch.isfinite(output).all()

    rich_inputs = torch.randn(2, 2, 32, 64)
    rich_vit = build_model(
        "vit_deep", input_size=(32, 64), patch_size=(4, 8)
    )
    assert rich_vit(rich_inputs).shape == (2, 2)
    assert build_model("cnn_residual")(rich_inputs).shape == (2, 2)


def test_complex_models_have_equivalent_sizes():
    cnn = build_model("cnn_residual")
    vit = build_model("vit_deep")
    cnn_parameters = sum(parameter.numel() for parameter in cnn.parameters())
    vit_parameters = sum(parameter.numel() for parameter in vit.parameters())
    relative_difference = abs(cnn_parameters - vit_parameters) / max(
        cnn_parameters, vit_parameters
    )
    assert cnn_parameters > 400_000
    assert vit_parameters > 400_000
    assert relative_difference < 0.02


def test_shared_dataset_bundle_and_experiment_config():
    bundle, splits = load_csi_splits(DATA_PATH)
    assert bundle.input_size == (32, 64)
    assert splits["train"].bundle is splits["validation"].bundle
    assert splits["validation"].bundle is splits["test"].bundle
    assert sum(len(split) for split in splits.values()) == len(bundle.features)

    config = load_experiment_config(
        "configs/experiments/city_0_newyork_3p5_max_32x64.json"
    )
    paths = experiment_paths(config)
    assert config["dataset"]["deepmimo"]["ofdm_limit"] == 64
    assert paths["processed"].name == "city_0_newyork_3p5_max_32x64.npz"


def test_random_complex_awgn_level_and_clean_probability():
    torch.manual_seed(123)
    features = torch.ones(128, 2, 16, 32)
    mean = torch.zeros(1, 2, 1, 1)
    std = torch.ones(1, 2, 1, 1)
    noisy = add_random_complex_awgn(features, mean, std, 10.0, 10.0)
    signal_power = (features[:, 0].square() + features[:, 1].square()).mean()
    noise = noisy - features
    noise_power = (noise[:, 0].square() + noise[:, 1].square()).mean()
    measured_snr = 10.0 * torch.log10(signal_power / noise_power)
    assert 9.8 < float(measured_snr) < 10.2

    clean = add_random_complex_awgn(
        features, mean, std, 0.0, 30.0, clean_probability=1.0
    )
    assert torch.equal(clean, features)
