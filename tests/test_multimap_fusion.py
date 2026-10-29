"""Controlled fusion contracts, including the historical checkpoint interface."""

from pathlib import Path

import pytest
import torch

from src.models.multimap import MultiMapLocalizer


@pytest.fixture
def inputs():
    torch.manual_seed(17)
    return (
        torch.randn(2, 3, 2, 32, 64),
        torch.randn(2, 3, 3),
        torch.tensor([[True, False, True], [False, True, False]]),
        torch.randn(2, 2, 64, 64),
    )


@pytest.mark.parametrize("fusion", ["add", "concat_mlp", "gated"])
def test_fusion_invariance_padding_and_gradients(fusion, inputs):
    model = MultiMapLocalizer(32, 4, 1, fusion=fusion).eval()
    prediction = model(*inputs)
    assert prediction.shape == (2, 2)
    assert torch.isfinite(prediction).all()
    order = torch.tensor([2, 0, 1])
    permuted = tuple(x[:, order] for x in inputs[:3]) + (inputs[3],)
    torch.testing.assert_close(prediction, model(*permuted), atol=1e-5, rtol=1e-5)
    changed = [x.clone() for x in inputs]
    changed[0][~inputs[2]] = 100
    changed[1][~inputs[2]] = -100
    torch.testing.assert_close(prediction, model(*changed), atol=1e-5, rtol=1e-5)
    # Appending inactive observations must not change the radio context.
    padded = (
        torch.cat((inputs[0], inputs[0][:, :1]), 1),
        torch.cat((inputs[1], inputs[1][:, :1]), 1),
        torch.cat((inputs[2], torch.zeros(2, 1, dtype=torch.bool)), 1),
        inputs[3],
    )
    torch.testing.assert_close(prediction, model(*padded), atol=1e-5, rtol=1e-5)
    prediction.square().sum().backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
    if fusion != "add":
        assert any(p.grad.abs().sum() > 0 for p in model.fusion_network.parameters())
    mask = inputs[2].clone()
    mask[0] = False
    with pytest.raises(ValueError, match="at least one active"):
        model(inputs[0], inputs[1], mask, inputs[3])


def test_invalid_fusion():
    with pytest.raises(ValueError, match="Unknown CSI/BTS fusion"):
        MultiMapLocalizer(fusion="invalid")


def test_seeded_shared_parameters_are_identical():
    torch.manual_seed(23)
    baseline = MultiMapLocalizer(32, 4, 1)
    baseline_next_random = torch.rand(16)
    for fusion in ("concat_mlp", "gated"):
        torch.manual_seed(23)
        candidate = MultiMapLocalizer(32, 4, 1, fusion=fusion)
        torch.testing.assert_close(torch.rand(16), baseline_next_random, rtol=0, atol=0)
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(value, candidate.state_dict()[name], rtol=0, atol=0)


def test_original_checkpoint_strict_add_and_default_outputs(inputs):
    path = Path("outputs/multimap/training/clean_v0/best.pt")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    default = MultiMapLocalizer(**checkpoint["config"]["model"]).eval()
    explicit = MultiMapLocalizer(**checkpoint["config"]["model"], fusion="add").eval()
    default.load_state_dict(checkpoint["model"], strict=True)
    explicit.load_state_dict(checkpoint["model"], strict=True)
    assert not any(name.startswith("fusion_network.") for name in default.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(default(*inputs), explicit(*inputs), rtol=0, atol=0)


@pytest.mark.parametrize("fusion", ["concat_mlp", "gated"])
def test_fusion_equation(fusion, inputs):
    model = MultiMapLocalizer(32, 4, 1, fusion=fusion).eval()
    captured = []
    handle = model.station_set_encoder.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].detach())
    )
    try:
        with torch.no_grad():
            c = model.csi_encoder(inputs[0])
            p = model.bs_position_encoder(inputs[1])
            joint = torch.cat((c, p), -1)
            learned = model.fusion_network(joint)
            expected = learned if fusion == "concat_mlp" else (
                torch.sigmoid(learned) * c + (1 - torch.sigmoid(learned)) * p
            )
            model(*inputs)
        torch.testing.assert_close(captured[0][:, 1:], expected)
    finally:
        handle.remove()
