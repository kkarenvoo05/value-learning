"""ValueModel: shapes, both cameras used, normalization inside, expected value and loss arithmetic."""

import math

import pytest
import torch

from vl.model import ValueModel, value_loss

CAMS = ["wrist_left", "side_1"]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return ValueModel(CAMS, pretrained=False).eval()


def images(b=2, size=64, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {c: torch.randint(0, 256, (b, 3, size, size), dtype=torch.uint8, generator=g) for c in CAMS}


def test_output_shape(model):
    with torch.no_grad():
        assert model(images(b=3)).shape == (3, 201)


def test_one_shared_backbone(model):
    resnets = [m for m in model.modules() if m.__class__.__name__ == "ResNet"]
    assert len(resnets) == 1
    assert model.head[0].in_features == 2 * 512


def test_both_cameras_matter_and_order_is_fixed(model):
    x = images()
    with torch.no_grad():
        out = model(x)
        swapped = model({"wrist_left": x["side_1"], "side_1": x["wrist_left"]})
        reordered = model({"side_1": x["side_1"], "wrist_left": x["wrist_left"]})   # dict order only
    assert not torch.allclose(out, swapped)          # wrist and side features go to different slots
    assert torch.equal(out, reordered)               # model reads cameras by name, not position


@pytest.mark.parametrize("cam", CAMS)
def test_changing_one_camera_changes_the_output(model, cam):
    x = images()
    y = dict(x, **{cam: images(seed=1)[cam]})        # replace only this camera's images
    with torch.no_grad():
        assert not torch.allclose(model(x), model(y))


def test_normalization_happens_inside(model):
    seen = {}
    h = model.backbone.register_forward_pre_hook(lambda m, a: seen.setdefault("x", a[0]))
    mean_img = (model.mean.view(3, 1, 1) * 255).round().to(torch.uint8).expand(3, 8, 8)
    with torch.no_grad():
        model({c: mean_img.unsqueeze(0).clone() for c in CAMS})
    h.remove()
    assert seen["x"].dtype == torch.float32
    assert seen["x"].abs().max() < 0.01              # the ImageNet mean image normalizes to ~0


def test_expected_value_of_one_hot_is_bin_center(model):
    for b in [0, 57, 100, 200]:
        logits = torch.full((1, 201), -1e4)
        logits[0, b] = 0.0
        assert model.expected_value(logits).item() == pytest.approx(-1.0 + b * 0.005, abs=1e-6)


def test_uniform_prediction_gives_mid_value_and_ln_bins_loss(model):
    logits = torch.zeros(4, 201)
    assert model.expected_value(logits).allclose(torch.full((4,), -0.5), atol=1e-6)
    assert value_loss(logits, torch.tensor([0, 50, 150, 200])).item() == pytest.approx(math.log(201), abs=1e-5)


def test_a_few_steps_fit_one_batch():
    """Gradients reach backbone and head, and the loss on a fixed batch falls."""
    torch.manual_seed(0)
    m = ValueModel(CAMS, pretrained=False).train()
    x, y = images(b=4, size=64), torch.tensor([10, 60, 120, 190])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    losses = []
    for _ in range(15):
        loss = value_loss(m(x), y)
        opt.zero_grad()
        loss.backward()
        losses.append(loss.item())
        if not losses[1:]:
            assert m.backbone.conv1.weight.grad.abs().sum() > 0 and m.head[2].weight.grad.abs().sum() > 0
        opt.step()
    assert losses[-1] < 0.5 * losses[0]
