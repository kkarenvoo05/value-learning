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


@pytest.mark.parametrize("cam", CAMS)
def test_single_camera_model_reads_only_its_camera(cam):
    """Stage 5 camera ablation: a one-camera model takes a 512-d feature and ignores other views."""
    torch.manual_seed(0)
    m = ValueModel([cam], pretrained=False).eval()
    assert m.head[0].in_features == 512
    x = images(b=2)
    other = next(c for c in CAMS if c != cam)
    with torch.no_grad():
        a = m(x)
        b = m({**x, other: torch.zeros_like(x[other])})
        c = m({**x, cam: torch.zeros_like(x[cam])})
    assert a.shape == (2, 201)
    assert torch.equal(a, b)
    assert not torch.allclose(a, c)


def test_frozen_backbone_trains_only_the_head():
    """Stage 5 encoder ablation: after training steps the backbone's weights *and* BatchNorm
    running statistics are unchanged (it stays in eval mode under model.train()); the head moves."""
    torch.manual_seed(0)
    m = ValueModel(CAMS, pretrained=False).freeze_backbone().train()
    before = {k: v.clone() for k, v in m.backbone.state_dict().items()}
    head_before = m.head[2].weight.clone()
    assert m.training and not m.backbone.training
    assert all(not p.requires_grad for p in m.backbone.parameters())
    opt = torch.optim.AdamW(m.head.parameters(), lr=1e-3)
    x, y = images(b=4, size=64), torch.tensor([10, 60, 120, 190])
    for _ in range(3):
        loss = value_loss(m(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    m.eval().train()                                 # eval/train round trip keeps the backbone in eval
    assert not m.backbone.training
    assert all(torch.equal(v, before[k]) for k, v in m.backbone.state_dict().items())
    assert not torch.equal(m.head[2].weight, head_before)


def proprio_model():
    torch.manual_seed(0)
    return ValueModel(CAMS, pretrained=False, proprio_mean=[1.0] * 10, proprio_std=[2.0] * 10).eval()


def test_proprio_is_used_and_normalized_inside():
    m = proprio_model()
    assert m.head[0].in_features == 2 * 512 + 10
    x, p = images(), torch.randn(2, 10)
    with torch.no_grad():
        out = m(x, p)
        assert out.shape == (2, 201)
        assert not torch.allclose(out, m(x, p + 1))                       # proprio changes the output
        # z-scoring inside: a model with mean 0 / std 1 fed the z-scored input gives the same logits.
        m0 = ValueModel(CAMS, pretrained=False, proprio_mean=[0.0] * 10, proprio_std=[1.0] * 10).eval()
        m0.load_state_dict({k: v for k, v in m.state_dict().items() if not k.startswith("proprio_")}, strict=False)
        assert torch.allclose(out, m0(x, (p - 1) / 2), atol=1e-5)


def test_proprio_presence_must_match_the_model(model):
    with pytest.raises(ValueError):
        proprio_model()(images())
    with pytest.raises(ValueError):
        model(images(), torch.zeros(2, 10))


def test_image_only_state_dict_unchanged(model):
    assert not any(k.startswith("proprio") for k in model.state_dict())   # old checkpoints still load
