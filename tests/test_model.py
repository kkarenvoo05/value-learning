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


# Stage 5 stronger backbone: DINOv2 ViT-S/14 with registers (random init here, except the weight test).
DINO_SIZE = 56                                       # 4 x 4 patches of 14 px: small and fast on CPU


@pytest.fixture(scope="module")
def dino():
    torch.manual_seed(0)
    return ValueModel(CAMS, pretrained=False, backbone="dinov2_vits14_reg", image_size=DINO_SIZE).eval()


def test_dino_shape_and_one_shared_backbone(dino):
    with torch.no_grad():
        assert dino(images(b=3, size=DINO_SIZE)).shape == (3, 201)
    vits = [m for m in dino.modules() if m.__class__.__name__ == "VisionTransformer"]
    assert len(vits) == 1
    assert dino.head[0].in_features == 2 * 768       # [CLS, mean patch] per camera
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in dino.modules())


@pytest.mark.parametrize("cam", CAMS)
def test_dino_each_camera_changes_the_output(dino, cam):
    x = images(size=DINO_SIZE)
    with torch.no_grad():
        assert not torch.allclose(dino(x), dino({**x, cam: torch.zeros_like(x[cam])}))


def test_dino_readout_is_cls_and_mean_patch(dino):
    x = torch.randn(2, 3, DINO_SIZE, DINO_SIZE)
    with torch.no_grad():
        t = dino.backbone.vit.forward_features(x)
        f = dino.backbone(x)
    assert t.shape[1] == 1 + 4 + (DINO_SIZE // 14) ** 2            # CLS + 4 registers + patches
    assert torch.equal(f[:, :384], t[:, 0])
    assert torch.allclose(f[:, 384:], t[:, 5:].mean(1))


def test_dino_pretrained_weights_are_loaded():
    """The pinned DINOv2 file really ends up in the backbone (not a silent random init)."""
    from huggingface_hub import try_to_load_from_cache
    from safetensors.torch import load_file
    from vl.model import DINOV2_HF
    f = try_to_load_from_cache(DINOV2_HF[0], "model.safetensors", revision=DINOV2_HF[1])
    if not isinstance(f, str):
        pytest.skip("DINOv2 weights not in the local HF cache")
    sd = load_file(f)
    m = ValueModel(CAMS, pretrained=True, backbone="dinov2_vits14_reg", image_size=224)
    vit = m.backbone.vit.state_dict()
    for k in ["patch_embed.proj.weight", "blocks.0.attn.qkv.weight", "blocks.11.mlp.fc2.weight", "norm.weight", "reg_token"]:
        assert torch.equal(vit[k], sd[k]), k
    assert sd["pos_embed"].shape[1] == 37 * 37 and vit["pos_embed"].shape[1] == 16 * 16   # resampled for 224


def test_run_data_config_reads_the_backbone(tmp_path):
    import json
    from vl.model import run_data_config
    base = {"cameras": CAMS, "image_size": 224}
    (tmp_path / "config.json").write_text(json.dumps({"cameras": CAMS}))      # run from before the option
    assert run_data_config(base, tmp_path)["backbone"] == "resnet18"
    (tmp_path / "config.json").write_text(json.dumps({"cameras": CAMS, "backbone": "dinov2_vits14_reg"}))
    assert run_data_config(base, tmp_path)["backbone"] == "dinov2_vits14_reg"


def test_layer_decay_groups_cover_the_backbone_once_with_decaying_lrs(dino, model):
    from vl.model import layer_decay_groups
    groups = layer_decay_groups(dino, 1e-5, 0.75)
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(dino.backbone.parameters()))   # each param exactly once
    head = {id(p) for p in dino.head.parameters()}
    assert not head & set(ids)                                                 # head is separate
    lr = {g["layer"]: g["lr"] for g in groups}
    assert sorted(lr) == list(range(14))                                       # embeddings, 12 blocks, norm
    assert lr[13] == pytest.approx(1e-5)                                       # final norm: top LR
    assert lr[12] == pytest.approx(7.5e-6)                                     # last block
    assert lr[0] == pytest.approx(1e-5 * 0.75 ** 13)                           # embeddings
    vit = dino.backbone.vit
    emb = {id(p) for p in [vit.cls_token, vit.reg_token, vit.pos_embed, *vit.patch_embed.parameters()]}
    assert emb == {id(p) for p in next(g for g in groups if g["layer"] == 0)["params"]}
    with pytest.raises(ValueError):
        layer_decay_groups(model, 1e-5, 0.75)                                  # ResNet: not defined
