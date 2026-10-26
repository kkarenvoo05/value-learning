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


def test_run_data_config_reads_crops_and_resize(tmp_path):
    import json
    from vl.model import STAGE3_CROPS, run_data_config
    later = {"wrist_left": [160, 0, 640, 480], "side_1": [140, 0, 560, 420]}    # data.yaml after a Gate 0 change
    base = {"cameras": CAMS, "image_size": 224, "crops": later}
    (tmp_path / "config.json").write_text(json.dumps({"cameras": CAMS}))      # run from before the option
    c = run_data_config(base, tmp_path)
    assert c["crops"] == STAGE3_CROPS and c["resize"] == {}                    # what it was trained on, not data.yaml
    rec = {"cameras": CAMS, "crops": later, "resize": {"side_1": "letterbox"}}
    (tmp_path / "config.json").write_text(json.dumps(rec))
    c = run_data_config(dict(base, crops=STAGE3_CROPS), tmp_path)
    assert c["crops"] == later and c["resize"] == {"side_1": "letterbox"}


def test_run_data_config_reads_the_input_size(tmp_path):
    import json
    from vl.model import run_data_config
    base = {"cameras": CAMS, "image_size": 320}                               # data.yaml after a size change
    (tmp_path / "config.json").write_text(json.dumps({"cameras": CAMS}))      # run from before the option
    assert run_data_config(base, tmp_path)["image_size"] == 224
    (tmp_path / "config.json").write_text(json.dumps({"cameras": CAMS, "image_size": 320}))
    assert run_data_config(dict(base, image_size=224), tmp_path)["image_size"] == 320


def per_camera_model(pretrained=False):
    torch.manual_seed(0)
    return ValueModel(CAMS, pretrained=pretrained, per_camera_backbone=True).eval()


def test_per_camera_backbones_are_distinct_modules():
    """Stage 5.5 C1: one ResNet-18 per camera, no parameter or BatchNorm buffer shared."""
    m = per_camera_model()
    resnets = [mod for mod in m.modules() if mod.__class__.__name__ == "ResNet"]
    assert len(resnets) == 2 and set(m.backbone.keys()) == set(CAMS)
    assert m.head[0].in_features == 2 * 512
    a, b = (dict(m.backbone[c].named_parameters()) for c in CAMS)
    assert all(a[k].data_ptr() != b[k].data_ptr() for k in a)
    ba, bb = (dict(m.backbone[c].named_buffers()) for c in CAMS)
    assert all(ba[k].data_ptr() != bb[k].data_ptr() for k in ba)
    n_shared = sum(p.numel() for p in ValueModel(CAMS, pretrained=False).backbone.parameters())
    assert sum(p.numel() for p in m.backbone.parameters()) == 2 * n_shared


def test_per_camera_backbones_start_from_the_same_pretrained_weights():
    m = per_camera_model(pretrained=True)
    a, b = (m.backbone[c].state_dict() for c in CAMS)
    assert all(torch.equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("cam", CAMS)
def test_per_camera_input_changes_only_its_backbone_features(cam):
    """Changing one camera's images changes that backbone's feature and leaves the other's unchanged."""
    m = per_camera_model()
    x = images()
    y = dict(x, **{cam: images(seed=1)[cam]})
    with torch.no_grad():
        fx = {c: m.backbone[c](m.normalize(x[c])) for c in CAMS}
        fy = {c: m.backbone[c](m.normalize(y[c])) for c in CAMS}
        assert not torch.allclose(m(x), m(y))
    other = next(c for c in CAMS if c != cam)
    assert not torch.allclose(fx[cam], fy[cam])
    assert torch.equal(fx[other], fy[other])


def test_per_camera_forward_is_concat_of_each_backbone_in_camera_order():
    m = per_camera_model()
    x = images()
    with torch.no_grad():
        f = torch.cat([m.backbone[c](m.normalize(x[c])) for c in CAMS], dim=1)
        assert torch.allclose(m(x), m.head(f))
        assert torch.equal(m(x), m({"side_1": x["side_1"], "wrist_left": x["wrist_left"]}))


def test_per_camera_training_updates_each_backbone_from_its_own_camera():
    """Gradients reach both backbones, and each one's BatchNorm statistics come from its own camera."""
    m = per_camera_model().train()
    before = {c: {k: v.clone() for k, v in m.backbone[c].state_dict().items()} for c in CAMS}
    x, y = images(b=4, size=64), torch.tensor([10, 60, 120, 190])
    loss = value_loss(m(x), y)
    loss.backward()
    for c in CAMS:
        assert m.backbone[c].conv1.weight.grad.abs().sum() > 0
    after = {c: m.backbone[c].state_dict() for c in CAMS}
    for c in CAMS:                                   # running stats updated from this camera's batch only
        k = "bn1.running_mean"
        assert not torch.equal(after[c][k], before[c][k])
    assert not torch.equal(after["wrist_left"]["bn1.running_mean"], after["side_1"]["bn1.running_mean"])


def test_run_data_config_reads_per_camera_flag(tmp_path):
    import json
    from vl.model import run_data_config
    base = {"cameras": CAMS, "proprio": {"enabled": False}}
    (tmp_path / "config.json").write_text(json.dumps({"per_camera_backbone": True}))
    assert run_data_config(base, tmp_path)["per_camera_backbone"] is True
    (tmp_path / "config.json").write_text(json.dumps({}))       # runs from before C1: shared
    assert run_data_config(base, tmp_path)["per_camera_backbone"] is False


# Stage 5.5 B2: one past frame per camera through the same backbone, masked without a past.
def history_model():
    torch.manual_seed(0)
    return ValueModel(CAMS, pretrained=False, history=True).eval()


def past(x, has):
    return {"images": x, "has_past": torch.tensor(has, dtype=torch.float32)}


def test_history_head_sees_current_past_and_flag():
    m = history_model()
    assert m.head[0].in_features == 2 * 2 * 512 + 1
    assert len([k for k in m.modules() if k.__class__.__name__ == "ResNet"]) == 1   # shared across time
    x, xp = images(), images(seed=1)
    with torch.no_grad():
        f = torch.cat([m.features(x), m.features(xp), torch.ones(2, 1)], dim=1)
        assert torch.allclose(m(x, past=past(xp, [1, 1])), m.head(f), atol=1e-5)


def test_history_past_frame_changes_the_output_only_when_present():
    m = history_model()
    x, a, b = images(), images(seed=1), images(seed=2)
    with torch.no_grad():
        assert not torch.allclose(m(x, past=past(a, [1, 1])), m(x, past=past(b, [1, 1])))
        # has_past = 0: the placeholder image is ignored, the head gets zeros + flag 0
        assert torch.allclose(m(x, past=past(a, [0, 0])), m(x, past=past(b, [0, 0])), atol=1e-6)
        f = torch.cat([m.features(x), torch.zeros(2, 2 * 512 + 1)], dim=1)
        assert torch.allclose(m(x, past=past(a, [0, 0])), m.head(f), atol=1e-5)
        # the mask is per item
        mixed, ref = m(x, past=past(a, [1, 0])), m(x, past=past(b, [1, 0]))
        assert not torch.allclose(mixed[0], ref[0]) and torch.allclose(mixed[1], ref[1], atol=1e-6)


def test_history_presence_must_match_the_model(model):
    with pytest.raises(ValueError):
        history_model()(images())
    with pytest.raises(ValueError):
        model(images(), past=past(images(), [1, 1]))


def test_history_gradients_reach_the_backbone_through_the_past_frame():
    m = history_model().train()
    x, xp, y = images(b=4), images(b=4, seed=1), torch.tensor([10, 60, 120, 190])
    value_loss(m(x, past=past(xp, [1, 1, 1, 1])), y).backward()
    g_both = m.backbone.conv1.weight.grad.clone()
    m.zero_grad()
    value_loss(m(x, past=past(xp, [0, 0, 0, 0])), y).backward()
    assert not torch.allclose(g_both, m.backbone.conv1.weight.grad)


def test_model_inputs_and_run_data_config_carry_history(tmp_path):
    import json
    from vl.model import model_inputs, run_data_config
    b = {"images": images(), "past": past(images(seed=1), [1, 0])}
    inp = model_inputs(b, "cpu")
    assert inp["proprio"] is None and torch.equal(inp["past"]["has_past"], torch.tensor([1.0, 0.0]))
    assert "past" not in model_inputs({"images": images()}, "cpu")
    base = {"cameras": CAMS, "proprio": {"enabled": False}}
    (tmp_path / "config.json").write_text(json.dumps({"history": {"gap_s": 2.0}}))
    assert run_data_config(base, tmp_path)["history"] == {"gap_s": 2.0}
    (tmp_path / "config.json").write_text(json.dumps({}))       # runs from before B2: no history
    assert run_data_config(base, tmp_path)["history"] is None


# Stage 5.5 B3 follow-up: seconds still as one extra head input (log1p), an upper bound for history.
def test_still_time_head_gets_log1p_seconds():
    torch.manual_seed(0)
    m = ValueModel(CAMS, pretrained=False, still_time=True).eval()
    assert m.head[0].in_features == 2 * 512 + 1
    x, s = images(), torch.tensor([[0.0], [12.0]])
    with torch.no_grad():
        f = torch.cat([m.features(x), torch.log1p(s)], dim=1)
        assert torch.allclose(m(x, still=s), m.head(f), atol=1e-5)
        assert not torch.allclose(m(x, still=s), m(x, still=torch.zeros(2, 1)))
    with pytest.raises(ValueError):
        m(x)
    with pytest.raises(ValueError):
        ValueModel(CAMS, pretrained=False).eval()(x, still=s)


def test_model_inputs_and_run_data_config_carry_still_time(tmp_path):
    import json
    from vl.model import model_inputs, run_data_config
    assert torch.equal(model_inputs({"images": images(), "still": torch.ones(2, 1)}, "cpu")["still"], torch.ones(2, 1))
    assert "still" not in model_inputs({"images": images()}, "cpu")
    base = {"cameras": CAMS, "proprio": {"enabled": False}, "still_time": {"enabled": False, "still_mps": 0.01}}
    (tmp_path / "config.json").write_text(json.dumps({"still_time": True}))
    assert run_data_config(base, tmp_path)["still_time"] == {"enabled": True, "still_mps": 0.01}
    (tmp_path / "config.json").write_text(json.dumps({}))       # earlier runs: off
    assert run_data_config(base, tmp_path)["still_time"]["enabled"] is False


# Stage 5.5 C3: attention fusion head. Camera features -> tokens, a learned readout attends to them.
def fusion_head(n_time=1):
    from vl.model import TokenFusionHead
    torch.manual_seed(0)
    return TokenFusionHead(2, 512, 201, n_time=n_time).eval()


def test_fusion_head_output_shape():
    with torch.no_grad():
        assert fusion_head()(torch.randn(4, 1024)).shape == (4, 201)


def test_fusion_attention_is_a_distribution_over_image_tokens():
    with torch.no_grad():
        _, attn = fusion_head()(torch.randn(4, 1024), return_attn=True)
    assert attn.shape == (4, 2, 2)                      # (B, layers, tokens)
    assert (attn >= 0).all() and torch.allclose(attn.sum(-1), torch.ones(4, 2), atol=1e-5)


def test_fusion_head_knows_which_camera_is_which():
    h, f = fusion_head(), torch.randn(4, 1024)
    with torch.no_grad():
        assert not torch.allclose(h(f), h(torch.cat([f[:, 512:], f[:, :512]], dim=1)))


def test_fusion_attention_depends_on_the_frame():
    h = fusion_head()
    with torch.no_grad():
        _, a = h(torch.randn(4, 1024), return_attn=True)
        _, b = h(torch.randn(4, 1024), return_attn=True)
    assert not torch.allclose(a, b)


def test_fusion_masked_token_is_ignored():
    h, f, g = fusion_head(n_time=2), torch.randn(4, 2048), torch.randn(4, 1024)
    mask = torch.tensor([[False, False, True, True]] * 4)          # time-major: [now cams, past cams]
    with torch.no_grad():
        out, attn = h(f, key_padding_mask=mask, return_attn=True)
        out2 = h(torch.cat([f[:, :1024], g], dim=1), key_padding_mask=mask)
    assert torch.allclose(out, out2, atol=1e-6)
    assert torch.all(attn[..., 2:] == 0)


def test_fusion_fully_masked_row_is_refused():
    mask = torch.tensor([[False, False, True, True], [True, True, True, True]])
    with pytest.raises(ValueError):
        fusion_head(n_time=2)(torch.randn(2, 2048), key_padding_mask=mask)


def test_fusion_gradients_reach_every_parameter():
    h = fusion_head().train()
    h(torch.randn(4, 1024)).sum().backward()
    for name, p in h.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name


# Stage 5.5 C3 wired into ValueModel: fusion="attention" replaces the concat head.
def attention_model():
    torch.manual_seed(0)
    return ValueModel(CAMS, pretrained=False, per_camera_backbone=True, fusion="attention").eval()


def test_attention_fusion_model_uses_the_token_head():
    from vl.model import TokenFusionHead
    m = attention_model()
    assert m.fusion == "attention" and isinstance(m.head, TokenFusionHead)
    with torch.no_grad():
        assert m(images()).shape == (2, 201)


@pytest.mark.parametrize("cam", CAMS)
def test_attention_fusion_each_camera_changes_the_output(cam):
    m, x = attention_model(), images()
    with torch.no_grad():
        assert not torch.allclose(m(x), m(dict(x, **{cam: images(seed=1)[cam]})))


def test_attention_fusion_fits_one_batch():
    m = attention_model().train()
    x, y = images(b=4), torch.tensor([10, 60, 120, 190])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    losses = []
    for _ in range(15):
        loss = value_loss(m(x), y)
        opt.zero_grad()
        loss.backward()
        losses.append(loss.item())
        opt.step()
    assert losses[-1] < 0.5 * losses[0]


def test_attention_fusion_refuses_unbuilt_inputs():
    for kw in ({"history": True}, {"still_time": True}, {"proprio_mean": [0.0] * 10, "proprio_std": [1.0] * 10}):
        with pytest.raises(ValueError):
            ValueModel(CAMS, pretrained=False, fusion="attention", **kw)
    with pytest.raises(ValueError):
        ValueModel(CAMS, pretrained=False, fusion="sum")


def test_run_data_config_reads_fusion(tmp_path):
    import json
    from vl.model import run_data_config
    base = {"cameras": CAMS, "proprio": {"enabled": False}}
    (tmp_path / "config.json").write_text(json.dumps({"fusion": "attention"}))
    assert run_data_config(base, tmp_path)["fusion"] == "attention"
    (tmp_path / "config.json").write_text(json.dumps({}))       # runs from before C3: concat
    assert run_data_config(base, tmp_path)["fusion"] == "concat"


# Stage 5.5 C3 regularised: smaller head (matched to C1's concat head) and dropout, via fusion_cfg.
SMALL = {"d": 128, "n_layers": 1, "ff": 256}


def test_small_fusion_head_is_about_the_size_of_the_concat_head():
    from vl.model import TokenFusionHead
    n = sum(p.numel() for p in TokenFusionHead(2, 512, 201, **SMALL).parameters())
    concat = sum(p.numel() for p in ValueModel(CAMS, pretrained=False).head.parameters())
    assert 0.8 * concat < n < 1.2 * concat, (n, concat)


def test_fusion_dropout_is_active_in_train_and_off_in_eval():
    from vl.model import TokenFusionHead
    torch.manual_seed(0)
    h, f = TokenFusionHead(2, 512, 201, dropout=0.1, **SMALL), torch.randn(4, 1024)
    h.train()
    assert not torch.allclose(h(f), h(f))
    h.eval()
    with torch.no_grad():
        assert torch.allclose(h(f), h(f))
        _, attn = h(f, return_attn=True)
    assert torch.allclose(attn.sum(-1), torch.ones(4, 1), atol=1e-5)


def test_value_model_passes_fusion_cfg_to_the_head():
    m = ValueModel(CAMS, pretrained=False, per_camera_backbone=True, fusion="attention",
                   fusion_cfg=dict(SMALL, dropout=0.1))
    assert len(m.head.blocks) == 1 and m.head.proj.out_features == 128
    assert m.head.blocks[0].attn.dropout == 0.1
    with pytest.raises(ValueError):
        ValueModel(CAMS, pretrained=False, fusion="concat", fusion_cfg={"d": 128})   # concat has no such knobs


def test_run_data_config_reads_fusion_cfg(tmp_path):
    import json
    from vl.model import run_data_config
    base = {"cameras": CAMS, "proprio": {"enabled": False}}
    (tmp_path / "config.json").write_text(json.dumps({"fusion": "attention", "fusion_cfg": {"d": 128}}))
    assert run_data_config(base, tmp_path)["fusion_cfg"] == {"d": 128}
    (tmp_path / "config.json").write_text(json.dumps({"fusion": "attention"}))   # c3_attn: head defaults
    assert run_data_config(base, tmp_path)["fusion_cfg"] == {}


def test_default_fusion_head_keys_match_the_c3_attn_checkpoints():
    """Regularisation knobs must not rename parameters: c3_attn_s{0,1,2}/best.pt load strict."""
    from vl.model import TokenFusionHead
    keys = set(TokenFusionHead(2, 512, 201).state_dict())
    assert {"blocks.0.ff.0.weight", "blocks.0.ff.2.weight", "blocks.1.ff.2.bias", "out.3.weight"} <= keys
    assert not any(k.startswith("blocks.0.ff.3") for k in keys)


# Camera dropout (wrist-optimism follow-up to C4 [ours]): training-only, per-sample zeroing of one
# camera's backbone feature before the head.

def test_camera_dropout_zeroes_only_that_cameras_block_per_sample():
    torch.manual_seed(0)
    m = ValueModel(CAMS, pretrained=False, cam_dropout={"wrist_left": 0.5}).train()
    f = torch.ones(4000, 2 * 512)
    out = m.drop_cameras(f)
    wrist, side = out[:, :512], out[:, 512:]
    assert torch.equal(side, f[:, 512:])
    row_zero = (wrist == 0).all(1)
    assert torch.equal(row_zero | (wrist == 1).all(1), torch.ones(4000, dtype=torch.bool))   # whole block or nothing
    assert abs(row_zero.float().mean().item() - 0.5) < 0.03


def test_camera_dropout_is_off_in_eval():
    torch.manual_seed(0)
    m = ValueModel(CAMS, pretrained=False, cam_dropout={"wrist_left": 1.0}).eval()
    ref = ValueModel(CAMS, pretrained=False).eval()
    ref.load_state_dict(m.state_dict())               # no new parameters or buffers
    x = images()
    with torch.no_grad():
        assert torch.equal(m(x), ref(x))


def test_camera_dropout_p1_in_train_feeds_the_head_zeros_for_that_camera():
    torch.manual_seed(0)
    m = ValueModel(CAMS, pretrained=False, cam_dropout={"wrist_left": 1.0}).train()
    m.backbone.eval()       # train-mode BatchNorm normalizes wrist + side together, which couples the cameras
    x = images(b=4)
    with torch.no_grad():
        f = m.features(x)
        f[:, :512] = 0
        assert torch.allclose(m(x), m.head(f), atol=1e-5)
        x2 = dict(x, wrist_left=images(b=4, seed=1)["wrist_left"])
        assert torch.allclose(m(x2), m(x), atol=1e-5)  # the wrist image no longer reaches the head


def test_camera_dropout_refuses_bad_settings():
    for kw in ({"cam_dropout": {"wrist_right": 0.2}}, {"cam_dropout": {"wrist_left": 1.5}},
               {"cam_dropout": {"wrist_left": 0.2}, "history": True},
               {"cam_dropout": {"wrist_left": 0.2}, "fusion": "attention"}):
        with pytest.raises(ValueError):
            ValueModel(CAMS, pretrained=False, **kw)


def test_run_data_config_reads_cam_dropout(tmp_path):
    import json
    from vl.model import build_model, run_data_config
    base = {"cameras": CAMS, "proprio": {"enabled": False}, "imagenet_mean": [0.485, 0.456, 0.406],
            "imagenet_std": [0.229, 0.224, 0.225], "image_size": 224}
    (tmp_path / "config.json").write_text(json.dumps({"cam_dropout": {"wrist_left": 0.25}}))
    cfg = run_data_config(base, tmp_path)
    assert cfg["cam_dropout"] == {"wrist_left": 0.25}
    assert build_model(cfg, {"n_bins": 201, "v_min": -1.0, "v_max": 0.0}, pretrained=False).cam_dropout == {"wrist_left": 0.25}
    (tmp_path / "config.json").write_text(json.dumps({}))       # earlier runs: none
    assert run_data_config(base, tmp_path)["cam_dropout"] == {}
