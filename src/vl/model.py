"""State-value model V(o): two camera images -> distribution over 201 value bins.

    wrist_left ─┐                        ┌─ 512 ─┐
                ├─ shared ResNet-18 ─────┤       ├─ concat 1024 → 256 → ReLU → 201 logits
    side_1 ─────┘   (ImageNet weights)   └─ 512 ─┘
    proprio (optional, 10) ── z-score ──────── 10 ─┘   (Stage 5 ablation 2: concat 1034 → 256 → 201)

- Categorical head [paper: RECAP, B = 201 bins on the normalized value range]; architecture
  details (shared backbone, MLP sizes) are ours.
- Trained with cross-entropy against the hard target bin (`vl.targets`).
- Scalar prediction V = Σ_b p_b · v_b, the expected value under the predicted distribution,
  with v_b the 201 bin centers evenly spaced on [-1, 0].
- Input is the uint8 images (as `vl.dataset` produces them); ImageNet normalization happens
  inside `forward`, so a caller cannot forget it or apply it twice. Optional raw proprio
  [ours]: z-scored inside `forward` with training-split statistics stored as buffers, then
  concatenated with the image features; no separate proprio encoder (simplest version).
- Backbone (Stage 5 "stronger backbone" [ours]): `resnet18` (default, the baseline) or
  `dinov2_vits14_reg`, DINOv2 ViT-S/14 with 4 registers (timm weights pinned to a commit). Its
  per-camera feature is [CLS token, mean of the patch tokens] (2 × 384 = 768), the readout of
  DINOv2's linear evaluation. Same ImageNet mean/std, same 224×224 input (16×16 patches; timm
  resamples the 37×37 position embeddings). LayerNorm only, so no BatchNorm statistics are shared
  between the two cameras.
- Per-camera backbones (Stage 5.5 C1 [ours; Mulligan's Insert Marker critic uses one ResNet-18 per
  camera]): `per_camera_backbone=True` gives each camera its own copy of the backbone, each from the
  same pretrained weights, then the same concat -> head. Nothing is shared between cameras, including
  BatchNorm statistics (the shared ResNet normalizes over the wrist + side batch together).
  `self.backbone` is then an nn.ModuleDict keyed by camera.
- History (Stage 5.5 B2 [ours; Lars: share the backbone across time steps]): `history=True` adds one
  past frame per camera (fixed gap, `vl.dataset` `history`). The same backbone encodes the current
  and the past images; the head input is [current features (cams), has_past · past features (cams),
  has_past] (2 × 1024 + 1 for ResNet-18). Without a past (has_past = 0) the past features are zeroed,
  so the head sees zeros plus the flag whatever the placeholder image is.
- Still time (Stage 5.5 B3 follow-up [ours]): `still_time=True` appends log1p(seconds the gripper has
  been still) (`vl.dataset` `still_time`) to the head input, one extra number: an upper bound for
  what image history can tell the head about a stall.
- Attention fusion (Stage 5.5 C3 [Lars: one token per image + a learnable readout; sizes ours]):
  `fusion="attention"` replaces the concat MLP with `TokenFusionHead`: each camera feature becomes a
  256-d token with a learned camera embedding, a readout token attends to them (2 cross-attention
  blocks), readout -> 256 -> 201. Same `head(f)` input, so features / optimizer groups are unchanged.
  Images only for now; its time embedding and padding mask are there for the history variant.
  `fusion_cfg` (regularised C3 arms [ours]) overrides the head sizes (d, n_layers, ff, n_heads) and
  adds dropout (on image tokens, attention weights and both residual branches); {} = the defaults.
- Camera dropout (wrist-optimism follow-up to C4 [ours]): `cam_dropout={cam: p}` zeroes that camera's
  backbone feature for each training sample with probability p, before the head (no rescaling); off in
  eval, so evaluation and checkpoints are unchanged. Concat head without history only.

Sanity check on a real batch (CPU):
    .venv/bin/python -m vl.model
"""

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from vl.config import load_config


BACKBONES = ("resnet18", "dinov2_vits14_reg")
FUSIONS = ("concat", "attention")
DINOV2_HF = ("timm/vit_small_patch14_reg4_dinov2.lvd142m", "c04b5193082a8d5b0c4856c7937384a48136c5de")


class DinoV2Features(nn.Module):
    """DINOv2 ViT-S/14 with registers -> (B, 768) = [CLS, mean of patch tokens] after the final norm."""

    def __init__(self, pretrained: bool, image_size: int):
        super().__init__()
        import timm
        overlay = {}
        if pretrained:
            from huggingface_hub import hf_hub_download
            repo, rev = DINOV2_HF
            overlay = {"file": hf_hub_download(repo, "model.safetensors", revision=rev)}
        self.vit = timm.create_model("vit_small_patch14_reg4_dinov2", pretrained=pretrained, num_classes=0,
                                     img_size=image_size, pretrained_cfg_overlay=overlay)
        self.out_dim = 2 * self.vit.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.vit.forward_features(x)                  # (B, 1 CLS + 4 registers + 256 patches, 384)
        return torch.cat([t[:, 0], t[:, self.vit.num_prefix_tokens:].mean(1)], dim=1)


def layer_decay_groups(model: "ValueModel", lr: float, decay: float) -> list[dict]:
    """Layer-wise LR decay for the DINOv2 backbone (BEiT / DINOv2-style fine-tuning) [ours].
    Layer ids: embeddings (patch_embed, cls/reg tokens, pos_embed) 0, block i -> i + 1, final norm
    depth + 1. Parameters of layer l get lr * decay ** (depth + 1 - l): the final norm gets `lr`,
    the last block lr * decay, the embeddings lr * decay ** (depth + 1). The head is not included."""
    if model.backbone_name != "dinov2_vits14_reg" or model.per_camera_backbone:
        raise ValueError(f"layer decay is only defined for one shared ViT backbone, not {model.backbone_name}"
                         f"{' per camera' if model.per_camera_backbone else ''}")
    vit = model.backbone.vit
    depth = len(vit.blocks)
    by_layer: dict[int, list] = {}
    for name, p in vit.named_parameters():
        if name.startswith("blocks."):
            layer = int(name.split(".")[1]) + 1
        elif name.startswith(("patch_embed.", "cls_token", "reg_token", "pos_embed")):
            layer = 0
        else:                                             # final norm
            layer = depth + 1
        by_layer.setdefault(layer, []).append(p)
    return [{"params": ps, "lr": lr * decay ** (depth + 1 - layer), "layer": layer}
            for layer, ps in sorted(by_layer.items())]


class ReadoutBlock(nn.Module):
    """Pre-norm cross-attention block: only the readout is a query, image tokens are keys/values and
    are not updated. Returns the new readout and its attention over the tokens (mean over heads)."""

    def __init__(self, d: int, n_heads: int, ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm_q, self.norm_kv, self.norm_ff = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))   # indices fixed: c3_attn checkpoints
        self.drop = nn.Dropout(dropout)

    def forward(self, q, tokens, key_padding_mask=None):
        kv = self.norm_kv(tokens)
        a, w = self.attn(self.norm_q(q), kv, kv, key_padding_mask=key_padding_mask,
                         need_weights=True, average_attn_weights=True)
        q = q + self.drop(a)
        return q + self.drop(self.ff(self.norm_ff(q))), w[:, 0]          # (B, 1, d), (B, n_tokens)


class TokenFusionHead(nn.Module):
    """Stage 5.5 C3 attention fusion [Lars: one token per image, a learnable readout attends to all
    image tokens; sizes ours]. Input is the same concatenated feature vector the concat head takes,
    laid out time-major ([now cams..., past cams...], cameras in model order). Each feature -> Linear
    -> d-dim token + learned camera embedding + learned time embedding; a learned readout token goes
    through `n_layers` ReadoutBlocks; the readout -> LayerNorm -> d -> hidden -> ReLU -> n_bins.
    `key_padding_mask` (B, n_tokens), True = ignore (a missing past frame); a row may not mask every
    token. `return_attn` also returns the readout attention per layer, (B, n_layers, n_tokens)."""

    def __init__(self, n_tokens_per_time: int, feat: int, n_bins: int, n_time: int = 1, d: int = 256,
                 n_heads: int = 4, n_layers: int = 2, ff: int = 512, hidden: int = 256, dropout: float = 0.0):
        super().__init__()
        self.n_cams, self.feat, self.n_time = n_tokens_per_time, feat, n_time
        self.proj = nn.Linear(feat, d)
        self.cam_emb = nn.Parameter(0.02 * torch.randn(n_tokens_per_time, d))
        self.time_emb = nn.Parameter(0.02 * torch.randn(n_time, d))
        self.readout = nn.Parameter(0.02 * torch.randn(1, 1, d))
        self.blocks = nn.ModuleList(ReadoutBlock(d, n_heads, ff, dropout) for _ in range(n_layers))
        self.drop = nn.Dropout(dropout)                   # on the image tokens
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, n_bins))

    def forward(self, f: torch.Tensor, key_padding_mask: torch.Tensor | None = None, return_attn: bool = False):
        n_tok = self.n_time * self.n_cams
        if f.shape[1] != n_tok * self.feat:
            raise ValueError(f"expected {n_tok} x {self.feat} features, got {f.shape[1]}")
        if key_padding_mask is not None and key_padding_mask.all(dim=1).any():
            raise ValueError("key_padding_mask masks every token of a row")
        B = len(f)
        emb = self.cam_emb.repeat(self.n_time, 1) + self.time_emb.repeat_interleave(self.n_cams, dim=0)
        tokens = self.drop(self.proj(f.view(B, n_tok, self.feat)) + emb)
        q, attn = self.readout.expand(B, -1, -1), []
        for blk in self.blocks:
            q, w = blk(q, tokens, key_padding_mask)
            attn.append(w)
        logits = self.out(q[:, 0])
        return (logits, torch.stack(attn, dim=1)) if return_attn else logits


class ValueModel(nn.Module):
    def __init__(self, cams: list[str], n_bins: int = 201, v_min: float = -1.0, v_max: float = 0.0,
                 hidden: int = 256, pretrained: bool = True,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225),
                 proprio_mean=None, proprio_std=None, backbone: str = "resnet18", image_size: int = 224,
                 per_camera_backbone: bool = False, history: bool = False, still_time: bool = False,
                 fusion: str = "concat", fusion_cfg: dict | None = None, cam_dropout: dict | None = None):
        super().__init__()
        self.cams = list(cams)
        if backbone not in BACKBONES:
            raise ValueError(f"backbone {backbone!r} not in {BACKBONES}")
        if fusion not in FUSIONS:
            raise ValueError(f"fusion {fusion!r} not in {FUSIONS}")
        if fusion == "attention" and (history or still_time or proprio_mean is not None):
            raise ValueError("attention fusion takes images only (history / still time / proprio not built)")
        if fusion_cfg and fusion != "attention":
            raise ValueError(f"fusion_cfg {fusion_cfg} is only for attention fusion")
        self.fusion = fusion
        self.cam_dropout = dict(cam_dropout or {})
        if self.cam_dropout:
            if fusion != "concat" or history:
                raise ValueError("cam_dropout is built for the concat head without history only")
            for c, p in self.cam_dropout.items():
                if c not in self.cams or not 0.0 <= p <= 1.0:
                    raise ValueError(f"cam_dropout {c}={p}: camera not in {self.cams} or p outside [0, 1]")

        def make():
            if backbone == "resnet18":
                weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
                net = torchvision.models.resnet18(weights=weights)
                feat = net.fc.in_features     # 512
                net.fc = nn.Identity()        # keep the pooled 512-d feature, drop the 1000 classes
                return net, feat
            net = DinoV2Features(pretrained, image_size)
            return net, net.out_dim           # 768

        self.backbone_name = backbone
        self.per_camera_backbone = per_camera_backbone
        if per_camera_backbone:               # one backbone per camera (C1)
            nets = {c: make() for c in self.cams}
            self.backbone = nn.ModuleDict({c: n for c, (n, _) in nets.items()})
            feat = nets[self.cams[0]][1]
        else:
            net, feat = make()
            self.backbone = net               # one backbone, shared by both cameras
        self.proprio_dim = 0 if proprio_mean is None else len(proprio_mean)
        if self.proprio_dim:
            self.register_buffer("proprio_mean", torch.tensor(proprio_mean, dtype=torch.float32))
            self.register_buffer("proprio_std", torch.tensor(proprio_std, dtype=torch.float32))
        self.history = history
        self.still_time = still_time
        n_img = feat * len(self.cams) * (2 if history else 1) + (1 if history else 0)
        if fusion == "attention":             # C3: same head(f) interface, attention readout over cameras
            self.head = TokenFusionHead(len(self.cams), feat, n_bins, hidden=hidden, **(fusion_cfg or {}))
        else:
            self.head = nn.Sequential(
                nn.Linear(n_img + self.proprio_dim + int(still_time), hidden),
                nn.ReLU(),
                nn.Linear(hidden, n_bins),
            )
        self.register_buffer("bin_centers", torch.linspace(v_min, v_max, n_bins))
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))
        self.backbone_frozen = False

    def freeze_backbone(self) -> "ValueModel":
        """Stage 5 ablation 3: keep the pretrained backbone fixed, train only the head [ours].
        Frozen means no weight updates *and* no BatchNorm running-statistic updates, so the
        backbone stays in eval mode even when the model is put in train mode."""
        self.backbone.requires_grad_(False)
        self.backbone_frozen = True
        return self.train(self.training)

    def train(self, mode: bool = True) -> "ValueModel":
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self

    def features(self, images: dict[str, torch.Tensor]) -> torch.Tensor:
        """{cam: uint8 (B, 3, H, W)} -> (B, n_cams * feat), camera order fixed."""
        if self.per_camera_backbone:
            return torch.cat([self.backbone[c](self.normalize(images[c])) for c in self.cams], dim=1)
        f = self.backbone(self.normalize(torch.cat([images[c] for c in self.cams])))  # (n_cams * B, feat)
        return torch.cat(f.chunk(len(self.cams)), dim=1)

    def forward(self, images: dict[str, torch.Tensor], proprio: torch.Tensor | None = None,
                past: dict | None = None, still: torch.Tensor | None = None) -> torch.Tensor:
        """images: {cam: uint8 (B, 3, H, W)}, proprio: raw float (B, 10) iff the model has a
        proprio input, past: {"images": {cam: uint8 (B, 3, H, W)}, "has_past": (B,)} iff the model has
        history, still: raw seconds (B, 1) iff the model has still_time -> logits (B, n_bins)."""
        if (proprio is None) != (self.proprio_dim == 0):
            raise ValueError(f"model proprio_dim={self.proprio_dim} but proprio {'missing' if proprio is None else 'given'}")
        if (past is None) == self.history:
            raise ValueError(f"model history={self.history} but past {'missing' if past is None else 'given'}")
        if (still is None) == self.still_time:
            raise ValueError(f"model still_time={self.still_time} but still {'missing' if still is None else 'given'}")
        if self.history:
            B = len(images[self.cams[0]])
            both = {c: torch.cat([images[c], past["images"][c]]) for c in self.cams}   # one backbone pass
            f_now, f_past = self.features(both).split(B)
            m = past["has_past"].to(f_now.dtype).view(B, 1)
            f = torch.cat([f_now, f_past * m, m], dim=1)
        else:
            f = self.drop_cameras(self.features(images))
        if self.proprio_dim:
            z = (proprio.float() - self.proprio_mean) / self.proprio_std
            f = torch.cat([f, z.to(f.dtype)], dim=1)                # (B, n_cams * 512 + 10)
        if self.still_time:
            f = torch.cat([f, torch.log1p(still.float()).to(f.dtype)], dim=1)
        return self.head(f)

    def drop_cameras(self, f: torch.Tensor) -> torch.Tensor:
        """Training only: zero camera c's feature block of each row with probability cam_dropout[c]."""
        if not (self.training and self.cam_dropout):
            return f
        blocks = list(f.chunk(len(self.cams), dim=1))
        for i, c in enumerate(self.cams):
            if c in self.cam_dropout:
                keep = torch.rand(len(f), 1, device=f.device) >= self.cam_dropout[c]
                blocks[i] = blocks[i] * keep.to(f.dtype)
        return torch.cat(blocks, dim=1)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """uint8 images -> ImageNet-normalized float."""
        return (x.float() / 255.0 - self.mean) / self.std

    def expected_value(self, logits: torch.Tensor) -> torch.Tensor:
        """V = Σ_b softmax(logits)_b · v_b, shape (B,)."""
        return F.softmax(logits.float(), dim=-1) @ self.bin_centers


def value_loss(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against the hard target bin (mean over the batch)."""
    return F.cross_entropy(logits.float(), bins)


def build_model(data_cfg: dict, target_cfg: dict, pretrained: bool = True) -> ValueModel:
    pm = ps = None
    if data_cfg.get("proprio", {}).get("enabled"):
        st = json.loads(Path(data_cfg["proprio"]["stats"]).read_text())
        assert st["split"] == "train", st["split"]
        pm, ps = st["mean"], st["std"]
    # DINOv2 was trained with the same ImageNet mean/std; a backbone that isn't needs its own entry.
    return ValueModel(data_cfg["cameras"], target_cfg["n_bins"], target_cfg["v_min"], target_cfg["v_max"],
                      pretrained=pretrained, mean=data_cfg["imagenet_mean"], std=data_cfg["imagenet_std"],
                      proprio_mean=pm, proprio_std=ps, backbone=data_cfg.get("backbone", "resnet18"),
                      image_size=data_cfg["image_size"],
                      per_camera_backbone=bool(data_cfg.get("per_camera_backbone", False)),
                      history=(data_cfg.get("history") or {}).get("gap_s") is not None,
                      still_time=bool((data_cfg.get("still_time") or {}).get("enabled")),
                      fusion=data_cfg.get("fusion", "concat"), fusion_cfg=data_cfg.get("fusion_cfg") or {},
                      cam_dropout=data_cfg.get("cam_dropout") or {})


# The crop boxes every run used before runs recorded their own (Stage 3, configs/data.yaml until
# Gate 0). A run without "crops" in its config.json gets these, not whatever data.yaml says later.
STAGE3_CROPS = {"wrist_left": [80, 0, 560, 480], "side_1": [140, 0, 560, 420]}


def run_data_config(data_cfg: dict, run_dir: Path) -> dict:
    """The data config a finished run was trained with: its cameras, backbone, proprio switch, crop
    boxes and resize modes from the run's config.json (camera / backbone / proprio / Gate 0
    ablations, C1 per-camera backbones) and input size, everything else from `data_cfg`. Runs from before the backbone option
    are ResNet-18; runs from before the crop/resize option get STAGE3_CROPS with mode crop; runs from
    before the input-size option are 224."""
    meta = json.loads((Path(run_dir) / "config.json").read_text())
    cfg = dict(data_cfg, cameras=meta.get("cameras", data_cfg["cameras"]),
               backbone=meta.get("backbone", "resnet18"),
               per_camera_backbone=meta.get("per_camera_backbone", False), fusion=meta.get("fusion", "concat"),
               fusion_cfg=meta.get("fusion_cfg") or {}, cam_dropout=meta.get("cam_dropout") or {},
               crops=meta.get("crops", STAGE3_CROPS), resize=meta.get("resize", {}),
               image_size=meta.get("image_size", 224), history=meta.get("history"))
    if meta.get("proprio"):
        cfg["proprio"] = dict(data_cfg["proprio"], enabled=True)
    cfg["still_time"] = dict(data_cfg.get("still_time") or {}, enabled=bool(meta.get("still_time")))
    return cfg


def model_inputs(b: dict, device) -> dict:
    """A batch's model inputs on `device`, as keyword arguments: model(**model_inputs(b, device))."""
    inp = {"images": {c: v.to(device, non_blocking=True) for c, v in b["images"].items()},
           "proprio": b["proprio"].to(device, non_blocking=True) if "proprio" in b else None}
    if "past" in b:
        inp["past"] = {"images": {c: v.to(device, non_blocking=True) for c, v in b["past"]["images"].items()},
                       "has_past": b["past"]["has_past"].to(device, non_blocking=True)}
    if "still" in b:
        inp["still"] = b["still"].to(device, non_blocking=True)
    return inp


def main():
    from torch.utils.data import DataLoader
    from vl.dataset import FrameDataset, load_split

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--backbone", choices=BACKBONES, default="resnet18")
    args = ap.parse_args()
    dcfg, tcfg = load_config("configs/data.yaml"), load_config("configs/targets.yaml")
    dcfg["backbone"] = args.backbone
    torch.manual_seed(0)
    model = build_model(dcfg, tcfg).eval()
    n_backbone = sum(p.numel() for p in model.backbone.parameters())
    n_head = sum(p.numel() for p in model.head.parameters())
    print(f"parameters: backbone {n_backbone:,} + head {n_head:,} = {n_backbone + n_head:,}")
    ds = FrameDataset(load_split(dcfg, "train"), dcfg)
    b = next(iter(DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                             generator=torch.Generator().manual_seed(0))))
    with torch.no_grad():
        logits = model(b["images"])
        p = F.softmax(logits, -1)
        V = model.expected_value(logits)
        ce = value_loss(logits, b["bin"])
    print(f"batch: { {c: tuple(x.shape) for c, x in b['images'].items()} } -> logits {tuple(logits.shape)}")
    print(f"at initialization: CE {ce:.3f} (uniform = ln {tcfg['n_bins']} = {math.log(tcfg['n_bins']):.3f}), "
          f"max p {p.max():.4f} (uniform = {1 / tcfg['n_bins']:.4f}), "
          f"V mean {V.mean():.3f} range [{V.min():.3f}, {V.max():.3f}] (uniform = -0.5)")
    print(f"targets in batch: v mean {b['value'].mean():.3f}, range [{b['value'].min():.3f}, {b['value'].max():.3f}]")


if __name__ == "__main__":
    main()
