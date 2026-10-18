"""State-value model V(o): two camera images -> distribution over 201 value bins.

    wrist_left ─┐                        ┌─ 512 ─┐
                ├─ shared ResNet-18 ─────┤       ├─ concat 1024 → 256 → ReLU → 201 logits
    side_1 ─────┘   (ImageNet weights)   └─ 512 ─┘

- Categorical head [paper: RECAP, B = 201 bins on the normalized value range]; architecture
  details (shared backbone, MLP sizes) are ours.
- Trained with cross-entropy against the hard target bin (`vl.targets`).
- Scalar prediction V = Σ_b p_b · v_b, the expected value under the predicted distribution,
  with v_b the 201 bin centers evenly spaced on [-1, 0].
- Input is the uint8 images only (as `vl.dataset` produces them); ImageNet normalization happens
  inside `forward`, so a caller cannot forget it or apply it twice.

Sanity check on a real batch (CPU):
    .venv/bin/python -m vl.model
"""

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from vl.config import load_config


class ValueModel(nn.Module):
    def __init__(self, cams: list[str], n_bins: int = 201, v_min: float = -1.0, v_max: float = 0.0,
                 hidden: int = 256, pretrained: bool = True,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        super().__init__()
        self.cams = list(cams)
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = torchvision.models.resnet18(weights=weights)
        feat = backbone.fc.in_features        # 512
        backbone.fc = nn.Identity()           # keep the pooled 512-d feature, drop the 1000 classes
        self.backbone = backbone              # one backbone, shared by both cameras
        self.head = nn.Sequential(
            nn.Linear(feat * len(self.cams), hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_bins),
        )
        self.register_buffer("bin_centers", torch.linspace(v_min, v_max, n_bins))
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, images: dict[str, torch.Tensor]) -> torch.Tensor:
        """images: {cam: uint8 (B, 3, H, W)} -> logits (B, n_bins)."""
        x = torch.cat([images[c] for c in self.cams])               # (n_cams * B, 3, H, W)
        x = (x.float() / 255.0 - self.mean) / self.std
        f = self.backbone(x)                                        # (n_cams * B, 512)
        f = torch.cat(f.chunk(len(self.cams)), dim=1)               # (B, n_cams * 512), cam order fixed
        return self.head(f)

    def expected_value(self, logits: torch.Tensor) -> torch.Tensor:
        """V = Σ_b softmax(logits)_b · v_b, shape (B,)."""
        return F.softmax(logits.float(), dim=-1) @ self.bin_centers


def value_loss(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Cross-entropy against the hard target bin (mean over the batch)."""
    return F.cross_entropy(logits.float(), bins)


def build_model(data_cfg: dict, target_cfg: dict, pretrained: bool = True) -> ValueModel:
    return ValueModel(data_cfg["cameras"], target_cfg["n_bins"], target_cfg["v_min"], target_cfg["v_max"],
                      pretrained=pretrained, mean=data_cfg["imagenet_mean"], std=data_cfg["imagenet_std"])


def main():
    from torch.utils.data import DataLoader
    from vl.dataset import FrameDataset, load_split

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    dcfg, tcfg = load_config("configs/data.yaml"), load_config("configs/targets.yaml")
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
