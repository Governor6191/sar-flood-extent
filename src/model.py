"""Model builder for SAR flood-extent segmentation."""
from __future__ import annotations

import segmentation_models_pytorch as smp
import torch.nn as nn


def build_model(
    arch: str = "unet",
    encoder: str = "resnet34",
    in_channels: int = 2,
    classes: int = 2,
    encoder_weights: str | None = "imagenet",
) -> nn.Module:
    """Build a segmentation model via segmentation_models_pytorch.

    Args:
        arch: Decoder architecture ("unet", "unetplusplus", "fpn", ...).
        encoder: Encoder backbone ("resnet34", "resnet50", ...).
        in_channels: Number of input channels. 2 for VV+VH SAR.
        classes: Number of output classes. 2 for {dry, water}.
        encoder_weights: Pretrained encoder weights, or None. "imagenet"
            adapts the 3-channel pretrained stem to in_channels.

    Returns:
        nn.Module producing (B, classes, H, W) logits.
    """
    return smp.create_model(
        arch=arch,
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
