---
license: mit
library_name: pytorch
pipeline_tag: image-segmentation
tags:
  - segmentation
  - remote-sensing
  - sar
  - sentinel-1
  - flood-mapping
  - unet
  - pytorch
datasets:
  - pdosquet/sen1floods11-preprocessed-dl
metrics:
  - iou
  - f1
---

# SAR Flood-Extent Segmentation (U-Net, ResNet34)

A U-Net with a ResNet34 encoder that segments flood water from Sentinel-1 Synthetic Aperture Radar (SAR) imagery. Input is the two SAR polarizations (VV, VH); output is a binary water mask. Trained on the Sen1Floods11 benchmark.

Code and full project: https://github.com/Governor6191/sar-flood-extent

## Model details

- Architecture: U-Net decoder, ResNet34 encoder (ImageNet-pretrained, stem adapted to 2 input channels), 2 output classes (dry, water), via `segmentation_models_pytorch`.
- Input: tensor of shape `(2, 512, 512)`, VV and VH, standardized the same way the Sen1Floods11 preprocessed-dl variant standardizes them (per-channel global z-score, mean 0, std 1).
- Output: per-pixel class logits; `argmax` gives `0` (dry) or `1` (water).
- Loss: cross-entropy + Dice, both with `ignore_index=-1` for unlabeled / nodata pixels.
- Training: trained on the full Sen1Floods11 train split (hand + weak labels), AdamW, cosine schedule, mixed precision, geometric augmentation only (flips and 90-degree rotations; no photometric augmentation, since SAR backscatter is a physical measurement).

## Results (held-out test split, water class)

Evaluated on the 105-scene all-hand Sen1Floods11 test split, never seen during training or checkpoint selection.

**Overall: IoU 0.67, F1 0.80.** Precision 86%, recall 74%. Training on hand + weak labels beat a hand-only baseline (test IoU 0.64), with the gain coming from improved recall.

Per-region test IoU (the test split spans 11 regions with different terrain):

| Region | IoU | F1 |
|---|---|---|
| Nigeria | 0.89 | 0.94 |
| Sri Lanka | 0.89 | 0.94 |
| Mekong | 0.87 | 0.93 |
| Spain | 0.70 | 0.83 |
| India | 0.69 | 0.81 |
| Paraguay | 0.68 | 0.81 |
| Bolivia | 0.65 | 0.79 |
| USA | 0.56 | 0.72 |
| Ghana | 0.55 | 0.71 |
| Somalia | 0.39 | 0.56 |
| Pakistan | 0.20 | 0.33 |

## Intended use and limitations

Intended for research and as a flood-mapping baseline on Sentinel-1 SAR. Not validated for operational emergency response.

Limitations to know before using it:

- It expects input already standardized like Sen1Floods11. Running it on a raw Sentinel-1 GRD scene requires applying the same preprocessing first; without that, predictions are unreliable.
- Performance varies sharply by region. It is strong on clear open-water flooding and weak on some terrain (Pakistan scores 0.20). The aggregate hides that range.
- It is SAR-only (no optical, no DEM, no land-cover priors).
- Trained and tested on Sen1Floods11; transfer to other sensors, resolutions, or regions is unverified.

## How to use

```python
import torch
import segmentation_models_pytorch as smp
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download("Governor6191/sar-flood-extent-unet-resnet34", "model.pt")
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

model = smp.create_model("unet", encoder_name="resnet34",
                         encoder_weights=None, in_channels=2, classes=2)
model.load_state_dict(ckpt["model_state"])
model.eval()

# x: (B, 2, 512, 512) standardized VV/VH SAR
# mask = model(x).argmax(dim=1)  # 0 = dry, 1 = water
```

## Training data

[Sen1Floods11](https://github.com/cloudtostreet/Sen1Floods11) (Bonafilia et al., 2020): 4,830 Sentinel-1 scenes across 11 flood events, with hand-labeled and weak (Otsu-thresholded) flood masks.

## Citation

```
Bonafilia, D., Tellman, B., Anderson, T., & Issenberg, E. (2020).
Sen1Floods11: a georeferenced dataset to train and test deep learning
flood algorithms for Sentinel-1. CVPR Workshops, 210-211.
```

## License

MIT.
