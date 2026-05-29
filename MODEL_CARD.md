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

## Real-disaster validation (Hurricane Harvey 2017)

The numbers above are on Sen1Floods11 data. To see how the model does on a hurricane it never trained on, I ran it on a Sentinel-1 GRD scene over Houston (the August 30, 2017 descending pass) and scored it against the Copernicus EMS EMSR229 Houston flood delineation, an independent radar-derived flood map (COSMO-SkyMed, August 28 and 30, 2017). JRC Global Surface Water permanent water (`occurrence >= 50`) is removed from both sides so it's flood against flood.

| Metric | Flood-only | Raw (all water) |
|---|---|---|
| IoU | 0.12 | 0.10 |
| F1 | 0.22 | 0.18 |
| Precision | 0.21 | 0.15 |
| Recall | 0.23 | 0.25 |

![Harvey validation against Copernicus EMS](harvey_validation.png)

*Sentinel-1 VV input, model prediction (red), Copernicus EMS observed flood (blue), and the agreement map with permanent water removed (green true positive, red false positive, blue false negative).*

This is well below the 0.67 benchmark IoU, and the reason is mostly a definition mismatch, not a registration error (alignment was checked with flip and shift tests). The model detects open water by its low radar backscatter, but a large share of the Copernicus flood is flooded vegetation and flooded urban land that stays bright in SAR (median VV near -14 dB under the Copernicus polygons versus -16 dB for the model's water, against -9 dB for dry ground). The model can't see flood that doesn't darken the return, and that's most of the recall gap. Houston is also dense urban, the model's weakest setting on the benchmark (USA 0.56 IoU), and this is a cross-sensor, cross-resolution comparison: 10 m Sentinel-1 C-band against a 1:440,000 COSMO-SkyMed X-band map. The takeaway is that open-water flood mapping carries over to a real unseen scene, while urban flood from C-band SAR alone does not.

## Intended use and limitations

Intended for research and as a flood-mapping baseline on Sentinel-1 SAR. Not validated for operational emergency response.

Limitations to know before using it:

- It expects input already standardized like Sen1Floods11. Running it on a raw Sentinel-1 GRD scene requires applying the same preprocessing first; without that, predictions are unreliable.
- Performance varies sharply by region. It is strong on clear open-water flooding and weak on some terrain (Pakistan scores 0.20). The aggregate hides that range.
- It is SAR-only (no optical, no DEM, no land-cover priors).
- It is an open-water detector, not an all-flood detector. On the Hurricane Harvey validation it underdetected urban and vegetated flood, which keep high SAR backscatter (flood-only IoU 0.12 against Copernicus EMS). See the validation section above.
- Trained and tested on Sen1Floods11; transfer to other sensors, resolutions, or regions is otherwise unverified.

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
