# Sentinel-1 SAR Flood-Extent Pipeline

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-WIP-orange.svg)](#roadmap)

A deep-learning segmentation pipeline that takes Sentinel-1 Synthetic Aperture Radar (SAR) imagery as input and produces binary flood-extent masks. Trained on the [Sen1Floods11][s1f] benchmark. Downstream validation runs against FEMA flood-extent polygons for Hurricane Ida (2021) and Hurricane Katrina (2005) in Harrison County, Mississippi.

![Sen1Floods11 sample with flood-extent overlay](docs/figures/hero_somalia_flood_sample.png)

*Left: VV polarization. Middle: VH polarization. Right: VV with hand-labeled flood overlay (red = water, grey = unlabeled). Sample from the Somalia region of the Sen1Floods11 hand-labeled training subset.*

---

## Why SAR for flood mapping?

Optical imagery (Sentinel-2, Landsat) gets blocked by storm cloud cover at exactly the moment a flood is most active. SAR penetrates clouds and works at night, so it's the only practical data source for near-real-time mapping during ongoing events.

The downstream goal is producing validated flood-extent polygons for two well-documented Mississippi Gulf Coast hurricanes: Ida (2021) and Katrina (2005). FEMA's published disaster-extent polygons serve as independent ground truth. Most Sen1Floods11 portfolio projects stop at the benchmark test split. This one carries through to a real US disaster validation.

---

## Dataset

[Sen1Floods11][s1f] (Bonafilia et al., 2020):

| Split | Samples | Hand-labeled | Weak-labeled | Regions |
|---|---|---|---|---|
| Train | 3,759 | 252 | 3,507 | 12 |
| Val   | 966   | 89  | 877   | 12 |
| Test  | 105   | 105 | 0     | 11 |

The test set is 100% hand-labeled, so reported test metrics are against pristine ground truth with no weak-label-noise caveats. Training is dominated by weak (Otsu-thresholded) labels with a small hand-labeled core. Both hand-only and curriculum (weak-pretrain, then hand-finetune) strategies are on the evaluation plan.

Region distribution is intentionally skewed. Mekong is 29% of train but only 5.7% of test. USA is balanced across splits. Africa is under-represented throughout. Per-region test metrics are reported alongside the aggregate.

![Train-split sanity check](docs/figures/dataset_sanity_check_grid.png)

*Four hand-labeled training samples (Pakistan, Ghana, Somalia, Sri Lanka). Per-image 2nd to 98th percentile contrast scaling, so each panel auto-adjusts to its own dynamic range.*

---

## Results

Baseline: U-Net with a ResNet34 encoder, trained on the 252 hand-labeled training scenes and evaluated on the 105-scene all-hand test split, which the model never saw during training or checkpoint selection.

**Overall test set, water class: IoU 0.64, F1 0.78.** The test number matches validation (0.6445 vs 0.6444), so the model generalizes rather than memorizing the validation set.

It is precision-leaning: 86% precision, 72% recall. When it calls a pixel water it is usually right, but it misses about a quarter of actual water. Raising recall is the main target for the next iteration.

Performance varies a lot by region, which is expected since the test split spans 11 regions with different flood and terrain types:

| Region | IoU | F1 |
|---|---|---|
| Nigeria | 0.90 | 0.95 |
| Sri Lanka | 0.86 | 0.92 |
| Mekong | 0.86 | 0.92 |
| Spain | 0.73 | 0.85 |
| Paraguay | 0.66 | 0.79 |
| India | 0.66 | 0.79 |
| Bolivia | 0.54 | 0.70 |
| USA | 0.54 | 0.70 |
| Ghana | 0.53 | 0.69 |
| Somalia | 0.41 | 0.58 |
| Pakistan | 0.17 | 0.29 |

Pakistan is the clear weak spot. The aggregate 0.64 hides a range from 0.17 to 0.90, which is why per-region reporting matters more than a single headline number.

![Test-set predictions](docs/figures/test_predictions.png)

*VV input, ground truth (blue), and prediction (red) on six test scenes. The model captures clear river and floodplain water well and leaves dry scenes blank, but over-predicts on some Pakistan terrain (the low-IoU region in the table).*

---

## Stack

- Python 3.11+ with [uv](https://github.com/astral-sh/uv)
- [PyTorch](https://pytorch.org/) and [segmentation_models_pytorch](https://github.com/qubvel-org/segmentation_models.pytorch)
- [rasterio](https://rasterio.readthedocs.io/) for geospatial I/O
- [albumentations](https://albumentations.ai/) for augmentation
- [Hugging Face `datasets`](https://huggingface.co/docs/datasets) for the Sen1Floods11 mirror

---

## Quick start

```bash
git clone https://github.com/Governor6191/sar-flood-extent
cd sar-flood-extent
uv sync
hf auth login   # needed for the Sen1Floods11 download

# Inspect the dataset and regenerate the visualizations in this README
uv run python scripts/explore_dataset.py
```

First run downloads Sen1Floods11 (about 80 GB) into `data/sen1floods11/`. Subsequent runs reuse the local cache. The exploration script prints split, region, and label-source statistics, runs tensor sanity checks, and writes per-sample and grid PNGs to `outputs/figures/`.

---

## Project layout

```
sar-flood-extent/
├── src/                 # Python modules (Sen1FloodsDataset, etc.)
├── scripts/             # Runnable entrypoints
├── notebooks/           # Jupyter notebooks (exploration, training)
├── data/                # Local dataset cache (gitignored, see data/README.md)
├── docs/figures/        # Curated portfolio visualizations (committed)
├── outputs/figures/     # Generated artifacts (gitignored, regenerable)
├── checkpoints/         # Model checkpoints (gitignored, released via HF Hub)
├── pyproject.toml
└── README.md
```

---

## Roadmap

- [x] Repository foundation and reproducible Python env
- [x] Sen1Floods11 acquisition and schema documentation
- [x] PyTorch `Dataset` class with hand/weak label-source filtering
- [x] Dataset exploration and sanity-check visualizations
- [x] Per-channel SAR normalization statistics (train split)
- [x] U-Net + ResNet34 baseline training (hand-only): val IoU 0.64, val F1 0.78
- [ ] Curriculum experiment (weak-pretrain, then hand-finetune)
- [ ] Test-set evaluation with per-region breakdown
- [ ] FEMA polygon validation: Hurricane Ida 2021 (Harrison County, MS)
- [ ] FEMA polygon validation: Hurricane Katrina 2005 (Harrison County, MS)
- [ ] Pretrained model release on Hugging Face Hub
- [ ] End-to-end inference tutorial notebook

---

## License

[MIT](LICENSE)

## Citation

If you use this work, please cite the underlying Sen1Floods11 dataset:

```
Bonafilia, D., Tellman, B., Anderson, T., & Issenberg, E. (2020).
Sen1Floods11: a georeferenced dataset to train and test deep learning
flood algorithms for Sentinel-1.
CVPR Workshops, 210-211.
```

---

[s1f]: https://github.com/cloudtostreet/Sen1Floods11
