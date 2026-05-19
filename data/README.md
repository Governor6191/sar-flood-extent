# Datasets

This folder contains downloaded datasets. **Contents are gitignored** (too large for git). Each dataset has its own subfolder.

## sen1floods11/

Source: https://huggingface.co/datasets/pdosquet/sen1floods11-preprocessed-dl
Citation: Bonafilia, D., Tellman, B., Anderson, T., & Issenberg, E. (2020). *Sen1Floods11: a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1*. CVPR Workshops. arXiv:2006.05509.

### Splits
- train: 3759 samples
- validation: 966 samples
- test: 105 samples
- Total: 4830 samples, ~80 GB on disk (preprocessed-dl variant includes full Sentinel-2 alongside SAR)

### Sample structure
- s1: (2, 512, 512) float64 — Sentinel-1 SAR (VV + VH polarizations), already normalized to dB-like scale
- s2: (13, 512, 512) float64 — Sentinel-2 optical (13 bands, normalized), unused by this project
- label: (512, 512) int64 — flood/water mask {-1: unlabeled, 0: dry, 1: water}
- jrc_mask: (512, 512) int64 — JRC permanent-water reference mask
- patch_id: str — unique patch identifier
- region: str — country/region tag (Bolivia, Ghana, USA, etc.)
- label_source: str — `hand` (hand-labeled) or `weak` (Otsu-threshold-derived)

### Re-download
```bash
uv run python -c "from datasets import load_dataset; load_dataset('pdosquet/sen1floods11-preprocessed-dl', cache_dir='data/sen1floods11')"
```
