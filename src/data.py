"""Sen1Floods11 PyTorch Dataset wrapper.

Wraps the locally-cached Hugging Face dataset
`pdosquet/sen1floods11-preprocessed-dl` as a `torch.utils.data.Dataset`.
SAR-only (Sentinel-2 channels dropped at load time), with optional
hand/weak label-source filtering and pluggable albumentations transforms.
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class Sen1FloodsDataset(Dataset):
    """Sentinel-1 SAR flood-extent segmentation dataset.

    Returns dicts with keys:
        image:        FloatTensor (2, 512, 512) — VV, VH polarizations (dB-scale)
        mask:         LongTensor  (512, 512)    — values in {-1, 0, 1}
                      (-1 = unlabeled/masked, 0 = dry, 1 = water)
        patch_id:     str
        region:       str
        label_source: str — "hand" or "weak"
    """

    SPLITS = ("train", "validation", "test")
    LABEL_SOURCES = ("hand", "weak", "all")
    HF_REPO = "pdosquet/sen1floods11-preprocessed-dl"

    def __init__(
        self,
        split: str = "train",
        cache_dir: str = "data/sen1floods11",
        label_source: str = "all",
        transform: Optional[Callable] = None,
    ):
        if split not in self.SPLITS:
            raise ValueError(f"split must be one of {self.SPLITS}, got {split!r}")
        if label_source not in self.LABEL_SOURCES:
            raise ValueError(
                f"label_source must be one of {self.LABEL_SOURCES}, got {label_source!r}"
            )

        ds = load_dataset(self.HF_REPO, cache_dir=cache_dir, split=split)

        # SAR-only: drop the 13-band Sentinel-2 optical column at load time
        if "s2" in ds.column_names:
            ds = ds.remove_columns(["s2"])

        # Optional hand-vs-weak label filter
        if label_source != "all":
            ds = ds.filter(lambda x: x["label_source"] == label_source)

        self.ds = ds
        self.transform = transform
        self.split = split
        self.label_source = label_source

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict:
        sample = self.ds[idx]

        # HF returns Python lists; convert to float32 / int64 numpy arrays
        image = np.asarray(sample["s1"], dtype=np.float32)   # (2, H, W)
        mask = np.asarray(sample["label"], dtype=np.int64)   # (H, W)

        if self.transform is not None:
            # Albumentations wants HWC for the image; mask stays HW
            image_hwc = image.transpose(1, 2, 0)
            out = self.transform(image=image_hwc, mask=mask)
            image = out["image"]
            mask = out["mask"]
            if isinstance(image, np.ndarray):
                image = image.transpose(2, 0, 1)

        return {
            "image": torch.from_numpy(image) if isinstance(image, np.ndarray) else image,
            "mask": torch.from_numpy(mask) if isinstance(mask, np.ndarray) else mask,
            "patch_id": sample["patch_id"],
            "region": sample["region"],
            "label_source": sample["label_source"],
        }
