"""Segmentation metrics (IoU, F1) for the water class, with ignore support."""
from __future__ import annotations

import torch


@torch.no_grad()
def confusion_counts(preds: torch.Tensor, target: torch.Tensor, ignore_index: int = -1):
    """Count TP, FP, FN, TN for the water class (index 1) over valid pixels.

    Args:
        preds: (B, H, W) predicted class indices.
        target: (B, H, W) labels in {-1, 0, 1}.
        ignore_index: label value to exclude from the counts.

    Returns:
        (tp, fp, fn, tn) as Python ints.
    """
    valid = target != ignore_index
    p = (preds == 1) & valid
    t = (target == 1) & valid
    tp = int((p & t).sum())
    fp = int((p & ~t & valid).sum())
    fn = int((~p & t).sum())
    tn = int((~p & ~t & valid).sum())
    return tp, fp, fn, tn


def iou_score(tp: int, fp: int, fn: int, eps: float = 1e-7) -> float:
    """Intersection over union for the positive (water) class."""
    return tp / (tp + fp + fn + eps)


def f1_score(tp: int, fp: int, fn: int, eps: float = 1e-7) -> float:
    """F1 (Dice) for the positive (water) class."""
    return 2 * tp / (2 * tp + fp + fn + eps)
