"""Train a U-Net baseline for SAR flood-extent segmentation.

Trains on the Sen1Floods11 hand-labeled subset by default with a
ResNet34-U-Net, combined cross-entropy + Dice loss (ignore_index=-1),
mixed precision, and per-epoch IoU/F1 on the validation split. Saves the
best checkpoint by validation IoU to checkpoints/.

Run from repo root:
    uv run python scripts/train.py            # full hand-only baseline
    uv run python scripts/train.py --smoke    # quick pipeline smoke test
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# data.py imports the HuggingFace datasets library (pyarrow backend) before
# torch. On Windows with CUDA torch, that load order is load-bearing: importing
# torch before pyarrow segfaults the process. So the local modules, which pull
# datasets in first, must be imported before torch/smp/albumentations below.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data import Sen1FloodsDataset  # noqa: E402
from model import build_model  # noqa: E402
from metrics import confusion_counts, iou_score, f1_score  # noqa: E402

import albumentations as A  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402
import segmentation_models_pytorch as smp  # noqa: E402

CKPT_DIR = REPO_ROOT / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def build_transforms() -> A.Compose:
    """Training augmentation: geometric only.

    No photometric augmentation. SAR backscatter is a physical measurement,
    not an RGB image, so brightness/contrast jitter would be meaningless.
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])


def run_epoch(model, loader, device, ce_loss, dice_loss, optimizer=None, scaler=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    n_seen = 0
    tp = fp = fn = 0

    for batch in loader:
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.autocast(device_type=device, enabled=(scaler is not None)):
                logits = model(image)                       # (B, 2, H, W)
                loss = ce_loss(logits, mask) + dice_loss(logits, mask)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        total_loss += loss.item() * image.size(0)
        n_seen += image.size(0)
        preds = logits.argmax(dim=1)
        b_tp, b_fp, b_fn, _ = confusion_counts(preds, mask)
        tp += b_tp
        fp += b_fp
        fn += b_fn

    return {
        "loss": total_loss / max(n_seen, 1),
        "iou": iou_score(tp, fp, fn),
        "f1": f1_score(tp, fp, fn),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--label-source", default="hand", choices=["hand", "weak", "all"])
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke", action="store_true", help="quick pipeline test")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: training on CPU will be very slow.")

    tfm = build_transforms()
    train_ds = Sen1FloodsDataset(split="train", label_source=args.label_source, transform=tfm)
    val_ds = Sen1FloodsDataset(split="validation", label_source="hand")

    if args.smoke:
        train_ds = Subset(train_ds, list(range(16)))
        val_ds = Subset(val_ds, list(range(8)))
        args.epochs = 2
        print("SMOKE TEST: 16 train / 8 val / 2 epochs")

    pin = device == "cuda"
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)
    print(f"train batches: {len(train_loader)}  val batches: {len(val_loader)}")

    model = build_model(in_channels=2, classes=2).to(device)
    ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    dice_loss = smp.losses.DiceLoss(mode="multiclass", ignore_index=-1, from_logits=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    best_iou = -1.0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_loader, device, ce_loss, dice_loss, optimizer, scaler)
        va = run_epoch(model, val_loader, device, ce_loss, dice_loss)
        scheduler.step()
        dt = time.time() - t0

        print(f"epoch {epoch:3d}/{args.epochs}  "
              f"train_loss {tr['loss']:.4f}  train_IoU {tr['iou']:.4f}  "
              f"val_loss {va['loss']:.4f}  val_IoU {va['iou']:.4f}  val_F1 {va['f1']:.4f}  "
              f"({dt:.1f}s)")

        if va["iou"] > best_iou and not args.smoke:
            best_iou = va["iou"]
            ckpt_path = CKPT_DIR / f"unet_resnet34_{args.label_source}_best.pt"
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_iou": best_iou,
                "arch": "unet",
                "encoder": "resnet34",
                "label_source": args.label_source,
            }, ckpt_path)
            print(f"  saved best checkpoint (val IoU {best_iou:.4f}) -> {ckpt_path.name}")

    print(f"\nDone. Best val IoU: {best_iou:.4f}")
    if args.smoke:
        print(f"Smoke test passed: pipeline runs end to end on {device}.")


if __name__ == "__main__":
    main()
