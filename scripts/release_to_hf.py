"""Release the trained checkpoint and model card to the Hugging Face Hub.

Requires `hf auth login` to have been run first. Creates (or reuses) the model
repo, uploads the checkpoint as model.pt, and uploads MODEL_CARD.md as the repo
README.md.

Run from repo root:
    uv run python scripts/release_to_hf.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "Governor6191/sar-flood-extent-unet-resnet34"
CHECKPOINT = REPO_ROOT / "checkpoints" / "unet_resnet34_all_best.pt"
MODEL_CARD = REPO_ROOT / "MODEL_CARD.md"


def main():
    if not CHECKPOINT.exists():
        sys.exit(f"checkpoint not found: {CHECKPOINT}")
    if not MODEL_CARD.exists():
        sys.exit(f"model card not found: {MODEL_CARD}")

    api = HfApi()
    print(f"Creating or reusing repo {REPO_ID} ...")
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

    print("Uploading checkpoint as model.pt ...")
    api.upload_file(
        path_or_fileobj=str(CHECKPOINT),
        path_in_repo="model.pt",
        repo_id=REPO_ID,
        repo_type="model",
    )

    print("Uploading model card as README.md ...")
    api.upload_file(
        path_or_fileobj=str(MODEL_CARD),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="model",
    )

    print(f"Done. https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
