# ---------------------------------------------------------------
# Unlabeled dataset for semi-supervised coronary artery segmentation.
# Loads images only (no annotations) and returns both weak and strong
# augmented views for EMA teacher-student training.
# ---------------------------------------------------------------

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.tv_tensors import Mask

from datasets.transforms import Transforms


class UnlabeledCoronaryDataset(Dataset):
    def __init__(
        self,
        img_dir: Path,
        img_size: tuple[int, int],
        strong_transforms: Transforms,
    ):
        super().__init__()
        self.img_dir = img_dir
        self.img_size = img_size
        self.strong_transforms = strong_transforms

        self.samples = sorted(img_dir.glob("*.png"))

    def __len__(self):
        return len(self.samples)

    def _weak_augment(self, img: torch.Tensor) -> torch.Tensor:
        """Weak augmentation: resize to fit img_size, horizontal flip, then pad."""
        h, w = img.shape[-2], img.shape[-1]
        target_h, target_w = self.img_size

        scale = min(target_h / h, target_w / w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = F.resize(img, [new_h, new_w])

        if torch.rand(()) < 0.5:
            img = F.horizontal_flip(img)

        pad_h = max(0, target_h - new_h)
        pad_w = max(0, target_w - new_w)
        img = F.pad(img, [0, 0, pad_w, pad_h])

        return img

    def _strong_augment(self, img: torch.Tensor) -> torch.Tensor:
        """Strong augmentation: reuse Transforms pipeline with a dummy target.

        Uses a full-image dummy mask so it survives any crop/scale operation
        without triggering the empty-mask retry in Transforms.forward().
        """
        h, w = img.shape[-2], img.shape[-1]
        dummy_target = {
            "masks": Mask(torch.ones(1, h, w, dtype=torch.bool)),
            "labels": torch.tensor([0]),
            "is_crowd": torch.tensor([False]),
        }
        for _ in range(10):
            try:
                img_aug, _ = self.strong_transforms(img.clone(), dummy_target)
                return img_aug
            except RecursionError:
                continue
        # Fallback: return weak-augmented version
        return self._weak_augment(img)

    def __getitem__(self, index):
        img_path = self.samples[index]
        img = tv_tensors.Image(Image.open(img_path).convert("RGB"))

        img_weak = self._weak_augment(img.clone())
        img_strong = self._strong_augment(img.clone())

        return img_weak, img_strong
