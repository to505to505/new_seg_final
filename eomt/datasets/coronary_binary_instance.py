# ---------------------------------------------------------------
# Dataset module for coronary artery binary segmentation.
# Same as coronary_instance but uses labels_binary/ (single class).
# ---------------------------------------------------------------

from pathlib import Path
from typing import Union

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors

from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import Transforms


class CoronaryBinaryDataset(Dataset):
    def __init__(self, img_dir: Path, label_dir: Path, transforms=None):
        super().__init__()
        self.transforms = transforms
        self.img_dir = img_dir
        self.label_dir = label_dir

        self.samples = []
        for img_path in sorted(img_dir.glob("*.png")):
            label_path = label_dir / f"{img_path.stem}.txt"
            if label_path.exists() and label_path.stat().st_size > 0:
                self.samples.append((img_path, label_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label_path = self.samples[index]

        img = tv_tensors.Image(Image.open(img_path).convert("RGB"))
        h, w = img.shape[-2], img.shape[-1]

        # Draw ALL polygons onto a single binary mask.
        # PIL uses even-odd fill, so bridge-cut holes are preserved.
        combined = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(combined)

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue

                coords = list(map(float, parts[1:]))

                polygon = []
                for i in range(0, len(coords) - 1, 2):
                    px = coords[i] * w
                    py = coords[i + 1] * h
                    polygon.append((px, py))

                if len(polygon) < 3:
                    continue

                draw.polygon(polygon, fill=1)

        mask = torch.from_numpy(np.array(combined, dtype=np.uint8)).bool()

        if not mask.any():
            mask = torch.zeros(h, w, dtype=torch.bool)

        target = {
            "masks": tv_tensors.Mask(tv_tensors.Mask(mask).unsqueeze(0)),
            "labels": torch.tensor([0]),
            "is_crowd": torch.tensor([False]),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class CoronaryBinaryInstance(LightningDataModule):
    def __init__(
        self,
        path,
        num_workers: int = 4,
        batch_size: int = 4,
        img_size: tuple[int, int] = (512, 512),
        num_classes: int = 1,
        color_jitter_enabled: bool = False,
        scale_range: tuple[float, float] = (0.5, 2.0),
        check_empty_targets: bool = True,
        stuff_classes: list[int] | None = None,
        skeleton_enabled: bool = False,
        skeleton_num_dilations: int = 2,
        extra_augmentations_enabled: bool = False,
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=img_size,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])

        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
            skeleton_enabled=skeleton_enabled,
            skeleton_num_dilations=skeleton_num_dilations,
            extra_augmentations_enabled=extra_augmentations_enabled,
        )

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        root = Path(self.path)
        self.train_dataset = CoronaryBinaryDataset(
            img_dir=root / "train" / "images",
            label_dir=root / "train" / "labels_binary",
            transforms=self.transforms,
        )
        self.val_dataset = CoronaryBinaryDataset(
            img_dir=root / "val" / "images",
            label_dir=root / "val" / "labels_binary",
        )
        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )
