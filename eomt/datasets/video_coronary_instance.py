# ---------------------------------------------------------------
# Video coronary instance dataset.
# Each "clip" is a directory of 5 frames (before2, before1, central,
# after1, after2). YOLO-format polygon labels exist for the central
# frame only; geometric augmentations are applied identically across
# all frames so the central-frame masks remain aligned.
# ---------------------------------------------------------------

import random
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from torchvision.tv_tensors import Mask

from datasets.lightning_data_module import LightningDataModule
from training.skeleton_utils import compute_instance_skeletons


FRAME_ORDER = ["before2", "before1", "central", "after1", "after2"]


def _load_clip_frames(clip_dir: Path) -> List[Image.Image]:
    """Load 5 PNG frames in canonical temporal order."""
    frames = []
    for tag in FRAME_ORDER:
        # File naming convention: <clip_id>_<tag>.png
        candidates = list(clip_dir.glob(f"*_{tag}.png"))
        if not candidates:
            raise FileNotFoundError(
                f"Missing frame '{tag}' in clip dir {clip_dir}"
            )
        frames.append(Image.open(candidates[0]).convert("RGB"))
    return frames


def _parse_yolo_polygons(label_path: Path, h: int, w: int):
    """Parse YOLO polygon labels into (masks, labels, is_crowd) lists."""
    masks, labels, is_crowd = [], [], []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            class_id = int(parts[0])
            coords = list(map(float, parts[1:]))
            polygon = []
            for i in range(0, len(coords) - 1, 2):
                polygon.append((coords[i] * w, coords[i + 1] * h))
            if len(polygon) < 3:
                continue
            mask_img = Image.new("L", (w, h), 0)
            ImageDraw.Draw(mask_img).polygon(polygon, fill=1)
            mask = torch.from_numpy(np.array(mask_img, dtype=np.uint8)).bool()
            if not mask.any():
                continue
            masks.append(mask)
            labels.append(class_id)
            is_crowd.append(False)
    return masks, labels, is_crowd


class VideoTransforms:
    """Resize-and-pad pipeline that applies identical geometric ops
    to every frame of a clip while keeping the central-frame mask
    aligned. Optional horizontal flip and scale jitter are sampled
    once per clip.
    """

    def __init__(
        self,
        img_size: Tuple[int, int],
        scale_range: Tuple[float, float] = (0.5, 2.0),
        hflip_prob: float = 0.5,
        scale_jitter_enabled: bool = True,
        random_crop_enabled: bool = True,
        skeleton_enabled: bool = False,
        skeleton_num_dilations: int = 2,
    ):
        self.img_size = img_size
        self.scale_range = scale_range
        self.hflip_prob = hflip_prob
        self.scale_jitter_enabled = scale_jitter_enabled
        self.random_crop_enabled = random_crop_enabled
        self.skeleton_enabled = skeleton_enabled
        self.skeleton_num_dilations = skeleton_num_dilations

    def __call__(self, frames: List[torch.Tensor], target: dict, training: bool):
        img_size = self.img_size
        masks = target["masks"]

        # 1. Optional horizontal flip (shared params).
        if training and torch.rand(()) < self.hflip_prob:
            frames = [F.horizontal_flip(f) for f in frames]
            masks = F.horizontal_flip(masks)

        # 2. Resize so the longer edge fits img_size, with optional scale jitter.
        h, w = frames[0].shape[-2], frames[0].shape[-1]
        if training and self.scale_jitter_enabled:
            scale = float(torch.empty(1).uniform_(*self.scale_range).item())
        else:
            scale = 1.0
        target_long = max(img_size)
        base_scale = target_long / max(h, w)
        scale = scale * base_scale
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))

        frames = [
            F.resize(f, [new_h, new_w], antialias=True)
            for f in frames
        ]
        masks = F.resize(masks, [new_h, new_w], interpolation=T.InterpolationMode.NEAREST)

        # 3. Pad up to img_size.
        pad_h = max(0, img_size[0] - new_h)
        pad_w = max(0, img_size[1] - new_w)
        if pad_h > 0 or pad_w > 0:
            padding = [0, 0, pad_w, pad_h]
            frames = [F.pad(f, padding) for f in frames]
            masks = F.pad(masks, padding)

        # 4. Random / centre crop to exactly img_size.
        ch, cw = frames[0].shape[-2], frames[0].shape[-1]
        crop_h, crop_w = img_size
        if ch > crop_h or cw > crop_w:
            if training and self.random_crop_enabled:
                top = int(torch.randint(0, ch - crop_h + 1, ()).item())
                left = int(torch.randint(0, cw - crop_w + 1, ()).item())
            else:
                top = (ch - crop_h) // 2
                left = (cw - crop_w) // 2
            frames = [F.crop(f, top, left, crop_h, crop_w) for f in frames]
            masks = F.crop(masks, top, left, crop_h, crop_w)

        # 5. Filter out empty masks.
        valid = masks.flatten(1).any(1)
        if not valid.any():
            # Keep one empty mask so downstream code does not crash;
            # dataset.__getitem__ retries on a different sample.
            target = dict(target)
            target["masks"] = tv_tensors.Mask(masks)
            target["_empty"] = True
            return frames, target

        target = dict(target)
        target["masks"] = tv_tensors.Mask(masks[valid])
        target["labels"] = target["labels"][valid]
        target["is_crowd"] = target["is_crowd"][valid]
        target["_empty"] = False

        if self.skeleton_enabled:
            target["skeletons"] = Mask(
                compute_instance_skeletons(
                    target["masks"], self.skeleton_num_dilations
                )
            )

        return frames, target


class VideoCoronaryDataset(Dataset):
    def __init__(
        self,
        img_root: Path,
        label_dir: Path,
        transforms: VideoTransforms,
        training: bool,
    ):
        super().__init__()
        self.transforms = transforms
        self.training = training
        self.img_root = img_root
        self.label_dir = label_dir

        self.samples: List[Tuple[Path, Path]] = []
        for clip_dir in sorted(p for p in img_root.iterdir() if p.is_dir()):
            label_path = label_dir / f"{clip_dir.name}.txt"
            if label_path.exists() and label_path.stat().st_size > 0:
                self.samples.append((clip_dir, label_path))

    def __len__(self):
        return len(self.samples)

    def _load_one(self, index: int):
        clip_dir, label_path = self.samples[index]
        pil_frames = _load_clip_frames(clip_dir)
        frames = [tv_tensors.Image(f) for f in pil_frames]
        h, w = frames[0].shape[-2], frames[0].shape[-1]

        masks, labels, is_crowd = _parse_yolo_polygons(label_path, h, w)
        if not masks:
            masks = [torch.zeros(h, w, dtype=torch.bool)]
            labels = [0]
            is_crowd = [False]

        target = {
            "masks": tv_tensors.Mask(torch.stack(masks)),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "is_crowd": torch.tensor(is_crowd, dtype=torch.bool),
        }
        return frames, target

    def __getitem__(self, index):
        for _ in range(8):
            frames, target = self._load_one(index)
            frames, target = self.transforms(frames, target, training=self.training)
            if not target.pop("_empty", False):
                break
            index = random.randrange(len(self.samples))

        # Stack frames into (T, 3, H, W) uint8 tensor.
        frames_t = torch.stack([f.as_subclass(torch.Tensor) for f in frames], dim=0)
        return frames_t, target


class VideoCoronaryInstance(LightningDataModule):
    def __init__(
        self,
        path,
        num_workers: int = 4,
        batch_size: int = 1,
        img_size: tuple[int, int] = (512, 512),
        num_classes: int = 9,
        T: int = 5,
        scale_range: tuple[float, float] = (0.5, 2.0),
        hflip_prob: float = 0.5,
        scale_jitter_enabled: bool = True,
        random_crop_enabled: bool = True,
        check_empty_targets: bool = True,
        stuff_classes: list[int] | None = None,
        label_subdir: str = "labels",
        skeleton_enabled: bool = False,
        skeleton_num_dilations: int = 2,
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

        if T != len(FRAME_ORDER):
            raise ValueError(
                f"VideoCoronaryInstance currently supports T={len(FRAME_ORDER)}; got T={T}"
            )
        self.T = T
        self.label_subdir = label_subdir

        self.train_transforms = VideoTransforms(
            img_size=img_size,
            scale_range=scale_range,
            hflip_prob=hflip_prob,
            scale_jitter_enabled=scale_jitter_enabled,
            random_crop_enabled=random_crop_enabled,
            skeleton_enabled=skeleton_enabled,
            skeleton_num_dilations=skeleton_num_dilations,
        )
        self.eval_transforms = VideoTransforms(
            img_size=img_size,
            scale_range=(1.0, 1.0),
            hflip_prob=0.0,
            scale_jitter_enabled=False,
            random_crop_enabled=False,
        )

    def setup(self, stage: Union[str, None] = None) -> "VideoCoronaryInstance":
        root = Path(self.path)
        self.train_dataset = VideoCoronaryDataset(
            img_root=root / "train" / "images",
            label_dir=root / "train" / self.label_subdir,
            transforms=self.train_transforms,
            training=True,
        )
        self.val_dataset = VideoCoronaryDataset(
            img_root=root / "val" / "images",
            label_dir=root / "val" / self.label_subdir,
            transforms=self.eval_transforms,
            training=False,
        )
        self.test_dataset = VideoCoronaryDataset(
            img_root=root / "test" / "images",
            label_dir=root / "test" / self.label_subdir,
            transforms=self.eval_transforms,
            training=False,
        )
        return self

    @staticmethod
    def video_train_collate(batch):
        frames, targets = [], []
        for f, t in batch:
            frames.append(f)
            targets.append(t)
        # (B, T, 3, H, W)
        return torch.stack(frames, dim=0), targets

    @staticmethod
    def video_eval_collate(batch):
        frames = tuple(b[0] for b in batch)
        targets = tuple(b[1] for b in batch)
        return frames, targets

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.video_train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.video_eval_collate,
            **self.dataloader_kwargs,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self.video_eval_collate,
            **self.dataloader_kwargs,
        )


# ---------------------------------------------------------------------------
# Flat-to-video wrapper: reads a plain image dataset (single PNG per sample)
# and repeats each frame T times to form a pseudo-clip.
# Useful for evaluating VideoEoMT on non-video datasets.
# ---------------------------------------------------------------------------

class FlatToVideoCoronaryDataset(Dataset):
    """Wraps a flat image directory as a video dataset by repeating the
    single frame T times.  Labels are YOLO polygon format (same as
    CoronaryDataset)."""

    def __init__(
        self,
        img_dir: Path,
        label_dir: Path,
        T: int,
        transforms: VideoTransforms,
        training: bool,
    ):
        super().__init__()
        self.T = T
        self.transforms = transforms
        self.training = training

        self.samples: List[Tuple[Path, Path]] = []
        for img_path in sorted(img_dir.glob("*.png")):
            label_path = label_dir / f"{img_path.stem}.txt"
            if label_path.exists() and label_path.stat().st_size > 0:
                self.samples.append((img_path, label_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label_path = self.samples[index]

        pil = Image.open(img_path).convert("RGB")
        frame = tv_tensors.Image(pil)
        frames = [frame] * self.T
        h, w = frame.shape[-2], frame.shape[-1]

        masks, labels, is_crowd = _parse_yolo_polygons(label_path, h, w)
        if not masks:
            masks = [torch.zeros(h, w, dtype=torch.bool)]
            labels = [0]
            is_crowd = [False]

        target = {
            "masks": tv_tensors.Mask(torch.stack(masks)),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "is_crowd": torch.tensor(is_crowd, dtype=torch.bool),
        }

        for _ in range(8):
            frames_out, target_out = self.transforms(frames, target, training=self.training)
            if not target_out.pop("_empty", False):
                break

        frames_t = torch.stack([f.as_subclass(torch.Tensor) for f in frames_out], dim=0)
        return frames_t, target_out


class FlatToVideoCoronaryInstance(LightningDataModule):
    """LightningDataModule for evaluating VideoEoMT on a plain flat dataset
    by repeating each image T times as a pseudo-clip."""

    def __init__(
        self,
        path,
        num_workers: int = 4,
        batch_size: int = 2,
        img_size: tuple[int, int] = (512, 512),
        num_classes: int = 9,
        T: int = 5,
        check_empty_targets: bool = True,
        label_subdir: str = "labels",
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
        self.T = T
        self.label_subdir = label_subdir

        self.eval_transforms = VideoTransforms(
            img_size=img_size,
            scale_range=(1.0, 1.0),
            hflip_prob=0.0,
            scale_jitter_enabled=False,
            random_crop_enabled=False,
        )
        self.train_transforms = VideoTransforms(
            img_size=img_size,
            scale_range=(0.5, 2.0),
            hflip_prob=0.5,
        )

    def setup(self, stage: Union[str, None] = None) -> "FlatToVideoCoronaryInstance":
        root = Path(self.path)
        self.train_dataset = FlatToVideoCoronaryDataset(
            img_dir=root / "train" / "images",
            label_dir=root / "train" / self.label_subdir,
            T=self.T,
            transforms=self.train_transforms,
            training=True,
        )
        self.val_dataset = FlatToVideoCoronaryDataset(
            img_dir=root / "val" / "images",
            label_dir=root / "val" / self.label_subdir,
            T=self.T,
            transforms=self.eval_transforms,
            training=False,
        )
        self.test_dataset = FlatToVideoCoronaryDataset(
            img_dir=root / "test" / "images",
            label_dir=root / "test" / self.label_subdir,
            T=self.T,
            transforms=self.eval_transforms,
            training=False,
        )
        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=VideoCoronaryInstance.video_train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=VideoCoronaryInstance.video_eval_collate,
            **self.dataloader_kwargs,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=VideoCoronaryInstance.video_eval_collate,
            **self.dataloader_kwargs,
        )
