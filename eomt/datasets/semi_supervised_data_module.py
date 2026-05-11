# ---------------------------------------------------------------
# Semi-supervised data module combining labeled and unlabeled data.
# Returns a dict of dataloaders for teacher-student training.
# ---------------------------------------------------------------

from pathlib import Path
from typing import Union

import torch
from torch.utils.data import DataLoader, Dataset

from datasets.coronary_instance import CoronaryDataset
from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import Transforms
from datasets.unlabeled_coronary import UnlabeledCoronaryDataset


class _RepeatDataset(Dataset):
    """Wraps a dataset so its effective length matches a target length."""

    def __init__(self, dataset: Dataset, target_len: int):
        self.dataset = dataset
        self.target_len = target_len

    def __len__(self):
        return self.target_len

    def __getitem__(self, index):
        return self.dataset[index % len(self.dataset)]


class SemiSupervisedDataModule(LightningDataModule):
    def __init__(
        self,
        path,
        unlabeled_path,
        num_workers: int = 4,
        batch_size: int = 4,
        img_size: tuple[int, int] = (512, 512),
        num_classes: int = 9,
        color_jitter_enabled: bool = False,
        scale_range: tuple[float, float] = (0.5, 2.0),
        check_empty_targets: bool = True,
        stuff_classes: list[int] | None = None,
        label_subdir: str = "labels",
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

        self.unlabeled_path = unlabeled_path
        self.label_subdir = label_subdir

        self.labeled_transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
            skeleton_enabled=skeleton_enabled,
            skeleton_num_dilations=skeleton_num_dilations,
            extra_augmentations_enabled=extra_augmentations_enabled,
        )

        self.strong_transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=True,
            scale_range=scale_range,
            skeleton_enabled=False,
            skeleton_num_dilations=0,
            extra_augmentations_enabled=True,
        )

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        root = Path(self.path)

        self.train_labeled = CoronaryDataset(
            img_dir=root / "train" / "images",
            label_dir=root / "train" / self.label_subdir,
            transforms=self.labeled_transforms,
        )
        self.val_dataset = CoronaryDataset(
            img_dir=root / "val" / "images",
            label_dir=root / "val" / self.label_subdir,
        )

        unlabeled_root = Path(self.unlabeled_path)
        self.train_unlabeled = UnlabeledCoronaryDataset(
            img_dir=unlabeled_root / "images",
            img_size=self.img_size,
            strong_transforms=self.strong_transforms,
        )

        # Match lengths by repeating the shorter dataset
        max_len = max(len(self.train_labeled), len(self.train_unlabeled))
        if len(self.train_labeled) < max_len:
            self.train_labeled = _RepeatDataset(self.train_labeled, max_len)
        if len(self.train_unlabeled) < max_len:
            self.train_unlabeled = _RepeatDataset(self.train_unlabeled, max_len)

        return self

    def train_dataloader(self):
        loader_l = DataLoader(
            self.train_labeled,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )
        loader_u = DataLoader(
            self.train_unlabeled,
            shuffle=True,
            drop_last=True,
            collate_fn=self._unlabeled_collate,
            **self.dataloader_kwargs,
        )
        return {"labeled": loader_l, "unlabeled": loader_u}

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )

    @staticmethod
    def _unlabeled_collate(batch):
        imgs_weak, imgs_strong = [], []
        for img_w, img_s in batch:
            imgs_weak.append(img_w)
            imgs_strong.append(img_s)
        return torch.stack(imgs_weak), torch.stack(imgs_strong)
