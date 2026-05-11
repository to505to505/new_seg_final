import os
import torch
import copy
import random
import numpy as np
import kornia.augmentation as K
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, PILToTensor
from PIL import Image


def seed_everything(seed: int = 1):
    """
    Seed everything for reproducibility

    Arguments:
        seed (int): Seed
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multiple GPUs

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CustomSegmentationDataset(Dataset):

    """Custom dataset class for loading images and masks from folders as tensors"""

    def __init__(self, main_dir: str, split: str):
        self.img_dir = os.path.join(main_dir, "images", split)
        self.seg_mask_dir = os.path.join(main_dir, "masks", split)
        self.img_name_net = sorted(os.listdir(self.img_dir))
        self.seg_mask_name_net = sorted(os.listdir(self.seg_mask_dir))
        self.convert_to_scaled_tensor = ToTensor()
        self.convert_to_tensor = PILToTensor()

    def __len__(self):
        return len(self.img_name_net)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_name_net[idx])
        seg_mask_path = os.path.join(self.seg_mask_dir, self.seg_mask_name_net[idx])

        img = self.convert_to_scaled_tensor(Image.open(img_path).convert("RGB"))
        seg_mask = (
            self.convert_to_tensor(Image.open(seg_mask_path))
            .to(torch.bool)
            .to(torch.float32)
        )

        return img, seg_mask


class CustomDataloader:

    """Custom data loader class for efficient application of Kornia's batch transforms"""

    def __init__(
        self,
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        transform_suite=None,
    ):
        self.pytorch_dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=pin_memory,
            num_workers=num_workers,
        )
        self.transform_suite = transform_suite
        self.dataset = dataset
        self.batch_size = batch_size

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for img_batch, seg_mask_batch in self.pytorch_dataloader:
            if self.transform_suite is not None:
                img_batch, seg_mask_batch = self.transform_suite(
                    img_batch, seg_mask_batch
                )

            yield img_batch, seg_mask_batch


def get_transform_suite(
    data_id=2,
    rotate=False,
    perspective=False,
    erase=False,
    scale=False,
    hflip=False,
    translate=False,
    hsv=False,
):
    """
    Establishes the set of transformations to be used during training, validation, and testing

    Arguments:
        data_id (int): ID of the dataset for retrieving stats for image normalization
        rotate (bool): Flag to decide whether/not to apply rotation
        perspective (bool): Flag to decide whether/not to apply a perspective transformation
        erase (bool): Flag to decide whether/not to apply erasing
        scale (bool): Flag to decide whether/not to apply scaling
        hflip (bool): Flag to decide whether/not to affect horizontal flips
        translate (bool): Flag to decide whether/not to affect translation
        hsv (bool): Flag to decide whether/not to apply color jiggling

    Returns:
        train_transform_suite (kornia.augmentation.container.AugmentationSequential): Suite of training set dynamic augmentations
        test_transform_suite (kornia.augmentation.container.AugmentationSequential): Suite of validation/test set augmentations
    """

    if data_id == 1:
        mean_net = torch.tensor([0.423, 0.423, 0.423])
        std_net = torch.tensor([0.142, 0.142, 0.142])

    elif data_id == 2:
        mean_net = torch.tensor([0.433, 0.433, 0.433])
        std_net = torch.tensor([0.118, 0.118, 0.118])

    selected_transform_suite = []

    if rotate:
        selected_transform_suite.append(
            K.RandomRotation(degrees=20.0, p=0.5, align_corners=True)
        )

    if perspective:
        selected_transform_suite.append(
            K.RandomPerspective(distortion_scale=0.001, p=0.5, align_corners=True)
        )

    if erase:
        selected_transform_suite.append(
            K.RandomErasing(scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.0, p=0.4)
        )

    if scale:
        selected_transform_suite.append(
            K.RandomAffine(degrees=0.0, scale=(0.5, 1.5), p=0.5, align_corners=True)
        )

    if hflip:
        selected_transform_suite.append(K.RandomHorizontalFlip(p=0.5))

    if translate:
        selected_transform_suite.append(
            K.RandomAffine(degrees=0.0, translate=(0.1, 0.0), p=0.5, align_corners=True)
        )

    if hsv:
        selected_transform_suite.append(
            K.ColorJiggle(
                brightness=0.4, contrast=0.0, saturation=0.7, hue=0.015, p=0.5
            )
        )

    if len(selected_transform_suite) != 0:
        train_transform_suite = K.container.AugmentationSequential(
            K.container.AugmentationSequential(
                *selected_transform_suite,
                data_keys=["image", "mask"],
                same_on_batch=False,
                random_apply=True
            ),
            K.Normalize(mean=mean_net, std=std_net),
            data_keys=["image", "mask"],
            same_on_batch=False,
            random_apply=False,
        )

    else:
        train_transform_suite = K.container.AugmentationSequential(
            K.Normalize(mean=mean_net, std=std_net),
            data_keys=["image", "mask"],
            same_on_batch=False,
            random_apply=False,
        )

    test_transform_suite = K.container.AugmentationSequential(
        K.Normalize(mean=mean_net, std=std_net),
        data_keys=["image", "mask"],
        same_on_batch=False,
        random_apply=False,
    )

    return train_transform_suite, test_transform_suite


def build_dataloaders(
    main_dir,
    data_id=2,
    transform=False,
    batch_size=64,
    num_workers=0,
    pin_memory=False,
    rotate=False,
    perspective=False,
    erase=False,
    scale=False,
    hflip=False,
    translate=False,
    hsv=False,
):
    """
    Build the dataloaders

    Arguments:
        main_dir (str): Main directory where the custom dataset is stored
        data_id (int): ID of the dataset for retrieving stats for image normalization
        transform (bool): Flag to decided whether/not to apply any augmentations
        batch_size (int): Batch size for data loaders
        num_workers (int): Number of subprocesses to use for data loading
        pin_memory (bool): Flag to decide whether/not to copy tensors into device/CUDA pinned memory before returning them
        rotate (bool): Flag to decide whether/not to apply rotation
        perspective (bool): Flag to decide whether/not to apply a perspective transformation
        erase (bool): Flag to decide whether/not to apply erasing
        scale (bool): Flag to decide whether/not to apply scaling
        hflip (bool): Flag to decide whether/not to affect horizontal flips
        translate (bool): Flag to decide whether/not to affect translation
        hsv (bool): Flag to decide whether/not to apply color jiggling

    Returns:
        train_dataloader (torch.utils.data.DataLoader): Training Set data loader
        val_dataloader (torch.utils.data.DataLoader): Validation Set data loader
        test_dataloader (torch.utils.data.DataLoader): Test Set data loader
    """

    train_set = CustomSegmentationDataset(main_dir, split="train")
    val_set = CustomSegmentationDataset(main_dir, split="val")
    test_set = CustomSegmentationDataset(main_dir, split="test")

    if not transform:
        train_time_transform, test_time_transform = None, None

    else:
        train_time_transform, test_time_transform = get_transform_suite(
            data_id=data_id,
            rotate=rotate,
            perspective=perspective,
            erase=erase,
            scale=scale,
            hflip=hflip,
            translate=translate,
            hsv=hsv,
        )

    train_dataloader = CustomDataloader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        transform_suite=train_time_transform,
    )

    val_dataloader = CustomDataloader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        transform_suite=test_time_transform,
    )

    test_dataloader = CustomDataloader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        transform_suite=test_time_transform,
    )

    return train_dataloader, val_dataloader, test_dataloader


def reverse_img_normalization(img, data_id=2):
    """
    Reversal of Image Normalization (for better visualization)

    Arguments:
        img (torch.tensor): Image
        data_id (int): ID of the dataset for retrieving stats for reversal of image normalization
    """

    if data_id == 1:
        reverse_norm_transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.0, 0.0, 0.0], std=[1 / 0.142, 1 / 0.142, 1 / 0.142]
                ),
                transforms.Normalize(
                    mean=[-0.423, -0.423, -0.423], std=[1.0, 1.0, 1.0]
                ),
            ]
        )

    elif data_id == 2:
        reverse_norm_transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.0, 0.0, 0.0], std=[1 / 0.118, 1 / 0.118, 1 / 0.118]
                ),
                transforms.Normalize(
                    mean=[-0.433, -0.433, -0.433], std=[1.0, 1.0, 1.0]
                ),
            ]
        )

    return reverse_norm_transform(img)
