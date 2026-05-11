import os
import torch
import copy
import random
import cv2
import numpy as np

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import kornia.augmentation as K

import albumentations as A
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

    def __init__(self, main_dir: str):
        self.img_dir = os.path.join(main_dir)
        self.img_name_net = sorted(os.listdir(self.img_dir))
        self.convert_to_scaled_tensor = ToTensor()

    def __len__(self):
        return len(self.img_name_net)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_name_net[idx])

        img = self.convert_to_scaled_tensor(Image.open(img_path).convert("RGB"))

        return img


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
        for img_batch in self.pytorch_dataloader:
            if self.transform_suite is not None:
                img_batch = self.transform_suite(img_batch)

            yield img_batch


def build_dataloader(
    main_dir, transform=False, batch_size=64, num_workers=0, pin_memory=False
):
    """
    Build the dataloader

    Arguments:
        main_dir (str): Main directory where the custom dataset is stored
        transform (bool): Flag to decided whether/not to apply any augmentations
        batch_size (int): Batch size for data loader
        num_workers (int): Number of subprocesses to use for data loading
        pin_memory (bool): Flag to decide whether/not to copy tensors into device/CUDA pinned memory before returning them

    Returns:
        ext_dataloader (torch.utils.data.DataLoader): Data loader
    """

    ext_set = CustomSegmentationDataset(main_dir)

    mean_net = torch.tensor([0.433, 0.433, 0.433])
    std_net = torch.tensor([0.118, 0.118, 0.118])

    if not transform:
        transform = None

    else:
        transform = K.Normalize(mean=mean_net, std=std_net)

    ext_dataloader = CustomDataloader(
        ext_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        transform_suite=transform,
    )

    return ext_dataloader


def reverse_img_normalization(img):
    """
    Reversal of Image Normalization (for better visualization)

    Arguments:
        img (torch.tensor): Image
    """

    reverse_norm_transform = transforms.Compose(
        [
            transforms.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1 / 0.118, 1 / 0.118, 1 / 0.118]
            ),
            transforms.Normalize(mean=[-0.433, -0.433, -0.433], std=[1.0, 1.0, 1.0]),
        ]
    )

    return reverse_norm_transform(img)


def resize(main_dir, img_file):
    """
    Resizing of input images to 512 x 512 for lesion detection

    Arguments:
        main_dir (str): Main dataset directory
        img_file (str): Image name
    """

    img_path = os.path.join(main_dir, img_file)

    orig_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    orig_img_w, orig_img_h = orig_img.shape[0], orig_img.shape[1]

    if orig_img_w != 512 or orig_img_h != 512:
        print(f"Resizing {img_path}")
        os.remove(img_path)
        r_img = A.Resize(p=1.0, height=512, width=512)(image=orig_img)["image"]
        cv2.imwrite(img_path, r_img)


def revert_to_orig_cropped_sizes(main_dir, orig_h, orig_w, lc_img_file, lc_mask_file):
    """
    Reverting lesion segmentation masks to their original (before resizing them) cropped sizes

    Arguments:
        main_dir (str): Main dataset directory
        orig_h (int): Original height of the cropped lesion
        orig_w (int): Original width of the cropped lesion
        lc_img_file (str): Image name of the cropped lesion
        lc_mask_file (str): Mask name of the cropped lesion
    """

    lc_img_path = os.path.join(main_dir, "cropped_lesion_images", lc_img_file)
    lc_mask_path = os.path.join(main_dir, "LSEG_RES", lc_mask_file)

    lc_img = cv2.imread(lc_img_path, cv2.IMREAD_GRAYSCALE)
    os.remove(lc_img_path)
    lc_mask = cv2.imread(lc_mask_path, cv2.IMREAD_GRAYSCALE)
    os.remove(lc_mask_path)

    r_set = A.Resize(p=1.0, height=orig_h, width=orig_w)(image=lc_img, mask=lc_mask)
    cv2.imwrite(lc_img_path, r_set["image"])
    cv2.imwrite(lc_mask_path, r_set["mask"])


def crop_and_resize(
    orig_img_pre_path,
    orig_img_fmtd_pre_path,
    bbox_coords_pre_path,
    bbox_coords_file,
    orig_img_file_ext,
):
    """
    Cropping detecting lesions and resizing them to 256 X 256 for lesion segmentation

    Arguments:
        orig_img_pre_path (str): Original image dataset directory
        orig_img_fmtd_pre_path (str): Dataset directory for cropped lesion images
        bbox_coords_pre_path (int): Dataset directory for detected bounding box coordinates
        bbox_coords_file (str): Detected bounding box coordinates file name
        orig_img_file_ext (str): Original image file extension
    """

    orig_img_file = bbox_coords_file[:-4] + orig_img_file_ext

    orig_img = cv2.imread(
        os.path.join(orig_img_pre_path, orig_img_file), cv2.IMREAD_GRAYSCALE
    )

    with open(
        os.path.join(bbox_coords_pre_path, bbox_coords_file), "r"
    ) as bbox_coords_file:
        txt_bbox_coords_net = bbox_coords_file.readlines()

    bbox_coords_net = []

    for i in range(len(txt_bbox_coords_net)):
        yolo_bbox_coords = list(map(float, txt_bbox_coords_net[i][:-1].split()))[1:5]

        x1 = int(256 * (2 * yolo_bbox_coords[0] - yolo_bbox_coords[2]))
        x2 = int(256 * (2 * yolo_bbox_coords[0] + yolo_bbox_coords[2]))
        y1 = int(256 * (2 * yolo_bbox_coords[1] - yolo_bbox_coords[3]))
        y2 = int(256 * (2 * yolo_bbox_coords[1] + yolo_bbox_coords[3]))

        bbox_coords_net.append([x1, x2, y1, y2])

    orig_lc_size_net = []
    resize_transform = A.Resize(p=1.0, height=256, width=256)

    for i in range(len(bbox_coords_net)):
        cropped_img = orig_img[
            bbox_coords_net[i][2] : bbox_coords_net[i][3],
            bbox_coords_net[i][0] : bbox_coords_net[i][1],
        ]

        r_img = resize_transform(image=cropped_img)["image"]

        img_tag = chr(97 + i).upper()

        new_img_file = orig_img_file[:-4] + f"_{img_tag}" + orig_img_file_ext

        orig_lc_size_net.append(
            {
                "image_name": new_img_file,
                "height": cropped_img.shape[0],
                "width": cropped_img.shape[1],
            }
        )

        cv2.imwrite(os.path.join(orig_img_fmtd_pre_path, new_img_file), r_img)

    return orig_lc_size_net
