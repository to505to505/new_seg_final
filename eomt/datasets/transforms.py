# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from Detectron2 by Facebook, Inc. and its affiliates,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------

import cv2
import numpy as np
import torch
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from torchvision.tv_tensors import wrap, TVTensor, Mask
from torch import nn, Tensor
from typing import Any, Union

from training.skeleton_utils import compute_instance_skeletons


class Transforms(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        color_jitter_enabled: bool,
        scale_range: tuple[float, float],
        max_brightness_delta: int = 32,
        max_contrast_factor: float = 0.5,
        saturation_factor: float = 0.5,
        max_hue_delta: int = 18,
        skeleton_enabled: bool = False,
        skeleton_num_dilations: int = 2,
        extra_augmentations_enabled: bool = False,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
    ):
        super().__init__()

        self.img_size = img_size
        self.color_jitter_enabled = color_jitter_enabled
        self.skeleton_enabled = skeleton_enabled
        self.skeleton_num_dilations = skeleton_num_dilations
        self.extra_augmentations_enabled = extra_augmentations_enabled
        self.max_brightness_factor = max_brightness_delta / 255.0
        self.max_contrast_factor = max_contrast_factor
        self.max_saturation_factor = saturation_factor
        self.max_hue_delta = max_hue_delta / 360.0

        self.random_horizontal_flip = T.RandomHorizontalFlip(p=hflip_prob)
        self.random_vertical_flip = T.RandomVerticalFlip(p=vflip_prob)
        self.scale_jitter = T.ScaleJitter(target_size=img_size, scale_range=scale_range)
        self.random_crop = T.RandomCrop(img_size)

    def _random_factor(self, factor: float, center: float = 1.0):
        return torch.empty(1).uniform_(center - factor, center + factor).item()

    def _brightness(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_brightness(
                img, self._random_factor(self.max_brightness_factor)
            )

        return img

    def _contrast(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_contrast(img, self._random_factor(self.max_contrast_factor))

        return img

    def _saturation_and_hue(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_saturation(
                img, self._random_factor(self.max_saturation_factor)
            )

        if torch.rand(()) < 0.5:
            img = F.adjust_hue(img, self._random_factor(self.max_hue_delta, center=0.0))

        return img

    def color_jitter(self, img):
        if not self.color_jitter_enabled:
            return img

        img = self._brightness(img)

        if torch.rand(()) < 0.5:
            img = self._contrast(img)
            img = self._saturation_and_hue(img)
        else:
            img = self._saturation_and_hue(img)
            img = self._contrast(img)

        return img

    # ------------------------------------------------------------------
    # Extra augmentations (enabled via extra_augmentations_enabled flag)
    # ------------------------------------------------------------------

    # --- photometric (image-only) ---

    def _extra_brightness_contrast(self, img):
        """RandomBrightnessContrast: brightness ±0.3, contrast ±0.3, p=0.5"""
        if torch.rand(()) < 0.5:
            bf = 1.0 + torch.empty(1).uniform_(-0.3, 0.3).item()
            img = F.adjust_brightness(img, bf)
            cf = 1.0 + torch.empty(1).uniform_(-0.3, 0.3).item()
            img = F.adjust_contrast(img, cf)
        return img

    def _clahe(self, img):
        """CLAHE on L-channel of LAB, clip_limit=4.0, p=0.3"""
        if torch.rand(()) < 0.3:
            np_img = img.permute(1, 2, 0).cpu().numpy()
            lab = cv2.cvtColor(np_img, cv2.COLOR_RGB2LAB)
            clahe_op = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe_op.apply(lab[:, :, 0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            img = wrap(torch.from_numpy(result).permute(2, 0, 1).to(img.device), like=img)
        return img

    def _random_gamma(self, img):
        """RandomGamma: gamma_range 80-120 -> 0.8-1.2, p=0.3"""
        if torch.rand(()) < 0.3:
            gamma = torch.empty(1).uniform_(0.8, 1.2).item()
            img = F.adjust_gamma(img, gamma)
        return img

    def _gaussian_blur(self, img):
        """GaussianBlur: kernel_size=3, p=0.2"""
        if torch.rand(()) < 0.2:
            img = F.gaussian_blur(img, kernel_size=3)
        return img

    def _gauss_noise(self, img):
        """GaussNoise: std 0.01-0.05 (relative to [0,255]), p=0.2"""
        if torch.rand(()) < 0.2:
            std = torch.empty(1).uniform_(0.01, 0.05).item() * 255.0
            noise = torch.randn_like(img.float()) * std
            img = wrap((img.float() + noise).clamp(0, 255).to(torch.uint8), like=img)
        return img

    def _sharpen(self, img):
        """Sharpen: alpha=(0.2,0.5) mapped to sharpness_factor=(1.2,1.5), p=0.2"""
        if torch.rand(()) < 0.2:
            alpha = torch.empty(1).uniform_(0.2, 0.5).item()
            factor = 1.0 + alpha  # [1.2, 1.5]
            img = F.adjust_sharpness(img, factor)
        return img

    def extra_photometric(self, img):
        if not self.extra_augmentations_enabled:
            return img
        img = self._extra_brightness_contrast(img)
        img = self._clahe(img)
        img = self._random_gamma(img)
        img = self._gaussian_blur(img)
        img = self._gauss_noise(img)
        img = self._sharpen(img)
        return img

    # --- geometric (image + masks) ---

    def extra_geometric(self, img, target):
        if not self.extra_augmentations_enabled:
            return img, target

        # VerticalFlip, p=0.5
        img, target = self.random_vertical_flip(img, target)

        # Rotate ±20°, p=0.3
        if torch.rand(()) < 0.3:
            angle = torch.empty(1).uniform_(-20, 20).item()
            img = F.rotate(img, angle, interpolation=T.InterpolationMode.BILINEAR, fill=0)
            target["masks"] = Mask(
                F.rotate(target["masks"], angle, interpolation=T.InterpolationMode.NEAREST, fill=0)
            )

        # Affine: scale 0.9-1.1, translate ±5%, p=0.3
        if torch.rand(()) < 0.3:
            scale = torch.empty(1).uniform_(0.9, 1.1).item()
            h, w = img.shape[-2], img.shape[-1]
            tx = torch.empty(1).uniform_(-0.05 * w, 0.05 * w).item()
            ty = torch.empty(1).uniform_(-0.05 * h, 0.05 * h).item()
            img = F.affine(
                img, angle=0, translate=(tx, ty), scale=scale, shear=0,
                interpolation=T.InterpolationMode.BILINEAR, fill=0,
            )
            target["masks"] = Mask(
                F.affine(
                    target["masks"], angle=0, translate=(tx, ty), scale=scale, shear=0,
                    interpolation=T.InterpolationMode.NEAREST, fill=0,
                )
            )

        # Perspective: distortion_scale sampled from (0.02, 0.05), p=0.15
        if torch.rand(()) < 0.15:
            h, w = img.shape[-2], img.shape[-1]
            distortion_scale = torch.empty(1).uniform_(0.02, 0.05).item()
            startpoints, endpoints = T.RandomPerspective.get_params(w, h, distortion_scale)
            img = F.perspective(img, startpoints, endpoints,
                                interpolation=T.InterpolationMode.BILINEAR, fill=0)
            target["masks"] = Mask(
                F.perspective(target["masks"], startpoints, endpoints,
                              interpolation=T.InterpolationMode.NEAREST, fill=0)
            )

        return img, target

    # ------------------------------------------------------------------

    def pad(
        self, img: Tensor, target: dict[str, Any]
    ) -> tuple[Tensor, dict[str, Union[Tensor, TVTensor]]]:
        pad_h = max(0, self.img_size[-2] - img.shape[-2])
        pad_w = max(0, self.img_size[-1] - img.shape[-1])
        padding = [0, 0, pad_w, pad_h]

        img = F.pad(img, padding)
        target["masks"] = F.pad(target["masks"], padding)

        return img, target

    def _filter(self, target: dict[str, Union[Tensor, TVTensor]], keep: Tensor) -> dict:
        return {k: wrap(v[keep], like=v) for k, v in target.items()}

    def forward(
        self, img: Tensor, target: dict[str, Union[Tensor, TVTensor]]
    ) -> tuple[Tensor, dict[str, Union[Tensor, TVTensor]]]:
        img_orig, target_orig = img, target

        target = self._filter(target, ~target["is_crowd"])

        img = self.color_jitter(img)
        img = self.extra_photometric(img)
        img, target = self.random_horizontal_flip(img, target)
        img, target = self.extra_geometric(img, target)
        img, target = self.scale_jitter(img, target)
        img, target = self.pad(img, target)
        img, target = self.random_crop(img, target)

        valid = target["masks"].flatten(1).any(1)
        if not valid.any():
            return self(img_orig, target_orig)

        target = self._filter(target, valid)

        if self.skeleton_enabled:
            target["skeletons"] = Mask(
                compute_instance_skeletons(target["masks"], self.skeleton_num_dilations)
            )

        return img, target
