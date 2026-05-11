"""
Connectivity Consistency Loss (Self-Supervised)

Differentiable Region Growing Loss that enforces connectivity of predictions.
Uses GT-guided seeding during training: the seed point is placed using the
Ground Truth mask, ensuring the "water flow" starts from the correct location.

The loss penalizes disconnected fragments in predictions by computing:
    loss = (total_predicted_mass - connected_mass) / total_predicted_mass

This forces the network to either:
1. Connect broken segments (so water can flow through)
2. Remove isolated noise (since it can't be reached from the GT center)

Works as a regularizer alongside supervised losses like Dice.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ConnectivityConsistencyLoss(nn.Module):
    """
    Differentiable Region Growing Loss for enforcing connectivity in predictions.

    Adapted for instance segmentation: operates on matched (M, H, W) pairs
    of predicted logits and ground truth masks after Hungarian matching.

    Args:
        iterations: Number of dilation iterations. If None, uses max(H, W) // 2.
        kernel_size: Size of the dilation kernel (default: 3x3).
        grow_threshold: Threshold for region growing (lower = more permissive).
    """

    def __init__(
        self,
        iterations: int = None,
        kernel_size: int = 3,
        grow_threshold: float = 0.3,
    ):
        super().__init__()
        self.iterations = iterations
        self.kernel_size = kernel_size
        self.grow_threshold = grow_threshold

        kernel = torch.ones(1, 1, kernel_size, kernel_size)
        self.register_buffer("kernel", kernel)

    def _get_gt_center_of_mass(self, gt_mask: Tensor) -> tuple:
        """Compute center of mass of a single GT mask (H, W)."""
        H, W = gt_mask.shape
        y_indices, x_indices = torch.where(gt_mask > 0.5)

        if len(y_indices) == 0:
            return H // 2, W // 2

        cy = y_indices.float().mean().long().item()
        cx = x_indices.float().mean().long().item()
        return cy, cx

    def _create_seed(self, pred: Tensor, gt_mask: Tensor) -> Tensor:
        """
        Create seed mask from intersection of GT and high-confidence predictions.

        Args:
            pred: Prediction probabilities (1, 1, H, W)
            gt_mask: Ground truth mask (1, 1, H, W)

        Returns:
            seed: (1, 1, H, W) seed region
        """
        H, W = pred.shape[-2:]

        gt_binary = (gt_mask > 0.5).float()
        high_conf_pred = (pred > self.grow_threshold).float()
        seed = gt_binary * high_conf_pred

        if seed.sum() < 1:
            cy, cx = self._get_gt_center_of_mass(gt_mask.squeeze(0).squeeze(0))
            cy = max(0, min(cy, H - 1))
            cx = max(0, min(cx, W - 1))
            seed = torch.zeros(1, 1, H, W, device=pred.device, dtype=pred.dtype)
            seed[0, 0, cy, cx] = 1.0

        return seed

    def _region_grow(self, pred: Tensor, seed: Tensor, iterations: int) -> Tensor:
        """
        Differentiable region growing from seed, constrained by prediction.

        Args:
            pred: Prediction probabilities (1, 1, H, W)
            seed: Seed mask (1, 1, H, W)
            iterations: Number of dilation iterations

        Returns:
            connected_region: (1, 1, H, W) reachable region masked by pred
        """
        padding = self.kernel_size // 2

        pred_detached = pred.detach()
        connectivity_mask = (pred_detached > self.grow_threshold).float()

        grown_region = seed.clone()

        for _ in range(iterations):
            dilated = F.conv2d(grown_region, self.kernel, padding=padding)
            dilated = (dilated > 0).float()
            grown_region = dilated * connectivity_mask

        # Final masking is differentiable through pred
        connected_region = grown_region.detach() * pred
        return connected_region

    def forward(self, pred_mask_logits: Tensor, gt_masks: Tensor) -> Tensor:
        """
        Compute connectivity consistency loss on matched instance pairs.

        Args:
            pred_mask_logits: (M, H, W) raw logits for M matched queries.
            gt_masks: (M, H, W) binary GT masks for M matched instances.

        Returns:
            Scalar loss value in [0, 1].
        """
        if pred_mask_logits.shape[0] == 0:
            return pred_mask_logits.sum() * 0.0

        pred_probs = pred_mask_logits.sigmoid()

        M, H, W = pred_probs.shape
        iterations = self.iterations if self.iterations is not None else max(H, W) // 2

        losses = []

        for i in range(M):
            pred_i = pred_probs[i : i + 1].unsqueeze(0)  # (1, 1, H, W)
            gt_i = gt_masks[i : i + 1].unsqueeze(0).float()  # (1, 1, H, W)

            seed = self._create_seed(pred_i, gt_i)
            grown_region = self._region_grow(pred_i, seed, iterations)

            total_mass = pred_i.sum()
            connected_mass = grown_region.sum()

            loss_i = (total_mass - connected_mass) / (total_mass + 1e-6)
            loss_i = loss_i.clamp(0, 1)
            losses.append(loss_i)

        return torch.stack(losses).mean()
