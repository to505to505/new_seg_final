import torch
import torch.nn as nn
from torch import Tensor


class SkeletonRecallLoss(nn.Module):
    """Skeleton Recall Loss adapted for instance segmentation.

    Computes soft recall of predicted mask logits on precomputed tubed
    skeleton ground truth masks. Operates on matched query-target pairs
    after Hungarian matching.

    Reference: "Skeleton Recall Loss for Connectivity Conserving and
    Resource Efficient Segmentation of Thin Tubular Structures" (ECCV 2024)
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred_mask_logits: Tensor, skeleton_targets: Tensor) -> Tensor:
        """Compute skeleton recall loss on matched instance pairs.

        Args:
            pred_mask_logits: (M, H, W) raw logits for M matched queries.
            skeleton_targets: (M, H, W) binary skeleton masks for M matched GT instances.

        Returns:
            Scalar loss value (negated mean recall).
        """
        if pred_mask_logits.shape[0] == 0:
            return pred_mask_logits.sum() * 0.0

        pred = pred_mask_logits.sigmoid()
        skel = skeleton_targets.float()

        # Per-instance: recall = sum(pred * skel) / sum(skel)
        axes = list(range(1, pred.ndim))  # spatial dims
        intersection = (pred * skel).sum(axes)
        sum_skel = skel.sum(axes)

        # Avoid division by zero for instances with empty skeletons
        recall = (intersection + self.smooth) / (sum_skel + self.smooth).clamp(min=1e-8)

        return -recall.mean()
