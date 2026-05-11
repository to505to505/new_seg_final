# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from the Hugging Face Transformers library,
# specifically from the Mask2Former loss implementation, which itself is based on
# Mask2Former and DETR by Facebook, Inc. and its affiliates.
# Used under the Apache 2.0 License.
# ---------------------------------------------------------------


from typing import List, Optional
import torch.distributed as dist
import torch
import torch.nn as nn
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerLoss,
    Mask2FormerHungarianMatcher,
)

from training.skeleton_recall_loss import SkeletonRecallLoss
from training.connectivity_loss import ConnectivityConsistencyLoss


class MaskClassificationLoss(Mask2FormerLoss):
    def __init__(
        self,
        num_points: int,
        oversample_ratio: float,
        importance_sample_ratio: float,
        mask_coefficient: float,
        dice_coefficient: float,
        class_coefficient: float,
        num_labels: int,
        no_object_coefficient: float,
        skeleton_coefficient: float = 0.0,
        connectivity_coefficient: float = 0.0,
    ):
        nn.Module.__init__(self)
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.mask_coefficient = mask_coefficient
        self.dice_coefficient = dice_coefficient
        self.class_coefficient = class_coefficient
        self.num_labels = num_labels
        self.eos_coef = no_object_coefficient
        empty_weight = torch.ones(self.num_labels + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        self.matcher = Mask2FormerHungarianMatcher(
            num_points=num_points,
            cost_mask=mask_coefficient,
            cost_dice=dice_coefficient,
            cost_class=class_coefficient,
        )

        self.skeleton_coefficient = skeleton_coefficient
        if skeleton_coefficient > 0:
            self.skeleton_recall_loss = SkeletonRecallLoss()

        self.connectivity_coefficient = connectivity_coefficient
        if connectivity_coefficient > 0:
            self.connectivity_loss = ConnectivityConsistencyLoss()

    @torch.compiler.disable
    def forward(
        self,
        masks_queries_logits: torch.Tensor,
        targets: List[dict],
        class_queries_logits: Optional[torch.Tensor] = None,
    ):
        mask_labels = [
            target["masks"].to(masks_queries_logits.dtype) for target in targets
        ]
        class_labels = [target["labels"].long() for target in targets]

        indices = self.matcher(
            masks_queries_logits=masks_queries_logits,
            mask_labels=mask_labels,
            class_queries_logits=class_queries_logits,
            class_labels=class_labels,
        )

        loss_masks = self.loss_masks(masks_queries_logits, mask_labels, indices)
        loss_classes = self.loss_labels(class_queries_logits, class_labels, indices)

        losses = {**loss_masks, **loss_classes}

        if self.skeleton_coefficient > 0 and "skeletons" in targets[0]:
            loss_skel = self._loss_skeleton_recall(
                masks_queries_logits, targets, indices
            )
            losses["skeleton_recall"] = loss_skel

        if self.connectivity_coefficient > 0:
            loss_conn = self._loss_connectivity(
                masks_queries_logits, targets, indices
            )
            losses["connectivity"] = loss_conn

        return losses

    def _loss_skeleton_recall(
        self,
        masks_queries_logits: torch.Tensor,
        targets: List[dict],
        indices: List[tuple],
    ) -> torch.Tensor:
        matched_pred_masks = []
        matched_skel_targets = []

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            matched_pred_masks.append(masks_queries_logits[b][src_idx])
            skel = targets[b]["skeletons"].to(masks_queries_logits.dtype)
            matched_skel_targets.append(skel[tgt_idx])

        if len(matched_pred_masks) == 0:
            return masks_queries_logits.sum() * 0.0

        pred = torch.cat(matched_pred_masks, dim=0)
        skel = torch.cat(matched_skel_targets, dim=0)

        # Upsample predictions to match skeleton resolution if sizes differ
        if pred.shape[-2:] != skel.shape[-2:]:
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=skel.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(1)

        return self.skeleton_recall_loss(pred, skel)

    def _loss_connectivity(
        self,
        masks_queries_logits: torch.Tensor,
        targets: List[dict],
        indices: List[tuple],
    ) -> torch.Tensor:
        matched_pred_masks = []
        matched_gt_masks = []

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            matched_pred_masks.append(masks_queries_logits[b][src_idx])
            gt = targets[b]["masks"].to(masks_queries_logits.dtype)
            matched_gt_masks.append(gt[tgt_idx])

        if len(matched_pred_masks) == 0:
            return masks_queries_logits.sum() * 0.0

        pred = torch.cat(matched_pred_masks, dim=0)
        gt = torch.cat(matched_gt_masks, dim=0)

        if pred.shape[-2:] != gt.shape[-2:]:
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1), size=gt.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(1)

        return self.connectivity_loss(pred, gt)

    def loss_masks(self, masks_queries_logits, mask_labels, indices):
        loss_masks = super().loss_masks(masks_queries_logits, mask_labels, indices, 1)

        num_masks = sum(len(tgt) for (_, tgt) in indices)
        num_masks_tensor = torch.as_tensor(
            num_masks, dtype=torch.float, device=masks_queries_logits.device
        )

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(num_masks_tensor)
            world_size = dist.get_world_size()
        else:
            world_size = 1

        num_masks = torch.clamp(num_masks_tensor / world_size, min=1)

        for key in loss_masks.keys():
            loss_masks[key] = loss_masks[key] / num_masks

        return loss_masks

    def loss_total(self, losses_all_layers, log_fn) -> torch.Tensor:
        loss_total = None
        for loss_key, loss in losses_all_layers.items():
            log_fn(f"losses/train_{loss_key}", loss, sync_dist=True)

            if "connectivity" in loss_key:
                weighted_loss = loss * self.connectivity_coefficient
            elif "skeleton_recall" in loss_key:
                weighted_loss = loss * self.skeleton_coefficient
            elif "mask" in loss_key:
                weighted_loss = loss * self.mask_coefficient
            elif "dice" in loss_key:
                weighted_loss = loss * self.dice_coefficient
            elif "cross_entropy" in loss_key:
                weighted_loss = loss * self.class_coefficient
            else:
                raise ValueError(f"Unknown loss key: {loss_key}")

            if loss_total is None:
                loss_total = weighted_loss
            else:
                loss_total = torch.add(loss_total, weighted_loss)

        log_fn("losses/train_loss_total", loss_total, sync_dist=True, prog_bar=True)

        return loss_total  # type: ignore
