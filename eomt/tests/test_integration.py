import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import pytest
from torchvision import tv_tensors

from datasets.transforms import Transforms
from training.mask_classification_loss import MaskClassificationLoss


def _make_dummy_targets(batch_size, num_instances, h, w, with_skeletons=False, num_classes=1):
    """Create dummy targets matching EoMT's expected format."""
    targets = []
    for _ in range(batch_size):
        masks = torch.zeros(num_instances, h, w, dtype=torch.bool)
        for i in range(num_instances):
            y0 = (i * 10) % (h - 20)
            masks[i, y0:y0+15, 10:40] = True
        target = {
            "masks": tv_tensors.Mask(masks),
            "labels": torch.randint(0, num_classes, (num_instances,), dtype=torch.long),
            "is_crowd": torch.zeros(num_instances, dtype=torch.bool),
        }
        if with_skeletons:
            # Simple skeleton: center line of each mask
            skels = torch.zeros(num_instances, h, w, dtype=torch.bool)
            for i in range(num_instances):
                y0 = (i * 10) % (h - 20)
                skels[i, y0+7, 12:38] = True
            target["skeletons"] = tv_tensors.Mask(skels)
        targets.append(target)
    return targets


class TestTransformsWithSkeleton:
    def test_skeleton_computed_in_transforms(self):
        transforms = Transforms(
            img_size=(64, 64),
            color_jitter_enabled=False,
            scale_range=(1.0, 1.0),
            skeleton_enabled=True,
            skeleton_num_dilations=2,
        )
        # Create a test image and target
        img = tv_tensors.Image(torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8))
        mask = torch.zeros(1, 64, 64, dtype=torch.bool)
        mask[0, 20:44, 10:54] = True
        target = {
            "masks": tv_tensors.Mask(mask),
            "labels": torch.tensor([0]),
            "is_crowd": torch.tensor([False]),
        }
        img_out, target_out = transforms(img, target)
        assert "skeletons" in target_out
        assert target_out["skeletons"].shape[0] == target_out["masks"].shape[0]
        assert target_out["skeletons"].shape[1:] == target_out["masks"].shape[1:]

    def test_skeleton_disabled_no_key(self):
        transforms = Transforms(
            img_size=(64, 64),
            color_jitter_enabled=False,
            scale_range=(1.0, 1.0),
            skeleton_enabled=False,
        )
        img = tv_tensors.Image(torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8))
        mask = torch.zeros(1, 64, 64, dtype=torch.bool)
        mask[0, 20:44, 10:54] = True
        target = {
            "masks": tv_tensors.Mask(mask),
            "labels": torch.tensor([0]),
            "is_crowd": torch.tensor([False]),
        }
        img_out, target_out = transforms(img, target)
        assert "skeletons" not in target_out


class TestMaskClassificationLossWithSkeleton:
    def test_skeleton_loss_computed_when_enabled(self):
        loss_fn = MaskClassificationLoss(
            num_points=1024,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            mask_coefficient=5.0,
            dice_coefficient=5.0,
            class_coefficient=2.0,
            num_labels=1,
            no_object_coefficient=0.1,
            skeleton_coefficient=1.0,
        )
        B, Q, H, W = 2, 10, 64, 64
        mask_logits = torch.randn(B, Q, H, W)
        class_logits = torch.randn(B, Q, 2)  # num_classes + 1
        targets = _make_dummy_targets(B, 3, H, W, with_skeletons=True)

        losses = loss_fn(
            masks_queries_logits=mask_logits,
            class_queries_logits=class_logits,
            targets=targets,
        )
        assert "skeleton_recall" in losses
        assert torch.isfinite(losses["skeleton_recall"])

    def test_skeleton_loss_with_resolution_mismatch(self):
        """Mask logits at lower resolution than skeleton targets (real scenario)."""
        loss_fn = MaskClassificationLoss(
            num_points=1024,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            mask_coefficient=5.0,
            dice_coefficient=5.0,
            class_coefficient=2.0,
            num_labels=1,
            no_object_coefficient=0.1,
            skeleton_coefficient=1.0,
        )
        B, Q = 2, 10
        # Model outputs at lower resolution (e.g. 72), targets at full res (512)
        mask_logits = torch.randn(B, Q, 72, 72)
        class_logits = torch.randn(B, Q, 2)
        targets = _make_dummy_targets(B, 3, 512, 512, with_skeletons=True)

        losses = loss_fn(
            masks_queries_logits=mask_logits,
            class_queries_logits=class_logits,
            targets=targets,
        )
        assert "skeleton_recall" in losses
        assert torch.isfinite(losses["skeleton_recall"])

    def test_no_skeleton_loss_when_disabled(self):
        loss_fn = MaskClassificationLoss(
            num_points=1024,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            mask_coefficient=5.0,
            dice_coefficient=5.0,
            class_coefficient=2.0,
            num_labels=1,
            no_object_coefficient=0.1,
            skeleton_coefficient=0.0,
        )
        B, Q, H, W = 2, 10, 64, 64
        mask_logits = torch.randn(B, Q, H, W)
        class_logits = torch.randn(B, Q, 2)
        targets = _make_dummy_targets(B, 3, H, W, with_skeletons=False)

        losses = loss_fn(
            masks_queries_logits=mask_logits,
            class_queries_logits=class_logits,
            targets=targets,
        )
        assert "skeleton_recall" not in losses

    def test_loss_total_includes_skeleton(self):
        loss_fn = MaskClassificationLoss(
            num_points=1024,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            mask_coefficient=5.0,
            dice_coefficient=5.0,
            class_coefficient=2.0,
            num_labels=1,
            no_object_coefficient=0.1,
            skeleton_coefficient=2.0,
        )
        B, Q, H, W = 2, 10, 64, 64
        mask_logits = torch.randn(B, Q, H, W, requires_grad=True)
        class_logits = torch.randn(B, Q, 2, requires_grad=True)
        targets = _make_dummy_targets(B, 3, H, W, with_skeletons=True)

        losses = loss_fn(
            masks_queries_logits=mask_logits,
            class_queries_logits=class_logits,
            targets=targets,
        )

        logged = {}
        def mock_log(key, val, **kwargs):
            logged[key] = val

        # Flatten to single block (no postfix)
        total = loss_fn.loss_total(losses, mock_log)
        assert torch.isfinite(total)
        assert "losses/train_skeleton_recall" in logged

    def test_backward_pass_with_skeleton(self):
        loss_fn = MaskClassificationLoss(
            num_points=1024,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            mask_coefficient=5.0,
            dice_coefficient=5.0,
            class_coefficient=2.0,
            num_labels=1,
            no_object_coefficient=0.1,
            skeleton_coefficient=1.0,
        )
        B, Q, H, W = 2, 10, 64, 64
        mask_logits = torch.randn(B, Q, H, W, requires_grad=True)
        class_logits = torch.randn(B, Q, 2, requires_grad=True)
        targets = _make_dummy_targets(B, 3, H, W, with_skeletons=True)

        losses = loss_fn(
            masks_queries_logits=mask_logits,
            class_queries_logits=class_logits,
            targets=targets,
        )

        def mock_log(key, val, **kwargs):
            pass

        total = loss_fn.loss_total(losses, mock_log)
        total.backward()
        assert mask_logits.grad is not None
        assert mask_logits.grad.abs().sum() > 0


class TestMultiClassSkeletonRecall:
    """Verify skeleton recall works with multiple classes (num_classes > 1)."""

    def test_multiclass_skeleton_loss_computed(self):
        num_classes = 9
        loss_fn = MaskClassificationLoss(
            num_points=1024,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            mask_coefficient=5.0,
            dice_coefficient=5.0,
            class_coefficient=2.0,
            num_labels=num_classes,
            no_object_coefficient=0.1,
            skeleton_coefficient=1.0,
        )
        B, Q, H, W = 2, 10, 64, 64
        mask_logits = torch.randn(B, Q, H, W)
        class_logits = torch.randn(B, Q, num_classes + 1)
        targets = _make_dummy_targets(B, 5, H, W, with_skeletons=True, num_classes=num_classes)

        losses = loss_fn(
            masks_queries_logits=mask_logits,
            class_queries_logits=class_logits,
            targets=targets,
        )
        assert "skeleton_recall" in losses
        assert torch.isfinite(losses["skeleton_recall"])

    def test_multiclass_backward_pass(self):
        num_classes = 9
        loss_fn = MaskClassificationLoss(
            num_points=1024,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            mask_coefficient=5.0,
            dice_coefficient=5.0,
            class_coefficient=2.0,
            num_labels=num_classes,
            no_object_coefficient=0.1,
            skeleton_coefficient=1.0,
        )
        B, Q, H, W = 2, 10, 64, 64
        mask_logits = torch.randn(B, Q, H, W, requires_grad=True)
        class_logits = torch.randn(B, Q, num_classes + 1, requires_grad=True)
        targets = _make_dummy_targets(B, 5, H, W, with_skeletons=True, num_classes=num_classes)

        losses = loss_fn(
            masks_queries_logits=mask_logits,
            class_queries_logits=class_logits,
            targets=targets,
        )

        def mock_log(key, val, **kwargs):
            pass

        total = loss_fn.loss_total(losses, mock_log)
        total.backward()
        assert mask_logits.grad is not None
        assert mask_logits.grad.abs().sum() > 0

    def test_multiclass_loss_total_weighted(self):
        num_classes = 9
        skel_coeff = 3.0
        loss_fn = MaskClassificationLoss(
            num_points=1024,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            mask_coefficient=5.0,
            dice_coefficient=5.0,
            class_coefficient=2.0,
            num_labels=num_classes,
            no_object_coefficient=0.1,
            skeleton_coefficient=skel_coeff,
        )
        B, Q, H, W = 2, 10, 64, 64
        mask_logits = torch.randn(B, Q, H, W)
        class_logits = torch.randn(B, Q, num_classes + 1)
        targets = _make_dummy_targets(B, 5, H, W, with_skeletons=True, num_classes=num_classes)

        losses = loss_fn(
            masks_queries_logits=mask_logits,
            class_queries_logits=class_logits,
            targets=targets,
        )

        logged = {}
        def mock_log(key, val, **kwargs):
            logged[key] = val

        total = loss_fn.loss_total(losses, mock_log)
        assert torch.isfinite(total)
        assert "losses/train_skeleton_recall" in logged

    def test_multiclass_transforms_skeleton(self):
        """Skeleton computation works with multi-class instance masks."""
        transforms = Transforms(
            img_size=(64, 64),
            color_jitter_enabled=False,
            scale_range=(1.0, 1.0),
            skeleton_enabled=True,
            skeleton_num_dilations=2,
        )
        img = tv_tensors.Image(torch.randint(0, 255, (3, 64, 64), dtype=torch.uint8))
        # 3 instances with different class labels
        masks = torch.zeros(3, 64, 64, dtype=torch.bool)
        masks[0, 10:30, 10:30] = True
        masks[1, 35:55, 10:30] = True
        masks[2, 10:30, 35:55] = True
        target = {
            "masks": tv_tensors.Mask(masks),
            "labels": torch.tensor([0, 3, 7]),  # Different classes
            "is_crowd": torch.tensor([False, False, False]),
        }
        img_out, target_out = transforms(img, target)
        assert "skeletons" in target_out
        n_out = target_out["masks"].shape[0]
        assert target_out["skeletons"].shape[0] == n_out
        # Each skeleton should have some non-zero pixels
        for i in range(n_out):
            if target_out["masks"][i].any():
                assert target_out["skeletons"][i].any(), f"Instance {i} has mask but empty skeleton"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
