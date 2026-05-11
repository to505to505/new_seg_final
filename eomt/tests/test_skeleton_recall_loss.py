import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest
from training.skeleton_recall_loss import SkeletonRecallLoss


class TestSkeletonRecallLoss:
    def setup_method(self):
        self.loss_fn = SkeletonRecallLoss(smooth=1.0)

    def test_perfect_prediction_gives_low_loss(self):
        """When prediction perfectly covers skeleton, loss should be near -1."""
        skel = torch.zeros(2, 32, 32)
        skel[:, 14:18, 8:24] = 1.0
        # Large positive logits → sigmoid ≈ 1.0 where skeleton is
        pred_logits = torch.full((2, 32, 32), -10.0)
        pred_logits[:, 14:18, 8:24] = 10.0
        loss = self.loss_fn(pred_logits, skel)
        # recall ≈ 1.0 → loss ≈ -1.0
        assert loss.item() < -0.9

    def test_zero_prediction_gives_high_loss(self):
        """When prediction is zero everywhere, recall is low."""
        skel = torch.zeros(2, 32, 32)
        skel[:, 14:18, 8:24] = 1.0
        pred_logits = torch.full((2, 32, 32), -10.0)  # sigmoid ≈ 0
        loss = self.loss_fn(pred_logits, skel)
        # recall ≈ 0 → loss ≈ -smooth/(sum_skel+smooth) ≈ small negative
        assert loss.item() > -0.1

    def test_random_prediction_intermediate_loss(self):
        skel = torch.zeros(3, 32, 32)
        skel[:, 10:22, 10:22] = 1.0
        pred_logits = torch.randn(3, 32, 32)
        loss = self.loss_fn(pred_logits, skel)
        assert -1.0 < loss.item() < 0.0

    def test_gradient_flow(self):
        skel = torch.zeros(2, 32, 32)
        skel[:, 14:18, 8:24] = 1.0
        pred_logits = torch.randn(2, 32, 32, requires_grad=True)
        loss = self.loss_fn(pred_logits, skel)
        loss.backward()
        assert pred_logits.grad is not None
        assert pred_logits.grad.abs().sum() > 0

    def test_empty_instances_returns_zero(self):
        pred_logits = torch.randn(0, 32, 32)
        skel = torch.zeros(0, 32, 32)
        loss = self.loss_fn(pred_logits, skel)
        assert loss.item() == 0.0

    def test_single_instance(self):
        skel = torch.zeros(1, 32, 32)
        skel[0, 15, 8:24] = 1.0
        pred_logits = torch.full((1, 32, 32), 10.0)
        loss = self.loss_fn(pred_logits, skel)
        assert loss.item() < -0.9

    def test_empty_skeleton_stability(self):
        """Instance with empty skeleton should not cause NaN/Inf."""
        skel = torch.zeros(2, 32, 32)
        # Only first instance has skeleton
        skel[0, 14:18, 8:24] = 1.0
        # Second has no skeleton at all
        pred_logits = torch.randn(2, 32, 32)
        loss = self.loss_fn(pred_logits, skel)
        assert torch.isfinite(loss)

    def test_loss_decreases_with_better_coverage(self):
        skel = torch.zeros(1, 32, 32)
        skel[0, 14:18, 8:24] = 1.0
        # Bad prediction: covers nothing
        bad_logits = torch.full((1, 32, 32), -10.0)
        # Good prediction: covers skeleton
        good_logits = torch.full((1, 32, 32), -10.0)
        good_logits[0, 14:18, 8:24] = 10.0
        loss_bad = self.loss_fn(bad_logits, skel)
        loss_good = self.loss_fn(good_logits, skel)
        # Better coverage → more negative loss (lower)
        assert loss_good.item() < loss_bad.item()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
