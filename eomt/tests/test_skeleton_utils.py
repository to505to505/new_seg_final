import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np
import pytest
from training.skeleton_utils import compute_instance_skeletons


def _make_circle_mask(h, w, cy, cx, r):
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= r ** 2


def _make_line_mask(h, w, thickness=3):
    mask = np.zeros((h, w), dtype=bool)
    cy = h // 2
    mask[cy - thickness // 2 : cy + thickness // 2 + 1, w // 4 : 3 * w // 4] = True
    return mask


def _make_l_shape_mask(h, w, thickness=5):
    mask = np.zeros((h, w), dtype=bool)
    # Vertical bar
    mask[h // 4 : 3 * h // 4, w // 4 : w // 4 + thickness] = True
    # Horizontal bar
    mask[3 * h // 4 - thickness : 3 * h // 4, w // 4 : 3 * w // 4] = True
    return mask


class TestComputeInstanceSkeletons:
    def test_circle_skeleton_subset_of_mask(self):
        mask = torch.from_numpy(_make_circle_mask(64, 64, 32, 32, 20)).unsqueeze(0)
        skel = compute_instance_skeletons(mask, num_dilations=0)
        # Raw skeleton must be a subset of the original mask
        assert (skel & ~mask).sum() == 0
        # Skeleton should be non-empty for a filled circle
        assert skel.any()

    def test_tubed_skeleton_wider_than_raw(self):
        mask = torch.from_numpy(_make_circle_mask(64, 64, 32, 32, 20)).unsqueeze(0)
        skel_raw = compute_instance_skeletons(mask, num_dilations=0)
        skel_tubed = compute_instance_skeletons(mask, num_dilations=2)
        assert skel_tubed.sum() >= skel_raw.sum()

    def test_tubed_skeleton_within_mask(self):
        mask = torch.from_numpy(_make_circle_mask(64, 64, 32, 32, 20)).unsqueeze(0)
        skel = compute_instance_skeletons(mask, num_dilations=2)
        # Tubed skeleton is clipped to original mask
        assert (skel & ~mask).sum() == 0

    def test_empty_mask_produces_empty_skeleton(self):
        mask = torch.zeros(1, 64, 64, dtype=torch.bool)
        skel = compute_instance_skeletons(mask)
        assert skel.sum() == 0

    def test_line_mask_skeleton(self):
        mask = torch.from_numpy(_make_line_mask(64, 64, thickness=5)).unsqueeze(0)
        skel = compute_instance_skeletons(mask, num_dilations=0)
        assert skel.any()
        assert (skel & ~mask).sum() == 0

    def test_l_shape_skeleton(self):
        mask = torch.from_numpy(_make_l_shape_mask(64, 64, thickness=7)).unsqueeze(0)
        skel = compute_instance_skeletons(mask, num_dilations=0)
        assert skel.any()
        assert (skel & ~mask).sum() == 0

    def test_multi_instance_independent(self):
        m1 = _make_circle_mask(64, 64, 16, 16, 10)
        m2 = _make_circle_mask(64, 64, 48, 48, 10)
        masks = torch.from_numpy(np.stack([m1, m2]))
        skels = compute_instance_skeletons(masks, num_dilations=0)
        assert skels.shape == (2, 64, 64)
        # Each skeleton only in its own mask region
        assert (skels[0] & torch.from_numpy(m2)).sum() == 0
        assert (skels[1] & torch.from_numpy(m1)).sum() == 0

    def test_zero_instances(self):
        masks = torch.zeros(0, 64, 64, dtype=torch.bool)
        skels = compute_instance_skeletons(masks)
        assert skels.shape == (0, 64, 64)

    def test_output_dtype_is_bool(self):
        mask = torch.from_numpy(_make_circle_mask(64, 64, 32, 32, 15)).unsqueeze(0)
        skel = compute_instance_skeletons(mask)
        assert skel.dtype == torch.bool


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
