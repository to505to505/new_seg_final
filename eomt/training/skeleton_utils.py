import numpy as np
import torch
from torch import Tensor
from skimage.morphology import skeletonize, dilation


def compute_instance_skeletons(masks: Tensor, num_dilations: int = 2) -> Tensor:
    """Compute tubed skeletons for each instance mask.

    Args:
        masks: (N, H, W) boolean instance masks.
        num_dilations: Number of dilation rounds to apply to the skeleton.

    Returns:
        (N, H, W) boolean skeleton masks.
    """
    if masks.numel() == 0:
        return masks.clone()

    masks_np = masks.numpy().astype(bool)
    skeletons = np.zeros_like(masks_np, dtype=bool)

    for i in range(masks_np.shape[0]):
        mask = masks_np[i]
        if not mask.any():
            continue
        skel = skeletonize(mask)
        for _ in range(num_dilations):
            skel = dilation(skel)
        # Keep skeleton within original mask
        skel = skel & mask
        skeletons[i] = skel

    return torch.from_numpy(skeletons)
