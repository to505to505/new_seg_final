import torch
import numpy as np
from torchmetrics import Metric
from scipy.ndimage import label as scipy_label


class LargestComponentRatio(Metric):
    """
    Largest Component Ratio (LCR) — measures per-instance mask connectivity.

    For each predicted instance mask, computes the ratio of the largest
    connected component area to the total mask area.
    LCR = 1.0 means a single connected component (no fragmentation).
    LCR < 1.0 means the mask is fragmented.
    """

    def __init__(self, connectivity: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.connectivity = connectivity
        if connectivity == 8:
            self.structure = np.ones((3, 3), dtype=np.int32)
        else:
            self.structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int32)

        self.add_state("lcr_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("lcr_count", default=torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum")

    def update(self, preds: list[dict]) -> None:
        for pred in preds:
            masks = pred["masks"]  # (N, H, W) bool tensor
            for i in range(masks.shape[0]):
                mask_np = masks[i].detach().cpu().numpy().astype(np.uint8)
                total_area = mask_np.sum()
                if total_area == 0:
                    continue

                labeled_array, num_features = scipy_label(mask_np, structure=self.structure)
                if num_features <= 1:
                    lcr = 1.0
                else:
                    component_sizes = np.bincount(labeled_array.ravel())[1:]
                    lcr = float(component_sizes.max()) / float(total_area)

                self.lcr_sum += lcr
                self.lcr_count += 1

    def compute(self) -> torch.Tensor:
        if self.lcr_count == 0:
            return torch.tensor(0.0)
        return self.lcr_sum / self.lcr_count.float()
