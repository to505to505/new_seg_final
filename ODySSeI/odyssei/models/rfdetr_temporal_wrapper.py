"""Wrapper around TemporalRFDETR for stenosis detection inference.

Loads the trained temporal RF-DETR model and provides a simple
detect() interface returning bounding boxes + scores.
"""

import sys
from pathlib import Path

import numpy as np
import torch

# Add rfdetr_temporal to sys.path so its imports resolve
_RFDETR_TEMPORAL_ROOT = Path("/home/dsa/stenosis")
if str(_RFDETR_TEMPORAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_RFDETR_TEMPORAL_ROOT))

from rfdetr_temporal.config import Config as RFDETRConfig
from rfdetr_temporal.model import TemporalRFDETR, _build_criterion

# Default paths
DEFAULT_RFDETR_CHECKPOINT = (
    "/home/dsa/stenosis/rfdetr_temporal/runs/cadica_temporal_v1/best.pth"
)


class StenosisDetector:
    """Temporal RF-DETR stenosis detector.

    Uses T=5 consecutive frames to detect stenosis lesions on the centre frame.
    Returns bounding boxes in absolute xyxy coordinates at the detector resolution.
    """

    def __init__(
        self,
        checkpoint_path: str = DEFAULT_RFDETR_CHECKPOINT,
        device: str = "cuda",
        score_thresh: float = 0.3,
        nms_thresh: float = 0.5,
    ):
        self.device = torch.device(device)
        self.score_thresh = score_thresh

        # Build config — override checkpoint path and thresholds
        cfg = RFDETRConfig()
        cfg.rfdetr_checkpoint = _find_base_rfdetr_checkpoint(cfg)
        cfg.score_thresh = score_thresh
        cfg.nms_thresh = nms_thresh

        # Build model
        self.model = TemporalRFDETR(cfg)

        # Load trained temporal weights
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device).eval()

        # Build postprocessor (for converting raw outputs → boxes)
        _, postprocessors = _build_criterion(cfg)
        self.postprocessor = postprocessors

        self.img_size = cfg.img_size

    @torch.no_grad()
    def detect(self, frames_tensor: torch.Tensor) -> list:
        """Run detection on a batch of temporal frame sequences.

        Args:
            frames_tensor: (B, T, 3, H, W) preprocessed float tensor

        Returns:
            List of dicts per batch item, each with:
                'boxes': np.ndarray (N, 4) absolute xyxy at detector resolution
                'scores': np.ndarray (N,) confidence scores
        """
        frames_tensor = frames_tensor.to(self.device)
        output = self.model(frames_tensor)

        B = frames_tensor.shape[0]
        orig_sizes = torch.tensor(
            [[self.img_size, self.img_size]] * B, device=self.device
        )

        # postprocessor expects orig_target_sizes
        results_raw = self.postprocessor(output, orig_sizes)

        results = []
        for r in results_raw:
            scores = r["scores"].cpu().numpy()
            boxes = r["boxes"].cpu().numpy()

            # Filter by score threshold
            keep = scores >= self.score_thresh
            results.append({
                "boxes": boxes[keep],
                "scores": scores[keep],
            })

        return results


def _find_base_rfdetr_checkpoint(cfg: RFDETRConfig) -> str:
    """Resolve the base RF-DETR checkpoint path.

    The temporal model needs the pretrained RF-DETR weights to build the
    backbone/decoder architecture. We look in known locations.
    """
    candidates = [
        Path(cfg.rfdetr_checkpoint),
        Path("/home/dsa/stenosis/rfdetr_runs/dataset2_augs/checkpoint_best_total.pth"),
        Path("/home/dsa/stenosis/rfdetr_runs/cadica_augs/checkpoint_best_total.pth"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    raise FileNotFoundError(
        "Cannot find base RF-DETR checkpoint. Looked in: "
        + ", ".join(str(c) for c in candidates)
    )
