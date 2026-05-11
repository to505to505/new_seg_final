"""Wrapper around EOMT for binary vessel segmentation inference.

Loads the trained EOMT lightning module and provides a simple segment()
interface returning binary masks from cropped patch images.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.amp.autocast_mode import autocast
from torch.nn import functional as F

# Add EOMT to sys.path
_EOMT_ROOT = Path("/home/dsa/new_seg_final/eomt")
if str(_EOMT_ROOT) not in sys.path:
    sys.path.insert(0, str(_EOMT_ROOT))

# Default paths
DEFAULT_EOMT_CONFIG = str(
    _EOMT_ROOT
    / "configs/dinov2/coronary/binary_instance/eomt_small_126_patch_dinov2_skelrecall.yaml"
)
DEFAULT_EOMT_CHECKPOINT = str(
    _EOMT_ROOT
    / "runs/coronary_binary_eomt_small_126_patch_dinov2_skelrecall/version_1/checkpoints/last.ckpt"
)


class VesselSegmentor:
    """EOMT-based binary vessel segmentation model.

    Operates on cropped patches (126×126) and produces binary masks
    by combining per-query class scores with mask logits.
    """

    def __init__(
        self,
        config_path: str = DEFAULT_EOMT_CONFIG,
        checkpoint_path: str = DEFAULT_EOMT_CHECKPOINT,
        device: str = "cuda",
        mask_threshold: float = 0.5,
    ):
        self.device = torch.device(device)
        self.mask_threshold = mask_threshold

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.img_size = tuple(
            config["data"].get("init_args", {}).get("img_size", [126, 126])
        )
        num_classes = 1  # binary segmentation

        # Build encoder
        encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
        enc_mod, enc_cls = encoder_cfg["class_path"].rsplit(".", 1)
        encoder = getattr(importlib.import_module(enc_mod), enc_cls)(
            img_size=self.img_size, **encoder_cfg.get("init_args", {})
        )

        # Build network
        net_cfg = config["model"]["init_args"]["network"]
        net_mod, net_cls = net_cfg["class_path"].rsplit(".", 1)
        net_kwargs = {k: v for k, v in net_cfg["init_args"].items() if k != "encoder"}
        network = getattr(importlib.import_module(net_mod), net_cls)(
            masked_attn_enabled=False,
            num_classes=num_classes,
            encoder=encoder,
            **net_kwargs,
        )

        # Build lightning module
        lit_mod, lit_cls = config["model"]["class_path"].rsplit(".", 1)
        model_kwargs = {
            k: v for k, v in config["model"]["init_args"].items() if k != "network"
        }
        LitClass = getattr(importlib.import_module(lit_mod), lit_cls)

        self.model = LitClass(
            img_size=self.img_size,
            num_classes=num_classes,
            network=network,
            **model_kwargs,
        )

        # Load checkpoint weights
        state_dict = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )["state_dict"]
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval().to(self.device)

    @torch.no_grad()
    def segment(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Segment vessels in a preprocessed patch.

        Args:
            image_tensor: (1, 3, H, W) float tensor with pixel values in [0, 255]

        Returns:
            Binary mask as np.ndarray (H, W) uint8 with values {0, 1}
            at the input resolution of the tensor (before EOMT internal resize)
        """
        # EOMT's resize_and_pad uses PIL Image.fromarray which requires uint8
        image_tensor = image_tensor.to(self.device).to(torch.uint8)
        input_h, input_w = image_tensor.shape[-2:]

        with autocast(dtype=torch.float16, device_type="cuda"):
            imgs = [image_tensor[0]]  # model expects list of (3, H, W)
            img_sizes = [img.shape[-2:] for img in imgs]

            transformed_imgs = self.model.resize_and_pad_imgs_instance_panoptic(imgs)
            mask_logits_per_layer, class_logits_per_layer = self.model(transformed_imgs)

            mask_logits = F.interpolate(
                mask_logits_per_layer[-1], self.img_size, mode="bilinear"
            )
            mask_logits = self.model.revert_resize_and_pad_logits_instance_panoptic(
                mask_logits, img_sizes
            )

            # Combine class scores × mask probs → binary prediction
            class_scores = class_logits_per_layer[-1][0].softmax(dim=-1)[
                :, :-1
            ]  # (num_q, 1)
            mask_probs = mask_logits[0].sigmoid()  # (num_q, H, W)
            combined = (class_scores[:, 0:1, None] * mask_probs).sum(
                dim=0
            )  # (H, W)
            pred_mask = (combined > self.mask_threshold).cpu().numpy().astype(np.uint8)

        return pred_mask
