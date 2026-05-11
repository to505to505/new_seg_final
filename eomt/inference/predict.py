#!/usr/bin/env python3
"""
CLI Inference for EoMT Coronary Artery Instance Segmentation.

Usage:
    python inference/predict.py --input <path_to_dicom_or_image> --output <output_dir>

Options:
    --input          Path to a single DICOM/image file or a folder of them
    --output         Directory for output images/masks
    --weights        Custom checkpoint path (default: best.ckpt from training run)
    --conf           Confidence threshold (default: 0.3)
    --save-masks     Save binary masks in addition to visualizations
    --frame N        Process only the frame at index N (DICOM only)
    --middle-frame   Process only the middle frame (DICOM only)
    (default)        Process all frames; results saved as <stem>_frame0001_segmentation.png
"""

import argparse
import importlib
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from PIL import Image
from scipy import ndimage
import yaml
import matplotlib.pyplot as plt

# ==============================================================================
# Path Setup
# ==============================================================================
SCRIPT_DIR = Path(__file__).resolve().parent
EOMT_ROOT = SCRIPT_DIR.parent
if str(EOMT_ROOT) not in sys.path:
    sys.path.insert(0, str(EOMT_ROOT))

# Postprocessing lives in demo_app/
DEMO_APP_DIR = EOMT_ROOT / "demo_app"
if str(DEMO_APP_DIR) not in sys.path:
    sys.path.insert(0, str(DEMO_APP_DIR))

from postprocessing import postprocess_instances

# ==============================================================================
# Constants
# ==============================================================================
DEFAULT_CONFIG = EOMT_ROOT / "configs/dinov2/coronary/instance/eomt_small_512_dinov2_skelrecall.yaml"
DEFAULT_CKPT = EOMT_ROOT / "runs/coronary_instance_eomt_small_512_dinov2_skelrecall/augs/checkpoints/best.ckpt"

TRAIN_SIZE = (512, 512)        # Resolution the checkpoint was trained at
INFERENCE_SIZE = (1024, 1024)  # Higher resolution for smoother masks

CLASS_NAMES = {
    0: "lad", 1: "lm", 2: "lcx", 3: "lad_b", 4: "lcx_b",
    5: "inter", 6: "rca", 7: "pda", 8: "pborca",
}
NUM_CLASSES = len(CLASS_NAMES)

CLASS_COLORS_FLOAT = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
CLASS_COLORS_BGR = [
    tuple(int(c * 255) for c in color[:3][::-1])
    for color in CLASS_COLORS_FLOAT
]
CLASS_COLORS_RGB = [
    tuple(int(c * 255) for c in color[:3])
    for color in CLASS_COLORS_FLOAT
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
DICOM_EXTENSIONS = {".dcm", ".dicom"}


# ==============================================================================
# Model Loading
# ==============================================================================
def load_model(weights_path: str = None, device: torch.device = None):
    """Load EoMT model from config + checkpoint, then upgrade to INFERENCE_SIZE."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ckpt_path = Path(weights_path) if weights_path else DEFAULT_CKPT

    with open(DEFAULT_CONFIG, "r") as f:
        config = yaml.safe_load(f)

    # Encoder — build at TRAIN_SIZE to match checkpoint
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    enc_mod, enc_cls = encoder_cfg["class_path"].rsplit(".", 1)
    encoder = getattr(importlib.import_module(enc_mod), enc_cls)(
        img_size=TRAIN_SIZE, **encoder_cfg.get("init_args", {})
    )

    # Network
    network_cfg = config["model"]["init_args"]["network"]
    net_mod, net_cls = network_cfg["class_path"].rsplit(".", 1)
    net_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
    network = getattr(importlib.import_module(net_mod), net_cls)(
        masked_attn_enabled=False, num_classes=NUM_CLASSES, encoder=encoder, **net_kwargs,
    )

    # Lightning module
    lit_mod, lit_cls = config["model"]["class_path"].rsplit(".", 1)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}
    model = getattr(importlib.import_module(lit_mod), lit_cls)(
        img_size=TRAIN_SIZE, num_classes=NUM_CLASSES, network=network, **model_kwargs,
    ).eval().to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"Model loaded from {ckpt_path}")

    # Upgrade to higher inference resolution
    if INFERENCE_SIZE != TRAIN_SIZE:
        _upgrade_resolution(model, INFERENCE_SIZE)
        print(f"Resolution upgraded to {INFERENCE_SIZE}")

    return model, device


def _upgrade_resolution(model, new_size):
    """Upgrade model to higher inference resolution by interpolating pos_embed."""
    backbone = model.network.encoder.backbone
    patch_size = backbone.patch_embed.patch_size
    new_grid = (new_size[0] // patch_size[0], new_size[1] // patch_size[1])

    pos_embed = backbone.pos_embed
    num_prefix = backbone.num_prefix_tokens
    cls_tokens = pos_embed[:, :num_prefix, :]
    patch_tokens = pos_embed[:, num_prefix:, :]

    old_grid = backbone.patch_embed.grid_size
    C = patch_tokens.shape[-1]
    patch_tokens = patch_tokens.reshape(1, old_grid[0], old_grid[1], C).permute(0, 3, 1, 2)
    patch_tokens = F.interpolate(patch_tokens.float(), size=new_grid, mode="bicubic", align_corners=False)
    patch_tokens = patch_tokens.permute(0, 2, 3, 1).reshape(1, -1, C).to(pos_embed.dtype)

    backbone.pos_embed = nn.Parameter(torch.cat([cls_tokens, patch_tokens], dim=1))
    backbone.patch_embed.grid_size = new_grid
    backbone.patch_embed.img_size = new_size
    model.img_size = new_size


# ==============================================================================
# Image / DICOM Helpers
# ==============================================================================
def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-6:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


def to_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return np.stack([frame] * 3, axis=-1)
    return frame


def load_dicom_frames(path: Path) -> list[np.ndarray]:
    """Load all frames from a DICOM file as RGB uint8 arrays."""
    import pydicom
    ds = pydicom.dcmread(str(path))
    pixel_array = ds.pixel_array

    if pixel_array.ndim == 2:
        return [to_rgb(normalize_to_uint8(pixel_array))]

    return [to_rgb(normalize_to_uint8(pixel_array[i])) for i in range(pixel_array.shape[0])]


def load_image(path: Path) -> np.ndarray:
    """Load a standard image as RGB uint8."""
    return np.array(Image.open(path).convert("RGB"))


def numpy_to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    from torchvision import tv_tensors
    return tv_tensors.Image(Image.fromarray(img_rgb))


# ==============================================================================
# Inference
# ==============================================================================
@torch.no_grad()
def infer_instance(model, img_rgb: np.ndarray, device, conf_thresh: float = 0.3):
    """Run inference on a single RGB uint8 image. Returns (masks, labels, scores)."""
    img_tensor = numpy_to_tensor(img_rgb)
    device_type = "cuda" if device.type == "cuda" else "cpu"

    with autocast(dtype=torch.float16, device_type=device_type):
        imgs = [img_tensor.to(device)]
        img_sizes = [img_tensor.shape[-2:]]

        transformed = model.resize_and_pad_imgs_instance_panoptic(imgs)
        mlpl, clpl = model(transformed)

        ml = F.interpolate(mlpl[-1], model.img_size, mode="bilinear")
        ml = model.revert_resize_and_pad_logits_instance_panoptic(ml, img_sizes)
        cl = clpl[-1]

        ml0 = ml[0]
        scores = cl[0].softmax(dim=-1)[:, :-1]
        labels = (
            torch.arange(scores.shape[-1], device=scores.device)
            .unsqueeze(0).repeat(scores.shape[0], 1).flatten(0, 1)
        )

        topk_scores, topk_indices = scores.flatten(0, 1).topk(
            model.eval_top_k_instances, sorted=False,
        )
        labels = labels[topk_indices]
        topk_indices = topk_indices // scores.shape[-1]
        ml0 = ml0[topk_indices]

        masks = ml0 > 0
        mask_scores = (
            ml0.sigmoid().flatten(1) * masks.flatten(1)
        ).sum(1) / (masks.flatten(1).sum(1) + 1e-6)
        final_scores = topk_scores * mask_scores

        keep = final_scores > conf_thresh
        masks = masks[keep].cpu().numpy()
        labels = labels[keep].cpu().numpy()
        final_scores = final_scores[keep].cpu().numpy()

        order = np.argsort(-final_scores)
        return masks[order], labels[order], final_scores[order]


# ==============================================================================
# Visualization
# ==============================================================================
def visualize(image_rgb: np.ndarray, masks, labels, scores, alpha: float = 0.45) -> np.ndarray:
    """Draw coloured instance masks with contours and legend. Returns RGB uint8."""
    result = image_rgb.copy()
    legend_entries = {}

    for i in range(len(masks)):
        mask = masks[i]
        cls_id = int(labels[i])
        conf = float(scores[i])
        color_rgb = CLASS_COLORS_RGB[cls_id % NUM_CLASSES]
        color_bgr = CLASS_COLORS_BGR[cls_id % NUM_CLASSES]

        if mask.shape[:2] != image_rgb.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8),
                              (image_rgb.shape[1], image_rgb.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
        if not mask.any():
            continue

        coloured = np.zeros_like(result)
        coloured[mask] = color_rgb
        mask_3ch = np.stack([mask] * 3, axis=-1)
        result = np.where(mask_3ch,
                          (alpha * coloured + (1 - alpha) * result).astype(np.uint8),
                          result)

        eroded = ndimage.binary_erosion(mask, iterations=1)
        contour = mask & ~eroded
        result[contour] = color_rgb

        if cls_id not in legend_entries or conf > legend_entries[cls_id][1]:
            legend_entries[cls_id] = (color_bgr, conf)

    if legend_entries:
        result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        _draw_legend(result_bgr, legend_entries)
        result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    return result


def _draw_legend(img_bgr, legend_entries):
    sorted_entries = sorted(legend_entries.items())
    font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    lh, bs, pad, margin = 22, 14, 8, 10

    max_tw = 0
    for cid, (col, conf) in sorted_entries:
        t = f"{CLASS_NAMES.get(cid, str(cid))}: {conf:.0%}"
        (tw, _), _ = cv2.getTextSize(t, font, scale, thick)
        max_tw = max(max_tw, tw)

    lw = bs + pad + max_tw + pad * 2
    lh_total = len(sorted_entries) * lh + pad * 2

    ov = img_bgr.copy()
    cv2.rectangle(ov, (margin, margin), (margin + lw, margin + lh_total), (40, 40, 40), -1)
    cv2.addWeighted(ov, 0.7, img_bgr, 0.3, 0, dst=img_bgr)

    for i, (cid, (col, conf)) in enumerate(sorted_entries):
        yp = margin + pad + i * lh + lh // 2
        y1, y2 = yp - bs // 2, yp + bs // 2
        cv2.rectangle(img_bgr, (margin + pad, y1), (margin + pad + bs, y2), col, -1)
        cv2.rectangle(img_bgr, (margin + pad, y1), (margin + pad + bs, y2), (255, 255, 255), 1)
        t = f"{CLASS_NAMES.get(cid, str(cid))}: {conf:.0%}"
        cv2.putText(img_bgr, t, (margin + pad + bs + pad, yp + 4), font, scale,
                    (255, 255, 255), thick, cv2.LINE_AA)


# ==============================================================================
# Processing Logic
# ==============================================================================
def process_single_image(model, device, img_rgb, conf, save_masks, out_dir, stem, suffix=""):
    """Run inference on one image and save results."""
    masks, labels, scores = infer_instance(model, img_rgb, device, conf_thresh=conf)

    # Postprocessing: cross-class NMS + polygon filtering + dominant polygon
    if len(masks) > 0:
        masks, labels, scores = postprocess_instances(
            masks, labels, scores,
            orig_shape=img_rgb.shape[:2],
        )

    n = len(masks)
    print(f"  {stem}{suffix}: {n} detection(s)")

    # Save visualization
    vis = visualize(img_rgb, masks, labels, scores)
    vis_path = out_dir / f"{stem}{suffix}_segmentation.png"
    Image.fromarray(vis).save(vis_path)

    # Save binary masks
    if save_masks and n > 0:
        mask_dir = out_dir / "masks" / f"{stem}{suffix}"
        mask_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            cls_name = CLASS_NAMES.get(int(labels[i]), str(labels[i]))
            m = (masks[i].astype(np.uint8)) * 255
            cv2.imwrite(str(mask_dir / f"{i:03d}_{cls_name}_{scores[i]:.2f}.png"), m)

    return n


def process_file(model, device, filepath: Path, args):
    """Process a single file (image or DICOM)."""
    out_dir = Path(args.output)
    ext = filepath.suffix.lower()
    stem = filepath.stem

    if ext in DICOM_EXTENSIONS:
        frames = load_dicom_frames(filepath)
        num_frames = len(frames)
        print(f"DICOM: {filepath.name} ({num_frames} frames)")

        if args.frame is not None:
            idx = args.frame
            if idx < 0 or idx >= num_frames:
                print(f"  ERROR: frame {idx} out of range [0, {num_frames - 1}]")
                return
            process_single_image(
                model, device, frames[idx], args.conf, args.save_masks,
                out_dir, stem, suffix=f"_frame{idx:04d}",
            )
        elif args.middle_frame:
            idx = num_frames // 2
            process_single_image(
                model, device, frames[idx], args.conf, args.save_masks,
                out_dir, stem, suffix=f"_frame{idx:04d}",
            )
        else:
            # All frames
            total = 0
            for idx in range(num_frames):
                n = process_single_image(
                    model, device, frames[idx], args.conf, args.save_masks,
                    out_dir, stem, suffix=f"_frame{idx:04d}",
                )
                total += n
            print(f"  Total: {total} detection(s) across {num_frames} frames")

    elif ext in IMAGE_EXTENSIONS:
        img_rgb = load_image(filepath)
        print(f"Image: {filepath.name}")
        process_single_image(
            model, device, img_rgb, args.conf, args.save_masks, out_dir, stem,
        )
    else:
        print(f"Skipping unsupported file: {filepath.name}")


# ==============================================================================
# CLI Entry Point
# ==============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="EoMT Coronary Artery Instance Segmentation — CLI Inference",
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to a single DICOM/image file or a folder of them",
    )
    parser.add_argument(
        "--output", required=True,
        help="Directory for output images/masks",
    )
    parser.add_argument(
        "--weights", default=None,
        help=f"Custom checkpoint path (default: {DEFAULT_CKPT})",
    )
    parser.add_argument(
        "--conf", type=float, default=0.3,
        help="Confidence threshold (default: 0.3)",
    )
    parser.add_argument(
        "--save-masks", action="store_true",
        help="Save per-instance binary masks in addition to visualizations",
    )
    parser.add_argument(
        "--frame", type=int, default=None,
        help="Process only the frame at this index (DICOM only)",
    )
    parser.add_argument(
        "--middle-frame", action="store_true",
        help="Process only the middle frame (DICOM only)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  EoMT Coronary Artery Segmentation — CLI Inference")
    print("=" * 60)

    model, device = load_model(weights_path=args.weights)

    if input_path.is_file():
        process_file(model, device, input_path, args)
    elif input_path.is_dir():
        files = sorted(
            p for p in input_path.iterdir()
            if p.suffix.lower() in (IMAGE_EXTENSIONS | DICOM_EXTENSIONS)
        )
        if not files:
            print(f"No supported files found in {input_path}")
            return
        print(f"Found {len(files)} file(s) in {input_path}")
        for filepath in files:
            process_file(model, device, filepath, args)
    else:
        print(f"ERROR: {input_path} does not exist")
        return

    print(f"\nResults saved to: {out_dir}")


if __name__ == "__main__":
    main()
