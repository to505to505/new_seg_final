"""
Streamlit Web Demo for Coronary Artery Instance Segmentation (EoMT).

This application provides a web interface for running inference on coronary
angiography images using an Encoder-only Mask Transformer (EoMT) with a
DINOv2 ViT-Small backbone and Skeleton Recall loss.

Features:
- Upload PNG/JPG images or multi-frame DICOM files
- Navigate DICOM frames with a slider
- Adjustable confidence thresholds
- Side-by-side visualization of original and segmented images
- Per-instance detection details

Usage:
    streamlit run app.py
"""

import os
import sys
import tempfile
import time
import importlib
from pathlib import Path

import cv2
import numpy as np
import pydicom
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from PIL import Image
from scipy import ndimage
import yaml
import matplotlib.pyplot as plt

from postprocessing import postprocess_instances

# ==============================================================================
# Path Setup — add repo root so models.*, training.* resolve
# ==============================================================================
# Path Setup — add EoMT project root so models.*, training.* resolve
# ==============================================================================
EOMT_ROOT = Path(__file__).resolve().parent.parent
if str(EOMT_ROOT) not in sys.path:
    sys.path.insert(0, str(EOMT_ROOT))

# ==============================================================================
# Constants
# ==============================================================================
CONFIG_PATH = EOMT_ROOT / "configs/dinov2/coronary/instance/eomt_small_512_dinov2_skelrecall.yaml"
CKPT_PATH = EOMT_ROOT / "runs/coronary_instance_eomt_small_512_dinov2_skelrecall/augs/checkpoints/best.ckpt"

TRAIN_SIZE = (512, 512)        # Resolution the checkpoint was trained at
INFERENCE_SIZE = (1024, 1024)  # Higher resolution for smoother masks

CLASS_NAMES = {
    0: "lad", 1: "lm", 2: "lcx", 3: "lad_b", 4: "lcx_b",
    5: "inter", 6: "rca", 7: "pda", 8: "pborca",
}
NUM_CLASSES = len(CLASS_NAMES)

# matplotlib tab10 palette (RGB float 0-1)
CLASS_COLORS_FLOAT = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
CLASS_COLORS_BGR = [
    tuple(int(c * 255) for c in color[:3][::-1])
    for color in CLASS_COLORS_FLOAT
]
CLASS_COLORS_RGB = [
    tuple(int(c * 255) for c in color[:3])
    for color in CLASS_COLORS_FLOAT
]

DEFAULT_CONF_THRESH = 0.3
DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ==============================================================================
# Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="EoMT — Coronary Artery Segmentation",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 0rem;
        }
        header[data-testid="stHeader"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)


# ==============================================================================
# Model Loading with Caching
# ==============================================================================
@st.cache_resource
def load_model():
    """
    Load and cache the EoMT model.

    Builds at training resolution (512) to load checkpoint, then upgrades
    to INFERENCE_SIZE for higher-resolution masks.
    """
    st.write("Loading EoMT model... (first run only)")

    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # --- Encoder (ViT) — build at TRAIN_SIZE to match checkpoint ---
    encoder_cfg = config["model"]["init_args"]["network"]["init_args"]["encoder"]
    enc_module, enc_class = encoder_cfg["class_path"].rsplit(".", 1)
    encoder_cls = getattr(importlib.import_module(enc_module), enc_class)
    encoder = encoder_cls(img_size=TRAIN_SIZE, **encoder_cfg.get("init_args", {}))

    # --- Network (EoMT) ---
    network_cfg = config["model"]["init_args"]["network"]
    net_module, net_class = network_cfg["class_path"].rsplit(".", 1)
    network_cls = getattr(importlib.import_module(net_module), net_class)
    network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
    network = network_cls(
        masked_attn_enabled=False,
        num_classes=NUM_CLASSES,
        encoder=encoder,
        **network_kwargs,
    )

    # --- Lightning Module ---
    lit_module, lit_class = config["model"]["class_path"].rsplit(".", 1)
    lit_cls = getattr(importlib.import_module(lit_module), lit_class)
    model_kwargs = {k: v for k, v in config["model"]["init_args"].items() if k != "network"}

    model = lit_cls(
        img_size=TRAIN_SIZE,
        num_classes=NUM_CLASSES,
        network=network,
        **model_kwargs,
    ).eval()

    # Move to device
    device = torch.device(f"cuda:{DEVICE}" if isinstance(DEVICE, int) else DEVICE)
    model = model.to(device)

    # Load checkpoint (matches TRAIN_SIZE)
    ckpt = torch.load(str(CKPT_PATH), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=False)

    # --- Upgrade to INFERENCE_SIZE ---
    if INFERENCE_SIZE != TRAIN_SIZE:
        _upgrade_resolution(model, INFERENCE_SIZE)

    return model, device


def _upgrade_resolution(model, new_size):
    """Upgrade model to higher inference resolution by interpolating pos_embed and updating grid_size."""
    backbone = model.network.encoder.backbone
    patch_size = backbone.patch_embed.patch_size
    new_grid = (new_size[0] // patch_size[0], new_size[1] // patch_size[1])

    # Interpolate positional embeddings
    pos_embed = backbone.pos_embed  # [1, 1+H*W, C]
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
# DICOM / Image Loading Helpers
# ==============================================================================
def normalize_frame_to_uint8(frame: np.ndarray) -> np.ndarray:
    """Normalize a single raw DICOM frame to uint8."""
    arr = frame.astype(np.float32)
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-6:
        return np.zeros_like(frame, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255).astype(np.uint8)


def frame_to_rgb(frame_uint8: np.ndarray) -> np.ndarray:
    """Ensure a frame is RGB uint8 [H, W, 3]."""
    if frame_uint8.ndim == 2:
        return np.stack([frame_uint8] * 3, axis=-1)
    return frame_uint8


@st.cache_data
def load_dicom_from_upload(file_bytes: bytes):
    """
    Parse DICOM bytes, return (all_frames_rgb, num_frames).

    Cached by file content so re-parsing is avoided when switching frames.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        ds = pydicom.dcmread(tmp_path)
        pixel_array = ds.pixel_array

        if pixel_array.ndim == 2:
            return [frame_to_rgb(normalize_frame_to_uint8(pixel_array))], 1

        num_frames = pixel_array.shape[0]
        frames = []
        for i in range(num_frames):
            frames.append(frame_to_rgb(normalize_frame_to_uint8(pixel_array[i])))
        return frames, num_frames
    finally:
        os.remove(tmp_path)


def numpy_to_tensor(img_rgb: np.ndarray) -> torch.Tensor:
    """Convert RGB uint8 [H, W, 3] -> tensor [3, H, W] (uint8)."""
    from torchvision import tv_tensors
    return tv_tensors.Image(Image.fromarray(img_rgb))


# ==============================================================================
# Inference
# ==============================================================================
@torch.no_grad()
def infer_instance(model, img_tensor: torch.Tensor, device, conf_thresh: float = 0.3):
    """
    Run instance segmentation on a single image tensor [3, H, W] (uint8).

    Returns (masks, labels, scores) as numpy arrays.
    """
    device_type = "cuda" if isinstance(device, torch.device) and device.type == "cuda" else "cpu"
    with autocast(dtype=torch.float16, device_type=device_type):
        imgs = [img_tensor.to(device)]
        img_sizes = [img_tensor.shape[-2:]]

        transformed_imgs = model.resize_and_pad_imgs_instance_panoptic(imgs)
        mask_logits_per_layer, class_logits_per_layer = model(transformed_imgs)

        mask_logits = F.interpolate(
            mask_logits_per_layer[-1], model.img_size, mode="bilinear",
        )
        mask_logits = model.revert_resize_and_pad_logits_instance_panoptic(
            mask_logits, img_sizes,
        )

        class_logits = class_logits_per_layer[-1]

        ml = mask_logits[0]
        scores = class_logits[0].softmax(dim=-1)[:, :-1]
        labels = (
            torch.arange(scores.shape[-1], device=scores.device)
            .unsqueeze(0)
            .repeat(scores.shape[0], 1)
            .flatten(0, 1)
        )

        topk_scores, topk_indices = scores.flatten(0, 1).topk(
            model.eval_top_k_instances, sorted=False,
        )
        labels = labels[topk_indices]
        topk_indices = topk_indices // scores.shape[-1]
        ml = ml[topk_indices]

        masks = ml > 0
        mask_scores = (
            ml.sigmoid().flatten(1) * masks.flatten(1)
        ).sum(1) / (masks.flatten(1).sum(1) + 1e-6)
        final_scores = topk_scores * mask_scores

        keep = final_scores > conf_thresh
        masks = masks[keep].cpu().numpy()
        labels = labels[keep].cpu().numpy()
        final_scores = final_scores[keep].cpu().numpy()

        order = np.argsort(-final_scores)
        masks = masks[order]
        labels = labels[order]
        final_scores = final_scores[order]

    return masks, labels, final_scores


# ==============================================================================
# Visualization
# ==============================================================================
def visualize_predictions(image_rgb: np.ndarray, masks, labels, scores,
                          alpha: float = 0.45) -> np.ndarray:
    """Draw coloured instance masks on the image with contours and a legend."""
    result = image_rgb.copy()
    legend_entries = {}

    for i in range(len(masks)):
        mask = masks[i]
        cls_id = int(labels[i])
        conf = float(scores[i])
        color_rgb = CLASS_COLORS_RGB[cls_id % len(CLASS_COLORS_RGB)]
        color_bgr = CLASS_COLORS_BGR[cls_id % len(CLASS_COLORS_BGR)]

        if mask.shape[:2] != image_rgb.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (image_rgb.shape[1], image_rgb.shape[0]),
                              interpolation=cv2.INTER_NEAREST).astype(bool)

        if not mask.any():
            continue

        coloured = np.zeros_like(result)
        coloured[mask] = color_rgb
        mask_3ch = np.stack([mask] * 3, axis=-1)
        result = np.where(
            mask_3ch,
            (alpha * coloured + (1 - alpha) * result).astype(np.uint8),
            result,
        )

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


def _draw_legend(img_bgr: np.ndarray, legend_entries: dict):
    """Draw a semi-transparent legend in the top-left corner (in-place)."""
    sorted_entries = sorted(legend_entries.items())

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 22
    box_size = 14
    padding = 8
    margin = 10

    max_text_w = 0
    for cls_id, (color, conf) in sorted_entries:
        text = f"{CLASS_NAMES.get(cls_id, str(cls_id))}: {conf:.0%}"
        (tw, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
        max_text_w = max(max_text_w, tw)

    legend_w = box_size + padding + max_text_w + padding * 2
    legend_h = len(sorted_entries) * line_height + padding * 2

    overlay = img_bgr.copy()
    cv2.rectangle(overlay, (margin, margin),
                  (margin + legend_w, margin + legend_h), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.7, img_bgr, 0.3, 0, dst=img_bgr)

    for i, (cls_id, (color, conf)) in enumerate(sorted_entries):
        y_pos = margin + padding + i * line_height + line_height // 2
        box_y1 = y_pos - box_size // 2
        box_y2 = y_pos + box_size // 2

        cv2.rectangle(img_bgr, (margin + padding, box_y1),
                      (margin + padding + box_size, box_y2), color, -1)
        cv2.rectangle(img_bgr, (margin + padding, box_y1),
                      (margin + padding + box_size, box_y2), (255, 255, 255), 1)

        text = f"{CLASS_NAMES.get(cls_id, str(cls_id))}: {conf:.0%}"
        text_x = margin + padding + box_size + padding
        text_y = y_pos + 4
        cv2.putText(img_bgr, text, (text_x, text_y), font, font_scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)


# ==============================================================================
# Streamlit UI
# ==============================================================================
def main():
    st.title("🫀 Coronary Artery Segmentation (EoMT)")
    st.markdown(
        "Upload a coronary angiography image (PNG, JPG) or a multi-frame DICOM "
        "to segment coronary arteries using an **Encoder-only Mask Transformer** "
        "with DINOv2 ViT-Small backbone."
    )

    # ==================== Sidebar ====================
    st.sidebar.header("⚙️ Configuration")

    uploaded_file = st.sidebar.file_uploader(
        "Upload image / DICOM",
        type=["png", "jpg", "jpeg", "dcm", "dicom"],
        help="Coronary angiography image or multi-frame DICOM file",
    )

    st.sidebar.divider()
    st.sidebar.subheader("Confidence Threshold")

    conf_thresh = st.sidebar.slider(
        "Minimum confidence",
        min_value=0.0, max_value=1.0, value=DEFAULT_CONF_THRESH, step=0.05,
        help="Instances below this threshold are discarded",
    )

    st.sidebar.divider()

    analyze_btn = st.sidebar.button(
        "🔬 Analyze", type="primary", use_container_width=True,
        disabled=uploaded_file is None,
    )

    st.sidebar.divider()
    st.sidebar.subheader("📋 Class Legend")
    st.sidebar.markdown("""
    **Left Coronary:**
    - `lad` -- Left Anterior Descending
    - `lm` -- Left Main
    - `lcx` -- Left Circumflex
    - `lad_b` -- LAD Branches
    - `lcx_b` -- LCx Branches
    - `inter` -- Intermediate

    **Right Coronary:**
    - `rca` -- Right Coronary Artery
    - `pda` -- Posterior Descending
    - `pborca` -- Posterolateral Branch
    """)

    # ==================== Main Area ====================
    if uploaded_file is None:
        st.info("👈 Upload an image or DICOM file using the sidebar to get started.")
        return

    name_lower = uploaded_file.name.lower()
    is_dicom = name_lower.endswith((".dcm", ".dicom"))

    # ------------------------------------------------------------------
    # DICOM path: multi-frame navigation
    # ------------------------------------------------------------------
    if is_dicom:
        file_bytes = uploaded_file.read()
        frames, num_frames = load_dicom_from_upload(file_bytes)

        st.success(f"✅ Loaded DICOM: **{uploaded_file.name}** -- **{num_frames}** frame(s)")

        if num_frames > 1:
            frame_idx = st.slider(
                "Select frame", min_value=0, max_value=num_frames - 1,
                value=num_frames // 2, key="dicom_frame_slider",
                help="Navigate through DICOM frames",
            )
        else:
            frame_idx = 0

        img_rgb = frames[frame_idx]
        frame_caption = f"Frame {frame_idx} / {num_frames - 1}"
    # ------------------------------------------------------------------
    # Standard image path (PNG / JPG)
    # ------------------------------------------------------------------
    else:
        pil_img = Image.open(uploaded_file).convert("RGB")
        img_rgb = np.array(pil_img)
        h, w = img_rgb.shape[:2]
        st.success(f"✅ Loaded image: **{uploaded_file.name}** ({w}x{h})")
        frame_caption = uploaded_file.name

    # ==================== Two-column layout ====================
    _, col1, col2, _ = st.columns([0.5, 2, 2, 0.5])

    with col1:
        st.subheader("📷 Original Image")
        st.image(img_rgb, caption=frame_caption, use_container_width=True)

    # -------- Run inference on Analyze click --------
    if analyze_btn:
        with st.spinner("Running EoMT inference..."):
            model, device = load_model()
            img_tensor = numpy_to_tensor(img_rgb)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            masks, labels, scores = infer_instance(
                model, img_tensor, device, conf_thresh=conf_thresh,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_time = time.perf_counter() - t0

            # Postprocessing: cross-class NMS + polygon filtering + dominant polygon
            if len(masks) > 0:
                masks, labels, scores = postprocess_instances(
                    masks, labels, scores,
                    orig_shape=img_rgb.shape[:2],
                )

            t1 = time.perf_counter()
            vis_img = visualize_predictions(img_rgb, masks, labels, scores)
            vis_time = time.perf_counter() - t1

            st.session_state["vis_img"] = vis_img
            st.session_state["masks"] = masks
            st.session_state["labels"] = labels
            st.session_state["scores"] = scores
            st.session_state["inference_time"] = inference_time
            st.session_state["vis_time"] = vis_time
            st.session_state["analysis_done"] = True

    # -------- Show results --------
    with col2:
        st.subheader("🎯 Segmentation Result")

        if st.session_state.get("analysis_done", False):
            vis_img = st.session_state["vis_img"]
            masks = st.session_state["masks"]
            labels = st.session_state["labels"]
            scores = st.session_state["scores"]
            inf_t = st.session_state["inference_time"]
            vis_t = st.session_state["vis_time"]

            st.image(vis_img, caption="Predicted Segmentation", use_container_width=True)

            total_t = inf_t + vis_t
            st.info(
                f"⏱️ **Timing**: Inference **{inf_t*1000:.1f} ms** | "
                f"Visualisation **{vis_t*1000:.1f} ms** | "
                f"Total **{total_t*1000:.1f} ms**"
            )

            num_det = len(masks)
            if num_det > 0:
                st.caption(f"Found **{num_det}** instances")
                with st.expander("📊 Detection Details"):
                    for i in range(num_det):
                        cls_name = CLASS_NAMES.get(int(labels[i]), f"class_{labels[i]}")
                        st.write(f"- **{cls_name}**: {scores[i]:.2%} confidence")
            else:
                st.warning("No detections above the confidence threshold.")
        else:
            st.info("Click **Analyze** to run segmentation.")


# ==============================================================================
# Entry Point
# ==============================================================================
if __name__ == "__main__":
    if "analysis_done" not in st.session_state:
        st.session_state["analysis_done"] = False
    main()
