"""DICOM video loading and preprocessing utilities.

Handles reading multi-frame DICOM angiography videos, extracting temporal
windows, and preprocessing frames for the detection and segmentation models.
"""

import cv2
import numpy as np
import pydicom
import torch
from scipy import ndimage


def read_dicom(path: str) -> tuple:
    """Read a DICOM file and extract pixel data + metadata.

    Returns:
        (pixel_array, metadata) where:
        - pixel_array: np.ndarray of shape (num_frames, H, W) uint8
        - metadata: dict with keys 'primary_angle', 'secondary_angle', 'num_frames',
                    'original_height', 'original_width'
    """
    ds = pydicom.dcmread(path)
    pixel_array = ds.pixel_array

    if len(pixel_array.shape) == 2:
        pixel_array = pixel_array[np.newaxis, ...]

    pixel_array = _normalize_to_uint8(pixel_array)

    metadata = {
        "num_frames": pixel_array.shape[0],
        "original_height": pixel_array.shape[1],
        "original_width": pixel_array.shape[2],
        "primary_angle": None,
        "secondary_angle": None,
    }

    try:
        if hasattr(ds, "PositionerPrimaryAngle"):
            metadata["primary_angle"] = float(ds.PositionerPrimaryAngle)
        elif (0x0018, 0x1510) in ds:
            metadata["primary_angle"] = float(ds[0x0018, 0x1510].value)
    except (ValueError, TypeError):
        pass

    try:
        if hasattr(ds, "PositionerSecondaryAngle"):
            metadata["secondary_angle"] = float(ds.PositionerSecondaryAngle)
        elif (0x0018, 0x1511) in ds:
            metadata["secondary_angle"] = float(ds[0x0018, 0x1511].value)
    except (ValueError, TypeError):
        pass

    return pixel_array, metadata


def _normalize_to_uint8(pixel_array: np.ndarray) -> np.ndarray:
    """Min-max normalize pixel array to uint8."""
    img = pixel_array.astype(np.float32)
    vmin, vmax = img.min(), img.max()
    if vmax - vmin < 1e-6:
        return np.zeros_like(pixel_array, dtype=np.uint8)
    return ((img - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)


def extract_temporal_window(
    frames: np.ndarray, center_idx: int, T: int = 5
) -> list:
    """Extract T frames centered at center_idx with boundary clamping.

    Args:
        frames: (num_frames, H, W) uint8 array
        center_idx: index of the centre frame
        T: temporal window size

    Returns:
        List of T uint8 grayscale frames, each (H, W)
    """
    num_frames = frames.shape[0]
    half = T // 2
    window = []
    for offset in range(-half, half + 1):
        idx = max(0, min(center_idx + offset, num_frames - 1))
        window.append(frames[idx])
    return window


def preprocess_for_detection(
    window: list,
    img_size: int = 512,
    pixel_mean: tuple = (0.485, 0.456, 0.406),
    pixel_std: tuple = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """Preprocess a temporal window for the RF-DETR temporal detector.

    Converts grayscale frames to 3-channel, resizes, and applies ImageNet
    normalization (matching rfdetr_temporal training distribution).

    Args:
        window: list of T grayscale uint8 frames, each (H, W)
        img_size: target resolution (square)
        pixel_mean: ImageNet channel means
        pixel_std: ImageNet channel stds

    Returns:
        Tensor of shape (1, T, 3, img_size, img_size) float32
    """
    mean = torch.tensor(pixel_mean).view(3, 1, 1)
    std = torch.tensor(pixel_std).view(3, 1, 1)

    tensors = []
    for frame in window:
        resized = cv2.resize(frame, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        # Grayscale → 3-channel
        rgb = np.stack([resized] * 3, axis=0).astype(np.float32) / 255.0
        t = torch.from_numpy(rgb)
        t = (t - mean) / std
        tensors.append(t)

    # (T, 3, H, W) → (1, T, 3, H, W)
    return torch.stack(tensors, dim=0).unsqueeze(0)


def preprocess_contrast_fast(img_bgr: np.ndarray) -> np.ndarray:
    """White top-hat filtering + CLAHE contrast enhancement.

    Applies:
      1. Bitwise-NOT → morphological opening → white top-hat
      2. Subtract top-hat from original (removes uneven illumination)
      3. CLAHE for local contrast enhancement

    Args:
        img_bgr: (H, W, 3) uint8 BGR image (only channel 0 is used)

    Returns:
        (H, W, 3) uint8 BGR image with enhanced contrast
    """
    gray = img_bgr[:, :, 0]
    se = np.ones((50, 50), np.uint8)

    img_not = cv2.bitwise_not(gray)

    # Morphological opening via erosion + dilation (fast scipy path)
    eroded = ndimage.grey_erosion(img_not, footprint=se, mode="reflect")
    opened = ndimage.grey_dilation(eroded, footprint=se, mode="reflect")

    # White top-hat
    wth = img_not.astype(np.int32) - opened.astype(np.int32)
    wth = np.clip(wth, 0, 255).astype(np.uint8)

    # Subtract top-hat from original
    raw_minus = gray.astype(np.int32) - wth.astype(np.int32)
    raw_minus = ((raw_minus > 0) * raw_minus).astype(np.uint8)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    res = clahe.apply(raw_minus)

    return cv2.merge([res, res, res])


def preprocess_for_segmentation(
    crop: np.ndarray, img_size: int = 126
) -> torch.Tensor:
    """Preprocess a cropped region for EOMT vessel segmentation.

    Applies white top-hat + CLAHE contrast enhancement before resizing.
    EOMT normalizes internally, so we pass uint8-range values.

    Args:
        crop: (H, W) uint8 grayscale crop
        img_size: EOMT input resolution (default 126)

    Returns:
        Tensor of shape (1, 3, img_size, img_size) float [0,255] range
    """
    # Grayscale → 3-channel BGR for preprocessing
    bgr = cv2.merge([crop, crop, crop])

    # Apply white top-hat + CLAHE
    enhanced = preprocess_contrast_fast(bgr)

    # Resize to model input size
    resized = cv2.resize(enhanced, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

    # HWC → CHW tensor
    t = torch.from_numpy(resized).permute(2, 0, 1).float()  # (3, H, W) [0,255]
    return t.unsqueeze(0)  # (1, 3, H, W)


def crop_detection(
    frame: np.ndarray,
    box_xyxy: np.ndarray,
    padding: float = 0.15,
    source_size: int = 512,
) -> tuple:
    """Crop a detected region from the original-resolution frame.

    Maps detection coordinates (in detector resolution) back to the
    original frame, adds padding, and returns the crop.

    Args:
        frame: (H, W) original-resolution uint8 frame
        box_xyxy: (4,) detection box in detector-space coordinates [x1,y1,x2,y2]
        padding: fractional padding around the box
        source_size: resolution the detector operates at

    Returns:
        (crop, crop_coords) where crop is the uint8 image and
        crop_coords is (x1, y1, x2, y2) in original pixel space
    """
    orig_h, orig_w = frame.shape[:2]
    scale_x = orig_w / source_size
    scale_y = orig_h / source_size

    # Map to original coords
    x1 = box_xyxy[0] * scale_x
    y1 = box_xyxy[1] * scale_y
    x2 = box_xyxy[2] * scale_x
    y2 = box_xyxy[3] * scale_y

    # Add padding
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = bw * padding, bh * padding
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(orig_w, x2 + pad_x)
    y2 = min(orig_h, y2 + pad_y)

    ix1, iy1, ix2, iy2 = int(x1), int(y1), int(x2), int(y2)
    crop = frame[iy1:iy2, ix1:ix2]
    return crop, (ix1, iy1, ix2, iy2)
