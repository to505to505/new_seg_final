"""End-to-end stenosis analysis pipeline.

DICOM → detection (RF-DETR temporal) → segmentation (EOMT) → severity (MLD/DS)
"""

import csv
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .models.eomt_wrapper import VesselSegmentor, DEFAULT_EOMT_CONFIG, DEFAULT_EOMT_CHECKPOINT
from .models.rfdetr_temporal_wrapper import StenosisDetector, DEFAULT_RFDETR_CHECKPOINT
from .utils.dicom_utils import (
    read_dicom,
    extract_temporal_window,
    preprocess_for_detection,
    preprocess_for_segmentation,
    crop_detection,
)
from .utils.severity import compute_severity


# ── Result dataclasses ──────────────────────────────────────────────

@dataclass
class DetectionResult:
    box_xyxy: np.ndarray          # (4,) in original image coords
    score: float
    mask: np.ndarray              # binary mask at crop resolution
    crop_image: np.ndarray        # cropped grayscale patch (H, W) uint8
    crop_coords: tuple            # (x1, y1, x2, y2) in original image coords
    mld: float = float("nan")
    ds: float = float("nan")


@dataclass
class FrameResult:
    frame_idx: int
    detections: List[DetectionResult] = field(default_factory=list)


@dataclass
class PipelineResult:
    frame_results: List[FrameResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# ── Pipeline ────────────────────────────────────────────────────────

class StenosisPipeline:
    """End-to-end pipeline: DICOM → Detection → Segmentation → Severity."""

    def __init__(
        self,
        detector_ckpt: str = DEFAULT_RFDETR_CHECKPOINT,
        segmentor_config: str = DEFAULT_EOMT_CONFIG,
        segmentor_ckpt: str = DEFAULT_EOMT_CHECKPOINT,
        device: str = "cuda",
        score_thresh: float = 0.3,
    ):
        print("Loading stenosis detector...")
        self.detector = StenosisDetector(
            checkpoint_path=detector_ckpt,
            device=device,
            score_thresh=score_thresh,
        )
        print("Loading vessel segmentor...")
        self.segmentor = VesselSegmentor(
            config_path=segmentor_config,
            checkpoint_path=segmentor_ckpt,
            device=device,
        )
        self.device = device
        print("Pipeline ready.")

    def process_dicom(
        self,
        dicom_path: str,
        crop_padding: float = 0.15,
        frame_stride: int = 1,
    ) -> PipelineResult:
        """Process a single DICOM video file.

        Args:
            dicom_path: path to .dcm file
            crop_padding: fractional padding around detection boxes
            frame_stride: process every N-th frame (1 = all frames)

        Returns:
            PipelineResult with per-frame detections, masks, and severity
        """
        frames, metadata = read_dicom(dicom_path)
        num_frames = frames.shape[0]
        T = self.detector.model.T

        result = PipelineResult(metadata=metadata)

        for center_idx in range(0, num_frames, frame_stride):
            frame_result = self._process_frame(
                frames, center_idx, T, crop_padding
            )
            result.frame_results.append(frame_result)

        return result

    def process_single_frame(
        self,
        frames: np.ndarray,
        center_idx: int,
        crop_padding: float = 0.15,
    ) -> FrameResult:
        """Process a single frame from an already-loaded DICOM.

        Args:
            frames: (num_frames, H, W) uint8 array
            center_idx: which frame to analyze
            crop_padding: fractional padding around boxes

        Returns:
            FrameResult for the specified frame
        """
        T = self.detector.model.T
        return self._process_frame(frames, center_idx, T, crop_padding)

    def _process_frame(
        self,
        frames: np.ndarray,
        center_idx: int,
        T: int,
        crop_padding: float,
    ) -> FrameResult:
        """Internal: detect + segment + severity for one centre frame."""
        # 1. Extract temporal window and detect
        window = extract_temporal_window(frames, center_idx, T=T)
        det_tensor = preprocess_for_detection(window, img_size=self.detector.img_size)
        detections = self.detector.detect(det_tensor)[0]  # single batch

        frame_result = FrameResult(frame_idx=center_idx)
        center_frame = frames[center_idx]

        if len(detections["boxes"]) == 0:
            return frame_result

        # 2. For each detection: crop, segment, compute severity
        for i in range(len(detections["boxes"])):
            box = detections["boxes"][i]
            score = float(detections["scores"][i])

            crop, crop_coords = crop_detection(
                center_frame, box,
                padding=crop_padding,
                source_size=self.detector.img_size,
            )

            if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
                continue

            # Segment
            seg_tensor = preprocess_for_segmentation(crop)
            mask = self.segmentor.segment(seg_tensor)

            # Resize mask back to crop resolution
            crop_h, crop_w = crop.shape[:2]
            if mask.shape != (crop_h, crop_w):
                mask = cv2.resize(
                    mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST
                )

            # Severity
            severity = compute_severity(mask)

            det_result = DetectionResult(
                box_xyxy=box,
                score=score,
                mask=mask,
                crop_image=crop,
                crop_coords=crop_coords,
                mld=severity["mld"],
                ds=severity["ds"],
            )
            frame_result.detections.append(det_result)

        return frame_result


# ── Visualization helpers ───────────────────────────────────────────

def draw_frame_results(
    frame: np.ndarray, frame_result: FrameResult, alpha: float = 0.4
) -> np.ndarray:
    """Draw detection boxes, segmentation masks, and severity on a frame.

    Args:
        frame: (H, W) uint8 grayscale frame
        frame_result: FrameResult with detections
        alpha: overlay transparency

    Returns:
        (H, W, 3) uint8 BGR annotated image
    """
    if len(frame.shape) == 2:
        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        vis = frame.copy()

    for det in frame_result.detections:
        x1, y1, x2, y2 = det.crop_coords
        color = (0, 255, 0)

        # Draw box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Overlay mask
        mask_full = np.zeros(vis.shape[:2], dtype=np.uint8)
        crop_h, crop_w = det.mask.shape
        mask_full[y1:y1 + crop_h, x1:x1 + crop_w] = det.mask
        overlay = vis.copy()
        overlay[mask_full > 0] = (0, 200, 0)
        vis = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)

        # Label
        ds_pct = det.ds * 100 if not math.isnan(det.ds) else 0
        label = f"S:{det.score:.2f} DS:{ds_pct:.0f}%"
        cv2.putText(
            vis, label, (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )

    return vis


def save_results_csv(
    pipeline_result: PipelineResult, output_path: str
) -> None:
    """Save per-detection metrics to a CSV file."""
    rows = []
    for fr in pipeline_result.frame_results:
        for det in fr.detections:
            rows.append({
                "frame": fr.frame_idx,
                "score": f"{det.score:.4f}",
                "box_x1": int(det.crop_coords[0]),
                "box_y1": int(det.crop_coords[1]),
                "box_x2": int(det.crop_coords[2]),
                "box_y2": int(det.crop_coords[3]),
                "mld": f"{det.mld:.2f}" if not math.isnan(det.mld) else "",
                "ds": f"{det.ds:.4f}" if not math.isnan(det.ds) else "",
            })

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def save_results_video(
    frames: np.ndarray,
    pipeline_result: PipelineResult,
    output_path: str,
    fps: float = 15.0,
) -> None:
    """Save annotated video from pipeline results."""
    # Build a lookup from frame_idx to FrameResult
    fr_map = {fr.frame_idx: fr for fr in pipeline_result.frame_results}

    h, w = frames.shape[1], frames.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for idx in range(frames.shape[0]):
        frame = frames[idx]
        if idx in fr_map:
            vis = draw_frame_results(frame, fr_map[idx])
        else:
            vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if len(frame.shape) == 2 else frame
        writer.write(vis)

    writer.release()
