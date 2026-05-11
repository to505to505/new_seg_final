"""CLI for the ODySSeI stenosis analysis pipeline.

Usage:
    python -m odyssei.cli --dicom FILE.dcm --output-dir results/
    python -m odyssei.cli --dicom-dir folder/ --output-dir results/ --save-video
"""

import argparse
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="ODySSeI: Stenosis Detection + Segmentation + Severity Pipeline"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dicom", type=str, help="Path to a single DICOM file")
    group.add_argument("--dicom-dir", type=str, help="Directory of DICOM files for batch processing")

    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")

    # Model paths
    parser.add_argument("--detector-ckpt", type=str, default=None, help="RF-DETR temporal checkpoint")
    parser.add_argument("--segmentor-config", type=str, default=None, help="EOMT config yaml")
    parser.add_argument("--segmentor-ckpt", type=str, default=None, help="EOMT checkpoint")

    # Pipeline parameters
    parser.add_argument("--score-thresh", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--crop-padding", type=float, default=0.15, help="Fractional padding around detections")
    parser.add_argument("--frame-stride", type=int, default=1, help="Process every N-th frame")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda / cpu)")

    # Output options
    parser.add_argument("--save-video", action="store_true", help="Save annotated video")
    parser.add_argument("--save-masks", action="store_true", help="Save per-detection binary masks")
    parser.add_argument("--save-csv", action="store_true", help="Save metrics CSV")
    parser.add_argument("--save-frames", action="store_true", help="Save annotated frames as images")

    args = parser.parse_args()

    # Lazy import to avoid loading models before arg validation
    from .pipeline import (
        StenosisPipeline,
        PipelineResult,
        draw_frame_results,
        save_results_csv,
        save_results_video,
    )
    from .utils.dicom_utils import read_dicom
    from .models.rfdetr_temporal_wrapper import DEFAULT_RFDETR_CHECKPOINT
    from .models.eomt_wrapper import DEFAULT_EOMT_CONFIG, DEFAULT_EOMT_CHECKPOINT
    import cv2

    os.makedirs(args.output_dir, exist_ok=True)

    # Build pipeline
    pipeline = StenosisPipeline(
        detector_ckpt=args.detector_ckpt or DEFAULT_RFDETR_CHECKPOINT,
        segmentor_config=args.segmentor_config or DEFAULT_EOMT_CONFIG,
        segmentor_ckpt=args.segmentor_ckpt or DEFAULT_EOMT_CHECKPOINT,
        device=args.device,
        score_thresh=args.score_thresh,
    )

    # Collect DICOM files
    if args.dicom:
        dicom_files = [args.dicom]
    else:
        dicom_files = sorted(
            glob(os.path.join(args.dicom_dir, "**", "*.dcm"), recursive=True)
            + glob(os.path.join(args.dicom_dir, "**", "*.DCM"), recursive=True)
        )
        if not dicom_files:
            print(f"No DICOM files found in {args.dicom_dir}")
            sys.exit(1)

    print(f"Processing {len(dicom_files)} DICOM file(s)...")

    for dcm_path in dicom_files:
        dcm_name = Path(dcm_path).stem
        print(f"\n{'='*60}")
        print(f"Processing: {dcm_path}")

        result = pipeline.process_dicom(
            dcm_path,
            crop_padding=args.crop_padding,
            frame_stride=args.frame_stride,
        )

        # Count total detections
        total_dets = sum(len(fr.detections) for fr in result.frame_results)
        print(f"  Frames processed: {len(result.frame_results)}")
        print(f"  Total detections: {total_dets}")

        dcm_out_dir = os.path.join(args.output_dir, dcm_name)
        os.makedirs(dcm_out_dir, exist_ok=True)

        # Save CSV
        if args.save_csv:
            csv_path = os.path.join(dcm_out_dir, "metrics.csv")
            save_results_csv(result, csv_path)
            print(f"  CSV saved: {csv_path}")

        # Save annotated video
        if args.save_video:
            frames, _ = read_dicom(dcm_path)
            video_path = os.path.join(dcm_out_dir, "annotated.mp4")
            save_results_video(frames, result, video_path)
            print(f"  Video saved: {video_path}")

        # Save annotated frames
        if args.save_frames:
            frames, _ = read_dicom(dcm_path)
            frames_dir = os.path.join(dcm_out_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)
            for fr in result.frame_results:
                vis = draw_frame_results(frames[fr.frame_idx], fr)
                cv2.imwrite(
                    os.path.join(frames_dir, f"frame_{fr.frame_idx:04d}.png"), vis
                )
            print(f"  Frames saved: {frames_dir}")

        # Save masks
        if args.save_masks:
            masks_dir = os.path.join(dcm_out_dir, "masks")
            os.makedirs(masks_dir, exist_ok=True)
            for fr in result.frame_results:
                for j, det in enumerate(fr.detections):
                    mask_path = os.path.join(
                        masks_dir, f"frame_{fr.frame_idx:04d}_det_{j:02d}.png"
                    )
                    cv2.imwrite(mask_path, det.mask * 255)
            print(f"  Masks saved: {masks_dir}")

        # Print severity summary
        for fr in result.frame_results:
            for det in fr.detections:
                ds_str = f"{det.ds*100:.1f}%" if not np.isnan(det.ds) else "N/A"
                mld_str = f"{det.mld:.1f}px" if not np.isnan(det.mld) else "N/A"
                print(
                    f"  Frame {fr.frame_idx}: score={det.score:.2f} "
                    f"DS={ds_str} MLD={mld_str}"
                )

    print(f"\nDone. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
