"""Streamlit web app for ODySSeI stenosis analysis.

Upload a DICOM angiography video, detect stenoses, segment vessels,
and estimate severity — all in one interface.

Run:
    streamlit run odyssei/app.py
"""

import io
import math
import os
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Ensure ODySSeI package is importable
_ODYSSEI_ROOT = Path(__file__).resolve().parent.parent
if str(_ODYSSEI_ROOT) not in sys.path:
    sys.path.insert(0, str(_ODYSSEI_ROOT))

from odyssei.utils.dicom_utils import read_dicom, extract_temporal_window
from odyssei.pipeline import (
    StenosisPipeline,
    FrameResult,
    draw_frame_results,
    save_results_csv,
    save_results_video,
)


# ── Page config ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="ODySSeI — Stenosis Analysis",
    page_icon="🫀",
    layout="wide",
)

st.title("ODySSeI — Stenosis Detection, Segmentation & Severity")
st.markdown(
    "Upload a DICOM angiography video to detect stenoses, segment vessels, "
    "and estimate severity (MLD / DS)."
)


# ── Sidebar controls ────────────────────────────────────────────────

with st.sidebar:
    st.header("Settings")
    score_thresh = st.slider("Detection confidence threshold", 0.05, 0.95, 0.3, 0.05)
    crop_padding = st.slider("Crop padding around detections", 0.0, 0.5, 0.15, 0.05)
    frame_stride = st.number_input("Frame stride (1 = every frame)", 1, 10, 1)

    st.header("Model Paths")
    detector_ckpt = st.text_input(
        "Detector checkpoint",
        value="/home/dsa/stenosis/rfdetr_temporal/runs/cadica_temporal_v1/best.pth",
    )
    segmentor_config = st.text_input(
        "Segmentor config",
        value="/home/dsa/new_seg_final/eomt/configs/dinov2/coronary/binary_instance/eomt_small_126_patch_dinov2_skelrecall.yaml",
    )
    segmentor_ckpt = st.text_input(
        "Segmentor checkpoint",
        value="/home/dsa/new_seg_final/eomt/runs/coronary_binary_eomt_small_126_patch_dinov2_skelrecall/version_1/checkpoints/last.ckpt",
    )
    device = st.selectbox("Device", ["cuda", "cuda:0", "cuda:1", "cpu"], index=0)


# ── Model loading (cached) ─────────────────────────────────────────

@st.cache_resource
def load_pipeline(det_ckpt, seg_cfg, seg_ckpt, dev, s_thresh):
    return StenosisPipeline(
        detector_ckpt=det_ckpt,
        segmentor_config=seg_cfg,
        segmentor_ckpt=seg_ckpt,
        device=dev,
        score_thresh=s_thresh,
    )


# ── DICOM upload ────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Upload DICOM file", type=["dcm", "dicom", "DCM"], accept_multiple_files=False
)

if uploaded_file is not None:
    # Save to temp file (pydicom needs a path)
    with tempfile.NamedTemporaryFile(suffix=".dcm", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        frames, metadata = read_dicom(tmp_path)
    finally:
        os.unlink(tmp_path)

    num_frames = metadata["num_frames"]
    st.info(
        f"**{num_frames}** frames | "
        f"Resolution: {metadata['original_height']}×{metadata['original_width']} | "
        f"Angles: primary={metadata['primary_angle']}, secondary={metadata['secondary_angle']}"
    )

    # Frame selector
    if num_frames > 1:
        frame_idx = st.slider("Select frame", 0, num_frames - 1, num_frames // 2)
    else:
        frame_idx = 0

    # Show original frame
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Frame")
        st.image(frames[frame_idx], caption=f"Frame {frame_idx}", clamp=True)

    # ── Analysis ────────────────────────────────────────────────────

    col_btn1, col_btn2 = st.columns(2)
    analyze_one = col_btn1.button("🔍 Analyze This Frame")
    analyze_all = col_btn2.button("🔍 Analyze All Frames")

    if analyze_one or analyze_all:
        pipeline = load_pipeline(
            detector_ckpt, segmentor_config, segmentor_ckpt, device, score_thresh
        )

        if analyze_one:
            with st.spinner(f"Analyzing frame {frame_idx}..."):
                frame_result = pipeline.process_single_frame(
                    frames, frame_idx, crop_padding=crop_padding
                )

            with col2:
                st.subheader("Analysis Result")
                vis = draw_frame_results(frames[frame_idx], frame_result)
                st.image(
                    cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                    caption=f"Frame {frame_idx} — {len(frame_result.detections)} detection(s)",
                    clamp=True,
                )

            # Segmentation masks gallery + metrics
            if frame_result.detections:
                st.subheader(f"Stenosis Segmentation Masks ({len(frame_result.detections)} detected)")
                for i, det in enumerate(frame_result.detections):
                    ds_pct = f"{det.ds*100:.1f}%" if not math.isnan(det.ds) else "N/A"
                    mld_str = f"{det.mld:.1f} px" if not math.isnan(det.mld) else "N/A"

                    st.markdown(f"---")
                    st.markdown(f"**Stenosis #{i+1}** — Confidence: `{det.score:.3f}` | DS: `{ds_pct}` | MLD: `{mld_str}`")

                    mcol1, mcol2, mcol3 = st.columns(3)

                    # Crop image
                    with mcol1:
                        st.image(det.crop_image, caption="Crop", clamp=True, use_container_width=True)

                    # Binary mask
                    with mcol2:
                        st.image(det.mask * 255, caption="Segmentation Mask", clamp=True, use_container_width=True)

                    # Overlay: mask on crop
                    with mcol3:
                        crop_rgb = cv2.cvtColor(det.crop_image, cv2.COLOR_GRAY2RGB)
                        overlay = crop_rgb.copy()
                        overlay[det.mask > 0] = [0, 220, 0]
                        blended = cv2.addWeighted(overlay, 0.5, crop_rgb, 0.5, 0)
                        st.image(blended, caption="Overlay", clamp=True, use_container_width=True)
            else:
                st.warning("No stenoses detected in this frame.")

        if analyze_all:
            from odyssei.pipeline import PipelineResult

            progress = st.progress(0, text="Analyzing frames...")
            all_results = PipelineResult(metadata=metadata)

            indices = list(range(0, num_frames, frame_stride))
            for step, cidx in enumerate(indices):
                fr = pipeline.process_single_frame(
                    frames, cidx, crop_padding=crop_padding
                )
                all_results.frame_results.append(fr)
                progress.progress(
                    (step + 1) / len(indices),
                    text=f"Frame {cidx}/{num_frames - 1}...",
                )

            progress.empty()

            total_dets = sum(len(fr.detections) for fr in all_results.frame_results)
            st.success(
                f"Done! Processed {len(indices)} frames, found {total_dets} detections."
            )

            # Show annotated frame for current slider position
            fr_map = {fr.frame_idx: fr for fr in all_results.frame_results}
            if frame_idx in fr_map:
                with col2:
                    st.subheader("Analysis Result")
                    vis = draw_frame_results(frames[frame_idx], fr_map[frame_idx])
                    st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), clamp=True)

            # Download CSV
            csv_buf = io.StringIO()
            import csv as csv_mod

            rows = []
            for fr in all_results.frame_results:
                for det in fr.detections:
                    rows.append({
                        "frame": fr.frame_idx,
                        "score": f"{det.score:.4f}",
                        "ds": f"{det.ds:.4f}" if not math.isnan(det.ds) else "",
                        "mld": f"{det.mld:.2f}" if not math.isnan(det.mld) else "",
                    })
            if rows:
                writer = csv_mod.DictWriter(csv_buf, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            st.download_button(
                "📥 Download Metrics CSV",
                csv_buf.getvalue(),
                file_name="stenosis_metrics.csv",
                mime="text/csv",
            )

            # Download annotated video
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as vtmp:
                save_results_video(frames, all_results, vtmp.name)
                with open(vtmp.name, "rb") as vf:
                    st.download_button(
                        "📥 Download Annotated Video",
                        vf.read(),
                        file_name="annotated.mp4",
                        mime="video/mp4",
                    )
                os.unlink(vtmp.name)

else:
    st.info("👆 Upload a DICOM file to get started.")
