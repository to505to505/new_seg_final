#!/usr/bin/env python3
"""
Fragmentation rate evaluation on raw video clips (no labels).

Loads either the 2D EoMT model or the Video EoMT model from a YAML config,
runs inference on every frame of every clip in a video directory, and
measures temporal stability of the predictions via "fragmentation rate"
(adapted from /home/dsa/stenosis/_eval_stfs_ablations.py).

Tracks are built from predictions over consecutive frames by linking
instances whose bounding boxes have IoU >= LINK_IOU.  A track is "lost"
on a frame whenever no prediction matches the track's last seen box with
IoU >= MATCH_IOU and confidence >= score_thr.  Each transition from a
"lost" gap back to a hit (after the track has already started) counts
as one fragmentation event.

  fragmentation_rate = total_frag_events / total_track_frames

Lower is better (0 means predictions are perfectly stable across time).

Usage:
    python inference/eval_fragmentation.py \\
        --config configs/dinov2/coronary/instance/test_2d_model_on_single_dataset.yaml \\
        --ckpt   runs/coronary_instance_eomt_small_512_dinov2/version_0/checkpoints/best.ckpt \\
        --videos /home/dsa/new_seg_final/just_videos_newsequentialdata \\
        --output runs/coronary_instance_eomt_small_512_dinov2/version_0/fragmentation.txt

Use ``--video`` instead of ``--ckpt`` autodetection if you want to force
the video model to be evaluated with sliding T-frame windows.
"""

import argparse
import importlib
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch.amp.autocast_mode import autocast
from torchvision import tv_tensors

SCRIPT_DIR = Path(__file__).resolve().parent
EOMT_ROOT = SCRIPT_DIR.parent
if str(EOMT_ROOT) not in sys.path:
    sys.path.insert(0, str(EOMT_ROOT))


# --------------------------------------------------------------------------
# Tunables (mirroring _eval_stfs_ablations.py)
# --------------------------------------------------------------------------
LINK_IOU = 0.3       # IoU to link predictions across consecutive frames
MATCH_IOU = 0.3      # IoU to declare a track-frame as "hit"
MAX_GAP = 5          # frames a track may be missing before being terminated
MIN_TRACK_LEN = 3    # ignore very short tracks (matches reference)
DEFAULT_SCORE_THR = 0.3
DEFAULT_T = 5        # window size for the video model


# --------------------------------------------------------------------------
# Geometry helpers (bbox-based, like the reference)
# --------------------------------------------------------------------------
def _masks_to_boxes(masks: np.ndarray) -> np.ndarray:
    """(N, H, W) bool -> (N, 4) [x0, y0, x1, y1]; empty masks -> zeros."""
    if masks.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    boxes = np.zeros((masks.shape[0], 4), dtype=np.float32)
    for i, m in enumerate(masks):
        ys, xs = np.where(m)
        if ys.size == 0:
            continue
        boxes[i] = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    return boxes


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    a_area = (a[:, 2] - a[:, 0]).clip(0) * (a[:, 3] - a[:, 1]).clip(0)
    b_area = (b[:, 2] - b[:, 0]).clip(0) * (b[:, 3] - b[:, 1]).clip(0)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clip(0)
    inter = wh[..., 0] * wh[..., 1]
    union = a_area[:, None] + b_area[None, :] - inter
    return inter / np.maximum(union, 1e-6)


# --------------------------------------------------------------------------
# Track building from predictions (no GT)
# --------------------------------------------------------------------------
def build_pred_tracks(boxes_per_frame: List[np.ndarray],
                      max_gap: int = MAX_GAP) -> List[List[Tuple[int, np.ndarray]]]:
    """Greedy IoU linking that survives short prediction gaps.

    A track is kept "alive" for up to ``max_gap`` frames after its last
    detection.  This lets the metric below count those missing frames as
    fragmentation events instead of treating each disconnected detection
    burst as a brand-new track.
    """
    tracks: List[List[Tuple[int, np.ndarray]]] = []
    # active entries: (track_idx, last_seen_frame, last_seen_box)
    active: List[Tuple[int, int, np.ndarray]] = []

    for f_idx, boxes in enumerate(boxes_per_frame):
        # Drop tracks that have been silent for longer than the gap budget.
        active = [a for a in active if (f_idx - a[1]) <= max_gap]

        if boxes.shape[0] == 0:
            continue

        if not active:
            for box in boxes:
                tracks.append([(f_idx, box.copy())])
                active.append((len(tracks) - 1, f_idx, box.copy()))
            continue

        last_boxes = np.stack([a[2] for a in active], axis=0)
        ious = _iou_matrix(last_boxes, boxes)
        flat = [
            (ious[i, j], i, j)
            for i in range(ious.shape[0])
            for j in range(ious.shape[1])
            if ious[i, j] >= LINK_IOU
        ]
        flat.sort(reverse=True)
        used_track_local = set()
        used_box = set()
        new_active: List[Tuple[int, int, np.ndarray]] = []
        for _, i, j in flat:
            if i in used_track_local or j in used_box:
                continue
            ti = active[i][0]
            tracks[ti].append((f_idx, boxes[j].copy()))
            new_active.append((ti, f_idx, boxes[j].copy()))
            used_track_local.add(i)
            used_box.add(j)
        # Carry forward unmatched but still-alive tracks (gap continues).
        for k, a in enumerate(active):
            if k not in used_track_local:
                new_active.append(a)
        # Brand-new tracks for unmatched detections.
        for j in range(boxes.shape[0]):
            if j in used_box:
                continue
            tracks.append([(f_idx, boxes[j].copy())])
            new_active.append((len(tracks) - 1, f_idx, boxes[j].copy()))
        active = new_active
    return tracks


def compute_video_frag(
    boxes_per_frame: List[np.ndarray],
    scores_per_frame: List[np.ndarray],
    score_thr: float,
) -> Tuple[int, int]:
    """Returns (fragmentation_events, track_frames_total).

    For each track that survived ``MAX_GAP`` linking, walk frame-by-frame
    over its life span ``[first_f, last_f]``.  A frame is a "hit" if the
    track has a detection there; otherwise we look at all predictions on
    that frame and accept it as a hit if any of them matches the track's
    last seen box (IoU >= MATCH_IOU, score >= score_thr).  Each transition
    from a "gap" back to a hit (after the track has already started)
    increments the fragmentation counter.
    """
    tracks = build_pred_tracks(boxes_per_frame)
    frag_total = 0
    track_frames_total = 0
    for trk in tracks:
        if len(trk) < MIN_TRACK_LEN:
            track_frames_total += len(trk)
            continue
        first_f = trk[0][0]
        last_f = trk[-1][0]
        present = {f: b for f, b in trk}
        last_seen_box = trk[0][1]
        status = []
        for f in range(first_f, last_f + 1):
            if f in present:
                last_seen_box = present[f]
                status.append(1)
                continue
            cand_boxes = boxes_per_frame[f]
            cand_scores = scores_per_frame[f]
            keep = cand_scores >= score_thr
            if cand_boxes.size == 0 or not keep.any():
                status.append(0)
                continue
            ious = _iou_matrix(last_seen_box[None, :], cand_boxes[keep])[0]
            status.append(1 if (ious >= MATCH_IOU).any() else 0)

        seen_one = False
        in_gap = False
        for s in status:
            if s == 1:
                if in_gap and seen_one:
                    frag_total += 1
                in_gap = False
                seen_one = True
            else:
                if seen_one:
                    in_gap = True
        track_frames_total += len(status)
    return frag_total, track_frames_total


# --------------------------------------------------------------------------
# Model loading (config-driven)
# --------------------------------------------------------------------------
def load_model_from_config(config_path: Path, ckpt_path: Path, device: torch.device):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    network_cfg = cfg["model"]["init_args"]["network"]
    encoder_cfg = network_cfg["init_args"]["encoder"]
    img_size = tuple(cfg["model"]["init_args"].get("img_size", (512, 512)))
    num_classes = int(cfg["model"]["init_args"].get("num_classes", 9))

    enc_mod, enc_cls = encoder_cfg["class_path"].rsplit(".", 1)
    encoder = getattr(importlib.import_module(enc_mod), enc_cls)(
        img_size=img_size, **encoder_cfg.get("init_args", {})
    )
    net_mod, net_cls = network_cfg["class_path"].rsplit(".", 1)
    net_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
    network = getattr(importlib.import_module(net_mod), net_cls)(
        masked_attn_enabled=False, num_classes=num_classes, encoder=encoder, **net_kwargs,
    )
    lit_mod, lit_cls = cfg["model"]["class_path"].rsplit(".", 1)
    model_kwargs = {
        k: v for k, v in cfg["model"]["init_args"].items()
        if k not in ("network", "ckpt_path")
    }
    model_kwargs.setdefault("num_classes", num_classes)
    model_kwargs.setdefault("img_size", img_size)
    model = getattr(importlib.import_module(lit_mod), lit_cls)(
        network=network, **model_kwargs,
    ).eval().to(device)

    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=False)

    is_video = "video_mask_classification" in cfg["model"]["class_path"].lower()
    return model, img_size, is_video


# --------------------------------------------------------------------------
# Inference helpers
# --------------------------------------------------------------------------
def _load_frame(path: Path) -> torch.Tensor:
    return tv_tensors.Image(Image.open(path).convert("RGB"))


@torch.no_grad()
def predict_2d(model, frame: torch.Tensor, device, conf_thresh: float):
    img_sizes = [frame.shape[-2:]]
    transformed = model.resize_and_pad_imgs_instance_panoptic([frame.to(device)])
    with autocast(dtype=torch.float16, device_type=device.type):
        mlpl, clpl = model(transformed)
    ml = F.interpolate(mlpl[-1], model.img_size, mode="bilinear")
    ml = model.revert_resize_and_pad_logits_instance_panoptic(ml, img_sizes)
    cl = clpl[-1]
    return _decode(model, ml[0], cl[0], conf_thresh)


@torch.no_grad()
def predict_video(model, frames: List[torch.Tensor], device, conf_thresh: float):
    """frames: list of T tensors (C, H, W). Returns predictions for the central frame."""
    clip = torch.stack([f.to(device) for f in frames], dim=0)  # (T, 3, H, W)
    clips_tensor, img_sizes = model._resize_and_pad_clips([clip])
    B, T = clips_tensor.shape[0], clips_tensor.shape[1]
    with autocast(dtype=torch.float16, device_type=device.type):
        mlpl, clpl = model(clips_tensor)
    ml = model._slice_central(mlpl[-1], B, T)
    cl = model._slice_central(clpl[-1], B, T)
    ml = F.interpolate(ml, model.img_size, mode="bilinear")
    ml = model.revert_resize_and_pad_logits_instance_panoptic(ml, img_sizes)
    return _decode(model, ml[0], cl[0], conf_thresh)


def _decode(model, mask_logits: torch.Tensor, class_logits: torch.Tensor, conf_thresh: float):
    """Common top-K + sigmoid decoding."""
    scores = class_logits.softmax(dim=-1)[:, :-1]
    labels = (
        torch.arange(scores.shape[-1], device=scores.device)
        .unsqueeze(0).repeat(scores.shape[0], 1).flatten(0, 1)
    )
    topk_scores, topk_indices = scores.flatten(0, 1).topk(
        model.eval_top_k_instances, sorted=False
    )
    labels = labels[topk_indices]
    topk_indices = topk_indices // scores.shape[-1]
    mask_logits = mask_logits[topk_indices]
    masks = mask_logits > 0
    mask_scores = (
        mask_logits.sigmoid().flatten(1) * masks.flatten(1)
    ).sum(1) / (masks.flatten(1).sum(1) + 1e-6)
    final_scores = topk_scores * mask_scores
    keep = final_scores > conf_thresh
    masks_np = masks[keep].cpu().numpy()
    boxes_np = _masks_to_boxes(masks_np)
    return boxes_np, final_scores[keep].cpu().numpy()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def list_clips(root: Path) -> List[Path]:
    return sorted(p for p in root.iterdir() if p.is_dir())


def list_frames(clip_dir: Path) -> List[Path]:
    return sorted(clip_dir.glob("*.png")) + sorted(clip_dir.glob("*.jpg"))


def evaluate_clip_2d(model, frames: List[Path], device, conf_thresh: float):
    boxes_per_frame, scores_per_frame = [], []
    for fp in frames:
        frame = _load_frame(fp)
        b, s = predict_2d(model, frame, device, conf_thresh)
        boxes_per_frame.append(b)
        scores_per_frame.append(s)
    return boxes_per_frame, scores_per_frame


def evaluate_clip_video(model, frames: List[Path], device, conf_thresh: float, T: int):
    """Slide a window of T frames; central prediction per window."""
    central = T // 2
    loaded = [_load_frame(fp) for fp in frames]
    boxes_per_frame, scores_per_frame = [], []
    for i in range(len(loaded)):
        idxs = [min(max(i - central + k, 0), len(loaded) - 1) for k in range(T)]
        clip = [loaded[k] for k in idxs]
        b, s = predict_video(model, clip, device, conf_thresh)
        boxes_per_frame.append(b)
        scores_per_frame.append(s)
    return boxes_per_frame, scores_per_frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--videos", required=True, type=Path,
                   help="Directory containing one sub-folder per video clip.")
    p.add_argument("--output", required=True, type=Path,
                   help="Path to write the fragmentation report (txt).")
    p.add_argument("--score-thr", type=float, default=DEFAULT_SCORE_THR)
    p.add_argument("--T", type=int, default=DEFAULT_T,
                   help="Window size for the video model.")
    args = p.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, img_size, is_video = load_model_from_config(args.config, args.ckpt, device)
    print(f"Loaded {'video' if is_video else '2D'} model from {args.ckpt}")

    clips = list_clips(args.videos)
    print(f"Found {len(clips)} clips in {args.videos}")

    per_clip = []
    total_frag = 0
    total_track_frames = 0
    for clip_dir in clips:
        frames = list_frames(clip_dir)
        if len(frames) < 3:
            continue
        if is_video:
            boxes, scores = evaluate_clip_video(model, frames, device, args.score_thr, args.T)
        else:
            boxes, scores = evaluate_clip_2d(model, frames, device, args.score_thr)

        frag, tf = compute_video_frag(boxes, scores, args.score_thr)
        rate = frag / tf if tf > 0 else 0.0
        per_clip.append((clip_dir.name, len(frames), frag, tf, rate))
        total_frag += frag
        total_track_frames += tf
        print(f"  {clip_dir.name:<20s}  frames={len(frames):3d}  frag={frag:3d}  "
              f"track_frames={tf:4d}  rate={rate:.4f}")

    overall = total_frag / total_track_frames if total_track_frames > 0 else 0.0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        f.write("Fragmentation rate evaluation\n")
        f.write("=============================\n")
        f.write(f"config:        {args.config}\n")
        f.write(f"ckpt:          {args.ckpt}\n")
        f.write(f"videos:        {args.videos}\n")
        f.write(f"score_thr:     {args.score_thr}\n")
        f.write(f"LINK_IOU:      {LINK_IOU}\n")
        f.write(f"MATCH_IOU:     {MATCH_IOU}\n")
        f.write(f"model_kind:    {'video' if is_video else '2D'}\n")
        if is_video:
            f.write(f"T:             {args.T}\n")
        f.write("\nPer-clip results:\n")
        f.write(f"{'clip':<20s}  {'frames':>6s}  {'frag':>5s}  {'tframes':>7s}  {'rate':>7s}\n")
        for name, nf, fr, tf, rate in per_clip:
            f.write(f"{name:<20s}  {nf:>6d}  {fr:>5d}  {tf:>7d}  {rate:>7.4f}\n")
        f.write("\n=============================\n")
        f.write(f"TOTAL frag events:    {total_frag}\n")
        f.write(f"TOTAL track frames:   {total_track_frames}\n")
        f.write(f"OVERALL frag rate:    {overall:.4f}\n")
    print(f"\nOverall fragmentation rate: {overall:.4f}")
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
