"""
Visualise random patches from patch_dataset.
Loads 16 random patches, overlays the YOLO mask, shows a 4x4 grid.
Saves to patch_examples.png.
"""

import cv2
import numpy as np
from pathlib import Path
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PATCH_DIR = Path("/home/dsa/new_seg_final/patch_dataset")
SPLIT = "train"
N_SAMPLES = 16
COLS = 4
ROWS = 4
OUT_PATH = Path("/home/dsa/new_seg_final/patch_examples.png")


def parse_yolo_label(label_path):
    """Parse YOLO segmentation label → list of Nx2 normalised arrays."""
    polygons = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            coords = list(map(float, parts[1:]))
            pts = np.array(coords).reshape(-1, 2)
            polygons.append(pts)
    return polygons


def rasterise_polygons(polygons, h, w):
    """Normalised polygons → binary mask."""
    mask = np.zeros((h, w), dtype=np.uint8)
    for pts in polygons:
        pixel_pts = pts.copy()
        pixel_pts[:, 0] *= w
        pixel_pts[:, 1] *= h
        pixel_pts = pixel_pts.astype(np.int32)
        cv2.fillPoly(mask, [pixel_pts], 255)
    return mask


def overlay_mask(img, mask, color=(0, 0, 255), alpha=0.4):
    """Overlay binary mask on image with semi-transparent colour."""
    overlay = img.copy()
    overlay[mask > 0] = (
        overlay[mask > 0].astype(np.float32) * (1 - alpha)
        + np.array(color, dtype=np.float32) * alpha
    ).astype(np.uint8)
    # draw contour outline
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 1)
    return overlay


def main():
    random.seed(123)
    img_dir = PATCH_DIR / SPLIT / "images"
    lbl_dir = PATCH_DIR / SPLIT / "labels"

    img_files = sorted(img_dir.glob("*.png"))
    if not img_files:
        print(f"No images found in {img_dir}")
        return

    samples = random.sample(img_files, min(N_SAMPLES, len(img_files)))

    fig, axes = plt.subplots(ROWS, COLS, figsize=(14, 14))
    fig.suptitle("Patch examples (red = vessel mask overlay)", fontsize=14, y=0.98)

    for idx, ax in enumerate(axes.flat):
        if idx >= len(samples):
            ax.axis("off")
            continue

        img_path = samples[idx]
        lbl_path = lbl_dir / f"{img_path.stem}.txt"

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        if lbl_path.exists():
            polygons = parse_yolo_label(lbl_path)
            mask = rasterise_polygons(polygons, h, w)
            vis = overlay_mask(img, mask, color=(255, 50, 50), alpha=0.45)
        else:
            vis = img
            mask = np.zeros((h, w), dtype=np.uint8)

        vessel_pct = 100.0 * np.count_nonzero(mask) / max(mask.size, 1)
        ax.imshow(vis)
        ax.set_title(f"{img_path.stem}\n{w}x{h}  vessel={vessel_pct:.1f}%", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(str(OUT_PATH), dpi=150, bbox_inches="tight")
    print(f"Saved visualisation to {OUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
