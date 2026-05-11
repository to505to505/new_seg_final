"""
Extract vessel-centered patches from single_dataset.

For each image:
  1. Resize image to 512x512
  2. Rasterise binary YOLO label → binary mask 512x512
  3. Skeletonise mask → 1px centerline
  4. Sample 5-10 points on skeleton with min spacing ~40px
  5. For each point: random patch size S in [24, 72], crop SxS,
     resize to 128x128, convert mask back to YOLO polygon
  6. Save to patch_dataset/{split}/images/ and labels/
"""

import cv2
import numpy as np
from pathlib import Path
from skimage.morphology import skeletonize
from PIL import Image
import random
import argparse

SRC_DIR = Path("/home/dsa/new_seg_final/single_dataset")
DST_DIR = Path("/home/dsa/new_seg_final/patch_dataset")
SPLITS = ["train", "val", "test"]

IMG_SIZE = 512         # resize source images to this
PATCH_MIN = 24         # min patch side
PATCH_MAX = 72         # max patch side
OUTPUT_SIZE = 126      # resize all patches to this (divisible by 14 for DINOv2)
PATCHES_PER_IMAGE = (5, 10)  # (min, max) patches per image
MIN_SPACING = 40       # min px between sampled skeleton points
MIN_VESSEL_RATIO = 0.05  # skip patch if vessel < 5% area
SIMPLIFY_EPS = 0.01    # contour simplification (fraction of arc length)
MIN_CONTOUR_AREA = 4   # min contour area in output-size pixels


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


def rasterise_polygons(polygons, size):
    """Normalised polygons → binary mask (0/255)."""
    mask = np.zeros((size, size), dtype=np.uint8)
    for pts in polygons:
        pixel_pts = (pts * size).astype(np.int32)
        cv2.fillPoly(mask, [pixel_pts], 255)
    return mask


def sample_skeleton_points(mask, n_target, min_spacing):
    """Skeletonise binary mask and sample points with min spacing."""
    binary = (mask > 0).astype(np.uint8)
    skel = skeletonize(binary).astype(np.uint8)
    ys, xs = np.where(skel > 0)
    if len(xs) == 0:
        return []

    coords = np.stack([xs, ys], axis=1)  # Nx2 (x, y)
    # shuffle and greedily pick with min spacing
    indices = list(range(len(coords)))
    random.shuffle(indices)

    selected = []
    for idx in indices:
        pt = coords[idx]
        if all(np.linalg.norm(pt - s) >= min_spacing for s in selected):
            selected.append(pt)
        if len(selected) >= n_target:
            break
    return selected


def mask_to_yolo_polygons(mask_patch):
    """Convert binary mask patch → list of YOLO-format strings (class 0)."""
    h, w = mask_patch.shape[:2]
    contours, _ = cv2.findContours(mask_patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue
        eps = SIMPLIFY_EPS * cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, eps, True)
        if len(cnt) < 3:
            continue
        pts = cnt.reshape(-1, 2).astype(np.float64)
        pts[:, 0] /= w
        pts[:, 1] /= h
        pts = np.clip(pts, 0.0, 1.0)
        coord_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
        lines.append(f"0 {coord_str}")
    return lines


def extract_patch(img, mask, cx, cy, patch_size, output_size):
    """Crop patch centred at (cx, cy), resize to output_size.  Returns (img_patch, mask_patch) or None."""
    h, w = img.shape[:2]
    half = patch_size // 2

    x1 = cx - half
    y1 = cy - half
    x2 = x1 + patch_size
    y2 = y1 + patch_size

    # clamp to image bounds
    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(w, x2)
    y2c = min(h, y2)

    if x2c - x1c < patch_size // 2 or y2c - y1c < patch_size // 2:
        return None

    img_crop = img[y1c:y2c, x1c:x2c]
    mask_crop = mask[y1c:y2c, x1c:x2c]

    # check vessel coverage
    vessel_ratio = np.count_nonzero(mask_crop) / max(mask_crop.size, 1)
    if vessel_ratio < MIN_VESSEL_RATIO:
        return None

    # resize to output size
    img_out = cv2.resize(img_crop, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    mask_out = cv2.resize(mask_crop, (output_size, output_size), interpolation=cv2.INTER_NEAREST)

    return img_out, mask_out


def process_image(img_path, label_path, dst_images, dst_labels):
    """Process single image: extract patches, save."""
    img = cv2.imread(str(img_path))
    if img is None:
        return 0
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)

    polygons = parse_yolo_label(label_path)
    if not polygons:
        return 0
    mask = rasterise_polygons(polygons, IMG_SIZE)

    n_target = random.randint(*PATCHES_PER_IMAGE)
    points = sample_skeleton_points(mask, n_target * 3, MIN_SPACING)
    if not points:
        return 0

    stem = img_path.stem
    count = 0
    random.shuffle(points)

    for pt in points:
        if count >= n_target:
            break
        cx, cy = int(pt[0]), int(pt[1])
        patch_size = random.randint(PATCH_MIN, PATCH_MAX)
        result = extract_patch(img, mask, cx, cy, patch_size, OUTPUT_SIZE)
        if result is None:
            continue
        img_out, mask_out = result

        yolo_lines = mask_to_yolo_polygons(mask_out)
        if not yolo_lines:
            continue

        out_name = f"{stem}_p{count}"
        cv2.imwrite(str(dst_images / f"{out_name}.png"), img_out)
        (dst_labels / f"{out_name}.txt").write_text("\n".join(yolo_lines) + "\n")
        count += 1

    return count


def process_split(split):
    src_images = SRC_DIR / split / "images"
    src_labels = SRC_DIR / split / "labels_binary"
    dst_images = DST_DIR / split / "images"
    dst_labels = DST_DIR / split / "labels_binary"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    img_files = sorted(src_images.glob("*.png"))
    total = 0
    skipped = 0

    for i, img_path in enumerate(img_files):
        label_path = src_labels / f"{img_path.stem}.txt"
        if not label_path.exists():
            skipped += 1
            continue
        n = process_image(img_path, label_path, dst_images, dst_labels)
        total += n
        if (i + 1) % 100 == 0:
            print(f"  [{split}] {i+1}/{len(img_files)} images, {total} patches so far")

    print(f"  {split}: {total} patches from {len(img_files)} images ({skipped} skipped)")
    return total


def write_dataset_yaml():
    yaml_path = DST_DIR / "dataset.yaml"
    yaml_path.write_text(
        f"path: {DST_DIR.resolve()}\n"
        f"train: train/images\n"
        f"val: val/images\n"
        f"test: test/images\n"
        f"\n"
        f"names:\n"
        f"  0: vessel\n"
    )
    print(f"Saved {yaml_path}")


def main():
    random.seed(42)
    np.random.seed(42)
    print(f"Extracting patches from {SRC_DIR} → {DST_DIR}")
    print(f"  Image resize: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Patch size: {PATCH_MIN}-{PATCH_MAX}px → resize to {OUTPUT_SIZE}x{OUTPUT_SIZE}")
    print(f"  Patches per image: {PATCHES_PER_IMAGE[0]}-{PATCHES_PER_IMAGE[1]}")
    print()

    grand_total = 0
    for split in SPLITS:
        print(f"Processing {split}...")
        grand_total += process_split(split)

    write_dataset_yaml()
    print(f"\nDone! Total patches: {grand_total}")


if __name__ == "__main__":
    main()
