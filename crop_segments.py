"""
Crop segments from stenosis_arcade dataset.
For each polygon segment in a label file:
  1. Compute bounding box around the polygon
  2. Add padding
  3. Crop the image to that bbox
  4. Recalculate polygon coordinates relative to the crop
  5. Save cropped image + label
"""

import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

SRC_DIR = Path("/home/dsa/new_seg_final/stenosis_arcade")
DST_DIR = Path("/home/dsa/new_seg_final/stenosis_cropped")
PADDING_RATIO = 0.1  # 10% padding around bbox


def parse_label_line(line):
    """Parse a YOLO segmentation label line: class x1 y1 x2 y2 ..."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None, None
    cls = int(parts[0])
    coords = list(map(float, parts[1:]))
    # coords are x1,y1,x2,y2,... (normalized)
    xs = coords[0::2]
    ys = coords[1::2]
    return cls, list(zip(xs, ys))


def process_split(split):
    src_images = SRC_DIR / split / "images"
    src_labels = SRC_DIR / split / "labels"
    dst_images = DST_DIR / split / "images"
    dst_labels = DST_DIR / split / "labels"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    label_files = sorted(src_labels.glob("*.txt"))
    total_crops = 0

    for label_file in label_files:
        stem = label_file.stem
        img_path = src_images / f"{stem}.png"
        if not img_path.exists():
            print(f"  WARNING: image not found for {label_file.name}, skipping")
            continue

        img = Image.open(img_path)
        img_w, img_h = img.size

        lines = label_file.read_text().strip().split("\n")
        lines = [l for l in lines if l.strip()]

        for seg_idx, line in enumerate(lines):
            cls, polygon = parse_label_line(line)
            if polygon is None:
                continue

            # Polygon in pixel coords
            px = [x * img_w for x, y in polygon]
            py = [y * img_h for x, y in polygon]

            # Bounding box
            x_min, x_max = min(px), max(px)
            y_min, y_max = min(py), max(py)
            bbox_w = x_max - x_min
            bbox_h = y_max - y_min

            # Add padding
            pad_x = bbox_w * PADDING_RATIO
            pad_y = bbox_h * PADDING_RATIO
            x_min_pad = max(0, x_min - pad_x)
            y_min_pad = max(0, y_min - pad_y)
            x_max_pad = min(img_w, x_max + pad_x)
            y_max_pad = min(img_h, y_max + pad_y)

            # Integer crop coordinates
            crop_x1 = int(x_min_pad)
            crop_y1 = int(y_min_pad)
            crop_x2 = int(np.ceil(x_max_pad))
            crop_y2 = int(np.ceil(y_max_pad))

            crop_w = crop_x2 - crop_x1
            crop_h = crop_y2 - crop_y1
            if crop_w < 2 or crop_h < 2:
                continue

            # Crop image
            cropped = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))

            # Recalculate polygon relative to crop, normalized
            new_polygon = []
            for x, y in polygon:
                px_abs = x * img_w
                py_abs = y * img_h
                nx = (px_abs - crop_x1) / crop_w
                ny = (py_abs - crop_y1) / crop_h
                # Clamp to [0, 1]
                nx = max(0.0, min(1.0, nx))
                ny = max(0.0, min(1.0, ny))
                new_polygon.append((nx, ny))

            # Use class 0 for single-class segmentation
            coord_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in new_polygon)
            label_line = f"0 {coord_str}\n"

            # Save with unique name: originalname_segidx
            out_name = f"{stem}_{seg_idx}"
            cropped.save(dst_images / f"{out_name}.png")
            (dst_labels / f"{out_name}.txt").write_text(label_line)
            total_crops += 1

    print(f"  {split}: {total_crops} crops from {len(label_files)} images")
    return total_crops


def main():
    print(f"Source: {SRC_DIR}")
    print(f"Destination: {DST_DIR}")
    total = 0
    for split in ["train", "val", "test"]:
        total += process_split(split)
    print(f"Total: {total} cropped segments")

    # Create data.yaml
    yaml_content = f"""train: {DST_DIR}/train/images
val: {DST_DIR}/val/images
test: {DST_DIR}/test/images

nc: 1
names: ['stenosis']
"""
    (DST_DIR / "data.yaml").write_text(yaml_content)
    print("Created data.yaml")


if __name__ == "__main__":
    main()
