"""
Merge multi-class YOLO segmentation labels into binary (single class).
Rasterises all polygons -> morphological closing -> extracts contours WITH
hierarchy (RETR_CCOMP) so that holes are preserved.

Each outer contour and its child holes are encoded as a single composite
polygon using "bridge-cut" lines.  When PIL rasterises such a polygon with
the even-odd fill rule the hole area stays empty — exactly what we need for
training.

Saves to labels_binary/ directories.
"""

import cv2
import numpy as np
from pathlib import Path

DATASET_ROOT = Path("/home/dsa/new_seg_final/single_dataset")
RASTER_SIZE = 2048       # high res for precision
CLOSING_KERNEL = 5       # morphological closing kernel (merges small gaps)
CLOSING_ITERS = 1        # closing iterations
MIN_CONTOUR_AREA = 50    # skip tiny noise outer contours  (raster px²)
MIN_HOLE_AREA = 100      # skip tiny holes (raster px²)
SIMPLIFY_EPS = 0.001     # contour simplification factor (fraction of arc length)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def parse_yolo_label(label_path):
    """Parse YOLO segmentation label file.  Returns list of Nx2 normalised arrays."""
    polygons = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # class + at least 3 points (6 coords)
                continue
            coords = list(map(float, parts[1:]))
            pts = np.array(coords).reshape(-1, 2)
            polygons.append(pts)
    return polygons


# ---------------------------------------------------------------------------
# Mask operations
# ---------------------------------------------------------------------------

def polygons_to_binary_mask(polygons, size=RASTER_SIZE):
    """Rasterise normalised polygons to a binary mask (0/255)."""
    mask = np.zeros((size, size), dtype=np.uint8)
    for pts in polygons:
        pixel_pts = (pts * size).astype(np.int32)
        cv2.fillPoly(mask, [pixel_pts], 255)
    return mask


def close_mask(mask, kernel_size=CLOSING_KERNEL, iterations=CLOSING_ITERS):
    """Morphological closing — merges small gaps without expanding outward."""
    if kernel_size <= 0 or iterations <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)


# ---------------------------------------------------------------------------
# Contour extraction with holes
# ---------------------------------------------------------------------------

def _simplify(cnt):
    """Simplify contour with Douglas–Peucker."""
    eps = SIMPLIFY_EPS * cv2.arcLength(cnt, True)
    return cv2.approxPolyDP(cnt, eps, True)


def _closest_pair_idx(pts_a, pts_b):
    """Return (idx_a, idx_b) of the closest point pair (Euclidean)."""
    # vectorised pairwise distances
    a = np.asarray(pts_a, dtype=np.float64)
    b = np.asarray(pts_b, dtype=np.float64)
    dists = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
    flat = np.argmin(dists)
    return int(flat // b.shape[0]), int(flat % b.shape[0])


def build_composite_polygon(outer_cnt, hole_cnts, size):
    """Build a single YOLO polygon that traces the outer boundary + holes.

    For each hole a *bridge cut* connects the outer boundary to the hole:
        …outer → bridge → hole (full loop) → bridge back → outer…

    When rasterised with even-odd fill (PIL default) the hole area remains
    empty.  Returns normalised Nx2 float array.
    """
    outer_pts = outer_cnt.reshape(-1, 2).tolist()

    for hole_cnt in hole_cnts:
        hole_pts = hole_cnt.reshape(-1, 2).tolist()

        # closest point pair between current outer_pts and hole
        o_idx, h_idx = _closest_pair_idx(outer_pts, hole_pts)

        # reorder hole to start from closest point
        hole_reordered = hole_pts[h_idx:] + hole_pts[:h_idx]

        # splice:  outer[:o+1] ++ hole_loop ++ bridge_back ++ outer[o+1:]
        outer_pts = (
            outer_pts[: o_idx + 1]
            + hole_reordered
            + [hole_reordered[0]]       # close hole loop
            + [outer_pts[o_idx]]        # bridge back to outer
            + outer_pts[o_idx + 1:]
        )

    pts = np.array(outer_pts, dtype=np.float64) / size
    return pts


def extract_polygons_with_holes(mask, size=RASTER_SIZE):
    """Extract contours using RETR_CCOMP and return YOLO-format lines.

    RETR_CCOMP gives a two-level hierarchy:
        level 0  — outer boundaries  (parent == -1)
        level 1  — holes             (parent == outer index)
    """
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_L1
    )
    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]  # shape (N, 4): [next, prev, first_child, parent]
    lines = []

    for idx in range(len(contours)):
        # only process outer contours (no parent)
        if hierarchy[idx][3] != -1:
            continue

        outer_cnt = contours[idx]
        if cv2.contourArea(outer_cnt) < MIN_CONTOUR_AREA:
            continue

        outer_cnt = _simplify(outer_cnt)
        if len(outer_cnt) < 3:
            continue

        # --- collect child holes ---
        hole_cnts = []
        child_idx = hierarchy[idx][2]           # first child
        while child_idx >= 0:
            h_cnt = contours[child_idx]
            if cv2.contourArea(h_cnt) >= MIN_HOLE_AREA:
                h_cnt = _simplify(h_cnt)
                if len(h_cnt) >= 3:
                    hole_cnts.append(h_cnt)
            child_idx = hierarchy[child_idx][0]  # next sibling hole

        # --- build composite polygon ---
        pts = build_composite_polygon(outer_cnt, hole_cnts, size)
        coord_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in pts)
        lines.append(f"0 {coord_str}")

    return lines


# ---------------------------------------------------------------------------
# Per-split processing
# ---------------------------------------------------------------------------

def process_split(split_name):
    labels_dir = DATASET_ROOT / split_name / "labels"
    output_dir = DATASET_ROOT / split_name / "labels_binary"
    output_dir.mkdir(exist_ok=True)

    if not labels_dir.exists():
        print(f"  Skipping {split_name}: no labels dir")
        return

    label_files = sorted(labels_dir.glob("*.txt"))
    print(f"  {split_name}: {len(label_files)} files")

    for lf in label_files:
        polygons = parse_yolo_label(lf)
        if not polygons:
            (output_dir / lf.name).write_text("")
            continue

        mask = polygons_to_binary_mask(polygons)
        mask = close_mask(mask)
        lines = extract_polygons_with_holes(mask)

        (output_dir / lf.name).write_text("\n".join(lines) + ("\n" if lines else ""))

    print(f"  {split_name}: done -> {output_dir}")


def main():
    print(
        f"Raster size: {RASTER_SIZE}, "
        f"closing kernel: {CLOSING_KERNEL}, iters: {CLOSING_ITERS}"
    )
    for split in ["train", "val", "test"]:
        process_split(split)
    print("All done!")


if __name__ == "__main__":
    main()
