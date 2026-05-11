"""
Postprocessing for EoMT instance segmentation.

Replicates the key operations from PostprocessObject (segmentation_modules):
1. Cross-class NMS (mask IoU based)
2. Mask → Shapely polygon conversion
3. Small polygon filtering (area < max_area / min_ratio)
4. Crux cordis center finding (distance_transform_edt)
5. Dominant polygon selection near center
6. Polygon → binary mask conversion

No dependency on segmentation_modules or Detectron2.
"""

import warnings
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from shapely.geometry import GeometryCollection, MultiPolygon, Point, Polygon


# ==========================================================================
# Cross-class NMS
# ==========================================================================

def cross_class_nms(
    masks: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove lower-confidence masks when IoU > threshold between different classes."""
    n = len(masks)
    if n <= 1:
        return masks, labels, scores

    keep = [True] * n
    sorted_indices = sorted(range(n), key=lambda i: scores[i], reverse=True)

    for i_pos, i in enumerate(sorted_indices):
        if not keep[i]:
            continue
        mask_i = masks[i].astype(bool)
        area_i = mask_i.sum()
        if area_i == 0:
            continue

        for j in sorted_indices[i_pos + 1:]:
            if not keep[j]:
                continue
            if labels[i] == labels[j]:
                continue

            mask_j = masks[j].astype(bool)
            area_j = mask_j.sum()
            if area_j == 0:
                continue

            intersection = np.logical_and(mask_i, mask_j).sum()
            union = np.logical_or(mask_i, mask_j).sum()
            if union > 0 and (intersection / union) > iou_threshold:
                keep[j] = False

    sel = [i for i in range(n) if keep[i]]
    return masks[sel], labels[sel], scores[sel]


# ==========================================================================
# Mask ↔ Polygon conversions
# ==========================================================================

def mask_to_polygons(
    mask: np.ndarray,
    min_contour_area: float = 1.0,
) -> List[Polygon]:
    """Convert a binary mask [H, W] to a list of Shapely Polygons."""
    if mask.dtype != np.uint8:
        mask = (mask > 0).astype(np.uint8) * 255

    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours or hierarchy is None:
        return []

    polygons: List[Polygon] = []

    for i, contour in enumerate(contours):
        # Only outer contours (no parent)
        if hierarchy[0][i][3] != -1:
            continue

        if cv2.contourArea(contour) < min_contour_area:
            continue

        shell = contour.squeeze(axis=1)
        if len(shell) < 3:
            continue

        # Collect child contours (holes)
        interiors = []
        child_idx = hierarchy[0][i][2]
        while child_idx != -1:
            hole = contours[child_idx].squeeze(axis=1)
            if len(hole) >= 3:
                interiors.append(hole)
            child_idx = hierarchy[0][child_idx][0]

        try:
            poly = Polygon(shell=shell, holes=interiors or None)
            if not poly.is_valid:
                poly = poly.buffer(0)

            if poly.is_empty:
                continue

            if isinstance(poly, Polygon):
                polygons.append(poly)
            elif isinstance(poly, (MultiPolygon, GeometryCollection)):
                for p in poly.geoms:
                    if isinstance(p, Polygon) and p.is_valid and not p.is_empty:
                        polygons.append(p)
        except Exception:
            pass

    return polygons


def polygons_to_mask(
    polygons: Union[List[Polygon], Polygon],
    height: int,
    width: int,
) -> np.ndarray:
    """Rasterize Shapely polygon(s) into a binary uint8 mask."""
    if isinstance(polygons, Polygon):
        polygons = [polygons]

    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        if poly is None or poly.is_empty:
            continue
        ext = np.round(np.array(poly.exterior.coords, dtype=np.float32)).astype(np.int32)
        if ext.shape[0] >= 3:
            cv2.fillPoly(mask, [ext], color=1)
        for interior in poly.interiors:
            hole = np.round(np.array(interior.coords, dtype=np.float32)).astype(np.int32)
            if hole.shape[0] >= 3:
                cv2.fillPoly(mask, [hole], color=0)
    return mask


# ==========================================================================
# Polygon filtering
# ==========================================================================

def filter_small_polygons(
    polygons: List[Polygon],
    min_ratio: float = 20,
) -> List[Polygon]:
    """Keep polygons whose area >= max_area / min_ratio."""
    if not polygons:
        return []

    areas = []
    for p in polygons:
        if isinstance(p, Polygon) and not p.is_empty:
            areas.append(p.area)
        else:
            areas.append(0.0)

    max_area = max(areas) if areas else 0.0
    if max_area == 0:
        return []

    min_allowed = max_area / min_ratio
    return [p for p, a in zip(polygons, areas) if a >= min_allowed]


# ==========================================================================
# Crux cordis / dominant polygon
# ==========================================================================

def _find_minimum_touching_circle(masks: List[np.ndarray]) -> Optional[Tuple[int, int]]:
    """Find center (x, y) of the minimum enclosing circle touching all masks."""
    if not masks:
        return None

    distance_transforms = [distance_transform_edt(m == 0) for m in masks]
    max_dist_map = np.maximum.reduce(distance_transforms)
    center_yx = np.unravel_index(np.argmin(max_dist_map), max_dist_map.shape)
    return (center_yx[1], center_yx[0])


def find_crux_cordis(
    polygons_by_class: List[List[Polygon]],
    canvas_shape: Tuple[int, int] = (512, 512),
) -> Optional[Tuple[int, int]]:
    """Find crux cordis center from class-grouped polygon lists.

    Returns (x, y) or None if input is empty.
    """
    if not polygons_by_class:
        return None

    masks: List[np.ndarray] = []
    for class_polygons in polygons_by_class:
        class_mask = np.zeros(canvas_shape, dtype=np.uint8)
        for poly in class_polygons:
            ext = np.array(poly.exterior.coords, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(class_mask, [ext], color=1)
            if poly.interiors:
                for interior in poly.interiors:
                    hole = np.array(interior.coords, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.fillPoly(class_mask, [hole], color=0)
        masks.append(class_mask)

    return _find_minimum_touching_circle(masks)


def find_dominant_polygon(
    polygons: List[Polygon],
    center_xy: Tuple[float, float],
    ratio: float = 2,
) -> Optional[Polygon]:
    """Select the dominant polygon nearest to center, or a larger one if it exists.

    Returns a single dominant Polygon or None.
    """
    valid = [p for p in polygons if p.is_valid and not p.is_empty]
    if not valid:
        return None

    center_point = Point(center_xy)
    closest = min(valid, key=lambda p: p.distance(center_point))

    dominant_candidates = [
        p for p in valid
        if p is not closest and p.area > closest.area * ratio
    ]

    if dominant_candidates:
        return max(dominant_candidates, key=lambda p: p.area)
    return closest


# ==========================================================================
# Main postprocessing pipeline
# ==========================================================================

def postprocess_instances(
    masks: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    orig_shape: Tuple[int, int],
    iou_threshold: float = 0.35,
    min_ratio: float = 20,
    dominant_ratio: float = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full postprocessing pipeline matching the reference demo app.

    Steps:
    1. Cross-class NMS  (remove overlapping masks from different classes)
    2. Per-instance polygon cleaning:
       a. mask → polygons
       b. filter small polygons
       c. find crux cordis center
       d. keep dominant polygon per instance
       e. polygon → mask
    """
    if len(masks) == 0:
        return masks, labels, scores

    # --- Step 1: Cross-class NMS ---
    masks, labels, scores = cross_class_nms(masks, labels, scores, iou_threshold)

    if len(masks) == 0:
        return masks, labels, scores

    h, w = orig_shape

    # --- Step 2a-b: Convert each mask to polygons, filter small ones ---
    instance_data = []  # list of (idx, polygons)
    all_polygons = []

    for idx in range(len(masks)):
        mask_uint8 = masks[idx].astype(np.uint8) * 255 if masks[idx].max() <= 1 else masks[idx]
        polys = mask_to_polygons(mask_uint8)
        if polys:
            polys = filter_small_polygons(polys, min_ratio=min_ratio)
        if polys:
            instance_data.append((idx, polys))
            all_polygons.extend(polys)

    if not instance_data:
        return masks, labels, scores

    # --- Step 2c: Find crux cordis ---
    crux = find_crux_cordis([all_polygons], canvas_shape=(h, w))

    # --- Step 2d-e: dominant polygon → new mask ---
    new_masks = []
    new_labels = []
    new_scores = []

    for idx, polys in instance_data:
        if len(polys) > 1 and crux is not None:
            dominant = find_dominant_polygon(polys, crux, ratio=dominant_ratio)
            if dominant is None:
                dominant = max(polys, key=lambda p: p.area)
            final_poly = dominant
        elif len(polys) > 1:
            final_poly = max(polys, key=lambda p: p.area)
        else:
            final_poly = polys[0]

        new_mask = polygons_to_mask(final_poly, height=h, width=w)
        if new_mask.any():
            new_masks.append(new_mask.astype(bool))
            new_labels.append(labels[idx])
            new_scores.append(scores[idx])

    if not new_masks:
        return (
            np.empty((0, h, w), dtype=bool),
            np.empty(0, dtype=labels.dtype),
            np.empty(0, dtype=scores.dtype),
        )

    return np.array(new_masks), np.array(new_labels), np.array(new_scores)
