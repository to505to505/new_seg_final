"""Severity estimation from binary segmentation masks.

Thin wrapper around ODySSeI's existing compute_per_lesion_severity()
providing a cleaner interface for the pipeline.
"""

import cv2
import numpy as np
from scipy.signal import find_peaks
from skimage.morphology import skeletonize


def compute_severity(binary_mask: np.ndarray) -> dict:
    """Estimate stenosis severity from a binary vessel segmentation mask.

    Uses skeleton + distance transform to compute:
    - MLD (Minimum Lumen Diameter): narrowest point between reference points
    - DS (Diameter Stenosis): 1 - MLD / max_diameter

    Args:
        binary_mask: (H, W) uint8 array with values {0, 1}

    Returns:
        dict with keys: 'mld', 'ds', 'skeleton', 'diameter_profile'
        mld/ds may be NaN if the mask is too small or has no valid skeleton.
    """
    result = {"mld": float("nan"), "ds": float("nan"), "skeleton": None, "diameter_profile": None}

    if binary_mask.sum() < 10:
        return result

    mask = binary_mask.astype(np.uint8)

    # Distance transform — distance from each foreground pixel to nearest edge
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dist = dist.T  # match ODySSeI convention

    # Skeletonize
    skel = skeletonize(mask)
    result["skeleton"] = skel

    # Get skeleton points sorted by x-coordinate
    pts = np.column_stack(np.where(skel.T == 1))
    if len(pts) < 3:
        return result

    sorted_pts = sorted(pts.tolist(), key=lambda p: p[0])

    # Compute diameter at each skeleton point (2 × distance = diameter)
    diameters = np.array([2 * dist[int(p[0]), int(p[1])] for p in sorted_pts])
    result["diameter_profile"] = diameters

    # Find peaks (local maxima = reference diameters)
    peaks, _ = find_peaks(diameters)

    if len(peaks) == 0:
        return result
    elif len(peaks) == 1:
        mld = float(np.min(diameters[peaks[0]:]))
    else:
        mld = float(np.min(diameters[peaks[0]:peaks[-1]]))

    max_d = float(np.max(diameters))
    ds = 1.0 - (mld / max_d) if max_d > 0 else float("nan")

    result["mld"] = mld
    result["ds"] = ds
    return result
