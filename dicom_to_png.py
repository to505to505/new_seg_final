"""
Convert multi-frame DICOMs from NewSequentialData into per-sequence PNG folders.

Source structure:  {src}/{patient_id}/{seq_id}/*.dcm
Output structure:  {dst}/{patient_id}-{seq_id}/frame_0000.png  ...
"""

import sys
from pathlib import Path
import numpy as np
import pydicom
from PIL import Image

SRC = Path("/home/dsa/segmentation/segmentation_modules/MaskDino/data/NewSequentialData")
DST = Path("/home/dsa/new_seg_final/just_videos_newsequentialdata")


def convert_dicom(dcm_path: Path, out_dir: Path) -> int:
    ds = pydicom.dcmread(str(dcm_path))
    # Some files are missing PhotometricInterpretation — inject it
    if not hasattr(ds, "PhotometricInterpretation"):
        ds.PhotometricInterpretation = "MONOCHROME2"

    bits = int(ds.BitsAllocated)
    n_frames = int(ds.get("NumberOfFrames", 1))
    rows = int(ds.Rows)
    cols = int(ds.Columns)

    dtype = np.uint8 if bits == 8 else np.uint16
    raw = np.frombuffer(ds.PixelData, dtype=dtype)
    arr = raw.reshape(n_frames, rows, cols)

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(arr):
        if bits == 16:
            # Normalise to 8-bit for PNG
            mn, mx = frame.min(), frame.max()
            if mx > mn:
                frame = ((frame.astype(np.float32) - mn) / (mx - mn) * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        img = Image.fromarray(frame, mode="L")
        img.save(out_dir / f"frame_{i:04d}.png")

    return n_frames


def main():
    if not SRC.exists():
        print(f"ERROR: source not found: {SRC}", file=sys.stderr)
        sys.exit(1)

    DST.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted(p for p in SRC.iterdir() if p.is_dir())
    total_sequences = 0
    total_frames = 0

    for patient_dir in patient_dirs:
        for seq_dir in sorted(p for p in patient_dir.iterdir() if p.is_dir()):
            dcm_files = sorted(seq_dir.glob("*.dcm"))
            if not dcm_files:
                continue

            out_name = f"{patient_dir.name}-{seq_dir.name}"
            out_dir = DST / out_name

            # Use the first (usually only) DCM in the sequence folder
            dcm_path = dcm_files[0]
            print(f"  {patient_dir.name}/{seq_dir.name}  →  {out_name}/  ", end="", flush=True)
            n = convert_dicom(dcm_path, out_dir)
            print(f"{n} frames")
            total_sequences += 1
            total_frames += n

    print(f"\nDone: {total_sequences} sequences, {total_frames} frames total → {DST}")


if __name__ == "__main__":
    main()
