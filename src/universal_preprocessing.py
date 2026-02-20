"""
Universal CT Image Preprocessor.

Normalises ANY CT input (DICOM, JPG, PNG, NIfTI, numpy, 3-D volume, 2-D slice)
into a consistent representation that matches what the ResNet feature extractor
was trained on (LIDC-IDRI DICOM HU-windowed lung tissue).

Key steps:
  1. Format-agnostic loading   (delegates to load_2d_image / sitk)
  2. Auto-detect & apply lung HU window for DICOM; CLAHE + contrast stretch
     for non-DICOM (JPEG/PNG) so they approximate HU-windowed appearance.
  3. Auto-crop black borders   (common in screenshot CT images)
  4. Normalise to [0, 1]       (consistent range for ResNet preprocessing)
"""

import numpy as np
from pathlib import Path

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import SimpleITK as sitk
    HAS_SITK = True
except ImportError:
    HAS_SITK = False


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def universal_preprocess(image: np.ndarray,
                         source_format: str = "auto",
                         hu_window: tuple = (-1000, 400)) -> np.ndarray:
    """
    Convert any 2-D CT image array into a normalised [0, 1] grayscale array
    that approximates the appearance of HU-windowed LIDC-IDRI slices.

    Args:
        image:        (H, W) float or uint, already loaded from any format.
        source_format: "dicom" | "jpeg" | "png" | "auto"
                       Controls which contrast-enhancement path to take.
        hu_window:     (lo, hi) Hounsfield window for DICOM inputs.

    Returns:
        (H, W) float32 array in [0, 1].
    """
    img = image.copy().astype(np.float32)

    # ── 1.  Determine if the pixel values look like raw HU ──────────────
    is_hu = _looks_like_hounsfield(img) if source_format == "auto" else (source_format == "dicom")

    if is_hu:
        # Apply lung HU windowing
        lo, hi = hu_window
        img = np.clip(img, lo, hi)
        img = (img - lo) / (hi - lo + 1e-8)
    else:
        # Non-DICOM path: map to [0, 1] first
        if img.max() > 1.0:
            img = img / 255.0 if img.max() <= 255.0 else img / img.max()
        img = np.clip(img, 0.0, 1.0)

        # ── 2.  Auto-crop black/white borders ───────────────────────────
        img = _auto_crop(img)

        # ── 3.  CLAHE contrast enhancement ──────────────────────────────
        img = _enhance_contrast(img)

    # ── 4.  Final [0, 1] normalisation ──────────────────────────────────
    if img.max() > img.min():
        img = (img - img.min()) / (img.max() - img.min())
    else:
        img = np.zeros_like(img)

    return img.astype(np.float32)


def detect_source_format(file_path: str) -> str:
    """Infer source format from file extension."""
    ext = Path(file_path).suffix.lower()
    if ext in (".dcm", ".dicom"):
        return "dicom"
    elif ext in (".jpg", ".jpeg"):
        return "jpeg"
    elif ext in (".png",):
        return "png"
    elif ext in (".nii", ".nii.gz"):
        return "nifti"
    elif ext in (".npy", ".npz"):
        return "numpy"
    return "auto"


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _looks_like_hounsfield(img: np.ndarray) -> bool:
    """
    Heuristic: raw HU arrays typically have negative values (air = -1000)
    and/or values >> 255.  Screenshot JPEGs are in [0, 255].
    """
    if img.min() < -100:
        return True
    if img.max() > 2000:
        return True
    return False


def _auto_crop(img: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Remove black borders where the row/column mean is below *threshold*."""
    row_means = img.mean(axis=1)
    col_means = img.mean(axis=0)
    row_mask = row_means > threshold
    col_mask = col_means > threshold
    if row_mask.any() and col_mask.any():
        r0, r1 = np.where(row_mask)[0][[0, -1]]
        c0, c1 = np.where(col_mask)[0][[0, -1]]
        if (r1 - r0) > 30 and (c1 - c0) > 30:
            img = np.ascontiguousarray(img[r0:r1 + 1, c0:c1 + 1])
    return img


def _enhance_contrast(img: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (OpenCV) or percentile stretch (fallback) to approximate the
    appearance of HU-windowed DICOM images.
    """
    if HAS_CV2:
        try:
            img_u8 = np.ascontiguousarray(
                np.clip(img * 255, 0, 255).astype(np.uint8)
            )
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img_u8 = clahe.apply(img_u8)
            return img_u8.astype(np.float32) / 255.0
        except Exception:
            pass  # fall through to percentile stretch

    # Pure-numpy fallback — always works
    p2, p98 = np.percentile(img, (2, 98))
    if p98 > p2:
        img = np.clip((img - p2) / (p98 - p2), 0.0, 1.0)
    return img
