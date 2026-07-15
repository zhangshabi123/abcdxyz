"""Real-image inputs for detection models.

On uniform noise a detector emits ~0 boxes, so reference and candidate
trivially agree on "nothing" and the detection path is never measured.
These models therefore take a handful of real COCO val2017 images instead.

Usage (one-time, needs network):
    python3 -m precision_matching.real_images assets/real_images
Then in run_inference, for trial i:
    img = load_image_for_trial("assets/real_images", trial_index=i,
                               shape=(1, 3, 800, 800))
Everything is deterministic: trial i always maps to the same file
(sorted order, cycled), so reference and candidate see the same pixels.
"""

from __future__ import annotations

import os
import urllib.request

import numpy as np

# Fixed COCO val2017 ids -- multi-object scenes, stable official URLs.
COCO_VAL2017_IDS = [139, 285, 632, 724, 776, 39769]
_URL = "http://images.cocodataset.org/val2017/{:012d}.jpg"


def fetch_coco_images(out_dir, ids=None):
    """Download the pinned COCO images into out_dir (skips existing files).
    Returns the list of local paths. Needs network; run once and keep the
    files with the harness so evaluation itself stays offline."""
    ids = ids if ids is not None else COCO_VAL2017_IDS
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i in ids:
        path = os.path.join(out_dir, f"{i:012d}.jpg")
        if not os.path.exists(path):
            urllib.request.urlretrieve(_URL.format(i), path)
        with open(path, "rb") as f:
            if f.read(2) != b"\xff\xd8":  # JPEG magic
                raise IOError(f"{path} is not a valid JPEG, delete and re-fetch")
        paths.append(path)
    return paths


def load_image(path, shape, low=0.0, high=1.0, dtype="float32"):
    """Load one image as an array of exactly `shape` = (B, 3, H, W), resized
    with bilinear interpolation, pixel values scaled to [low, high].
    torchvision detection models expect (B, 3, H, W) in [0, 1] -- the default."""
    from PIL import Image

    b, c, h, w = (int(s) for s in shape)
    if c != 3:
        raise ValueError(f"expected 3-channel image shape, got {shape}")
    with Image.open(path) as im:
        im = im.convert("RGB").resize((w, h), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float64) / 255.0
    arr = (low + arr * (high - low)).transpose(2, 0, 1)  # HWC -> CHW
    return np.broadcast_to(arr, (b, c, h, w)).astype(dtype)


def load_image_for_trial(image_dir, trial_index, shape, **kwargs):
    """Deterministically pick the image for a trial: sorted files, cycled."""
    files = sorted(
        f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not files:
        raise FileNotFoundError(
            f"no images in {image_dir}; run fetch_coco_images first"
        )
    path = os.path.join(image_dir, files[trial_index % len(files)])
    return load_image(path, shape, **kwargs)


if __name__ == "__main__":
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else "assets/real_images"
    for p in fetch_coco_images(out):
        print("ok", p)
