"""Rank the bundled real images by how much signal they give the R-CNN models.

With the project running a SINGLE trial, only one image is ever used: the
first file in sorted order inside the image directory. Run this once on a
machine with torch/torchvision (weights download on first run):

    python3 -m precision_matching.check_images assets/real_images

and keep the suggested image first (or alone) in the directory. A good image
must yield many confident Mask R-CNN detections AND several persons --
Keypoint R-CNN only detects people, so a people-free image is a blank sheet
for it no matter how rich it looks.
"""

from __future__ import annotations

import os
import sys


def main(image_dir):
    import torch
    from torchvision.models.detection import (
        KeypointRCNN_ResNet50_FPN_Weights,
        MaskRCNN_ResNet50_FPN_V2_Weights,
        keypointrcnn_resnet50_fpn,
        maskrcnn_resnet50_fpn_v2,
    )

    from .real_images import load_image

    mask_model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    ).eval()
    kpt_model = keypointrcnn_resnet50_fpn(
        weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT
    ).eval()

    files = sorted(
        f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not files:
        raise SystemExit(f"no images in {image_dir}")

    print(f"{'file':<24}{'maskrcnn boxes>=0.5':>20}{'persons>=0.5':>14}")
    results = []
    for f in files:
        x = torch.from_numpy(load_image(os.path.join(image_dir, f), (1, 3, 800, 800)))
        with torch.no_grad():
            det = mask_model(x)[0]
            kpt = kpt_model(x)[0]
        n_boxes = int((det["scores"] >= 0.5).sum())
        n_persons = int((kpt["scores"] >= 0.5).sum())
        results.append((f, n_boxes, n_persons))
        print(f"{f:<24}{n_boxes:>20}{n_persons:>14}")

    # An image with no persons is useless for keypointrcnn, however box-rich.
    best = max(results, key=lambda r: (r[2] > 0, r[1] + 2 * r[2]))
    print(f"\nsuggested single-trial image: {best[0]} "
          f"({best[1]} boxes, {best[2]} persons)")
    if best[2] == 0:
        print("WARNING: no image has confident persons -- add a people-rich "
              "image (crowd/sports scene) before trusting keypointrcnn numbers")
    elif best[0] != files[0]:
        print(f"NOTE: {files[0]} sorts first and would be used as-is; keep only "
              f"{best[0]} in the directory (or make it sort first)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "assets/real_images")
